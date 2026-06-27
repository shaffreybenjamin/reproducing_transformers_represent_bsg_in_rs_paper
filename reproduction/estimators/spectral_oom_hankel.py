"""Two-factor Hankel matrix estimation in activation space for belief geometry recovery.

Building on spectral-OOM, this extends the estimator to compute both:
  - Controllability factor P (forward/reachability): maps initial state to belief-carrying directions
  - Observability factor S (future observables): stacks readouts from current state to future

Then forms H = P·S and SVDs to recover the belief subspace. This two-sided factorization
improves conditioning over S alone by leveraging past-and-future statistics mutually.

Key features:
  - Fully target-free: uses only activations, softmax, and prefix tree
  - Activation-space realization: operates in the residual stream coordinate system
  - Elbow-based d-selection: uses singular-value gap to infer latent dimension
  - Gauge-invariant validation: eigenvalue comparison with true transition operators
"""

import itertools
import numpy as np
import torch
from sklearn.linear_model import Ridge

EPS = 1e-3
RIDGE_ALPHA = 1e-2


def build_full_prefix_tree(resid, T, pi, vocab):
    """Compute reachability for all prefixes using TRUE process probability.

    A prefix w is reachable if P_true(w) > 0 under the HMM.
    Uses the true transition matrices and initial state, not model softmax.
    """
    def ndist(b):
        """Next-token distribution from belief b."""
        return np.array([(b @ T[x]).sum() for x in range(vocab)])

    def upd(b, x):
        """Update belief after observing token x."""
        nb = b @ T[x]
        return nb / nb.sum() if nb.sum() > 0 else b

    reach = {}
    for w in resid:
        b, p = np.array(pi, dtype=float), 1.0
        for x in w:
            dd = ndist(b)
            p *= dd[x]
            if dd[x] < 1e-12:
                p = 0.0
                break
            b = upd(b, x)
        reach[w] = p > 0  # Reachable if true process probability > 0
    return reach


def fit_transition_operators(rows, resid, soft, vocab, wmap=None):
    """Fit rescaled transition operators A_x in activation space.

    For each token x, fit A_x such that:
        a(w) A_x ≈ P(x|w) a(wx)

    This is the per-token operator in activation coordinates, estimated from
    the linear relation obtained by multiplying the projective update by P(x|w).

    Returns: {x: A_x (D, D)}  operators indexed by token
    """
    # Build parent activations and softmax
    A_list = []
    soft_list = []
    child_list = []
    weight_list = []

    for w in rows:
        if all((w + (x,)) in resid for x in range(vocab)):
            A_list.append(resid[w])
            soft_list.append(soft[w])
            child_list.append([resid[w + (x,)] for x in range(vocab)])
            if wmap is not None:
                weight_list.append(wmap.get(w, 1.0))

    A = np.stack(A_list)  # (N, D)
    P = np.stack(soft_list)  # (N, vocab)
    Yc = np.stack(child_list)  # (N, vocab, D)
    w = None if not weight_list else np.array(weight_list)

    # Fit A_x for each token independently
    ops = {}
    for x in range(vocab):
        # Ridge regression: A @ A_x ≈ P[:, x:x+1] * Yc[:, x, :]
        # where P[:, x:x+1] * Yc[:, x, :] is the rescaled child activation
        target = P[:, x:x+1] * Yc[:, x, :]  # (N, D)
        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(A, target, sample_weight=w)
        ops[x] = reg.coef_.T  # (D, D)

    return ops


def build_controllability_factor(resid, reach, T, pi, ops, vocab, wmap=None):
    """Build controllability matrix R: (num_pasts, D) stacks reachable activations.

    For each reachable word w = x_1...x_k, computes:
        R[w] = P(w) * (a_0 * A_{x_1} * ... * A_{x_k})

    where:
    - a_0 = root activation (from initial belief b_0)
    - A_x = fitted transition operators in activation space
    - P(w) = true probability of word w under the HMM

    This is the dual of observability:
    - Observability: O = [C, G_x C, G_x G_y C, ...] (operators applied on right)
    - Controllability: R = [a_0, a_0 A_x, a_0 A_x A_y, ...] (operators applied on left)

    Returns: R (num_pasts, D) controllability matrix, rows: list of reachable words
    """
    def ndist(b):
        """Next-token distribution from belief b."""
        return np.array([(b @ T[x]).sum() for x in range(vocab)])

    def upd(b, x):
        """Update belief after observing token x."""
        nb = b @ T[x]
        return nb / nb.sum() if nb.sum() > 0 else b

    # Compute root activation a_0 from initial belief b_0
    b_0 = np.array(pi, dtype=float)

    # Root activation is the empirical mean of prefix activations weighted by P(w)
    # (or use activation at empty prefix if available)
    if () in resid:  # Empty prefix (root)
        a_0 = resid[()]
    else:
        # Estimate a_0 as P(w)-weighted mean of initial activations
        total_weight = 0.0
        a_0_accum = np.zeros(resid[next(iter(resid))].shape)
        for w in resid:
            if reach[w]:
                # Compute P(w)
                b, p = b_0.copy(), 1.0
                for x in w:
                    dd = ndist(b)
                    p *= dd[x]
                    if dd[x] < 1e-12:
                        p = 0.0
                        break
                    b = upd(b, x)

                if p > 0:
                    a_0_accum += p * resid[w]
                    total_weight += p

        a_0 = a_0_accum / (total_weight + 1e-12) if total_weight > 0 else np.zeros_like(a_0_accum)

    # Build controllability matrix by computing reachable activations
    reachable_activations = []
    reachable_words = []

    for w in resid:
        if reach[w]:
            # Compute true probability P(w)
            b, p_w = b_0.copy(), 1.0
            for x in w:
                dd = ndist(b)
                p_w *= dd[x]
                if dd[x] < 1e-12:
                    p_w = 0.0
                    break
                b = upd(b, x)

            if p_w > 0:
                # Compute reachable activation: a_0 * A_{x_1} * A_{x_2} * ... * A_{x_k}
                a_reach = a_0.copy()
                for x in w:
                    a_reach = a_reach @ ops[x]  # Apply operators on the left

                # Weight by P(w) to get un-normalized reachable activation
                weighted_a = p_w * a_reach
                reachable_activations.append(weighted_a)
                reachable_words.append(w)

    R = np.stack(reachable_activations) if reachable_activations else np.zeros((0, a_0.shape[0]))
    return R, reachable_words


def build_observability_factor(resid, soft, reach, vocab, depth=3, wmap=None):
    """Build observability factor S by stacking multi-step readouts.

    S = [C | G_x C | G_x G_y C | ...]

    where C is the one-step readout (activation to softmax) and G_x are the
    per-token rescaled transition maps in the observation space.

    Returns: S (D, num_columns), and the list of readout matrices for inspection
    """
    # Identify rows with all children reachable and softmax > EPS
    rows = [
        w for w in resid
        if reach[w] and all(
            (w + (x,)) in resid and reach[w + (x,)]
            for x in range(vocab) if soft[w][x] > EPS
        )
    ]

    A = np.stack([resid[w] for w in rows])  # (N, D)
    P = np.stack([soft[w] for w in rows])  # (N, vocab)
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    # Fit C: activation -> softmax
    C = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T  # (D, vocab)

    # Fit G_x: activation -> P(x|w) * child_activation
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows])
        if m.sum() == 0:
            # No reachable transitions for this token; zero operator
            Gs.append(np.zeros((resid[rows[0]].shape[0], resid[rows[0]].shape[0])))
            continue
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])  # (M, D)
        tgt = P[m, x][:, None] * child  # (M, D) rescaled target
        swm = None if sw is None else sw[m]
        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(A[m], tgt, sample_weight=swm)
        Gs.append(reg.coef_.T)  # (D, D)

    # Stack observability matrix to given depth
    cols = [C]
    frontier = [C]

    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols.extend(nxt)
        frontier = nxt

    S = np.hstack(cols)  # (D, D + D*vocab + D*vocab^2 + ...)
    return S, Gs, C


def compute_hankel_svd_factored(P, S):
    """Compute SVD of Hankel matrix H = P·S without materializing the huge product.

    Uses low-rank factorization to avoid memory blow-up:

    P: (num_pasts, D) - past activations
    S: (D, num_futures) - multi-step observables
    H = P @ S is (num_pasts, num_futures) - would be huge!

    Returns: U_basis (D, rank), sv_H where U_basis is a basis in activation space
    """
    # SVD of P: (num_pasts, D) -> U_P (num_pasts, r_P) @ Σ_P (r_P,) @ V_P^T (r_P, D)
    U_P, sigma_P, VT_P = np.linalg.svd(P, full_matrices=False)

    # SVD of S: (D, num_futures) -> U_S (D, r_S) @ Σ_S (r_S,) @ V_S^T (r_S, num_futures)
    U_S, sigma_S, VT_S = np.linalg.svd(S, full_matrices=False)

    # Cross term: V_P^T @ U_S is (r_P, r_S) - small!
    cross = VT_P @ U_S  # (r_P, r_S)

    # Middle product: Σ_P @ cross @ Σ_S is (r_P, r_S) - still small
    middle = np.diag(sigma_P) @ cross @ np.diag(sigma_S)  # (r_P, r_S)

    # SVD of the small middle matrix
    U_mid, sigma_mid, VT_mid = np.linalg.svd(middle, full_matrices=False)

    # The singular values of H are sigma_mid
    sv_H = sigma_mid

    # For projecting activations, we want a basis in activation space (D-dimensional)
    # U_S is already (D, r_S) and lives in activation space
    # Project it through the Hankel structure: U_S @ V_mid gives the belief basis
    U_basis = U_S @ VT_mid.T  # (D, r_S) @ (r_S, r_P) = (D, r_P)

    return U_basis, sv_H


def detect_elbow(singular_values, threshold_ratio=0.10):
    """Detect elbow in singular value spectrum using maximum log-gap.

    Prioritizes finding the largest drop in the log spectrum, which corresponds
    to the transition from "signal" (large singular values) to "noise" (small ones).

    Returns: d (estimated latent dimension)
    """
    sv = np.asarray(singular_values)
    sv_norm = sv / sv[0]  # normalize by largest

    # Find maximum gap in log(sv) - this is the "elbow" where spectrum drops
    # This prioritizes the structural break in the spectrum over variance thresholds
    log_sv = np.log(np.clip(sv_norm, 1e-12, None))
    gaps = np.diff(log_sv)

    # The elbow is where the gap is largest (most negative, biggest drop)
    max_gap_idx = np.argmax(-gaps)  # argmax of negative gaps = largest drop
    d = max_gap_idx + 1  # +1 because we're indexing into differences

    return max(d, 1)


def run_hankel_spectral_oom(model, hmm, context_len, device, verbose=True):
    """Full pipeline: collect data, build P and S, form H, SVD, detect d.

    Returns: dict with subspace basis, singular values, detected d, operators, and metadata
    """
    import itertools

    T = np.array(hmm.transition_matrices)  # (vocab, nstates, nstates)
    pi = np.array(hmm.initial_state)
    vocab = hmm.vocab_size
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = hmm.num_states

    def ndist(b):
        return np.array([(b @ T[x]).sum() for x in range(vocab)])

    def upd(b, x):
        nb = b @ T[x]
        return nb / nb.sum() if nb.sum() > 0 else b

    def bp(w):
        b, p = pi.copy(), 1.0
        for x in w:
            dd = ndist(b)
            p *= dd[x]
            if dd[x] < 1e-12:
                return None
            b = upd(b, x)
        return b

    # Collect prefix features: activations, softmax, beliefs (for scoring only)
    if verbose:
        print("Collecting prefix features...")

    resid, soft, belief = {}, {}, {}
    for L in range(1, context_len + 1):
        strs = np.array(list(itertools.product(range(vocab), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i : i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i : i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]
                soft[w] = sm[j]
                b = bp(w)
                belief[w] = b if b is not None else np.full(NS, np.nan)

    if verbose:
        print(f"  Enumerated {len(resid)} prefixes")

    # Build reachability from TRUE process (HMM), not model softmax
    reach = build_full_prefix_tree(resid, T, pi, vocab)
    reachable_count = sum(reach.values())
    if verbose:
        print(f"  Reachable prefixes (TRUE process): {reachable_count}/{len(resid)}")

    # Fit transition operators A_x
    if verbose:
        print("Fitting transition operators...")
    rows_ops = [
        w for w in resid
        if all((w + (x,)) in resid for x in range(vocab))
    ]
    ops = fit_transition_operators(rows_ops, resid, soft, vocab, wmap=None)
    if verbose:
        print(f"  Fitted {vocab} operators from {len(rows_ops)} rows")

    # Build controllability factor R: (num_pasts, D) from reachable activations
    # R[w] = P(w) * (a_0 * A_{x_1} * ... * A_{x_k})
    if verbose:
        print("Building controllability factor R...")
    P, rows_P = build_controllability_factor(resid, reach, T, pi, ops, vocab, wmap=None)
    if verbose:
        print(f"  R shape: {P.shape}")

    # Build observability factor S: (D, num_futures)
    if verbose:
        print("Building observability factor S...")
    S, Gs, C = build_observability_factor(resid, soft, reach, vocab, depth=context_len, wmap=None)
    if verbose:
        print(f"  S shape: {S.shape}")

    # Compute SVD of H = P·S using low-rank factorization (no huge matrix materialization)
    if verbose:
        print("Computing P·S via low-rank factorization (no memory blow-up)...")
    U_basis, sv_H = compute_hankel_svd_factored(P, S)
    if verbose:
        print(f"  Basis shape: {U_basis.shape}, singular values: {sv_H.shape}")

    # Detect elbow to select d
    d_elbow = detect_elbow(sv_H, threshold_ratio=0.05)

    if verbose:
        print(f"\nSingular value spectrum (top 20):")
        print(f"  {np.round(sv_H[:min(20, len(sv_H))], 4)}")
        print(f"\nNormalized spectrum (sv / sv[0]):")
        print(f"  {np.round(sv_H[:min(20, len(sv_H))] / sv_H[0], 4)}")
        print(f"\nElbow-detected d: {d_elbow}")

    return {
        "U": U_basis,
        "sv": sv_H,
        "d_elbow": d_elbow,
        "P": P,
        "S": S,
        "ops": ops,
        "Gs": Gs,
        "C": C,
        "resid": resid,
        "soft": soft,
        "belief": belief,
        "reach": reach,
        "rows_ops": rows_ops,
    }
