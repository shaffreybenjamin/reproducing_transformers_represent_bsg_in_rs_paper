"""Minimal spectral-OOM using observability matrix only (no Hankel two-factor).

This is a sanity check to verify the observability matrix implementation.
If this produces the same results as the Hankel version, the implementation is sound.
If not, there's a bug to identify.
"""

import itertools
import numpy as np
import torch
from sklearn.linear_model import Ridge

EPS = 1e-3
RIDGE_ALPHA = 1e-2


def build_full_prefix_tree(resid, T, pi, vocab):
    """Compute reachability for all prefixes using TRUE process probability."""
    def ndist(b):
        return np.array([(b @ T[x]).sum() for x in range(vocab)])

    def upd(b, x):
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
        reach[w] = p > 0
    return reach


def fit_transition_operators(rows, resid, soft, vocab, wmap=None):
    """Fit rescaled transition operators A_x in activation space."""
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

    A = np.stack(A_list)
    P = np.stack(soft_list)
    Yc = np.stack(child_list)
    w = None if not weight_list else np.array(weight_list)

    ops = {}
    for x in range(vocab):
        target = P[:, x:x+1] * Yc[:, x, :]
        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(A, target, sample_weight=w)
        ops[x] = reg.coef_.T

    return ops


def build_observability_matrix(resid, soft, reach, vocab, depth=3, wmap=None):
    """Build observability matrix S directly (no controllability factor).

    S = [C | G_x C | G_x G_y C | ...]

    Then SVD on S directly to get singular values.
    """
    rows = [
        w for w in resid
        if reach[w] and all(
            (w + (x,)) in resid and reach[w + (x,)]
            for x in range(vocab) if soft[w][x] > EPS
        )
    ]

    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    # Fit C: activation -> softmax
    C = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T

    # Fit G_x: activation -> P(x|w) * child_activation
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows])
        if m.sum() == 0:
            Gs.append(np.zeros((resid[rows[0]].shape[0], resid[rows[0]].shape[0])))
            continue
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else sw[m]
        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(A[m], tgt, sample_weight=swm)
        Gs.append(reg.coef_.T)

    # Stack observability matrix to given depth
    cols = [C]
    frontier = [C]

    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols.extend(nxt)
        frontier = nxt

    S = np.hstack(cols)
    return S, Gs, C


def detect_elbow(sv):
    """Detect elbow using log-gap method."""
    if len(sv) < 2:
        return 1

    log_sv = np.log(sv + 1e-12)
    gaps = np.diff(log_sv)

    if len(gaps) == 0:
        return 1

    max_gap_idx = np.argmax(-gaps)
    d = max_gap_idx + 1
    return min(d, len(sv) - 1)


def run_spectral_oom_observability_only(resid, soft, T, pi, vocab, true_d=None, wmap=None):
    """Run spectral-OOM using only observability matrix.

    Returns:
      - sv: singular values from SVD of observability matrix
      - d_detected: detected dimensionality
      - d_true: ground truth dimensionality
    """
    print(f"Building prefix tree and reachability...")
    reach = build_full_prefix_tree(resid, T, pi, vocab)

    # Identify rows for operator fitting
    rows = [w for w in resid if reach[w]]
    rows.sort(key=lambda w: (len(w), w))
    rows = rows[:8000]  # Limit for tractability

    print(f"Fitting transition operators from {len(rows)} rows...")
    ops = fit_transition_operators(rows, resid, soft, vocab, wmap=wmap)

    print(f"Building observability matrix...")
    S, Gs, C = build_observability_matrix(resid, soft, reach, vocab, depth=3, wmap=wmap)

    print(f"Observability matrix shape: {S.shape}")
    print(f"Computing SVD of observability matrix...")

    # Direct SVD on S
    U, sv, _ = np.linalg.svd(S, full_matrices=False)

    print(f"\nSingular value spectrum (top 20):")
    print(f"  {sv[:20]}")

    print(f"\nNormalized spectrum (sv / sv[0]):")
    normalized = sv / (sv[0] + 1e-12)
    print(f"  {normalized[:20]}")

    # Detect elbow
    d_detected = detect_elbow(sv)

    print(f"\nElbow-detected d: {d_detected}")
    if true_d is not None:
        print(f">>> Elbow-detected d: {d_detected}   True d: {true_d}   Match: {d_detected == true_d}")

    return sv, d_detected, U, C, ops, S


def run_hankel_spectral_oom(model, hmm, context_len, device, verbose=True):
    """Full pipeline matching hankel module interface: collect data, build S, SVD, detect d.

    Returns: dict with basis, singular values, detected d, and metadata (compatible with Hankel module).
    """
    import itertools

    T = np.array(hmm.transition_matrices)
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

    # Collect prefix features
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

    # Build reachability
    reach = build_full_prefix_tree(resid, T, pi, vocab)
    if verbose:
        reachable_count = sum(reach.values())
        print(f"  Reachable prefixes (TRUE process): {reachable_count}/{len(resid)}")

    # Fit transition operators
    if verbose:
        print("Fitting transition operators...")
    rows_ops = [w for w in resid if all((w + (x,)) in resid for x in range(vocab))]
    ops = fit_transition_operators(rows_ops, resid, soft, vocab, wmap=None)
    if verbose:
        print(f"  Fitted {vocab} operators from {len(rows_ops)} rows")

    # Build observability matrix only (no controllability factor)
    # Use limited depth to keep spectrum manageable for elbow detection
    # Deeper matrices (depth=context_len) yield too many singular values with no clear structure
    if verbose:
        print("Building observability matrix...")
    S, Gs, C = build_observability_matrix(resid, soft, reach, vocab, depth=3, wmap=None)
    if verbose:
        print(f"  S shape: {S.shape}")

    # Direct SVD on observability matrix (no Hankel factorization)
    if verbose:
        print("Computing SVD of observability matrix...")
    U_S, sv_S, _ = np.linalg.svd(S, full_matrices=False)

    if verbose:
        print(f"\nSingular value spectrum (top 20):")
        print(f"  {sv_S[:min(20, len(sv_S))]}")
        print(f"\nNormalized spectrum (sv / sv[0]):")
        print(f"  {sv_S[:min(20, len(sv_S))] / sv_S[0]}")

    # Detect elbow
    d_elbow = detect_elbow(sv_S)
    if verbose:
        print(f"\nElbow-detected d: {d_elbow}")

    return {
        "d_elbow": d_elbow,
        "U": U_S,
        "sv": sv_S,
        "belief": belief,
        "resid": resid,
        "soft": soft,
        "ops": ops,
        "S": S,
    }


if __name__ == "__main__":
    import pickle
    import sys

    # Example usage - this will be called from individual process scripts
    print("Spectral-OOM observability-only module loaded")
