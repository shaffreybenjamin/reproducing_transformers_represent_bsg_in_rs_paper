"""CCA / Reduced-Rank Regression estimator: direct multi-step readouts + HKZ-style whitening.

This implements Suggestions 1+2 from the Claude research report: instead of composing
estimated operators (which compounds error multiplicatively), fit each multi-step
readout C_k directly to empirically measured future-token distributions P(x_{t+1:t+k}|w),
then stack [C_1 | C_2 | ... | C_L] and whiten by past-future covariance (proper CCA/RRR).

Pipeline:
  1. Enumerate future-token distributions: for each prefix w and horizon k, compute
     P(x_{t+1:t+k}|w) empirically by enumerating continuations through the model softmax.
  2. Direct multi-step readouts: fit C_k by ridge regression a(w) @ C_k ≈ P_k(futures|w)
     independently for each k, no operator composition.
  3. Stack observability matrix: O = [C_1 | C_2 | ... | C_L]
  4. Whiten by past-future covariance (HKZ/CCA style): compute generalized SVD that
     scores directions by belief-relevant (past-future) variance, not absolute variance.
  5. Recover subspace, fit operators, evaluate geometry.

Key differences from fig14_observable_oom.py:
  - No operator composition: C_k is fit directly to true future distributions.
  - Proper CCA whitening: whitens by past-future correlation, not input covariance alone.
  - Expect horizon-based improvement: RRXOR's degenerate beliefs separate at horizon ≈2-3
    in the future distribution, so L=2,3,4 should show improvement vs flat plateau.
"""

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.transition_matrices import mess3, rrxor

MAX_LEN = 10
EPS = 1e-3            # model-softmax reachability threshold
RIDGE = 1e-2
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def enumerate_future_dist(soft, w, vocab, max_horizon=4):
    """Enumerate empirical P(x_{t+1:t+k}|w) for horizons k=1..max_horizon.

    Returns dict {k: future_probs} where future_probs is shape (n_distinct_futures_k, ),
    indexed by the flattened tuple of k tokens, e.g. (0,0), (0,1), (1,0), (1,1) for k=2, vocab=2.

    For efficiency: only enumerate futures that are reachable (P > EPS) from w in the model.
    """
    out = {}
    for k in range(1, max_horizon + 1):
        # Enumerate all length-k continuations
        future_dist = {}
        for future in itertools.product(range(vocab), repeat=k):
            # Check reachability and compute joint probability P(future | w)
            curr_prefix = w
            prob = 1.0
            reachable = True
            for token in future:
                if curr_prefix not in soft or soft[curr_prefix][token] <= EPS:
                    reachable = False
                    break
                prob *= soft[curr_prefix][token]
                curr_prefix = curr_prefix + (token,)
            if reachable and prob > 0:
                future_dist[future] = prob
        out[k] = future_dist
    return out


def direct_multistep_readouts(resid, soft, reach, vocab, max_horizon=4, wmap=None):
    """Build multi-step observables via direct regression onto empirical future distributions.

    Returns:
      rows: list of prefixes used
      A: activations (n_rows × D)
      P: next-token softmax (n_rows × vocab)
      C_ks: list of fitted C_k matrices for each horizon k
      O: stacked observability matrix [C_1 | C_2 | ... | C_L] (D × total_feature_width)
      sv: singular values of O
      U: left singular vectors of O (D × D)
    """
    # Filter to reachable prefixes with valid children at each horizon
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C_ks = []
    future_collections = []

    for k in range(1, max_horizon + 1):
        # Enumerate futures for each row and build future-distribution matrix
        futures_per_prefix = []
        future_vocab_list = []  # track which futures appear

        for i, w in enumerate(rows):
            future_dists = enumerate_future_dist(soft, w, vocab, max_horizon=k)
            futures_per_prefix.append(future_dists[k])
            if i == 0:  # collect all possible futures from first row to establish ordering
                future_vocab_list = sorted(futures_per_prefix[0].keys())

        # Standardize: use the union of all futures seen
        all_futures = set()
        for fp in futures_per_prefix:
            all_futures.update(fp.keys())
        future_vocab_list = sorted(all_futures)

        # Build matrix: each row is a prefix, each column is a future, entries are P(future|w)
        Phi_k = np.zeros((len(rows), len(future_vocab_list)))
        for i, fp in enumerate(futures_per_prefix):
            for j, future in enumerate(future_vocab_list):
                Phi_k[i, j] = fp.get(future, 0.0)

        # Fit C_k: A @ C_k ≈ Phi_k
        C_k = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, Phi_k, sample_weight=sw).coef_.T  # (D, #futures_k)
        C_ks.append(C_k)
        future_collections.append(Phi_k)

    # Stack all C_k into observability matrix O
    O = np.hstack(C_ks)  # (D, sum of #futures across horizons)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)

    return rows, A, P, C_ks, O, U, sv, future_collections, sw


def cca_rrr_subspace(A, future_collections, ridge_cca=0.01, sw=None):
    """Compute CCA/reduced-rank regression subspace via past-future whitening.

    Instead of plain SVD of O, whiten by past and future covariances to score
    directions by belief-relevant (past-future correlation) variance.

    Args:
      A: past activations (n × D)
      future_collections: list of Phi_k matrices (each n × feature_width_k)
      ridge_cca: ridge parameter for covariance inversion
      sw: optional sample weights (n,) for P(w)-weighted computation

    Returns:
      U_wh: whitened past basis (D × D) ranked by past-future correlation
      sv: singular values of the whitened correlation
    """
    n = A.shape[0]

    # Stack all futures
    Phi = np.hstack(future_collections)  # (n, total_features)

    # Compute weighted mean and centering
    if sw is not None:
        sw_sum = np.sum(sw)
        A_mean = np.average(A, axis=0, weights=sw)
        Phi_mean = np.average(Phi, axis=0, weights=sw)
        A_cent = A - A_mean
        Phi_cent = Phi - Phi_mean
        # Weighted covariances
        A_w = A_cent * np.sqrt(sw)[:, None]
        Phi_w = Phi_cent * np.sqrt(sw)[:, None]
        Cov_aa = A_w.T @ A_w / sw_sum
        Cov_pp = Phi_w.T @ Phi_w / sw_sum
        Cov_ap = A_w.T @ Phi_w / sw_sum
    else:
        A_cent = A - A.mean(0)
        Phi_cent = Phi - Phi.mean(0)
        Cov_aa = A_cent.T @ A_cent / n
        Cov_pp = Phi_cent.T @ Phi_cent / n
        Cov_ap = A_cent.T @ Phi_cent / n

    # Add ridge for numerical stability
    Cov_aa_ridge = Cov_aa + ridge_cca * np.trace(Cov_aa) / Cov_aa.shape[0] * np.eye(Cov_aa.shape[0])
    Cov_pp_ridge = Cov_pp + ridge_cca * np.trace(Cov_pp) / Cov_pp.shape[0] * np.eye(Cov_pp.shape[0])

    # Compute whitening matrices
    val_a, vec_a = np.linalg.eigh(Cov_aa_ridge)
    val_p, vec_p = np.linalg.eigh(Cov_pp_ridge)

    # Whitening: Sigma^{-1/2}
    Whiten_a = vec_a @ np.diag(1.0 / np.sqrt(np.clip(val_a, 1e-12, None))) @ vec_a.T
    Whiten_p = vec_p @ np.diag(1.0 / np.sqrt(np.clip(val_p, 1e-12, None))) @ vec_p.T

    # Whitened correlation matrix
    M = Whiten_a @ Cov_ap @ Whiten_p

    # SVD of whitened correlation
    U_w, sv, V_w = np.linalg.svd(M, full_matrices=False)

    # Transform back to original space: U_wh @ sv @ V_wh^T = whitened correlation
    # For regression, we want the left singular vectors in the original space
    U_wh = Whiten_a @ U_w

    return U_wh, sv


def fit_at_dim(rows, A, P, resid, vocab, B):
    """Project to subspace B (D x d), fit operators A_x and eval functional e."""
    s_all = A @ B
    ops = {}
    for x in range(vocab):
        m = np.array([P[i, x] > EPS for i in range(len(rows))])
        sw = s_all[m]
        sc = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]]) @ B
        ops[x] = np.linalg.lstsq(sw, P[m, x][:, None] * sc, rcond=None)[0]
    e = np.linalg.lstsq(s_all, np.ones(len(s_all)), rcond=None)[0]
    return s_all, ops, e


def analytic_prefix_probs(resid, T, pi):
    """{prefix: P(w)} via the belief-update product, for transition tensor T and start pi.
    Forbidden prefixes get 0. Used as the P(w) sample-weights for the subspace/operator fits."""
    out = {}
    for w in resid:
        b, p = np.array(pi, dtype=float), 1.0
        for x in w:
            d = np.array([(b @ T[i]).sum() for i in range(T.shape[0])])
            if d[x] < 1e-12:
                p = 0.0; break
            p *= float(d[x]); b = b @ T[x] / d[x]
        out[w] = p
    return out


def run(name, ckpt, proc_T, stationary, max_horizon=3, use_cca_whitening=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load(ckpt, device)
    resid, soft, belief, pp = _collect(m, proc_T, stationary, device)

    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    vocab = proc_T.shape[0]
    print(f"\n=== {name} (CCA/RRR) ===  reachable nodes: {sum(reach.values())}/{len(reach)}")

    # Compute P(w) weights from ground-truth DGP for proper spectral-learning weighting
    prefix_probs = analytic_prefix_probs(resid, proc_T, stationary)

    rows, A, P, C_ks, O, U_plain, sv_plain, future_collections, sw = direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=max_horizon, wmap=prefix_probs
    )
    print(f"  rows={len(rows)}  D={A.shape[1]}  max_horizon={max_horizon}  O shape={O.shape}")
    print(f"  plain SVD spectrum (first 12): {np.round(sv_plain[:12] / sv_plain[0], 3)}")

    if use_cca_whitening:
        U_cca, sv_cca = cca_rrr_subspace(A, future_collections, ridge_cca=0.01, sw=sw)
        print(f"  CCA/RRR spectrum (first 12): {np.round(sv_cca[:12] / sv_cca[0] if len(sv_cca) > 0 else [], 3)}")
        U = U_cca
        sv = sv_cca
    else:
        U = U_plain
        sv = sv_plain

    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)

    # For state-level R² computation: map each prefix to its belief state index
    # Get unique belief states and their indices
    unique_beliefs = {}
    belief_indices = []
    for i, b in enumerate(Yb):
        if fin[i]:
            b_tuple = tuple(np.round(b, 5))
            if b_tuple not in unique_beliefs:
                unique_beliefs[b_tuple] = len(unique_beliefs)
            belief_indices.append(unique_beliefs[b_tuple])
        else:
            belief_indices.append(-1)
    belief_indices = np.array(belief_indices)

    decs = []
    state_decs = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :d])

        # Per-position decode R²
        dec = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
        decs.append((d, dec))

        # State-level (center-of-mass) R²
        valid_states = set(belief_indices[fin])
        state_coms_pred = np.array([s_all[belief_indices == s].mean(0)
                                     for s in valid_states if (belief_indices == s).any()])
        state_coms_true = np.array([Yb[belief_indices == s][0]
                                     for s in valid_states if (belief_indices == s).any()])
        if len(state_coms_pred) > 1:
            state_dec = LinearRegression().fit(state_coms_pred, state_coms_true).score(state_coms_pred, state_coms_true)
            state_decs.append((d, state_dec))

    print("  per-position R^2 vs d:", {d: round(r, 3) for d, r in decs})
    if state_decs:
        print("  state-level R^2 vs d:  ", {d: round(r, 3) for d, r in state_decs})

    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])
    dstar = proc_T.shape[1]
    s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :dstar])
    reg = LinearRegression().fit(s_all[fin], Yb[fin]); decode = reg.score(s_all[fin], Yb[fin])

    # Compute state-level R² at d=dstar
    valid_states = set(belief_indices[fin])
    state_coms_pred = np.array([s_all[belief_indices == s].mean(0)
                                 for s in valid_states if (belief_indices == s).any()])
    state_coms_true = np.array([Yb[belief_indices == s][0]
                                 for s in valid_states if (belief_indices == s).any()])
    state_decode = LinearRegression().fit(state_coms_pred, state_coms_true).score(state_coms_pred, state_coms_true) if len(state_coms_pred) > 1 else np.nan

    print(f"  supervised ceiling={ceiling:.3f};  at d={dstar}: per-position R^2={decode:.3f}, state-level R^2={state_decode:.3f}; eig(A^x) vs eig(T^x):")
    for x in range(vocab):
        evt = np.sort(np.linalg.eigvals(proc_T[x]).real)
        eva = np.sort(np.linalg.eigvals(ops[x]).real)
        print(f"    token {x}: eig(T)={np.round(evt,3)}  eig(A)={np.round(eva,3)}")

    return dict(resid=resid, soft=soft, belief=belief, rows=rows, A=A, P=P, C_ks=C_ks, U=U,
                fin=fin, Yb=Yb, reg=reg, dstar=dstar, decode=decode, decs=decs, ceiling=ceiling, vocab=vocab)


def _load(ckpt, device):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    c = torch.load(MODEL_DIR / ckpt, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(c["cfg"]); cfg.device = device
    mm = HookedTransformer(cfg); mm.load_state_dict(c["state_dict"]); mm.to(device).eval()
    return mm


def _collect(model, T, stationary, device):
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = T.shape[1]

    def ndist(b): return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])
    def upd(b, x): nb = b @ T[x]; return nb / nb.sum()
    def bp(w):
        b, p = stationary, 1.0
        for x in w:
            dd = ndist(b); p *= dd[x]
            if dd[x] < 1e-12: return None
            b = upd(b, x)
        return b

    resid, soft, belief, pp = {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]; soft[w] = sm[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan); pp[w] = 0.0
    return resid, soft, belief, pp


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Testing direct multi-step readouts + CCA/RRR whitening")
    print("="*80)

    # Test on RRXOR with different horizons
    print("\n### RRXOR: varying max_horizon ###")
    rrxor_results = {}
    for h in [1, 2, 3, 4]:
        R = run("RRXOR", "rrxor_transformer.pt", np.array(rrxor(0.5, 0.5)),
                np.array([2, 1, 1, 1, 1]) / 6.0, max_horizon=h, use_cca_whitening=True)
        rrxor_results[h] = R

    print("\n### Mess3: control sanity check ###")
    M = run("Mess3", "mess3_transformer.pt", np.array(mess3(x=0.05, a=0.85)),
            np.array([1, 1, 1]) / 3.0, max_horizon=3, use_cca_whitening=True)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: RRXOR horizons
    for h in sorted(rrxor_results.keys()):
        res = rrxor_results[h]
        ds = [d for d, _ in res["decs"]]; rr = [r for _, r in res["decs"]]
        axes[0].plot(ds, rr, "o-", label=f"RRXOR (max_horizon={h})")
        axes[0].axhline(res["ceiling"], color="gray", ls="--", lw=0.5, alpha=0.5)

    axes[0].set_xlabel("subspace dim d")
    axes[0].set_ylabel("belief-decode R$^2$")
    axes[0].set_title("CCA/RRR: RRXOR recovery vs horizon length")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=.3)

    # Right: comparison of d=5 results across horizons
    horizons = sorted(rrxor_results.keys())
    d5_scores = []
    for h in horizons:
        res = rrxor_results[h]
        d5_score = next((r for d, r in res["decs"] if d == 5), None)
        d5_scores.append(d5_score if d5_score is not None else 0)

    axes[1].plot(horizons, d5_scores, "o-", color="black", markersize=8, linewidth=2, label="RRXOR (d=5)")
    axes[1].axhline(rrxor_results[1]["ceiling"], color="red", ls="--", lw=1.5, alpha=0.7, label="supervised ceiling")
    axes[1].set_xlabel("max_horizon")
    axes[1].set_ylabel("belief-decode R$^2$ at d=5")
    axes[1].set_title("CCA/RRR: RRXOR improvement with horizon")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=.3)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "fig14_cca_rrr_estimator.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
