"""Stage 1 (Regularized CCA/RRR): Isolate belief-bearing region.

Modified from fig14_cca_rrr_estimator.py with:
  - Ledoit-Wolf shrinkage on future covariance (variance control, not d-selection)
  - h=2 horizon (avoid finite-sample noise from longer horizons)
  - Generous d_stage1=8–10 candidate subspace
  - Logging of CCA spectrum

The goal of Stage 1 is to isolate the belief-bearing region of activation space
and filter pure junk. It is *not* expected to solve the d-selection problem — that
is Stage 2's job via cluster-separation ranking.

Based on: direct multi-step readouts (C_k fit directly to empirical future
distributions) + HKZ-style whitening (score by past-future correlation).

Output: Stage-1 subspace basis (8–10D), spectrum, and activations projected into it.
"""

from pathlib import Path
import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

from simplexity.generative_processes.transition_matrices import rrxor

MAX_LEN = 10
EPS = 1e-3  # model-softmax reachability threshold
RIDGE_REGRESSION = 1e-2
RIDGE_CCA = 0.01  # ridge for covariance inversion in CCA whitening
MAX_HORIZON = 4  # h=4: test if longer horizon helps despite sparsity
D_STAGE1 = 10  # generous candidate subspace dimension
LEDOIT_WOLF_SHRINKAGE = 0.15  # stronger shrinkage for h=4 sparsity

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"

T = np.array(rrxor(0.5, 0.5))
STATIONARY = np.array([2, 1, 1, 1, 1]) / 6.0


def enumerate_future_dist(soft, w, vocab, max_horizon=4):
    """Enumerate empirical P(x_{t+1:t+k}|w) for horizons k=1..max_horizon.

    Returns dict {k: future_probs} where future_probs is shape (n_distinct_futures_k, ),
    indexed by the flattened tuple of k tokens.
    """
    out = {}
    for k in range(1, max_horizon + 1):
        future_dist = {}
        for future in itertools.product(range(vocab), repeat=k):
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
      O: stacked observability matrix [C_1 | C_2 | ... | C_L]
      U: left singular vectors of O
      sv: singular values of O
      future_collections: list of Phi_k matrices (each n_rows × feature_width_k)
      sw: sample weights (prefix probabilities)
    """
    rows = [
        w
        for w in resid
        if reach[w]
        and all(
            (w + (x,)) in resid and reach[w + (x,)]
            for x in range(vocab)
            if soft[w][x] > EPS
        )
    ]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C_ks = []
    future_collections = []

    for k in range(1, max_horizon + 1):
        futures_per_prefix = []
        future_vocab_list = []

        for i, w in enumerate(rows):
            future_dists = enumerate_future_dist(soft, w, vocab, max_horizon=k)
            futures_per_prefix.append(future_dists[k])
            if i == 0:
                future_vocab_list = sorted(futures_per_prefix[0].keys())

        all_futures = set()
        for fp in futures_per_prefix:
            all_futures.update(fp.keys())
        future_vocab_list = sorted(all_futures)

        Phi_k = np.zeros((len(rows), len(future_vocab_list)))
        for i, fp in enumerate(futures_per_prefix):
            for j, future in enumerate(future_vocab_list):
                Phi_k[i, j] = fp.get(future, 0.0)

        C_k = Ridge(alpha=RIDGE_REGRESSION, fit_intercept=False).fit(A, Phi_k, sample_weight=sw).coef_.T
        C_ks.append(C_k)
        future_collections.append(Phi_k)

    O = np.hstack(C_ks)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)

    return rows, A, P, C_ks, O, U, sv, future_collections, sw


def cca_rrr_subspace_ledoit(A, future_collections, ridge_cca=0.01, sw=None, ledoit_shrinkage=0.5):
    """Compute CCA/RRR subspace via past-future whitening with Ledoit-Wolf shrinkage.

    Adds Ledoit-Wolf shrinkage to the future covariance (Cov_pp) before whitening,
    targeting the shrinkage towards an identity matrix scaled by the trace. This
    reduces amplification of small eigenvalues in CCA inversion.

    Args:
      A: past activations (n × D)
      future_collections: list of Phi_k matrices (each n × feature_width_k)
      ridge_cca: additional ridge for numerical stability (post-Ledoit)
      sw: optional sample weights (n,) for P(w)-weighted computation
      ledoit_shrinkage: Ledoit-Wolf shrinkage intensity (0 = no shrinkage, 1 = identity)

    Returns:
      U_wh: whitened past basis (D × D) ranked by past-future correlation
      sv: singular values of the whitened correlation
    """
    n = A.shape[0]
    Phi = np.hstack(future_collections)

    # Compute weighted mean and centering
    if sw is not None:
        sw_sum = np.sum(sw)
        A_mean = np.average(A, axis=0, weights=sw)
        Phi_mean = np.average(Phi, axis=0, weights=sw)
        A_cent = A - A_mean
        Phi_cent = Phi - Phi_mean
        A_w = A_cent * np.sqrt(sw)[:, None]
        Phi_w = Phi_cent * np.sqrt(sw)[:, None]
        Cov_aa = A_w.T @ A_w / sw_sum
        Cov_pp_raw = Phi_w.T @ Phi_w / sw_sum
        Cov_ap = A_w.T @ Phi_w / sw_sum
    else:
        A_cent = A - A.mean(0)
        Phi_cent = Phi - Phi.mean(0)
        Cov_aa = A_cent.T @ A_cent / n
        Cov_pp_raw = Phi_cent.T @ Phi_cent / n
        Cov_ap = A_cent.T @ Phi_cent / n

    # Apply Ledoit-Wolf shrinkage to future covariance
    # Target is identity scaled by trace (to preserve scale)
    trace_pp = np.trace(Cov_pp_raw)
    target = (trace_pp / Cov_pp_raw.shape[0]) * np.eye(Cov_pp_raw.shape[0])
    Cov_pp_shrunk = (1.0 - ledoit_shrinkage) * Cov_pp_raw + ledoit_shrinkage * target

    # Add ridge for numerical stability
    Cov_aa_ridge = Cov_aa + ridge_cca * np.trace(Cov_aa) / Cov_aa.shape[0] * np.eye(Cov_aa.shape[0])
    Cov_pp_ridge = Cov_pp_shrunk + ridge_cca * np.trace(Cov_pp_shrunk) / Cov_pp_shrunk.shape[0] * np.eye(Cov_pp_shrunk.shape[0])

    # Compute whitening matrices
    val_a, vec_a = np.linalg.eigh(Cov_aa_ridge)
    val_p, vec_p = np.linalg.eigh(Cov_pp_ridge)

    Whiten_a = vec_a @ np.diag(1.0 / np.sqrt(np.clip(val_a, 1e-12, None))) @ vec_a.T
    Whiten_p = vec_p @ np.diag(1.0 / np.sqrt(np.clip(val_p, 1e-12, None))) @ vec_p.T

    M = Whiten_a @ Cov_ap @ Whiten_p
    U_w, sv, V_w = np.linalg.svd(M, full_matrices=False)
    U_wh = Whiten_a @ U_w

    return U_wh, sv


def fit_at_dim(rows, A, P, resid, vocab, B):
    """Project to subspace B (D x d), fit operators A_x."""
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
    """{prefix: P(w)} via the belief-update product."""
    out = {}
    for w in resid:
        b, p = np.array(pi, dtype=float), 1.0
        for x in w:
            d = np.array([(b @ T[i]).sum() for i in range(T.shape[0])])
            if d[x] < 1e-12:
                p = 0.0
                break
            p *= float(d[x])
            b = b @ T[x] / d[x]
        out[w] = p
    return out


def run_stage1(name, ckpt, proc_T, stationary, max_horizon=MAX_HORIZON):
    """Run Stage 1: regularized CCA/RRR with Ledoit-Wolf shrinkage.

    Args:
        name: process name (e.g., "RRXOR")
        ckpt: checkpoint filename
        proc_T: transition matrix (if None, loads from spec)
        stationary: stationary distribution (if None, loads from spec)
        max_horizon: max future horizon (default 2)

    Returns:
        outputs: dict with subspace, spectrum, and activations for Stage 2
    """
    # Use defaults if not provided
    if proc_T is None:
        proc_T = T
    if stationary is None:
        stationary = STATIONARY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== {name} (CCA/RRR + Ledoit-Wolf) ===")

    # Load model and collect activations
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    ckpt_path = MODEL_DIR / ckpt
    if not ckpt_path.exists():
        ckpt_path = Path(ckpt)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"])
    cfg.device = device
    m = HookedTransformer(cfg)
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()

    nL = m.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = proc_T.shape[1]

    def ndist(b):
        return np.array([(b @ proc_T[x]).sum() for x in range(proc_T.shape[0])])

    def upd(b, x):
        nb = b @ proc_T[x]
        return nb / nb.sum()

    def bp(w):
        b, p = stationary, 1.0
        for x in w:
            dd = ndist(b)
            p *= dd[x]
            if dd[x] < 1e-12:
                return None
            b = upd(b, x)
        return b

    resid, soft, belief, pp = {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(proc_T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i : i + 4096]).to(device)
            with torch.no_grad():
                logits, c = m.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i : i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]
                soft[w] = sm[j]
                b = bp(w)
                belief[w] = b if b is not None else np.full(NS, np.nan)
                pp[w] = 0.0

    # Compute reachability and prefix probabilities
    reach = {}
    for w in resid:
        ok = True
        pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    vocab = proc_T.shape[0]
    prefix_probs = analytic_prefix_probs(resid, proc_T, stationary)

    # Run direct multi-step readouts + CCA/RRR with Ledoit-Wolf
    rows, A, P, C_ks, O, U_plain, sv_plain, future_collections, sw = direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=max_horizon, wmap=prefix_probs
    )
    print(f"  rows={len(rows)}  D={A.shape[1]}  max_horizon={max_horizon}  O shape={O.shape}")
    print(f"  plain SVD spectrum (first 12): {np.round(sv_plain[:12] / sv_plain[0], 3)}")

    # CCA/RRR with Ledoit-Wolf
    U_cca, sv_cca = cca_rrr_subspace_ledoit(
        A, future_collections, ridge_cca=RIDGE_CCA, sw=sw, ledoit_shrinkage=LEDOIT_WOLF_SHRINKAGE
    )
    print(f"  CCA/RRR spectrum (first 12): {np.round(sv_cca[:12] / sv_cca[0] if len(sv_cca) > 0 else [], 3)}")

    U = U_cca
    sv = sv_cca

    # Extract Stage-1 subspace (generous d=8-10)
    B_stage1 = U[:, :D_STAGE1]
    print(f"  Stage-1 subspace: d={D_STAGE1}")

    outputs = {
        "rows": rows,
        "A": A,
        "P": P,
        "U": U,
        "sv": sv,
        "B_stage1": B_stage1,
        "D_stage1": D_STAGE1,
        "resid": resid,
        "soft": soft,
        "belief": belief,
        "vocab": vocab,
        "reach": reach,
        "prefix_probs": prefix_probs,
    }

    return outputs


def main():
    print("\n" + "=" * 80)
    print("STAGE 1: Regularized CCA/RRR (Ledoit-Wolf + h=2)")
    print("=" * 80)

    # Check if Stage 0 output exists
    stage0_out = OUT_DIR / "stage0_gate_output.pkl"
    if stage0_out.exists():
        with open(stage0_out, "rb") as f:
            stage0_results = pickle.load(f)
        print(f"\nLoaded Stage 0 results from {stage0_out}")
        print(f"  Gate pass: {stage0_results['gate_pass']}")
        print(f"  Separation ratio: {stage0_results['separation_ratio']:.4f}")
    else:
        print(f"\nWarning: Stage 0 output not found at {stage0_out}")
        print("Running Stage 0 first...")
        import fig_stage0_gate_rrxor as stage0
        stage0_results = stage0.main()

    # Run Stage 1
    results = run_stage1(
        "RRXOR",
        "rrxor_transformer.pt",
        T,
        STATIONARY,
        max_horizon=MAX_HORIZON,
    )

    # Save Stage 1 outputs for Stage 2
    stage1_out = OUT_DIR / "stage1_cca_output.pkl"
    with open(stage1_out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved Stage 1 outputs to {stage1_out}")

    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
