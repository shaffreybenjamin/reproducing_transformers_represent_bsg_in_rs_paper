"""Generalized Stage 1 CCA/RRR for any process.

Minimal wrapper that adapts the RRXOR Stage 1 code to work with any HMM process.
Used by fig1_*_principled_stage1.py plotting scripts.
"""
import itertools
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

EPS = 1e-3
RIDGE_REGRESSION = 1e-2
RIDGE_CCA = 0.01
MAX_HORIZON = 3
D_STAGE1 = 10
LEDOIT_WOLF_SHRINKAGE = 0.05


def enumerate_future_dist(soft, w, vocab, max_horizon=4):
    """Enumerate empirical P(futures|w) for horizons k=1..max_horizon."""
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


def direct_multistep_readouts(resid, soft, reach, vocab, max_horizon=3, wmap=None):
    """Build multi-step observables via direct regression."""
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


def cca_rrr_subspace_ledoit(A, future_collections, ridge_cca=0.01, sw=None, ledoit_shrinkage=0.05):
    """Compute CCA/RRR subspace with Ledoit-Wolf shrinkage."""
    n = A.shape[0]
    Phi = np.hstack(future_collections)

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

    # Ledoit-Wolf shrinkage
    trace_pp = np.trace(Cov_pp_raw)
    target = (trace_pp / Cov_pp_raw.shape[0]) * np.eye(Cov_pp_raw.shape[0])
    Cov_pp_shrunk = (1.0 - ledoit_shrinkage) * Cov_pp_raw + ledoit_shrinkage * target

    Cov_aa_ridge = Cov_aa + ridge_cca * np.trace(Cov_aa) / Cov_aa.shape[0] * np.eye(Cov_aa.shape[0])
    Cov_pp_ridge = Cov_pp_shrunk + ridge_cca * np.trace(Cov_pp_shrunk) / Cov_pp_shrunk.shape[0] * np.eye(Cov_pp_shrunk.shape[0])

    val_a, vec_a = np.linalg.eigh(Cov_aa_ridge)
    val_p, vec_p = np.linalg.eigh(Cov_pp_ridge)

    Whiten_a = vec_a @ np.diag(1.0 / np.sqrt(np.clip(val_a, 1e-12, None))) @ vec_a.T
    Whiten_p = vec_p @ np.diag(1.0 / np.sqrt(np.clip(val_p, 1e-12, None))) @ vec_p.T

    M = Whiten_a @ Cov_ap @ Whiten_p
    U_w, sv, V_w = np.linalg.svd(M, full_matrices=False)
    U_wh = Whiten_a @ U_w

    return U_wh, sv


def run_stage1(hmm, model, device, max_len=10, max_horizon=3):
    """Run Stage 1 CCA/RRR for a given HMM and model.

    Returns: (B_stage1, rows, A, P, resid, soft, belief)
    """
    T = hmm.transition_matrix  # (vocab, nstates, nstates)
    pi = hmm.initial_state_distribution
    vocab = hmm.vocab_size
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = T.shape[1]

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

    resid, soft, belief = {}, {}, {}
    for L in range(1, max_len + 1):
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

    # Compute reachability
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

    # Run CCA/RRR
    rows, A, P, C_ks, O, U_plain, sv_plain, future_collections, sw = direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=max_horizon, wmap=None
    )

    U_cca, sv_cca = cca_rrr_subspace_ledoit(
        A, future_collections, ridge_cca=RIDGE_CCA, sw=sw, ledoit_shrinkage=LEDOIT_WOLF_SHRINKAGE
    )

    B_stage1 = U_cca[:, :D_STAGE1]

    return B_stage1, rows, A, P, resid, soft, belief
