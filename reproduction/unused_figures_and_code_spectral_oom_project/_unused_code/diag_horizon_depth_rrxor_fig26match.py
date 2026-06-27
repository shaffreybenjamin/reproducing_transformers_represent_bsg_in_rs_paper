"""Diagnostic: test horizon L on RRXOR using fig26's exact data pipeline.

Builds observability matrix at different horizons L=3,4,5,6,7,8,10,12,15 and scores
on fig26's full length-10 enumeration dataset with per-state-equal weighting.
Should match fig26's baseline (state-level R²=0.816) at L=3, d=5.
"""

from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14
import fig10_rrxor_representation as F10

MODEL_DIR = Path(__file__).parent / "models"
FIG_DIR = Path(__file__).parent / "figures"
D = 5


def observable_subspace_at_depth(resid, soft, reach, vocab, depth=3, wmap=None):
    """Build observability matrix O at specified depth with P(w)-weighting."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > F14.EPS)]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C = F14.Ridge(alpha=F14.RIDGE, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > F14.EPS for w in rows])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else sw[m]
        Gs.append(F14.Ridge(alpha=F14.RIDGE, fit_intercept=False).fit(A[m], tgt, sample_weight=swm).coef_.T)

    cols, frontier = [C], [C]
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols += nxt
        frontier = nxt
    O = np.hstack(cols)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)
    return U, sv


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)

    # ---- Build observability subspace (using prefix tree, P(w)-weighted) ----
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    resid, soft, belief_u, _ = F14._collect(model, T, pi, device)
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    pw = F14.analytic_prefix_probs(resid, T, pi)

    # ---- Score on fig26's exact data: full length-10 enumeration ----
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    concat = np.concatenate([acts[h] for h in hooks], axis=-1)  # (N, 10, 384)
    Xf = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, B36.shape[1])
    fidx = idx.reshape(-1)

    print(f"Full enumeration: {len(Xf)} (input,position) points across {len(B36)} states")

    # Compute per-state-equal weights (same as fig26)
    counts = np.bincount(fidx, minlength=len(B36))
    sw = 1.0 / np.clip(counts[fidx], 1, None)
    sw = sw / sw.mean()

    # Supervised ceiling
    sup = LinearRegression().fit(Xf, Y, sample_weight=sw)
    r2_sup = sup.score(Xf, Y, sample_weight=sw)
    print(f"Supervised ceiling (full 384-D): {r2_sup:.3f}\n")

    # Define COM-geom metric (same as fig26 line 84-86)
    def com_r2(S, pred):
        """State-level R²: collapse predictions to 36 state COMs, score against true beliefs."""
        com = np.array([pred[fidx == s].mean(0) for s in range(len(B36)) if (fidx == s).any()])
        bel = np.array([Y[fidx == s][0] for s in range(len(B36)) if (fidx == s).any()])
        return LinearRegression().fit(com, bel).score(com, bel)

    # Test different horizons
    horizons = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    print("=== HORIZON TEST (scored on fig26's full enumeration) ===\n")

    results_pp = {}  # per-position
    results_cl = {}  # state-level (COM)

    for L in horizons:
        print(f"L={L}:")
        U, sv = observable_subspace_at_depth(resid, soft, reach, 2, depth=L, wmap=pw)

        decs_pp = []
        decs_cl = []
        for d in [2, 3, 4, 5, 6, 8, 10]:
            if d > len(sv):
                continue
            B = U[:, :d]
            S = Xf @ B

            # Per-position R²
            dec_pp = LinearRegression().fit(S, Y, sample_weight=sw).score(S, Y, sample_weight=sw)
            decs_pp.append((d, dec_pp))

            # State-level (COM-geom) R²
            reg = LinearRegression().fit(S, Y, sample_weight=sw)
            pred = reg.predict(S)
            dec_cl = com_r2(S, pred)
            decs_cl.append((d, dec_cl))

        results_pp[L] = decs_pp
        results_cl[L] = decs_cl
        print(f"  Per-position R^2 vs d: {[(d, round(r, 3)) for d, r in decs_pp]}")
        print(f"  State-level R^2 vs d:  {[(d, round(r, 3)) for d, r in decs_cl]}")

    # Summary at d=5 (the key dimension)
    print("\n=== SUMMARY: STATE-LEVEL R² AT d=5 ===")
    print("L\tState-level R² at d=5")
    for L in horizons:
        r2_d5 = [r for d, r in results_cl[L] if d == 5][0]
        print(f"{L}\t{r2_d5:.3f}")

    print(f"\nFig26 baseline: per-position R²=0.651, state-level R²=0.816 at L=3, d=5")


if __name__ == "__main__":
    main()
