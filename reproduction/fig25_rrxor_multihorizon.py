"""fig24 analogue for RRXOR, using MULTI-HORIZON RIDGE RRR (the 0.669 method) instead of
plain OBS-OOM (0.635). Same family (deep input-whitened predictive subspace), but stacks
futures from horizons h=1..H and adds a light ridge, which edges the raw recovery up.

Panels (separate file, fig25_rrxor_multihorizon.png):
  1  ground truth (PCA 1 / PCA 3)
  2  raw-activation recovery on the multi-horizon RRR subspace   (decode -> belief -> PCA)
  3  operator rollout in that subspace                           (collapses -- nilpotent ops)
"""
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9
import unsupervised_belief_oom as U

FIG_DIR = Path(__file__).parent / "figures"
VOCAB, D, H, RIDGE = 2, 5, 3, 0.03


def whiten_inv_sqrt(C, ridge):
    C = C + ridge * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    val, vec = np.linalg.eigh(C)
    return (vec / np.sqrt(np.clip(val, 1e-12, None))) @ vec.T


def com_geometry(S, Yb, idx):
    groups = defaultdict(list)
    for i, s in enumerate(idx):
        groups[s].append(i)
    com = np.array([S[g].mean(0) for g in groups.values()])
    bel = np.array([Yb[g[0]] for g in groups.values()])
    return LinearRegression().fit(com, bel).score(com, bel), groups


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    model = F14._load("rrxor_transformer.pt", "cpu")
    resid, soft, belief, _ = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Aall = np.stack([resid[w] for w in allw]); Yb = np.stack([belief[w] for w in allw])
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]

    # ---- multi-horizon ridge RRR subspace ----
    paths = [p for L in range(H + 1) for p in itertools.product(range(VOCAB), repeat=L)]
    fitw = [w for w in allw if len(w) <= F14.MAX_LEN - H and all((w + p) in soft for p in paths)]
    Af = np.stack([resid[w] for w in fitw])
    Phi = np.stack([np.concatenate([soft[w + p] for p in paths]) for w in fitw])
    Winv = whiten_inv_sqrt(Af.T @ Af / len(Af), RIDGE)
    Uu = np.linalg.svd((Af @ Winv).T @ Phi / len(Af), full_matrices=False)[0]
    B = Winv @ Uu[:, :D]

    # ---- raw recovery ----
    S_raw = Aall @ B
    cg_raw, groups = com_geometry(S_raw, Yb, idx)
    xy_raw = pca.transform(LinearRegression().fit(S_raw, Yb).predict(S_raw))[:, [0, 2]]

    # ---- operator rollout (same rescaled-children operators; collapses on RRXOR) ----
    rows = [w for w in resid if reach[w] and len(w) < F14.MAX_LEN
            and all((w + (x,)) in resid for x in range(VOCAB))]
    X = np.stack([resid[w] for w in rows]); P = np.stack([soft[w] for w in rows])
    Yc = np.stack([[resid[w + (x,)] for x in range(VOCAB)] for w in rows])
    Wt = np.array([1.0 if reach[w] else 0.0 for w in rows])
    ops, _ = U.fit_operators(X @ B, P, Yc @ B, Wt)
    proj = {w: resid[w] @ B for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    roll = U.rollout_states(proj, ops, e, F14.MAX_LEN, VOCAB)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S_roll = np.stack([roll[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, groups_r = com_geometry(S_roll, Yb_r, idx_r)
    xy_roll = pca.transform(LinearRegression().fit(S_roll, Yb_r).predict(S_roll))[:, [0, 2]]
    print(f"multi-horizon ridge RRR  raw COM-geom={cg_raw:.3f}  rollout COM-geom={cg_roll:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax[0].set_title("RRXOR ground truth\n(PCA 1 / PCA 3)")
    for panel, xy, grp, ttl in [
        (ax[1], xy_raw, groups, f"Multi-horizon ridge RRR - raw\nCOM-geom R$^2$={cg_raw:.2f}  (OBS-OOM 0.64)"),
        (ax[2], xy_roll, groups_r, f"Multi-horizon ridge RRR - operator rollout\nCOM-geom R$^2$={cg_roll:.2f}  (~raw; compressed, outer states unreached)"),
    ]:
        for s, g in grp.items():
            panel.scatter(xy[g, 0], xy[g, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
            panel.scatter(*xy[g].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
        panel.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        panel.set_title(ttl)
    for a in ax:
        a.set_aspect("equal"); a.spines[["top", "right"]].set_visible(False)
        for setl, v in ((a.set_xlim, tgt[:, 0]), (a.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        a.set_xlabel("PCA 1"); a.set_ylabel("PCA 3")
    out = FIG_DIR / "fig25_rrxor_multihorizon.png"
    fig.tight_layout(); fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
