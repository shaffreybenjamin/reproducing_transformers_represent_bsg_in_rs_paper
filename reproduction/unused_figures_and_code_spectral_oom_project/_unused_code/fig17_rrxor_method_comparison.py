"""RRXOR belief geometry recovered by the TWO unsupervised methods, for comparison.

Two figures, each: (left) ground-truth PCA (PC1/PC3, as fig09); (right) belief decoded
from that method's unsupervised subspace, projected with the SAME panel-B PCA
(small dots = per-prefix, large = per-state centre of mass, grey rings = ground truth).

Method A: dynamics-consistent predictive CCA (+ PCA pre-reduction), as in fig04/fig13.
Method B: observability OOM (observable anchor + operator closure), as in fig14/fig16.
Both on the same RRXOR concatenated residual stream (RRXOR's belief is spread across
layers, so the concat is the meaningful representation -- matching supervised fig10).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9
import reproduction.estimators.unsupervised_belief_oom as U

FIG_DIR = Path(__file__).parent / "figures"
D = 5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    T = np.array(rrxor(0.5, 0.5)); pi = np.array([2, 1, 1, 1, 1]) / 6.0
    model = F14._load("rrxor_transformer.pt", "cpu")
    resid, soft, belief, pp = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok

    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, 2)
    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    rowsf = [w for w, f in zip(rows, fin) if f]
    B36, index = F13.msp_index()
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in rowsf])
    colors = np.array(B9.distinct_colors(len(B36)))

    groups = {}
    for i, s in enumerate(idx):
        groups.setdefault(s, []).append(i)

    def evaluate(B):
        s = A @ B
        reg = LinearRegression().fit(s[fin], Yb[fin])
        pred = reg.predict(s[fin])
        xy = pca.transform(pred)[:, [0, 2]]
        per_prefix = reg.score(s[fin], Yb[fin])
        com = np.array([s[fin][g].mean(0) for g in groups.values()])
        bel = np.array([Yb[fin][g[0]] for g in groups.values()])
        com_r2 = LinearRegression().fit(com, bel).score(com, bel)
        return xy, per_prefix, com_r2

    # --- Method B: observability OOM ---
    xy_obs, pp_obs, com_obs = evaluate(Uobs[:, :D])
    print(f"observability OOM: per-prefix R^2={pp_obs:.3f}  COM-geometry R^2={com_obs:.3f}")

    # --- Method A: dynamics-consistent predictive CCA (PCA-pre-reduced, uniform weights) ---
    Yc = np.stack([[resid[rows[i] + (x,)] for x in range(2)] for i in range(len(rows))])
    Bpca = PCA(n_components=20).fit(A).components_.T
    dirs, _ = U.predictive_cca(A @ Bpca, P, Yc @ Bpca, np.ones(len(rows)))
    B_cca = Bpca @ dirs[:, :D]
    xy_cca, pp_cca, com_cca = evaluate(B_cca)
    print(f"predictive CCA:    per-prefix R^2={pp_cca:.3f}  COM-geometry R^2={com_cca:.3f}")

    for xy, name, pp_, com_, fname in [
        (xy_cca, "Predictive CCA (fig04/fig13 method)", pp_cca, com_cca, "fig17_rrxor_cca_recovery.png"),
        (xy_obs, "Observability OOM (fig14/fig16 method)", pp_obs, com_obs, "fig18_rrxor_observability_recovery.png"),
    ]:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5.6))
        a0.scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
        a0.set_title("Ground truth  (PCA 1 / PCA 3)")
        for s, g in groups.items():
            a1.scatter(xy[g, 0], xy[g, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
            a1.scatter(*xy[g].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
        a1.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        a1.set_title(f"{name}\nper-prefix R$^2$={pp_:.2f}   COM-geometry R$^2$={com_:.2f}")
        for ax in (a0, a1):
            ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
            for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
                lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
            ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3")
        fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"saved -> {FIG_DIR / fname}")


if __name__ == "__main__":
    main()
