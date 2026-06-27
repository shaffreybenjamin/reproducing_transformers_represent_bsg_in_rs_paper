"""RRXOR belief-geometry comparison with EVERYTHING matched to the fig14 orphan scheme
(the opposite choice to fig26, which matched the supervised fig10 scheme).

fig14's scheme = the observability-OOM `rows`: reachable prefixes of length 1-9 that have
all children present (internal MSP nodes), each counted ONCE (so states are ~uniformly
weighted and the noisy deep length-10 leaves are dropped). Both panels use that SAME set,
the same panel-B PCA (PC1/PC3 on the 36 GT beliefs), the same per-prefix scatter + per-
state COM display, and the same label-fit simplex placement. The ONLY difference is the
affine-map input:

  panel 2 (supervised)   : full concatenated residual stream (384-D) -> belief
  panel 3 (unsupervised) : the recovered OBS-OOM 5-D subspace        -> belief
    (panel 3 reproduces fig14_observable_oom_rrxor.png by construction)

This is the fair version of fig14: the supervised panel is now built on fig14's own scheme,
so any "flattering" of the unsupervised look (uniform-over-states + dropped deep leaves) is
applied identically to the supervised baseline.
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

MODEL_DIR = Path(__file__).parent / "models"
FIG_DIR = Path(__file__).parent / "figures"
D = 5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    model = F14._load("rrxor_transformer.pt", device)
    resid, soft, belief, _ = F14._collect(model, T, pi, device)
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok

    # fig14's exact set: observability `rows` (internal nodes, lengths 1-9, each once)
    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, 2)
    B = Uobs[:, :D]                                          # unsupervised subspace (384,5)
    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    rowsf = [w for w, f in zip(rows, fin) if f]
    Af = A[fin]; Ybf = Yb[fin]
    print(f"fig14 scheme: {len(rowsf)} internal prefixes (lengths 1-9, each once)")

    B36, index = F13.msp_index()
    idxf = np.array([index[tuple(np.round(belief[w], 5))] for w in rowsf])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36)
    tgt = pca.transform(B36)[:, [0, 2]]

    # ---- affine readouts (label-fit, identical convention), only input differs ----
    sup = LinearRegression().fit(Af, Ybf); r2_sup = sup.score(Af, Ybf)
    xy_sup = pca.transform(sup.predict(Af))[:, [0, 2]]
    S = Af @ B
    uns = LinearRegression().fit(S, Ybf); r2_uns = uns.score(S, Ybf)
    xy_uns = pca.transform(uns.predict(S))[:, [0, 2]]

    def com_r2(feat):
        com = np.array([feat[idxf == s].mean(0) for s in np.unique(idxf)])
        bel = np.array([Ybf[idxf == s][0] for s in np.unique(idxf)])
        return LinearRegression().fit(com, bel).score(com, bel)
    print(f"per-prefix decode R^2:   supervised(384-D)={r2_sup:.3f}   unsupervised(5-D)={r2_uns:.3f}")
    print(f"state-COM geometry R^2:  supervised={com_r2(Af):.3f}   unsupervised={com_r2(S):.3f}")

    def draw(ax, xy, title, scatter=True):
        if scatter:
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c=colors[idxf], alpha=0.25, edgecolors="none")
            for s in np.unique(idxf):
                ax.scatter(*xy[idxf == s].mean(0), s=80, c=[colors[s]],
                           edgecolors="black", linewidths=0.5, zorder=3)
            ax.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
        ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo); setl(lo - pad, hi + pad)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
    draw(ax[1], xy_sup, f"Supervised: full residual stream (384-D)\ndecode R$^2$={r2_sup:.2f}  (fig14 scheme)")
    draw(ax[2], xy_uns, f"Unsupervised: OBS-OOM subspace (5-D)\ndecode R$^2$={r2_uns:.2f}  (= fig14 orphan)")
    out = FIG_DIR / "fig27_rrxor_fig14scheme.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
