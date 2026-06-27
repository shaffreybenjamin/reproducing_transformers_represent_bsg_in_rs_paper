"""RRXOR belief geometry from the OBSERVABILITY-OOM operator rollout (cf. fig20, which
did the same with CCA+ALS operators).

Learn operators via the observability OOM (fig14), roll them out to regenerate each
prefix's belief state (the "pre-PCA data" produced by the operators, not the raw
activations), gauge-align to belief coordinates, and project with the panel-B PCA.
Left = ground truth; right = observability-OOM operator rollout. Compare COM-geometry
vs the raw projection and vs fig20's CCA+ALS rollout (0.37).
"""

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
D = 5


def com_geometry(S, Yb, idx):
    groups = defaultdict(list)
    for i, s in enumerate(idx):
        groups[s].append(i)
    com = np.array([S[g].mean(0) for g in groups.values()])
    bel = np.array([Yb[g[0]] for g in groups.values()])
    return LinearRegression().fit(com, bel).score(com, bel), groups


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

    # observability OOM operators (fig14)
    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, 2)
    basis = Uobs[:, :D]
    s_all, ops, e_fit = F14.fit_at_dim(rows, A, P, resid, 2, basis)

    # roll the operators out -> regenerate each prefix's belief state
    proj = {w: resid[w] @ basis for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    sh = U.rollout_states(proj, ops, e, F14.MAX_LEN, 2)

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    B36, index = F13.msp_index()

    S_raw = np.stack([resid[w] @ basis for w in allw]); Yb = np.stack([belief[w] for w in allw])
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    cg_raw, _ = com_geometry(S_raw, Yb, idx)

    rw = [w for w in allw if w in sh and np.isfinite(sh[w]).all()]
    S_roll = np.stack([sh[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, groups = com_geometry(S_roll, Yb_r, idx_r)
    print(f"observability OOM  raw COM-geometry={cg_raw:.3f}   operator-rollout COM-geometry={cg_roll:.3f}  "
          f"(rolled {len(rw)}/{len(allw)} prefixes)   [fig20 CCA+ALS rollout was 0.37]")

    decode = LinearRegression().fit(S_roll, Yb_r)
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]
    xy = pca.transform(decode.predict(S_roll))[:, [0, 2]]
    colors = np.array(B9.distinct_colors(len(B36)))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5.6))
    a0.scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    a0.set_title("Ground truth  (PCA 1 / PCA 3)")
    for s, g in groups.items():
        a1.scatter(xy[g, 0], xy[g, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
        a1.scatter(*xy[g].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
    a1.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    a1.set_title(f"Observability-OOM operator rollout (RRXOR)\nCOM-geometry R$^2$={cg_roll:.2f} (raw {cg_raw:.2f})")
    for ax in (a0, a1):
        ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3")
    out = FIG_DIR / "fig21_rrxor_observability_rollout.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
