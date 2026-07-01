"""RRXOR belief geometry from the OPERATOR ROLLOUT (not raw activations).

Per the fig04 logic: the operators learned from the activations capture the dynamics,
so the right thing to visualise is the geometry the operators REGENERATE, not the raw
activation projections. We learn CCA + ALS operators on RRXOR, roll them out to
regenerate each prefix's belief state (the "pre-PCA data" -- now produced by the
operators), gauge-align those states to belief coordinates, and project with the
panel-B PCA (PC1/PC3). Left = ground truth (fig09); right = operator-rollout recovery.

Compare COM-geometry vs the RAW activation projections (CCA raw 0.57 / observability 0.68).
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
import reproduction.estimators.unsupervised_belief_oom as U

FIG_DIR = Path(__file__).parent / "figures"
D = 5
PRE_PCA_K = 20


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

    # CCA + ALS pipeline (the fig04 recipe), on the RRXOR concat
    prefix_prob = {w: (1.0 if reach[w] else 0.0) for w in resid}
    rows, X, P, Yc, Wt = U.build_transitions(resid, soft, prefix_prob, F14.MAX_LEN, 2)
    Bpca = PCA(n_components=PRE_PCA_K).fit(X).components_.T
    dirs, _ = U.predictive_cca(X @ Bpca, P, Yc @ Bpca, Wt)
    basis_r = U.als_refine_basis(X @ Bpca, P, Yc @ Bpca, Wt, dirs[:, :D], D)
    basis = Bpca @ basis_r
    ops, _ = U.fit_operators(X @ basis, P, Yc @ basis, Wt)

    # roll the operators out -> regenerate each prefix's belief state (the pre-PCA data)
    proj = {w: resid[w] @ basis for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    sh = U.rollout_states(proj, ops, e, F14.MAX_LEN, 2)

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    B36, index = F13.msp_index()

    # RAW baseline (CCA+ALS subspace, raw activation projection) for comparison
    raw_w = allw
    S_raw = np.stack([resid[w] @ basis for w in raw_w]); Yb = np.stack([belief[w] for w in raw_w])
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in raw_w])
    cg_raw, _ = com_geometry(S_raw, Yb, idx)

    # ROLLOUT recovery
    rw = [w for w in allw if w in sh and np.isfinite(sh[w]).all()]
    S_roll = np.stack([sh[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, groups = com_geometry(S_roll, Yb_r, idx_r)
    print(f"CCA+ALS  raw COM-geometry={cg_raw:.3f}   operator-rollout COM-geometry={cg_roll:.3f}  "
          f"(rolled {len(rw)}/{len(allw)} prefixes)")

    # gauge-align rolled states to belief coords, then panel-B PCA (PC1/PC3)
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
    a1.set_title(f"CCA+ALS operator rollout (RRXOR)\nCOM-geometry R$^2$={cg_roll:.2f} (raw {cg_raw:.2f})")
    for ax in (a0, a1):
        ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3")
    out = FIG_DIR / "fig20_rrxor_operator_rollout.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
