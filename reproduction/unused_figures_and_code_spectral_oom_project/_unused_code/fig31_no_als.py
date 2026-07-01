"""Operator rollout WITHOUT ALS, for Mess3 and RRXOR, to judge whether ALS is worth keeping.
Same recipe as fig23/fig24 (OBS subspace, rescaled operator fit, process-appropriate
weighting) but the ALS re-orientation is omitted. Compare:
  Mess3   no-ALS (here) vs ALS (fig23, 1.00)
  RRXOR   no-ALS (here) vs ALS (~0.19)
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.transition_matrices import rrxor
import reproduction.estimators.unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9
import fig23_unified_best as F23

FIG_DIR = Path(__file__).parent / "figures"


def mess3_fig():
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}
    rows, X, P, Yc = F23.transitions(resid, soft, reach, vocab, max_len)
    Wt = np.array([prefix_prob[w] for w in rows])
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab)
    B = Uobs[:, :3]
    ops, _ = U.fit_operators(X @ B, P, Yc @ B, Wt)            # NO ALS
    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    true_b = np.stack([belief[w] for w in allw]); true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1); wcol = np.array([prefix_prob[w] for w in allw])
    Xall = np.stack([resid[w] for w in allw])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)
    raw_r2 = U.belief_decode_r2(Xall @ B, true_b, wcol)
    dec = LinearRegression().fit(Xall @ B, true_b)
    raw_xy = U.simplex_to_xy(to_simplex(dec.predict(Xall @ B)))
    proj = {w: resid[w] @ B for w in resid}
    e, _ = U.recover_eval_functional(np.stack([proj[w] for w in allw]))
    roll = U.rollout_states(proj, ops, e, max_len, vocab)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S = np.stack([roll[w] for w in rw]); rb = np.stack([belief[w] for w in rw])
    roll_xy = U.simplex_to_xy(to_simplex(dec.predict(S)))
    roll_r2 = U.belief_decode_r2(S, rb, np.array([prefix_prob[w] for w in rw]))
    print(f"[Mess3 no-ALS]  raw R^2={raw_r2:.3f}  rollout R^2={roll_r2:.3f}  (ALS: 1.00)")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower"); ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(raw_xy, color, px=2), origin="lower")
    ax[1].set_title(f"OBS-OOM subspace (raw)\nR$^2$={raw_r2:.3f}")
    ax[2].imshow(U.rasterize_simplex(roll_xy, np.clip(rb, 0, 1), px=2), origin="lower")
    ax[2].set_title(f"Operator rollout, NO ALS\nbelief-decode R$^2$={roll_r2:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig31_mess3_no_als.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); print(f"saved -> {out}")


def rrxor_fig():
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
    rows, X, P, Yc = F23.transitions(resid, soft, reach, 2, F14.MAX_LEN)
    Wt = np.ones(len(rows))
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, 2)
    B = Uobs[:, :5]
    ops, _ = U.fit_operators(X @ B, P, Yc @ B, Wt)            # NO ALS
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]

    def com_geom(S, Y, ix):
        g = defaultdict(list)
        for i, s in enumerate(ix):
            g[s].append(i)
        com = np.array([S[v].mean(0) for v in g.values()]); bel = np.array([Y[v[0]] for v in g.values()])
        return LinearRegression().fit(com, bel).score(com, bel), g

    S_raw = np.stack([resid[w] for w in allw]) @ B
    cg_raw, g_raw = com_geom(S_raw, Yb, idx)
    xy_raw = pca.transform(LinearRegression().fit(S_raw, Yb).predict(S_raw))[:, [0, 2]]
    proj = {w: resid[w] @ B for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    roll = U.rollout_states(proj, ops, e, F14.MAX_LEN, 2)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S_roll = np.stack([roll[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, g_roll = com_geom(S_roll, Yb_r, idx_r)
    xy_roll = pca.transform(LinearRegression().fit(S_roll, Yb_r).predict(S_roll))[:, [0, 2]]
    print(f"[RRXOR no-ALS]  raw COM={cg_raw:.3f}  rollout COM={cg_roll:.3f}  (ALS: ~0.19)")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax[0].set_title("RRXOR ground truth  (PCA 1 / PCA 3)")
    for panel, xy, grp, ttl in [
        (ax[1], xy_raw, g_raw, f"OBS-OOM subspace (raw)\nCOM-geom R$^2$={cg_raw:.2f}"),
        (ax[2], xy_roll, g_roll, f"Operator rollout, NO ALS\nCOM-geom R$^2$={cg_roll:.2f}"),
    ]:
        for s, gg in grp.items():
            panel.scatter(xy[gg, 0], xy[gg, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
            panel.scatter(*xy[gg].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
        panel.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        panel.set_title(ttl)
    for a in ax:
        a.set_aspect("equal"); a.spines[["top", "right"]].set_visible(False)
        for setl, v in ((a.set_xlim, tgt[:, 0]), (a.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        a.set_xlabel("PCA 1"); a.set_ylabel("PCA 3")
    out = FIG_DIR / "fig31_rrxor_no_als.png"
    fig.tight_layout(); fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white"); print(f"saved -> {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rrxor_fig()
    mess3_fig()
