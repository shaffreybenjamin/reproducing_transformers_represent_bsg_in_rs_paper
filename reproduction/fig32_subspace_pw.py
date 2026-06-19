"""Probability-weighted observability-OOM SUBSPACE recovery (vs the current unweighted one).

The OBS subspace comes from the SVD of the observability matrix [C, G_xC, ...], where C and
G_x are ridge regressions of the softmax / rescaled children on the activation. Those
regressions are currently UNWEIGHTED; here we weight them by P(w) (sample_weight), so the
operators -- and hence the subspace -- are estimated toward the high-probability prefixes.
Display/placement is held identical across panels so the only change is the subspace.

  fig32_mess3_subspace_pw.png   GT | unweighted subspace | P(w)-weighted subspace
  fig32_rrxor_subspace_pw.png   GT | unweighted subspace | P(w)-weighted subspace
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.transition_matrices import rrxor
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9

FIG_DIR = Path(__file__).parent / "figures"


def obs_subspace_w(resid, soft, reach, vocab, wmap, depth=3):
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
            for x in range(vocab) if soft[w][x] > F14.EPS)]
    A = np.stack([resid[w] for w in rows]); Pm = np.stack([soft[w] for w in rows])
    sw = np.array([max(wmap.get(w, 0.0), 1e-12) for w in rows])
    C = Ridge(alpha=F14.RIDGE, fit_intercept=False).fit(A, Pm, sample_weight=sw).coef_.T
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > F14.EPS for w in rows])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        Gs.append(Ridge(alpha=F14.RIDGE, fit_intercept=False)
                  .fit(A[m], Pm[m, x][:, None] * child, sample_weight=sw[m]).coef_.T)
    cols, fr = [C], [C]
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
    return np.linalg.svd(np.hstack(cols), full_matrices=False)[0]


def mess3_fig():
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}
    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    true_b = np.stack([belief[w] for w in allw]); true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1); wcol = np.array([prefix_prob[w] for w in allw])
    Xall = np.stack([resid[w] for w in allw])

    uni = {w: 1.0 for w in resid}
    B_u = obs_subspace_w(resid, soft, reach, vocab, uni)[:, :3]
    B_w = obs_subspace_w(resid, soft, reach, vocab, prefix_prob)[:, :3]

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    def panel(B):
        r2 = U.belief_decode_r2(Xall @ B, true_b, wcol)
        dec = LinearRegression().fit(Xall @ B, true_b)
        return U.simplex_to_xy(to_simplex(dec.predict(Xall @ B))), r2

    xy_u, r2_u = panel(B_u); xy_w, r2_w = panel(B_w)
    print(f"[Mess3] subspace decode R^2:  unweighted={r2_u:.3f}   P(w)-weighted={r2_w:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower"); ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(xy_u, color, px=2), origin="lower")
    ax[1].set_title(f"OBS subspace, unweighted\nR$^2$={r2_u:.3f}")
    ax[2].imshow(U.rasterize_simplex(xy_w, color, px=2), origin="lower")
    ax[2].set_title(f"OBS subspace, $P(w)$-weighted\nR$^2$={r2_w:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig32_mess3_subspace_pw.png"
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

    def pw(w):
        b, p = pi.copy(), 1.0
        for x in w:
            d = np.array([(b @ T[i]).sum() for i in range(T.shape[0])])
            if d[x] < 1e-12:
                return 0.0
            p *= float(d[x]); b = b @ T[x] / d[x]
        return p
    wmap = {w: pw(w) for w in resid}
    uni = {w: 1.0 for w in resid}

    B_u = obs_subspace_w(resid, soft, reach, 2, uni)[:, :5]
    B_w = obs_subspace_w(resid, soft, reach, 2, wmap)[:, :5]

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]

    def panel(B):
        S = np.stack([resid[w] for w in allw]) @ B
        g = defaultdict(list)
        for i, s in enumerate(idx):
            g[s].append(i)
        com = np.array([S[v].mean(0) for v in g.values()]); bel = np.array([Yb[v[0]] for v in g.values()])
        cg = LinearRegression().fit(com, bel).score(com, bel)
        xy = pca.transform(LinearRegression().fit(S, Yb).predict(S))[:, [0, 2]]
        return xy, cg, g

    xy_u, cg_u, g_u = panel(B_u); xy_w, cg_w, g_w = panel(B_w)
    print(f"[RRXOR] subspace COM-geom:  unweighted={cg_u:.3f}   P(w)-weighted={cg_w:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax[0].set_title("RRXOR ground truth  (PCA 1 / PCA 3)")
    for a, xy, g, cg, ttl in [
        (ax[1], xy_u, g_u, cg_u, f"OBS subspace, unweighted\nCOM-geom R$^2$={cg_u:.2f}"),
        (ax[2], xy_w, g_w, cg_w, f"OBS subspace, $P(w)$-weighted\nCOM-geom R$^2$={cg_w:.2f}"),
    ]:
        for s, gg in g.items():
            a.scatter(xy[gg, 0], xy[gg, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
            a.scatter(*xy[gg].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
        a.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        a.set_title(ttl)
    for a in ax:
        a.set_aspect("equal"); a.spines[["top", "right"]].set_visible(False)
        for setl, v in ((a.set_xlim, tgt[:, 0]), (a.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        a.set_xlabel("PCA 1"); a.set_ylabel("PCA 3")
    out = FIG_DIR / "fig32_rrxor_subspace_pw.png"
    fig.tight_layout(); fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white"); print(f"saved -> {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rrxor_fig()
    mess3_fig()
