"""Operator rollout from the SUPERVISED subspace (mixes supervised+unsupervised, on purpose).

fig23/fig24 fit operators inside the UNSUPERVISED observability-OOM subspace. Here we instead
take the SUPERVISED belief readout --- the activation->belief regression coefficients
B_sup --- as the subspace, then apply the *identical* learning procedure (ALS operator-
consistency refit of the rescaled operators a(w)B A_x ~ P(x|w) a(wx)B, recover the eval
functional, roll out from the root). The raw panel is the supervised decode itself; the
rollout panel asks whether the LEARNED operators reproduce the geometry when seeded only at
the root. If RRXOR collapses even here, the collapse is a property of the operator algebra
(T^1 nilpotent, T^0 defective), not of unsupervised subspace quality.

Outputs:
  fig29_mess3_supervised_rollout.png   GT | supervised decode | supervised-subspace rollout
  fig29_rrxor_supervised_rollout.png   GT | supervised decode | supervised-subspace rollout
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.transition_matrices import rrxor
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9
import fig23_unified_best as F23

FIG_DIR = Path(__file__).parent / "figures"


def supervised_pipeline(resid, soft, belief, reach, vocab, d, max_len, allw, Yb, prefix_prob):
    """B_sup = activation->belief regression; same ALS operator-consistency refit as fig23."""
    rows, X, P, Yc = F23.transitions(resid, soft, reach, vocab, max_len)
    Wt = np.array([prefix_prob.get(w, 1.0) for w in rows])
    Xall = np.stack([resid[w] for w in allw])
    w_all = np.array([prefix_prob.get(w, 1.0) for w in allw])  # P(w)-weighted regression (Simplex conv.)
    coef = LinearRegression().fit(Xall, Yb, sample_weight=w_all).coef_.T   # (d_model, n)
    B_dec = np.linalg.qr(coef)[0]                              # ORTHONORMAL supervised subspace
    # Exactly fig23's rollout recipe (ALS operator-consistency refit), but on the SUPERVISED
    # subspace instead of the observability-OOM one. Operators are learned via the rescaling
    # trick, not taken analytically.
    B_op = U.als_refine_basis(X, P, Yc, Wt, B_dec, B_dec.shape[1])
    ops, _ = U.fit_operators(X @ B_op, P, Yc @ B_op, Wt)
    proj = {w: resid[w] @ B_op for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    roll = U.rollout_states(proj, ops, e, max_len, vocab)
    return B_dec, B_op, roll


def mess3_fig():
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}
    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw]); wcol = np.array([prefix_prob[w] for w in allw])
    true_xy = U.simplex_to_xy(Yb); color = np.clip(Yb, 0, 1)

    B_dec, B_op, roll = supervised_pipeline(resid, soft, belief, reach, vocab, 3, max_len, allw, Yb, prefix_prob)

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)
    Xall = np.stack([resid[w] for w in allw])
    dec_raw = LinearRegression().fit(Xall @ B_dec, Yb)
    raw_xy = U.simplex_to_xy(to_simplex(dec_raw.predict(Xall @ B_dec)))
    dec_op = LinearRegression().fit(Xall @ B_op, Yb)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    roll_states = np.stack([roll[w] for w in rw]); roll_b = np.stack([belief[w] for w in rw])
    rwcol = np.array([prefix_prob[w] for w in rw])
    roll_xy = U.simplex_to_xy(to_simplex(dec_op.predict(roll_states)))
    roll_r2 = U.belief_decode_r2(roll_states, roll_b, rwcol)
    print(f"[Mess3] supervised-subspace operator rollout belief-decode R^2 = {roll_r2:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower"); ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(raw_xy, color, px=2), origin="lower")
    ax[1].set_title("Supervised decode (raw)")
    ax[2].imshow(U.rasterize_simplex(roll_xy, np.clip(roll_b, 0, 1), px=2), origin="lower")
    ax[2].set_title(f"Supervised subspace: operator rollout\nbelief-decode R$^2$={roll_r2:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig29_mess3_supervised_rollout.png"
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
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw]); wcol = np.ones(len(allw))
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]

    B_dec, B_op, roll = supervised_pipeline(resid, soft, belief, reach, 2, 5, F14.MAX_LEN, allw, Yb,
                                            F14.analytic_prefix_probs(resid, T, pi))

    def com_geom(S, Y, ix):
        g = defaultdict(list)
        for i, s in enumerate(ix):
            g[s].append(i)
        com = np.array([S[v].mean(0) for v in g.values()]); bel = np.array([Y[v[0]] for v in g.values()])
        return LinearRegression().fit(com, bel).score(com, bel), g

    S_raw = np.stack([resid[w] for w in allw]) @ B_dec
    cg_raw, g_raw = com_geom(S_raw, Yb, idx)
    xy_raw = pca.transform(LinearRegression().fit(S_raw, Yb).predict(S_raw))[:, [0, 2]]
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S_roll = np.stack([roll[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, g_roll = com_geom(S_roll, Yb_r, idx_r)
    xy_roll = pca.transform(LinearRegression().fit(S_roll, Yb_r).predict(S_roll))[:, [0, 2]]
    print(f"[RRXOR] supervised raw COM-geom={cg_raw:.3f}   supervised-subspace rollout COM-geom={cg_roll:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax[0].set_title("RRXOR ground truth  (PCA 1 / PCA 3)")
    for panel, xy, grp, ttl in [
        (ax[1], xy_raw, g_raw, f"Supervised decode (raw)\nCOM-geom R$^2$={cg_raw:.2f}"),
        (ax[2], xy_roll, g_roll, f"Supervised subspace: operator rollout\nCOM-geom R$^2$={cg_roll:.2f}"),
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
    out = FIG_DIR / "fig29_rrxor_supervised_rollout.png"
    fig.tight_layout(); fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white"); print(f"saved -> {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rrxor_fig()
    mess3_fig()
