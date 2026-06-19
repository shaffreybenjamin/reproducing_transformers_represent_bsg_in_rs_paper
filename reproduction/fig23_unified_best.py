"""The single best method (OBSERVABILITY OOM), rendered for visual verification, and a
direct same-data / same-metric comparison against the fig04 method (predictive CCA+ALS).

Why OBS-OOM (vs the horizon-h RRR I first tried): a clean operator ROLLOUT needs the
subspace to be OPERATOR-CONSISTENT -- built from the same rescaled-children relation the
operators use. OBS-OOM is (its observability matrix is [C, G_xC, ...] with G_x the
rescaled-children operators), so its rollout is stable. RRR-onto-softmax-future is great
for DECODING but operator-inconsistent, so its rollout diverged (the earlier broken panel).

Outputs (overwrite the earlier unified-method files):
  fig23_mess3_unified.png : Mess3 simplex fractal | OBS-OOM raw | OBS-OOM operator rollout
  fig24_rrxor_unified.png : RRXOR PCA(1,3) GT     | OBS-OOM raw | OBS-OOM operator rollout
Console: Mess3 raw & rollout align-R^2 for OBS-OOM vs CCA+ALS, on the SAME single-layer data.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor, mess3
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9
import fig16_oom_mess3_geometry as F16

FIG_DIR = Path(__file__).parent / "figures"


def transitions(resid, soft, reach, vocab, max_len):
    rows = [w for w in resid if reach[w] and len(w) < max_len
            and all((w + (x,)) in resid for x in range(vocab))]
    X = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    Yc = np.stack([[resid[w + (x,)] for x in range(vocab)] for w in rows])
    return rows, X, P, Yc


# --------------------------------------------------------------------------- #
# Mess3: OBS-OOM vs CCA+ALS on identical single-layer data
# --------------------------------------------------------------------------- #
def mess3_figure():
    from simplexity.generative_processes.builder import build_hidden_markov_model
    d = 3
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85})
    vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    # single layer + analytic prefix probabilities (the fig04 data path & weighting)
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}
    rows, X, P, Yc = transitions(resid, soft, reach, vocab, max_len)
    Wt = np.array([prefix_prob[w] for w in rows])

    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    true_b = np.stack([belief[w] for w in allw])
    true_xy = U.simplex_to_xy(true_b); color = np.clip(true_b, 0, 1)
    wcol = np.array([prefix_prob[w] for w in allw])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    def raw_and_rollout(B, ops, e):
        # FAIR prob-weighted belief-decode R^2 (fig04's metric); robust to the low-prob tail
        Araw = np.stack([resid[w] for w in allw]) @ B
        raw_r2 = U.belief_decode_r2(Araw, true_b, wcol)
        decode = LinearRegression().fit(Araw, true_b)              # readout for display
        raw_xy = U.simplex_to_xy(to_simplex(decode.predict(Araw)))
        proj = {w: resid[w] @ B for w in resid}
        roll = U.rollout_states(proj, ops, e, max_len, vocab)
        rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
        roll_states = np.stack([roll[w] for w in rw])
        roll_b = np.stack([belief[w] for w in rw]); rwcol = np.array([prefix_prob[w] for w in rw])
        roll_r2 = U.belief_decode_r2(roll_states, roll_b, rwcol)
        roll_xy = U.simplex_to_xy(to_simplex(decode.predict(roll_states)))
        rcol = np.clip(roll_b, 0, 1)
        return raw_r2, roll_r2, raw_xy, color, roll_xy, rcol

    # --- OBS+ALS: deep P(w)-weighted observability subspace, THEN ALS operator-consistency refit ---
    orows, A, oP, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)
    Bo = Uobs[:, :d]
    Boa = U.als_refine_basis(X, P, Yc, Wt, Bo, d)              # ALS = clean rollout (the fix)
    ops_o, _ = U.fit_operators(X @ Boa, P, Yc @ Boa, Wt)
    e_o, _ = U.recover_eval_functional(X @ Boa)
    obs = raw_and_rollout(Boa, ops_o, e_o)

    # --- CCA (+ALS if better held-out) = the fig04 method, prob-weighted per fig04 ---
    dirs, _ = U.predictive_cca(X, P, Yc, Wt)
    Bc = dirs[:, :d]
    rng = np.random.default_rng(0); val = rng.random(len(rows)) < 0.3; tr = ~val
    Bc_als = U.als_refine_basis(X[tr], P[tr], Yc[tr], Wt[tr], Bc, d)
    def hocheck(B):
        o, _ = U.fit_operators(X[tr] @ B, P[tr], Yc[tr] @ B, Wt[tr])
        return U.rescaled_r2(X[val] @ B, P[val], Yc[val] @ B, Wt[val], o)
    if hocheck(Bc_als) > hocheck(Bc) + 1e-4:
        Bc = U.als_refine_basis(X, P, Yc, Wt, Bc, d); cca_tag = "CCA+ALS"
    else:
        cca_tag = "CCA    "
    ops_c, _ = U.fit_operators(X @ Bc, P, Yc @ Bc, Wt)
    e_c, _ = U.recover_eval_functional(X @ Bc)
    cca = raw_and_rollout(Bc, ops_c, e_c)

    print("\n=== Mess3 (single layer, same data, prob-weighted belief-decode R^2) ===")
    print(f"  {cca_tag}  raw={cca[0]:.4f}   operator-rollout={cca[1]:.4f}   [fig04 method]")
    print(f"  OBS+ALS   raw={obs[0]:.4f}   operator-rollout={obs[1]:.4f}")

    raw_r2, roll_r2, raw_xy, col, roll_xy, rcol = obs
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower")
    ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(raw_xy, col, px=2), origin="lower")
    ax[1].set_title(f"Spectral-OOM - raw activations\n(belief-decode R$^2$={raw_r2:.3f})")
    ax[2].imshow(U.rasterize_simplex(roll_xy, rcol, px=2), origin="lower")
    ax[2].set_title(f"Spectral-OOM - operator rollout\n(belief-decode R$^2$={roll_r2:.3f})")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig23_mess3_unified.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- #
# RRXOR: OBS-OOM raw vs rollout
# --------------------------------------------------------------------------- #
def com_geometry(S, Yb, idx):
    groups = defaultdict(list)
    for i, s in enumerate(idx):
        groups[s].append(i)
    com = np.array([S[g].mean(0) for g in groups.values()])
    bel = np.array([Yb[g[0]] for g in groups.values()])
    return LinearRegression().fit(com, bel).score(com, bel), groups


def rrxor_figure():
    T, pi, d = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0, 5
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
    vocab = 2

    pw = F14.analytic_prefix_probs(resid, T, pi)             # P(w) sample weights
    orows, A, oP, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, vocab, wmap=pw)
    B = Uobs[:, :d]
    # consistent recipe: P(w)-weighted operators + ALS (rollout uses B_op; raw panel uses B)
    rows, X, Pm, Yc = transitions(resid, soft, reach, vocab, F14.MAX_LEN)
    Wt = np.array([pw[w] for w in rows])
    B_op = U.als_refine_basis(X, Pm, Yc, Wt, B, d)
    ops, _ = U.fit_operators(X @ B_op, Pm, Yc @ B_op, Wt)

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    colors = np.array(B9.distinct_colors(len(B36)))
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]

    S_raw = np.stack([resid[w] @ B for w in allw])
    cg_raw, groups = com_geometry(S_raw, Yb, idx)
    xy_raw = pca.transform(LinearRegression().fit(S_raw, Yb).predict(S_raw))[:, [0, 2]]

    proj = {w: resid[w] @ B_op for w in resid if reach[w]}
    e, _ = U.recover_eval_functional(np.stack(list(proj.values())))
    roll = U.rollout_states(proj, ops, e, F14.MAX_LEN, vocab)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S_roll = np.stack([roll[w] for w in rw]); Yb_r = np.stack([belief[w] for w in rw])
    idx_r = np.array([index[tuple(np.round(belief[w], 5))] for w in rw])
    cg_roll, groups_r = com_geometry(S_roll, Yb_r, idx_r)
    xy_roll = pca.transform(LinearRegression().fit(S_roll, Yb_r).predict(S_roll))[:, [0, 2]]
    print(f"\n=== RRXOR (OBS-OOM) ===  raw COM-geom={cg_raw:.3f}   operator-rollout COM-geom={cg_roll:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    ax[0].scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax[0].set_title("RRXOR ground truth\n(PCA 1 / PCA 3)")
    for panel, xy, grp, cg, ttl in [
        (ax[1], xy_raw, groups, cg_raw, f"Spectral-OOM - raw activations\nCOM-geom R$^2$={cg_raw:.2f}"),
        (ax[2], xy_roll, groups_r, cg_roll, f"Spectral-OOM - operator rollout\nCOM-geom R$^2$={cg_roll:.2f}"),
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
    out = FIG_DIR / "fig24_rrxor_unified.png"
    fig.tight_layout(); fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rrxor_figure()
    mess3_figure()
