"""Mess3 operator rollout with UNIFORM weighting + ALS (isolates the weighting knob).

Reference points:  uniform + no-ALS = fig16 (~0.80);  prob-weight + ALS = fig23 (1.00).
This figure keeps ALS but switches the operator-fit weighting to uniform, to see how much
of fig23's crispness is the ALS step versus the probability weighting.
GT | OBS-OOM subspace (raw) | operator rollout (uniform + ALS). R^2 scored prob-weighted
(belief-decode), identical to fig16/fig23 so the numbers are comparable.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig23_unified_best as F23

FIG_DIR = Path(__file__).parent / "figures"
D = 3


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}
    rows, X, P, Yc = F23.transitions(resid, soft, reach, vocab, max_len)
    Wt = np.ones(len(rows))                                   # UNIFORM operator-fit weighting

    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab)
    B_dec = Uobs[:, :D]
    B_op = U.als_refine_basis(X, P, Yc, Wt, B_dec, D)         # uniform + ALS
    ops, _ = U.fit_operators(X @ B_op, P, Yc @ B_op, Wt)

    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    true_b = np.stack([belief[w] for w in allw]); true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1); wcol = np.array([prefix_prob[w] for w in allw])
    Xall = np.stack([resid[w] for w in allw])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    raw_r2 = U.belief_decode_r2(Xall @ B_dec, true_b, wcol)
    dec_raw = LinearRegression().fit(Xall @ B_dec, true_b)
    raw_xy = U.simplex_to_xy(to_simplex(dec_raw.predict(Xall @ B_dec)))

    proj = {w: resid[w] @ B_op for w in resid}
    e, _ = U.recover_eval_functional(np.stack([proj[w] for w in allw]))
    roll = U.rollout_states(proj, ops, e, max_len, vocab)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    roll_states = np.stack([roll[w] for w in rw]); roll_b = np.stack([belief[w] for w in rw])
    rwcol = np.array([prefix_prob[w] for w in rw])
    dec_op = LinearRegression().fit(Xall @ B_op, true_b)
    roll_xy = U.simplex_to_xy(to_simplex(dec_op.predict(roll_states)))
    roll_r2 = U.belief_decode_r2(roll_states, roll_b, rwcol)
    print(f"Mess3 uniform+ALS:  raw R^2={raw_r2:.3f}   operator-rollout R^2={roll_r2:.3f}   "
          f"(cf. uniform+noALS~0.80 [fig16], prob+ALS=1.00 [fig23])")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower"); ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(raw_xy, color, px=2), origin="lower")
    ax[1].set_title(f"OBS-OOM subspace (raw)\nbelief-decode R$^2$={raw_r2:.3f}")
    ax[2].imshow(U.rasterize_simplex(roll_xy, np.clip(roll_b, 0, 1), px=2), origin="lower")
    ax[2].set_title(f"Operator rollout (uniform + ALS)\nbelief-decode R$^2$={roll_r2:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig30_mess3_uniform_als.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
