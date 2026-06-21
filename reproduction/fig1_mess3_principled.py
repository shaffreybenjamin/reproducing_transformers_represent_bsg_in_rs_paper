"""Mess3 GT | supervised | unsupervised comparison (the Mess3 analogue of fig26).

Everything identical except the affine-map input: panel 2 decodes the FULL single-layer
residual stream (64-D) onto belief; panel 3 decodes the recovered observability-OOM 5/3-D
subspace. Same enumerated prefixes, same simplex projection, same datashader raster, same
label-fit placement (display only). Demonstrates the unsupervised subspace reproduces the
supervised Mess3 fractal.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig8_10_unified_best as F23

FIG_DIR = Path(__file__).parent / "figures"
D = 3


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85})
    vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    reach = {w: True for w in resid}

    # unsupervised observability-OOM subspace (single layer), P(w)-weighted
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)
    B = Uobs[:, :D]

    seqs = [s for s in resid if s in belief]
    true_b = np.array([belief[s] for s in seqs])
    true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1)
    wcol = np.array([prefix_prob[s] for s in seqs])
    Xfull = np.stack([resid[s] for s in seqs])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    def decode_panel(feat):
        r2 = U.belief_decode_r2(feat, true_b, wcol)
        dec = LinearRegression().fit(feat, true_b)
        xy = U.simplex_to_xy(to_simplex(dec.predict(feat)))
        return xy, r2

    xy_sup, r2_sup = decode_panel(Xfull)
    xy_uns, r2_uns = decode_panel(Xfull @ B)
    print(f"Mess3 belief-decode R^2:  supervised(64-D)={r2_sup:.3f}   unsupervised({D}-D)={r2_uns:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower")
    ax[0].set_title("Mess3 ground truth")
    ax[1].imshow(U.rasterize_simplex(xy_sup, color, px=2), origin="lower")
    ax[1].set_title(f"Supervised: full residual stream (64-D)\nbelief-decode R$^2$={r2_sup:.3f}")
    ax[2].imshow(U.rasterize_simplex(xy_uns, color, px=2), origin="lower")
    ax[2].set_title(f"Unsupervised: spectral-OOM subspace ({D}-D)\nbelief-decode R$^2$={r2_uns:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig28_mess3_principled.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
