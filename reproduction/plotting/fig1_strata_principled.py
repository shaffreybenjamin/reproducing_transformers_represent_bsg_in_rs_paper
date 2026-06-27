"""Strata GT | supervised | unsupervised comparison.

The strata analogue of fig1_mess3_principled.py (Figure 1 in writeup_v1.tex): a 3-panel
belief-geometry comparison.
  * Panel 1: analytic ground-truth belief simplex.
  * Panel 2: supervised decode of belief from the FULL single-layer residual stream (64-D).
  * Panel 3: unsupervised decode from the observability-OOM subspace (D-D, D = #hidden states).

strata has 3 hidden states (vocab 2), so its belief lives on the same 2-simplex triangle as
Mess3 and reuses the identical projection / datashader raster / belief-decode metric. strata is
non-unifilar, so its belief should not collapse to the vertices.

Uses the strata checkpoint (the overnight run was interrupted by a Windows Update reboot at
~200k steps; the model is already at its optimal windowed loss).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import unsupervised_belief_oom as U
import fig14_observable_oom as F14

STRATA_PARAMS = {"a": 0.85, "t0": 0.5, "t1": 0.5}
FIG_DIR = Path(__file__).parent.parent / "figures"
MODEL_DIR = Path(__file__).parent.parent / "models"
D = 3   # = number of hidden states


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "strata_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device  # checkpoint trained on cuda; honor the local device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("strata", STRATA_PARAMS)
    vocab = hmm.vocab_size
    model, ctx, step = load_model(device)
    print(f"loaded strata checkpoint (step={step}, ctx={ctx}, device={device})")

    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, device)
    # keep only reachable prefixes (impossible token prefixes get prob 0 / NaN belief in the MSP tree)
    good = {
        w for w in resid
        if w in belief and np.all(np.isfinite(belief[w])) and prefix_prob.get(w, 0.0) > 1e-12
    }
    print(f"enumerated MSP prefixes: {len(resid)}  reachable (prob>0): {len(good)}  (lengths 1..{max_len})")
    reach = {w: (w in good) for w in resid}

    # unsupervised observability-OOM subspace (single layer), P(w)-weighted
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)
    B = Uobs[:, :D]

    seqs = [s for s in resid if s in good]
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
    print(f"strata belief-decode R^2:  supervised(64-D)={r2_sup:.3f}   unsupervised({D}-D)={r2_uns:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower")
    ax[0].set_title("Strata ground truth")
    ax[1].imshow(U.rasterize_simplex(xy_sup, color, px=2), origin="lower")
    ax[1].set_title(f"Supervised: full residual stream (64-D)\nbelief-decode R$^2$={r2_sup:.3f}")
    ax[2].imshow(U.rasterize_simplex(xy_uns, color, px=2), origin="lower")
    ax[2].set_title(f"Unsupervised: spectral-OOM subspace ({D}-D)\nbelief-decode R$^2$={r2_uns:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig1_strata_principled.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
