"""Figure 6 (top) reproduction: belief-geometry emergence over training.

Four training checkpoints (paper points, in trained tokens: 64000, 640000,
3187200, 629209600 -> steps ~100, 1000, 5000, 1000000 at 64x10=640 tok/step):
  - top row    : SUPERVISED residual-stream decode (per-checkpoint unweighted
                 linear regression activations -> belief, predicted beliefs).
  - bottom row : UNSUPERVISED raw activations projected onto the per-checkpoint
                 dynamics subspace (CCA -> ALS), affine-aligned for display.

Early steps 100/1000/5000 come from train_progression_checkpoints.py (same seed
42 / data order as the canonical run, so on the same trajectory); the final point
reuses the committed 1,000,000-step model. Same paper projection + datashader-
style rasterizer as fig03/fig04.
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

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
TOK_PER_STEP = 64 * 10  # batch x context
LATENT_D = 3

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"

CKPTS = [
    MODEL_DIR / "progression" / "step_100.pt",
    MODEL_DIR / "progression" / "step_1000.pt",
    MODEL_DIR / "progression" / "step_5000.pt",
    MODEL_DIR / "mess3_transformer.pt",
]


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["context_len"], ck["step"]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size

    fig, axes = plt.subplots(2, len(CKPTS), figsize=(4 * len(CKPTS), 8))
    for col, path in enumerate(CKPTS):
        model, context_len, step = load(path, device)
        resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(
            model, hmm, context_len, device
        )
        seqs = [s for s in resid if s in belief]
        true_b = np.array([belief[s] for s in seqs])
        true_xy = U.simplex_to_xy(true_b)
        color = np.clip(true_b, 0, 1)
        wcol = np.array([prefix_prob[s] for s in seqs])
        Xfull = np.stack([resid[s] for s in seqs])

        # --- top: supervised residual-stream decode (per-checkpoint regression) ---
        reg = LinearRegression().fit(Xfull, true_b)
        sup_xy = U.simplex_to_xy(reg.predict(Xfull))
        axes[0, col].imshow(U.rasterize_simplex(sup_xy, color, px=2, auto=True), origin="lower")

        # --- bottom: unsupervised raw activations in the recovered subspace ---
        # spectral-OOM + P(w) (consistent with the rest of the unsupervised section)
        reach = {w: True for w in resid}
        basis = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)[4][:, :LATENT_D]
        raw_xy = U.plane_coords(Xfull @ basis)
        raw_aligned, _ = U.affine_align(raw_xy, true_xy, wcol)
        axes[1, col].imshow(U.rasterize_simplex(raw_aligned, color, px=2, auto=True), origin="lower")

        tok = step * TOK_PER_STEP
        axes[0, col].set_title(f"step {step:,}\n({tok:.2e} tokens)")
        print(f"col {col}: step {step}, {len(seqs)} prefixes")

    axes[0, 0].set_ylabel("Supervised\n(residual stream)", fontsize=12)
    axes[1, 0].set_ylabel("Unsupervised\n(spectral-OOM subspace)", fontsize=12)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    # keep ylabels visible (axis off would hide them)
    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Belief-state geometry emerges over training  (Mess3)", fontsize=14)
    out = FIG_DIR / "fig06_training_progression_mess3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
