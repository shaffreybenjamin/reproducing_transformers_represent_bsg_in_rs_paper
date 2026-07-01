"""Wing GT | supervised | Stage 1 CCA/RRR (minimal wrapper).

Identical to fig1_wing_principled.py but uses Stage 1 CCA/RRR subspace.
To revert: delete this file.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U
import stage1_cca_general as S1

WING_PARAMS = {"x": 0.5, "y": 0.5}
FIG_DIR = Path(__file__).parent / "figures"
MODEL_DIR = Path(__file__).parent / "models"
D = 3


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "wing_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("wing", WING_PARAMS)
    vocab = hmm.vocab_size
    model, ctx, step = load_model(device)
    print(f"loaded wing checkpoint (step={step}, ctx={ctx}, device={device})")

    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, device)
    good = {
        w for w in resid
        if w in belief and np.all(np.isfinite(belief[w])) and prefix_prob.get(w, 0.0) > 1e-12
    }
    print(f"enumerated MSP prefixes: {len(resid)}  reachable (prob>0): {len(good)}")
    reach = {w: (w in good) for w in resid}

    # Stage 1 subspace
    B_stage1 = S1.run_stage1(resid, soft, reach, vocab, d_keep=D)
    print(f"Stage 1 subspace shape: {B_stage1.shape}")

    seqs = [s for s in resid if s in good]
    true_b = np.array([belief[s] for s in seqs])
    true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1)
    wcol = np.array([prefix_prob[s] for s in seqs])
    Xfull = np.stack([resid[s] for s in seqs])

    def to_simplex(p):
        p = np.clip(p, 0, None)
        return p / p.sum(1, keepdims=True)

    def decode_panel(feat):
        r2 = U.belief_decode_r2(feat, true_b, wcol)
        dec = LinearRegression().fit(feat, true_b)
        xy = U.simplex_to_xy(to_simplex(dec.predict(feat)))
        return xy, r2

    xy_sup, r2_sup = decode_panel(Xfull)
    xy_s1, r2_s1 = decode_panel(Xfull @ B_stage1)
    print(f"wing belief-decode R^2:  supervised(64-D)={r2_sup:.3f}   stage1({D}-D)={r2_s1:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(U.rasterize_simplex(true_xy, color, px=2), origin="lower")
    ax[0].set_title("Wing ground truth")
    ax[1].imshow(U.rasterize_simplex(xy_sup, color, px=2), origin="lower")
    ax[1].set_title(f"Supervised: full residual stream (64-D)\nbelief-decode R$^2$={r2_sup:.3f}")
    ax[2].imshow(U.rasterize_simplex(xy_s1, color, px=2), origin="lower")
    ax[2].set_title(f"Stage 1 CCA/RRR ({D}-D)\nbelief-decode R$^2$={r2_s1:.3f}")
    for a in ax:
        a.axis("off")
    out = FIG_DIR / "fig1_wing_principled_stage1.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
