"""Reproduce the paper's Mess3 belief-geometry figure (Shai et al.).

Pipeline:
  1. Enumerate ALL length-n_ctx token sequences (3^n_ctx) and run the model,
     taking per-position residual stream activations.
  2. Ground-truth belief at every position from the analytic MSP.
  3. P(w)-weighted linear regression: activations -> belief (consistent with fig10 and Section B).
  4. Project predicted/true beliefs to the 2-simplex and rasterize with datashader style.

Two panels: "Belief State Geometry" (ground truth) | "Residual Stream" (regression decode).
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import grey_dilation
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator

MESS3_PARAMS = {"x": 0.05, "a": 0.85}  # paper values (Appendix A.3)
BATCH = 4096
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def project_to_simplex(p: np.ndarray):
    """Paper's projection: state-2=blue top, state-0=red bottom-left, state-1=green bottom-right."""
    return p[:, 1] + 0.5 * p[:, 2], (np.sqrt(3) / 2.0) * p[:, 2]


def _disk(px: int) -> np.ndarray:
    r = np.arange(-px, px + 1)
    yy, xx = np.meshgrid(r, r)
    return (xx ** 2 + yy ** 2) <= px ** 2


def rasterize_simplex(x, y, color_rgb, px=2, W=1000, H=900):
    """Datashader-style render (matches the paper's tf.shade + tf.spread):
    bin points into a fixed WxH grid, colour each occupied cell by the mean RGB
    of points in it, then SPREAD (dilate) by `px` pixels so the fixed point set
    fills the body; empty cells stay white -> clean internal-triangle holes.
    Density no longer depends on figure size/DPI.
    """
    x0, x1 = -0.05, 1.05
    y0, y1 = -0.05, np.sqrt(3) / 2 + 0.05
    ix = np.clip(((x - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
    iy = np.clip(((y - y0) / (y1 - y0) * H).astype(int), 0, H - 1)
    sums = np.zeros((H, W, 3))
    cnt = np.zeros((H, W))
    np.add.at(sums, (iy, ix), color_rgb)
    np.add.at(cnt, (iy, ix), 1)
    occ = cnt > 0
    mean = np.zeros((H, W, 3))
    mean[occ] = sums[occ] / cnt[occ, None]
    fp = _disk(px)
    occ_d = grey_dilation(occ.astype(np.uint8), footprint=fp) > 0
    mean_d = np.stack([grey_dilation(mean[:, :, c], footprint=fp) for c in range(3)], axis=-1)
    img = np.ones((H, W, 3))                      # white background
    img[occ_d] = np.clip(mean_d[occ_d], 0, 1)
    return img


def load_model(device: str):
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def collect_all_sequences(model, hmm, context_len, device):
    """All length-context_len sequences, every position: activations + analytic belief."""
    vocab = hmm.vocab_size
    layer = model.cfg.n_layers - 1

    # prefix -> analytic belief, from the MSP (display/scoring target only)
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=context_len).generate()
    pref2bel = {
        tuple(int(t) for t in seq): np.asarray(v.belief_state, dtype=np.float32)
        for seq, v in tree.nodes.items() if len(seq) > 0
    }

    seqs = np.array(list(itertools.product(range(vocab), repeat=context_len)), dtype=np.int64)
    acts, bels = [], []
    for i in range(0, len(seqs), BATCH):
        chunk = seqs[i : i + BATCH]
        inp = torch.from_numpy(chunk).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, names_filter=f"blocks.{layer}.hook_resid_post")
        acts.append(cache["resid_post", layer].cpu().numpy())  # (b, L, d)
        bb = np.empty((len(chunk), context_len, 3), dtype=np.float32)
        for r, s in enumerate(chunk):
            pref = ()
            for j in range(context_len):
                pref = pref + (int(s[j]),)
                bb[r, j] = pref2bel[pref]
        bels.append(bb)

    A = np.concatenate(acts, 0).reshape(-1, model.cfg.d_model)
    Y = np.concatenate(bels, 0).reshape(-1, 3)

    # Compute prefix probabilities (consistent with fig10 and Section B)
    prefix_probs = np.empty(len(A))
    idx = 0
    for L in sorted(by_len):
        seqs, bels, probs = by_len[L]
        for i, (seq, p) in enumerate(zip(seqs, probs)):
            prefix_probs[idx] = p
            idx += 1
    return A, Y, prefix_probs


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model, context_len = load_model(device)

    A, Y, prefix_probs = collect_all_sequences(model, hmm, context_len, device)
    print(f"points (all positions of all 3^{context_len} sequences): {A.shape[0]}  d_model={A.shape[1]}")

    reg = LinearRegression().fit(A, Y, sample_weight=prefix_probs)
    Yhat = reg.predict(A)
    print(f"residual -> belief  R^2 = {reg.score(A, Y, sample_weight=prefix_probs):.4f}")

    tx, ty = project_to_simplex(Y)
    px_, py_ = project_to_simplex(Yhat)
    rgb = np.clip(Y, 0, 1)                        # colour by ground-truth belief (both panels)

    img_true = rasterize_simplex(tx, ty, rgb, px=2)
    img_pred = rasterize_simplex(px_, py_, rgb, px=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(img_true, origin="lower")
    axes[0].set_title("Belief State Geometry")
    axes[1].imshow(img_pred, origin="lower")
    axes[1].set_title("Residual Stream")
    for ax in axes:
        ax.axis("off")
    out = FIG_DIR / "fig03_residual_belief_geometry_mess3.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
