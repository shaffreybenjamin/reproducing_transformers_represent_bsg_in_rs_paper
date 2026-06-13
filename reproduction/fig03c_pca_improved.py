"""Step 3 (PCA refinement) - cleaner unsupervised PCA of the residual stream.

This explores whether two changes make the *unsupervised* PCA layout agree better
with the supervised regression (and the ground-truth fractal):

  1. Subtract the per-position mean. Each prefix's "position" is its length L
     (final token index L-1 in the model). Residual-stream norm/offset grows with
     position, injecting a large belief-irrelevant variance axis. Removing the
     per-position mean kills that nuisance direction.
  2. Weight PCA by prefix probability. Plain PCA is dominated by the dense common
     beliefs near the simplex center; probability weighting matches the regression's
     implicit weighting and the natural density of the process.

We still color by ground-truth belief. PCA remains unsupervised (Y is never used
to choose directions), so orientation/scale are still arbitrary - we only expect
the *structure* to sharpen, not to land in the exact simplex frame.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator

MESS3_PARAMS = {"x": 0.05, "a": 0.85}  # paper values (Appendix A.3)
MSP_DEPTH = 10
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def simplex_to_xy(beliefs: np.ndarray) -> np.ndarray:
    theta = np.pi / 3.0
    basis = np.array([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
    return beliefs[:, :2] @ basis


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    model = HookedTransformer(HookedTransformerConfig.from_dict(ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def enumerate_msp(hmm, depth):
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=depth).generate()
    by_len = defaultdict(lambda: ([], [], []))
    for seq, v in tree.nodes.items():
        if len(seq) == 0:
            continue
        seqs, bels, probs = by_len[len(seq)]
        seqs.append(seq)
        bels.append(v.belief_state)
        probs.append(v.probability)
    return {L: (np.array(s, dtype=np.int64), np.array(b), np.array(p)) for L, (s, b, p) in by_len.items()}


def collect_residuals(model, by_len, context_len, device):
    """Return residuals X, beliefs Y, weights w, and position (prefix length) pos."""
    layer = model.cfg.n_layers - 1
    Xs, Ys, ws, poss = [], [], [], []
    for L in sorted(by_len):
        if L > context_len:
            continue
        seqs, bels, probs = by_len[L]
        inp = torch.from_numpy(seqs).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, names_filter=f"blocks.{layer}.hook_resid_post")
        resid = cache["resid_post", layer][:, -1, :].cpu().numpy()
        Xs.append(resid)
        Ys.append(bels)
        ws.append(probs)
        poss.append(np.full(len(seqs), L))
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(ws), np.concatenate(poss))


def subtract_per_position_mean(X, w, pos):
    """Subtract, from each row, the prob-weighted mean residual of its position group."""
    Xout = X.copy()
    for L in np.unique(pos):
        m = pos == L
        wl = w[m]
        mean_l = np.average(X[m], axis=0, weights=wl)
        Xout[m] = X[m] - mean_l
    return Xout


def weighted_pca_2d(X, w):
    """Top-2 directions of the probability-weighted covariance of X."""
    mean = np.average(X, axis=0, weights=w)
    Xc = X - mean
    W = w.sum()
    cov = (Xc * w[:, None]).T @ Xc / W
    evals, evecs = np.linalg.eigh(cov)  # ascending
    top2 = evecs[:, ::-1][:, :2]  # largest two
    return Xc @ top2


def naive_pca_2d(X):
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def weighted_ols(X, Y, w):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=None)
    return Xb @ beta


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model, context_len = load_model(device)

    by_len = enumerate_msp(hmm, MSP_DEPTH)
    X, Y, w, pos = collect_residuals(model, by_len, context_len, device)
    print(f"prefixes: {X.shape[0]}  positions: {sorted(np.unique(pos).tolist())}")

    color = np.clip(Y, 0, 1)

    yhat = weighted_ols(X, Y, w)
    pcs_naive = naive_pca_2d(X)
    Xclean = subtract_per_position_mean(X, w, pos)
    pcs_improved = weighted_pca_2d(Xclean, w)

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    panels = [
        (simplex_to_xy(Y), "Ground truth (MSP)"),
        (simplex_to_xy(yhat), "Regression (supervised)"),
        (pcs_naive, "Naive PCA"),
        (pcs_improved, "PCA: per-position mean removed\n+ probability-weighted"),
    ]
    for ax, (xy, title) in zip(axes, panels):
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=2, edgecolors="none")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    out = FIG_DIR / "fig03c_pca_improved_mess3.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
