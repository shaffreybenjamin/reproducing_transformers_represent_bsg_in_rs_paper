"""Step 3 (sharpened) - Belief geometry in the residual stream, via MSP enumeration.

Same claim as fig03, but instead of *sampling* sequences we enumerate every
unique belief in the Mixed-State Presentation exactly once (grouped by length so
each group is a fixed-length batch). For each prefix we read the residual stream
at its final position. This removes sampling smear, so the recovered fractal is
as crisp as the model allows.

Output: 3-panel figure
  (1) ground-truth belief simplex (Figure 1, for reference)
  (2) belief decoded from residual stream by a linear map (prob-weighted OLS)
  (3) top-3 PCA of the residual stream, colored by ground-truth belief
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
    # Paper's projection (simplexity casper/analyses): state-2=blue top apex,
    # state-0=red bottom-left, state-1=green bottom-right.
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def load_model(device: str):
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device  # checkpoint was trained on cuda; honor the local device (e.g. cpu)
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def enumerate_msp(hmm, depth):
    """Return dict length -> (sequences[int array], beliefs, probs) for unique prefixes."""
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=depth).generate()
    by_len = defaultdict(lambda: ([], [], []))
    for seq, v in tree.nodes.items():
        if len(seq) == 0:
            continue
        seqs, bels, probs = by_len[len(seq)]
        seqs.append(seq)
        bels.append(v.belief_state)
        probs.append(v.probability)
    out = {}
    for L, (seqs, bels, probs) in by_len.items():
        out[L] = (np.array(seqs, dtype=np.int64), np.array(bels), np.array(probs))
    return out


def collect_residuals(model, by_len, context_len, device):
    """For each unique prefix, residual stream at its final position."""
    layer = model.cfg.n_layers - 1
    resid_list, belief_list, prob_list = [], [], []
    for L in sorted(by_len):
        if L > context_len:
            continue
        seqs, bels, probs = by_len[L]
        inp = torch.from_numpy(seqs).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, names_filter=f"blocks.{layer}.hook_resid_post")
        resid = cache["resid_post", layer][:, -1, :].cpu().numpy()  # final position
        resid_list.append(resid)
        belief_list.append(bels)
        prob_list.append(probs)
    X = np.concatenate(resid_list, 0)
    Y = np.concatenate(belief_list, 0)
    w = np.concatenate(prob_list, 0)
    return X, Y, w


def weighted_ols(X, Y, w):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=None)
    Yhat = Xb @ beta
    ss_res = (w[:, None] * (Y - Yhat) ** 2).sum()
    ss_tot = (w[:, None] * (Y - np.average(Y, axis=0, weights=w)) ** 2).sum()
    return Yhat, 1.0 - ss_res / ss_tot


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model, context_len = load_model(device)

    by_len = enumerate_msp(hmm, MSP_DEPTH)
    X, Y, w = collect_residuals(model, by_len, context_len, device)
    print(f"unique prefixes used: {X.shape[0]}  d_model={X.shape[1]}")

    Yhat, r2 = weighted_ols(X, Y, w)
    print(f"residual -> belief weighted R^2 = {r2:.4f}")

    color = np.clip(Y, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    xy_true = simplex_to_xy(Y)
    axes[0].scatter(xy_true[:, 0], xy_true[:, 1], c=color, s=2, edgecolors="none")
    axes[0].set_title("Ground truth (MSP)")

    xy_pred = simplex_to_xy(Yhat)
    axes[1].scatter(xy_pred[:, 0], xy_pred[:, 1], c=color, s=2, edgecolors="none")
    axes[1].set_title(f"Decoded from residual stream (R^2={r2:.3f})")

    Xc = X - np.average(X, axis=0, weights=w)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ vt[:2].T
    axes[2].scatter(pcs[:, 0], pcs[:, 1], c=color, s=2, edgecolors="none")
    axes[2].set_title("Residual-stream PCA (top 2)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    out = FIG_DIR / "fig03b_residual_belief_enumerated_mess3.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
