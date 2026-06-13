"""Step 3 (headline) - Belief-state geometry in the residual stream.

The paper's central claim: a transformer trained only on next-token prediction
linearly encodes the *entire* belief-state simplex (the fractal from Figure 1)
in its residual stream - not just the next-token distribution.

Method
------
1. Sample many mess3 sequences together with their ground-truth belief state at
   every position (simplexity gives us both).
2. Run the trained transformer with cache; grab the final-layer residual stream
   at every position. resid[:, t] and belief[:, t] both condition on tokens 0..t,
   so they are aligned.
3. Fit a single linear map  resid -> belief  (ordinary least squares). A high
   weighted R^2 means the belief is linearly readable from the residual stream.
4. Visualize: project the regression-predicted beliefs into the 2-simplex,
   colored by the ground-truth belief. If the fractal reappears, the network has
   internalized the belief geometry. We also show a raw PCA of the residual
   stream colored by belief.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch_with_full_history

MESS3_PARAMS = {"x": 0.05, "a": 0.85}  # paper values (Appendix A.3)
N_SEQUENCES = 40000
BATCH = 2000
SEED = 7

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def simplex_to_xy(beliefs: np.ndarray) -> np.ndarray:
    theta = np.pi / 3.0
    basis = np.array([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
    return beliefs[:, :2] @ basis


def load_model(device: str) -> tuple[HookedTransformer, int]:
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def collect_residuals(model, hmm, context_len, device):
    """Run sampled sequences through the model; return (resid, beliefs, probs)."""
    init_state = jnp.array(hmm.initial_state)
    key = jax.random.PRNGKey(SEED)
    layer = model.cfg.n_layers - 1

    resid_chunks, belief_chunks, prob_chunks = [], [], []
    done = 0
    while done < N_SEQUENCES:
        b = min(BATCH, N_SEQUENCES - done)
        key, bk = jax.random.split(key)
        gen_states = jnp.repeat(init_state[None, :], b, axis=0)
        data = generate_data_batch_with_full_history(gen_states, hmm, b, context_len + 1, bk, device=device)
        inputs = data["inputs"].long().to(device)  # (b, T)
        beliefs = np.asarray(data["belief_states"])  # (b, T, n_states)
        probs = np.asarray(data["prefix_probabilities"])  # (b, T)

        with torch.no_grad():
            _, cache = model.run_with_cache(inputs, names_filter=f"blocks.{layer}.hook_resid_post")
        resid = cache["resid_post", layer].cpu().numpy()  # (b, T, d_model)

        T = inputs.shape[1]
        resid_chunks.append(resid.reshape(-1, resid.shape[-1]))
        belief_chunks.append(beliefs[:, :T].reshape(-1, beliefs.shape[-1]))
        prob_chunks.append(probs[:, :T].reshape(-1))
        done += b

    X = np.concatenate(resid_chunks, 0)
    Y = np.concatenate(belief_chunks, 0)
    w = np.concatenate(prob_chunks, 0)
    return X, Y, w


def fit_linear(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS map X(+bias) -> Y. Returns predictions and weighted-by-uniform R^2."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yhat = Xb @ beta
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    return Yhat, float(r2)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model, context_len = load_model(device)

    X, Y, w = collect_residuals(model, hmm, context_len, device)
    print(f"collected {X.shape[0]} (position, sequence) points; d_model={X.shape[1]}")

    Yhat, r2 = fit_linear(X, Y)
    print(f"linear regression residual -> belief  R^2 = {r2:.4f}")

    # subsample for plotting density
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(60000, X.shape[0]), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # (a) regression-recovered belief simplex
    xy_pred = simplex_to_xy(Yhat[idx])
    color = np.clip(Y[idx], 0, 1)
    axes[0].scatter(xy_pred[:, 0], xy_pred[:, 1], c=color, s=2, edgecolors="none")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    axes[0].set_title(f"Belief recovered by linear map from residual stream\n(R^2 = {r2:.3f})")

    # (b) raw PCA of the residual stream, colored by ground-truth belief
    Xc = X[idx] - X[idx].mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ vt[:2].T
    axes[1].scatter(pcs[:, 0], pcs[:, 1], c=color, s=2, edgecolors="none")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].set_title("Top-2 PCA of residual stream\n(colored by ground-truth belief)")

    out = FIG_DIR / "fig03_residual_belief_geometry_mess3.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
