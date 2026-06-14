"""Step 3 (headline) - Belief-state geometry in the residual stream.

The paper's central claim: a transformer trained only on next-token prediction
linearly encodes the *entire* belief-state simplex (the fractal from Figure 1)
in its residual stream - not just the next-token distribution.

Method (matches Astera `casper/analyses`: sample -> dedup-by-prefix -> weighted)
------
1. Sample many mess3 sequences together with their ground-truth belief state at
   every position (simplexity gives us both).
2. Run the trained transformer with cache; grab the final-layer residual stream
   at every position. resid[:, t] and belief[:, t] both condition on tokens 0..t.
3. Deduplicate by prefix: causal attention makes a position's activation depend
   only on its prefix, so all occurrences of a prefix are identical. The row
   weight is the EMPIRICAL prefix frequency (occurrence count) — the natural data
   measure from sampling alone (no MSP enumeration, no analytic P(w)).
4. Fit a single linear map  resid -> belief  (probability-weighted least squares).
   A high weighted R^2 means the belief is linearly readable from the residual.
5. Visualize: project the regression-predicted beliefs into the 2-simplex,
   colored by ground-truth belief; plus a raw PCA of the residual stream.
"""

from collections import defaultdict
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
    cfg.device = device  # checkpoint was trained on cuda; honor the local device (e.g. cpu)
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def collect_residuals(model, hmm, context_len, device):
    """Sample sequences, dedup by prefix, return (X, Y, w) over unique prefixes.

    Causal attention => a position's activation/belief depend only on its prefix,
    so dedup is exact. Weight w = empirical prefix frequency (occurrence count,
    normalised) — the natural measure from sampling, no analytic P(w).
    """
    init_state = jnp.array(hmm.initial_state)
    key = jax.random.PRNGKey(SEED)
    layer = model.cfg.n_layers - 1

    resid, belief = {}, {}
    count = defaultdict(int)
    done = 0
    while done < N_SEQUENCES:
        b = min(BATCH, N_SEQUENCES - done)
        key, bk = jax.random.split(key)
        gen_states = jnp.repeat(init_state[None, :], b, axis=0)
        data = generate_data_batch_with_full_history(gen_states, hmm, b, context_len, bk, device=device)
        inputs = data["inputs"].long().to(device)  # (b, T)
        beliefs = np.asarray(data["belief_states"])  # (b, T, n_states)
        with torch.no_grad():
            _, cache = model.run_with_cache(inputs, names_filter=f"blocks.{layer}.hook_resid_post")
        z = cache["resid_post", layer].cpu().numpy()  # (b, T, d_model)
        toks = inputs.cpu().numpy()
        T = toks.shape[1]
        for si in range(b):
            pref = ()
            for pos in range(T):
                pref = pref + (int(toks[si, pos]),)
                count[pref] += 1
                if pref not in resid:
                    resid[pref] = z[si, pos]
                    belief[pref] = beliefs[si, pos]
        done += b

    keys = list(resid.keys())
    X = np.stack([resid[k] for k in keys])
    Y = np.stack([belief[k] for k in keys])
    total = sum(count.values())
    w = np.array([count[k] / total for k in keys])
    return X, Y, w


def fit_linear(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Probability-weighted OLS map X(+bias) -> Y. Returns predictions and weighted R^2."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=None)
    Yhat = Xb @ beta
    ss_res = (w[:, None] * (Y - Yhat) ** 2).sum()
    ss_tot = (w[:, None] * (Y - np.average(Y, axis=0, weights=w)) ** 2).sum()
    return Yhat, float(1.0 - ss_res / ss_tot)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model, context_len = load_model(device)

    X, Y, w = collect_residuals(model, hmm, context_len, device)
    print(f"unique prefixes from {N_SEQUENCES} sampled sequences: {X.shape[0]}; d_model={X.shape[1]}")

    Yhat, r2 = fit_linear(X, Y, w)
    print(f"weighted linear regression residual -> belief  R^2 = {r2:.4f}")

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
