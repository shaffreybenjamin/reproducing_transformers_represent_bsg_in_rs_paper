"""Figure 7C reproduction: RRXOR belief geometry in the residual stream.

Faithful to the authors' rrxor_simplex.ipynb: enumerate every positive-probability
length-n_ctx input, take the per-position ground-truth belief (5-D), regress the
(concatenated per-layer `ln1.hook_normalized`) activations onto belief, and project
the predicted beliefs with the SAME PCA used for panel B (the symmetric PC1/PC3).
Small dots = individual (input,position) predictions; large dots = per-belief-state
centre of mass. Coloured by the ground-truth belief state.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.transition_matrices import rrxor

P1 = P2 = 0.5
N_CTX = 10
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"

T = np.array(rrxor(P1, P2))                 # (2,5,5)  T[token, from, to]
NSTATES = T.shape[1]
STATIONARY = np.array([2, 1, 1, 1, 1]) / 6.0


def next_dist(b):
    return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])


def update(b, x):
    nb = b @ T[x]
    return nb / nb.sum()


def msp_states():
    """36 unique positive-probability MSP beliefs (incl. root), + {rounded->index}."""
    beliefs = {tuple(np.round(STATIONARY, 5)): STATIONARY}
    frontier = [STATIONARY]
    for _ in range(20):
        nxt = []
        for b in frontier:
            d = next_dist(b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                b2 = update(b, x)
                k = tuple(np.round(b2, 5))
                if k not in beliefs:
                    beliefs[k] = b2
                    nxt.append(b2)
        frontier = nxt
    keys = list(beliefs.keys())
    B = np.array([beliefs[k] for k in keys])
    index = {k: i for i, k in enumerate(keys)}
    return B, index


def enumerate_inputs(n_ctx, index):
    """All positive-prob length-n_ctx sequences, with per-position belief (after each
    token) and the 36-state index of each position."""
    frontier = [((), [])]                    # (seq, [belief after each token])
    for _ in range(n_ctx):
        nxt = []
        for seq, bels in frontier:
            b = STATIONARY if not bels else bels[-1]
            d = next_dist(b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                nxt.append((seq + (x,), bels + [update(b, x)]))
        frontier = nxt
    seqs = np.array([s for s, _ in frontier], dtype=np.int64)
    beliefs = np.array([b for _, b in frontier], dtype=np.float32)    # (N, n_ctx, 5)
    idx = np.array([[index[tuple(np.round(b, 5))] for b in bels] for _, bels in frontier])
    return seqs, beliefs, idx


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"]); cfg.device = device
    m = HookedTransformer(cfg); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m


def collect_activations(model, seqs, device, hooks):
    """Return {hook_name: (N, n_ctx, d)} for the requested hooks, batched."""
    out = {h: [] for h in hooks}
    inp = torch.from_numpy(seqs).to(device)
    for i in range(0, len(inp), 4096):
        with torch.no_grad():
            _, cache = model.run_with_cache(inp[i:i + 4096], names_filter=lambda n: n in hooks)
        for h in hooks:
            out[h].append(cache[h].cpu().numpy())
    return {h: np.concatenate(v, 0) for h, v in out.items()}


def distinct_colors(n):
    base = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return np.array([base[i % len(base)] for i in range(n)])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B36, index = msp_states()
    print(f"MSP states: {len(B36)}")
    seqs, beliefs, idx = enumerate_inputs(N_CTX, index)
    print(f"positive-prob length-{N_CTX} inputs: {len(seqs)}")

    model = load_model(MODEL_DIR / "rrxor_transformer.pt", device)
    layer_hooks = [f"blocks.{i}.ln1.hook_normalized" for i in range(model.cfg.n_layers)]
    acts = collect_activations(model, seqs, device, layer_hooks)

    # concat per-layer ln1 activations -> regress onto belief
    concat = np.concatenate([acts[h] for h in layer_hooks], axis=-1)   # (N, n_ctx, 4*d)
    X = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, NSTATES)
    pred = LinearRegression().fit(X, Y).predict(X)
    print(f"concat regression R^2 = {LinearRegression().fit(X, Y).score(X, Y):.4f}")

    # project with the panel-B PCA (fit on the 36 ground-truth beliefs), symmetric PC1/PC3
    pca = PCA(n_components=4).fit(B36)
    xy = pca.transform(pred)[:, [0, 2]]
    flat_idx = idx.reshape(-1)
    colors = distinct_colors(len(B36))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[flat_idx], alpha=0.25, edgecolors="none")
    # centre of mass per ground-truth belief state (large dots)
    for s in range(len(B36)):
        m = flat_idx == s
        if m.any():
            com = xy[m].mean(0)
            ax.scatter(com[0], com[1], s=90, c=[colors[s]], edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_title("Residual Stream Belief Representation  (RRXOR)")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig10_rrxor_residual_representation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
