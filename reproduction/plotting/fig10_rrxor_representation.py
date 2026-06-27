"""Figure 7C reproduction: RRXOR belief geometry in the residual stream.

Method: enumerate every positive-probability length-n_ctx input, take the per-position
ground-truth belief (5-D), regress the concatenated per-layer RESIDUAL STREAM
(`blocks.k.hook_resid_post`) onto belief, and project the predicted beliefs with the
SAME PCA used for panel B (the symmetric PC1/PC3). Small dots = individual
(input,position) predictions; large dots = per-belief-state centre of mass.

Note on the activation: the paper's TEXT says it concatenates the residual streams,
so we use the raw residual stream `hook_resid_post`. The authors' released notebook
(rrxor_simplex.ipynb) instead used `ln1.hook_normalized` -- the residual stream after
each block's input LayerNorm (mean-centred + scale-normalised). The raw residual
stream both matches the paper's wording and decodes belief more cleanly here
(R^2 ~ 0.88 vs ~ 0.81 for ln1), giving tighter clusters closer to the published figure.
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
    layer_hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = collect_activations(model, seqs, device, layer_hooks)

    concat = np.concatenate([acts[h] for h in layer_hooks], axis=-1)   # (N, n_ctx, 4*d)

    # DEDUPLICATE by token prefix and weight each unique prefix by its probability P(w).
    # Causal attention => every occurrence of a prefix has the identical activation, so we
    # keep the first occurrence and weight by the summed probability -- the convention of the
    # post-quantum paper (Riechers et al.): "retain the activation from the first occurrence,
    # sum probabilities across occurrences." Here P(w) is analytic (product of conditionals).
    def prob_of(w):
        b, p = STATIONARY, 1.0
        for x in w:
            d = next_dist(b); p *= float(d[x]); b = update(b, x)
        return p
    seen = {}
    for i in range(len(seqs)):
        for pos in range(N_CTX):
            w = tuple(int(t) for t in seqs[i][:pos + 1])
            if w not in seen:
                seen[w] = (concat[i, pos], beliefs[i, pos], int(idx[i, pos]))
    prefixes = list(seen)
    X = np.stack([seen[w][0] for w in prefixes])
    Y = np.stack([seen[w][1] for w in prefixes])
    flat_idx = np.array([seen[w][2] for w in prefixes])
    Wt = np.array([prob_of(w) for w in prefixes])
    print(f"unique prefixes: {len(prefixes)} (deduped from {len(seqs) * N_CTX} input-positions)")
    reg = LinearRegression().fit(X, Y, sample_weight=Wt)
    pred = reg.predict(X)
    print(f"concat regression R^2 (P(w)-weighted) = {reg.score(X, Y, sample_weight=Wt):.4f}")

    # project with the panel-B PCA (fit on the 36 ground-truth beliefs), symmetric PC1/PC3
    pca = PCA(n_components=4).fit(B36)
    xy = pca.transform(pred)[:, [0, 2]]
    tgt_xy = pca.transform(B36)[:, [0, 2]]          # fig09 ground-truth projection (for matching axes)
    colors = distinct_colors(len(B36))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.6))
    # left: ground truth (36 belief states)
    ax0.scatter(tgt_xy[:, 0], tgt_xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax0.set_title("Ground Truth Belief Geometry  (RRXOR)")
    # right: residual-stream representation, with ground-truth rings overlaid
    ax1.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[flat_idx], alpha=0.25, edgecolors="none")
    for s in range(len(B36)):
        m = flat_idx == s
        if m.any():
            ax1.scatter(*xy[m].mean(0), s=90, c=[colors[s]], edgecolors="black", linewidths=0.6, zorder=3)
    ax1.scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    ax1.set_title("Residual Stream Belief Representation  (RRXOR)")
    for ax in (ax0, ax1):
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        for set_lim, v in ((ax.set_xlim, tgt_xy[:, 0]), (ax.set_ylim, tgt_xy[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo)
            set_lim(lo - pad, hi + pad)
    out = FIG_DIR / "fig10_rrxor_residual_representation.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
