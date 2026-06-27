"""Figure 7E reproduction: belief-regression MSE per activation type.

For RRXOR and Mess3, regress each activation type onto the ground-truth belief and
report the MSE:  Embed, Layer 1..4 (blocks.k.hook_resid_post = the residual stream
after each block), LN Final (ln_final.hook_normalized), and Concat (all concatenated).

The paper's point: for RRXOR the geometry is spread across layers, so each single
layer has high MSE but the Concat is low; for Mess3 every layer already carries it
(all low).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.transition_matrices import mess3, rrxor

N_CTX = 10
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def proc(name):
    if name == "rrxor":
        return np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    return np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0


def ndist(T, b):
    return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])


def upd(T, b, x):
    nb = b @ T[x]
    return nb / nb.sum()


def enumerate_positive(T, pi, n_ctx):
    """All positive-probability length-n_ctx sequences + per-position beliefs."""
    frontier = [((), [])]
    for _ in range(n_ctx):
        nxt = []
        for seq, bels in frontier:
            b = pi if not bels else bels[-1]
            d = ndist(T, b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                nxt.append((seq + (x,), bels + [upd(T, b, x)]))
        frontier = nxt
    seqs = np.array([s for s, _ in frontier], dtype=np.int64)
    bels = np.array([b for _, b in frontier], dtype=np.float32)
    return seqs, bels


def sample_sequences(T, pi, n, n_ctx, seed=0):
    """Sample n sequences from the process + per-position beliefs (for full-support Mess3)."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, n_ctx), dtype=np.int64)
    bels = np.zeros((n, n_ctx, len(pi)), dtype=np.float32)
    for i in range(n):
        b = pi
        for j in range(n_ctx):
            d = ndist(T, b)
            x = rng.choice(len(d), p=d / d.sum())
            seqs[i, j] = x
            b = upd(T, b, x)
            bels[i, j] = b
    return seqs, bels


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"]); cfg.device = device
    m = HookedTransformer(cfg); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m


def activation_mses(model, seqs, bels, device):
    nL = model.cfg.n_layers
    hooks = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(nL)] + ["ln_final.hook_normalized"]
    inp = torch.from_numpy(seqs).to(device)
    cache_acc = {h: [] for h in hooks}
    for i in range(0, len(inp), 4096):
        with torch.no_grad():
            _, c = model.run_with_cache(inp[i:i + 4096], names_filter=lambda n: n in hooks)
        for h in hooks:
            cache_acc[h].append(c[h].cpu().numpy())
    acts = {h: np.concatenate(v, 0).reshape(-1, model.cfg.d_model) for h, v in cache_acc.items()}
    Y = bels.reshape(-1, bels.shape[-1])

    def mse(X):
        return float(np.mean((LinearRegression().fit(X, Y).predict(X) - Y) ** 2))

    labels = ["Embed"] + [f"Layer {i+1}" for i in range(nL)] + ["LN Final", "Concat"]
    vals = [mse(acts["hook_embed"])]
    vals += [mse(acts[f"blocks.{i}.hook_resid_post"]) for i in range(nL)]
    vals += [mse(acts["ln_final.hook_normalized"])]
    vals += [mse(np.concatenate(list(acts.values()), axis=-1))]
    return labels, vals


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for name, ckpt, sampler in [
        ("RRXOR", "rrxor_transformer.pt", "enum"),
        ("Mess3", "mess3_transformer.pt", "sample"),
    ]:
        T, pi = proc(name.lower())
        if sampler == "enum":
            seqs, bels = enumerate_positive(T, pi, N_CTX)
        else:
            seqs, bels = sample_sequences(T, pi, 6000, N_CTX)
        model = load_model(MODEL_DIR / ckpt, device)
        labels, vals = activation_mses(model, seqs, bels, device)
        results[name] = (labels, vals)
        print(f"{name}: inputs={len(seqs)}")
        print("   " + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in [("RRXOR", "black"), ("Mess3", "red")]:
        labels, vals = results[name]
        ax.plot(range(len(labels)), vals, "o", color=color, ms=8, label=name)
    ax.set_xticks(range(len(results["RRXOR"][0])))
    ax.set_xticklabels(results["RRXOR"][0], rotation=30, ha="right")
    ax.set_ylabel("Mean Squared Error"); ax.set_xlabel("Activation Type")
    ax.set_title("Belief-regression MSE per activation type")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    out = FIG_DIR / "fig11_rrxor_layerwise_mse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
