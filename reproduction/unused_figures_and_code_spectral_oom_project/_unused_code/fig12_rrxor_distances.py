"""Figure 7D reproduction: distance preservation in the belief representation.

For RRXOR and Mess3, take each ground-truth belief STATE, its model representation
(centre of mass of the concat-regression-predicted belief over all its occurrences),
and its optimal next-token distribution. Then, over all pairs of states, scatter:
  left : ground-truth belief distance        vs belief-representation distance
  right: ground-truth next-token-prob distance vs belief-representation distance

Paper's point: for RRXOR the representation preserves belief distances (high R^2)
but NOT next-token distances (low R^2 -> the geometry is more than next-token
predictions); for Mess3 both are ~1.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

import fig11_rrxor_layerwise_mse as A

N_CTX = 10
MAX_STATES = 400
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def states_for(name, ckpt, sampler, device, n_sample=6000):
    T, pi = A.proc(name.lower())
    if sampler == "enum":
        seqs, bels = A.enumerate_positive(T, pi, N_CTX)
    else:
        seqs, bels = A.sample_sequences(T, pi, n_sample, N_CTX)
    model = A.load_model(MODEL_DIR / ckpt, device)
    nL = model.cfg.n_layers
    hooks = ["hook_embed"] + [f"blocks.{i}.ln1.hook_normalized" for i in range(nL)] + ["ln_final.hook_normalized"]
    import torch
    inp = torch.from_numpy(seqs).to(device)
    accs = {h: [] for h in hooks}
    for i in range(0, len(inp), 4096):
        with torch.no_grad():
            _, c = model.run_with_cache(inp[i:i + 4096], names_filter=lambda n: n in hooks)
        for h in hooks:
            accs[h].append(c[h].cpu().numpy())
    concat = np.concatenate([np.concatenate(accs[h], 0) for h in hooks], axis=-1).reshape(-1, len(hooks) * model.cfg.d_model)
    Y = bels.reshape(-1, bels.shape[-1])
    pred = LinearRegression().fit(concat, Y).predict(concat)

    groups = defaultdict(list)
    for i, b in enumerate(Y):
        groups[tuple(np.round(b, 5))].append(i)
    true_b, rep, nt = [], [], []
    for k, idxs in groups.items():
        tb = Y[idxs[0]]
        true_b.append(tb)
        rep.append(pred[idxs].mean(0))
        d = A.ndist(T, tb)
        nt.append(d / d.sum())
    return np.array(true_b), np.array(rep), np.array(nt)


def pair_dists(true_b, rep, nt, seed=0):
    if len(true_b) > MAX_STATES:
        idx = np.random.default_rng(seed).choice(len(true_b), MAX_STATES, replace=False)
        true_b, rep, nt = true_b[idx], rep[idx], nt[idx]
    return pdist(true_b), pdist(nt), pdist(rep)


def r2(x, y):
    return float(linregress(x, y).rvalue ** 2)


def panel(ax, x, y, xlabel):
    ax.hexbin(x, y, gridsize=40, cmap="Greys", bins="log", mincnt=1)
    s = linregress(x, y)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, s.intercept + s.slope * xs, "r--", lw=1)
    ax.text(0.95, 0.06, f"$R^2$ = {s.rvalue**2:.2f}", transform=ax.transAxes, ha="right", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=9)


def main():
    import torch
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for r, (name, ckpt, sampler) in enumerate([("RRXOR", "rrxor_transformer.pt", "enum"),
                                               ("Mess3", "mess3_transformer.pt", "sample")]):
        tb, rep, nt = states_for(name, ckpt, sampler, device)
        d_belief, d_nt, d_rep = pair_dists(tb, rep, nt)
        print(f"{name}: {len(tb)} states  |  belief->rep R^2={r2(d_belief,d_rep):.2f}  nexttoken->rep R^2={r2(d_nt,d_rep):.2f}")
        panel(axes[r, 0], d_belief, d_rep, "Ground Truth Belief Distance")
        panel(axes[r, 1], d_nt, d_rep, "Ground Truth Distance Prob. Next Token")
        axes[r, 0].set_ylabel(f"{name}\nBelief Representation Distance", fontsize=9)
    fig.suptitle("Distance preservation in the belief representation", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig12_rrxor_distances.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
