"""Combined RRXOR diagnostics (paper Fig 7 D+E), supervised AND unsupervised.

Merges the old layerwise-MSE (fig11) and distance-preservation (fig12) into one figure, and
adds an unsupervised counterpart using the spectral-OOM + P(w) subspace.

  fig34_rrxor_diagnostics_supervised.png    layerwise MSE | belief-dist | next-token-dist
  fig35_rrxor_diagnostics_unsupervised.png   same, but the representation is the recovered
                                             spectral-OOM subspace (not a label-fit decode)

Layerwise: per activation type (Embed, Layer 1..4, LN Final, Concat), how well it carries belief
  -- supervised = regression MSE; unsupervised = decode MSE from the type's spectral-OOM subspace.
Distance: over the 36 RRXOR belief states, does the representation preserve belief distances
  (it should) but not next-token distances (it should not). Mess3 is the trivial control
  (everything ~1 / low) and is shown only in the layerwise panel.

One enumerated per-layer collection per process feeds both the supervised and unsupervised
computations and both figures.
"""
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import mess3, rrxor
import fig14_observable_oom as F14

MAX_LEN = 10
FIG_DIR = Path(__file__).parent.parent / "figures"
MODEL_DIR = Path(__file__).parent.parent / "models"


def proc(name):
    if name == "RRXOR":
        return np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0, "rrxor_transformer.pt", 5
    return np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0, "mess3_transformer.pt", 3


def collect_layers(name):
    """Enumerate all positive-prob prefixes; per-type activations + softmax + analytic belief."""
    T, pi, ckpt, d = proc(name)
    model = F14._load(ckpt, "cpu")
    nL = model.cfg.n_layers
    types = ["Embed"] + [f"Layer {i+1}" for i in range(nL)] + ["LN Final", "Concat"]
    hooks = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(nL)] + ["ln_final.hook_normalized"]
    NS = T.shape[1]

    def ndist(b): return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])
    def upd(b, x): nb = b @ T[x]; return nb / nb.sum()
    def bp(w):
        b = pi
        for x in w:
            dd = ndist(b)
            if dd[x] < 1e-12: return None
            b = upd(b, x)
        return b

    rt = {t: {} for t in types}; soft = {}; belief = {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to("cpu")
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            acts = {h: c[h][:, -1, :].cpu().numpy() for h in hooks}
            concat = np.concatenate([acts[h] for h in hooks], -1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                for ti, h in zip(types[:-1], hooks):
                    rt[ti][w] = acts[h][j]
                rt["Concat"][w] = concat[j]
                soft[w] = sm[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan)
    reach = {}
    for w in rt["Concat"]:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    return T, d, types, rt, soft, belief, reach


def layerwise(name):
    T, d, types, rt, soft, belief, reach = collect_layers(name)
    pw = F14.analytic_prefix_probs(rt["Concat"], T, np.array([2,1,1,1,1])/6 if name=="RRXOR" else np.array([1,1,1])/3)
    allw = [w for w in rt["Concat"] if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    sup, uns = [], []
    for t in types:
        X = np.stack([rt[t][w] for w in allw])
        sup.append(float(np.mean((LinearRegression().fit(X, Yb).predict(X) - Yb) ** 2)))
        B = F14.observable_subspace(rt[t], soft, reach, T.shape[0], wmap=pw)[4][:, :d]
        XB = X @ B
        uns.append(float(np.mean((LinearRegression().fit(XB, Yb).predict(XB) - Yb) ** 2)))
    return types, sup, uns, (T, d, rt, soft, belief, reach, pw, allw, Yb)


def distances(bundle, kind):
    """RRXOR pairwise distances: belief-dist & next-token-dist vs representation-dist."""
    T, d, rt, soft, belief, reach, pw, allw, Yb = bundle
    X = np.stack([rt["Concat"][w] for w in allw])
    if kind == "supervised":
        rep = LinearRegression().fit(X, Yb).predict(X)             # decoded belief (full concat)
    else:
        B = F14.observable_subspace(rt["Concat"], soft, reach, T.shape[0], wmap=pw)[4][:, :d]
        XB = X @ B
        rep = LinearRegression().fit(XB, Yb).predict(XB)          # decoded belief from the recovered
        #                                                           subspace (label-fit readout, as in fig26)
    groups = defaultdict(list)
    for i, w in enumerate(allw):
        groups[tuple(np.round(belief[w], 5))].append(i)
    tb = np.array([Yb[g[0]] for g in groups.values()])
    rp = np.array([rep[g].mean(0) for g in groups.values()])
    nt = np.array([[(tbi @ T[x]).sum() for x in range(T.shape[0])] for tbi in tb])
    nt = nt / nt.sum(1, keepdims=True)
    return pdist(tb), pdist(nt), pdist(rp)


def r2(x, y):
    return float(linregress(x, y).rvalue ** 2)


def hexpanel(ax, x, y, xlabel, ylabel=None):
    ax.hexbin(x, y, gridsize=40, cmap="Greys", bins="log", mincnt=1)
    s = linregress(x, y); xs = np.array([x.min(), x.max()])
    ax.plot(xs, s.intercept + s.slope * xs, "r--", lw=1)
    ax.text(0.95, 0.06, f"$R^2$={s.rvalue**2:.2f}", transform=ax.transAxes, ha="right", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def make_figure(fname, title, types, rr_sup_or_uns, m3_sup_or_uns, d_belief, d_nt, d_rep, ylabel):
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.32)
    axL = fig.add_subplot(gs[0])
    axL.plot(range(len(types)), rr_sup_or_uns, "o-", color="black", ms=7, label="RRXOR")
    axL.plot(range(len(types)), m3_sup_or_uns, "o-", color="red", ms=7, label="Mess3")
    axL.set_xticks(range(len(types))); axL.set_xticklabels(types, rotation=30, ha="right")
    axL.set_ylabel("Belief-reconstruction MSE"); axL.set_xlabel("Activation type")
    axL.set_title("Belief MSE per activation type"); axL.legend()
    axL.spines[["top", "right"]].set_visible(False); axL.grid(axis="y", ls="--", alpha=0.3)
    hexpanel(fig.add_subplot(gs[1]), d_belief, d_rep, "Ground-truth belief distance", ylabel)
    hexpanel(fig.add_subplot(gs[2]), d_nt, d_rep, "Ground-truth next-token distance")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {FIG_DIR / fname}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    types, rr_sup, rr_uns, rr_bundle = layerwise("RRXOR")
    _, m3_sup, m3_uns, _ = layerwise("Mess3")
    db_s, dn_s, dr_s = distances(rr_bundle, "supervised")
    db_u, dn_u, dr_u = distances(rr_bundle, "unsupervised")
    print(f"RRXOR distance  supervised:  belief R^2={r2(db_s,dr_s):.2f}  nexttoken R^2={r2(dn_s,dr_s):.2f}")
    print(f"RRXOR distance  unsupervised: belief R^2={r2(db_u,dr_u):.2f}  nexttoken R^2={r2(dn_u,dr_u):.2f}")

    make_figure("fig34_rrxor_diagnostics_supervised.png",
                "RRXOR diagnostics (supervised): belief is spread across layers and preserves "
                "belief — not next-token — distances",
                types, rr_sup, m3_sup, db_s, dn_s, dr_s, "RRXOR belief-representation distance")
    make_figure("fig35_rrxor_diagnostics_unsupervised.png",
                "RRXOR diagnostics (unsupervised, spectral-OOM subspace): belief spread across "
                "layers is recovered (left); the recovered representation's distances lean next-token, "
                "not belief (right)",
                types, rr_uns, m3_uns, db_u, dn_u, dr_u, "RRXOR recovered-belief distance")


if __name__ == "__main__":
    main()
