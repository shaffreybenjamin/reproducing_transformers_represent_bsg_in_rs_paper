"""Paper Fig 6 (supervised), as one figure: training progression (A) + Cross-Validation (B)
+ Shuffle control (C) + MSE bars (D). Mess3, residual-stream belief decode.

A  four training checkpoints (steps 100 / 1,000 / 5,000 / 1,000,000): per-checkpoint
   activation->belief regression, predicted belief projected to the simplex (datashader raster).
B  cross-validation on the final model: fit the decode on 20% of prefixes, decode the held-out
   80% -> the fractal persists (not memorisation).
C  shuffle control: permute belief labels w.r.t. activations, refit -> decode collapses to ~mean.
D  belief-reconstruction MSE at each checkpoint, plus cross-val and shuffle on the final model.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U

MESS3 = {"x": 0.05, "a": 0.85}
TRAIN_FRAC = 0.2
N_REPEAT = 30
SEED = 0
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
CKPTS = [
    ("100", MODEL_DIR / "progression" / "step_100.pt"),
    ("1,000", MODEL_DIR / "progression" / "step_1000.pt"),
    ("5,000", MODEL_DIR / "progression" / "step_5000.pt"),
    ("1,000,000", MODEL_DIR / "mess3_transformer.pt"),
]
TRI = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
EXT = (-0.05, 1.05, -0.05, np.sqrt(3) / 2 + 0.05)
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"]); cfg.device = device
    m = HookedTransformer(cfg); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, ck["context_len"], ck["step"]


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def to_simplex(p):
    p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)


def draw_triangle(ax, title=None):
    ax.plot(TRI[:, 0], TRI[:, 1], color="0.6", lw=1)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def broken_barh(axL, axR, vals, labels, wratio=3.0):
    """Horizontal bars with a broken x-axis: small bars on axL (fine scale), the outlier
    (shuffle) bar reaching into axR past a gap (paper Fig 6D style)."""
    y = np.arange(len(vals))[::-1]
    xmaxL = max(vals[:-1]) * 1.25
    big = vals[-1]
    for ax in (axL, axR):
        ax.barh(y, vals, color=BAR_COLORS[:len(vals)])
    axL.set_xlim(0, xmaxL); axR.set_xlim(big * 0.985, big * 1.01)
    axL.set_yticks(y); axL.set_yticklabels(labels); axR.set_yticks([])
    axL.spines[["top", "right"]].set_visible(False)
    axR.spines[["top", "left"]].set_visible(False); axR.tick_params(left=False)
    for yi, v in zip(y, vals):
        if v <= xmaxL:
            axL.text(v + xmaxL * 0.02, yi, f"{v:.4f}", va="center", fontsize=8)
        else:
            axR.text(big, yi, f"  {v:.4f}", va="center", fontsize=8)
    d = 0.02
    kw = dict(transform=axL.transAxes, color="k", clip_on=False, lw=1)
    axL.plot((1 - d, 1 + d), (-d, d), **kw); axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    dr = d * wratio
    kw = dict(transform=axR.transAxes, color="k", clip_on=False, lw=1)
    axR.plot((-dr, dr), (-d, d), **kw); axR.plot((-dr, dr), (1 - d, 1 + d), **kw)
    axL.set_xlabel("Mean Squared Error", x=0.7)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3)
    rng = np.random.default_rng(SEED)

    panelA, train_mse, Xf, Yf = [], [], None, None
    for label, path in CKPTS:
        model, ctx, step = load(path, device)
        resid, _, belief, _, _ = U.collect_prefix_features_enumerated(model, hmm, ctx, device)
        seqs = [s for s in resid if s in belief]
        X = np.stack([resid[s] for s in seqs]); Y = np.stack([belief[s] for s in seqs])
        reg = LinearRegression().fit(X, Y)
        train_mse.append(mse(reg.predict(X), Y))
        # RAW predictions (no clip-to-simplex), auto-framed -> early under-trained models show as
        # a small central blob (belief not yet differentiated), matching fig06.
        panelA.append((U.simplex_to_xy(reg.predict(X)), np.clip(Y, 0, 1), label, step))
        Xf, Yf = X, Y
        print(f"step {label}: MSE={train_mse[-1]:.5f}")

    N = Xf.shape[0]; rgb = np.clip(Yf, 0, 1)
    perm = rng.permutation(N); ntr = int(TRAIN_FRAC * N); tr, te = perm[:ntr], perm[ntr:]
    cv_pred = LinearRegression().fit(Xf[tr], Yf[tr]).predict(Xf[te])
    cv_img = U.rasterize_simplex(U.simplex_to_xy(cv_pred), rgb[te], px=2)
    cv_mse_pt = mse(cv_pred, Yf[te])
    sh_pred = LinearRegression().fit(Xf, Yf[rng.permutation(N)]).predict(Xf)
    sh_xy = U.simplex_to_xy(sh_pred)
    sh_img = U.rasterize_simplex(sh_xy, rgb, px=2)
    sh_inset = U.rasterize_simplex(sh_xy, rgb, px=3, auto=True)

    cv_errs, sh_errs = [], []
    for _ in range(N_REPEAT):
        p = rng.permutation(N); a, b = p[:ntr], p[ntr:]
        cv_errs.append(mse(LinearRegression().fit(Xf[a], Yf[a]).predict(Xf[b]), Yf[b]))
        sh_errs.append(mse(LinearRegression().fit(Xf, Yf[rng.permutation(N)]).predict(Xf), Yf))
    cv_mse, sh_mse = float(np.mean(cv_errs)), float(np.mean(sh_errs))
    print(f"cv={cv_mse:.5f} shuffle={sh_mse:.5f}")

    # ---------------- plot (paper Fig 6 layout) ----------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.05], hspace=0.32, wspace=0.6)
    for i, (xy, col, label, step) in enumerate(panelA):
        ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        ax.imshow(U.rasterize_simplex(xy, col, px=2, auto=True), origin="lower")
        ax.set_title(f"step {label}", fontsize=10); ax.axis("off")
    # "Training ->" arrow under panel A
    fig.text(0.13, 0.515, "Early model", ha="center", fontsize=9)
    fig.text(0.88, 0.515, "Final model", ha="center", fontsize=9)
    fig.text(0.5, 0.515, r"Training $\longrightarrow$", ha="center", fontsize=13, weight="bold")

    axB = fig.add_subplot(gs[1, 0:3])
    axB.imshow(cv_img, extent=EXT, origin="lower"); draw_triangle(axB, f"Cross Validation\n(MSE {cv_mse_pt:.4f})")
    axC = fig.add_subplot(gs[1, 3:6])
    axC.imshow(sh_img, extent=EXT, origin="lower"); draw_triangle(axC, "Shuffle control")
    axins = inset_axes(axC, width="42%", height="42%", loc="upper right")
    axins.imshow(sh_inset, origin="lower"); axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("0.5")

    labels = [f"step {l}" for l, _ in CKPTS] + ["Cross Val", "Shuffle"]
    vals = train_mse + [cv_mse, sh_mse]
    subD = gs[1, 7:12].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
    axDL = fig.add_subplot(subD[0]); axDR = fig.add_subplot(subD[1])
    broken_barh(axDL, axDR, vals, labels)
    axDL.set_title("Residual stream vs. controls", x=0.7, fontsize=10)

    fig.suptitle("Mess3: belief-state geometry emerges over training and is nontrivial  (supervised)",
                 fontsize=14, y=0.95)
    out = FIG_DIR / "fig33_mess3_fig6_supervised.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
