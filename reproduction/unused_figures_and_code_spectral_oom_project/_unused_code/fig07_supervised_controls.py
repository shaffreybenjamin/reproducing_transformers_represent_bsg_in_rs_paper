"""Figure 6 (B/C/D) reproduction — supervised controls (Shai et al.).

Faithful to the authors' epsilon-transformers mess3_simplex.ipynb (cells 41-42):
  B  Cross-validation : fit the activation->belief regression on a train subset of
                        prefixes, decode the held-out prefixes -> fractal persists
                        (not memorisation).
  C  Shuffle control  : permute the belief labels w.r.t. activations, refit; the
                        decode collapses to ~the mean belief -> the fractal is NOT
                        an artefact of projecting 64-D activations onto a fractal-
                        shaped target.
  D  MSE bars         : residual-stream MSE at each training checkpoint, plus the
                        cross-validation and shuffle MSE on the final model.

All on blocks.3.hook_resid_post, unweighted LinearRegression, paper projection,
rasterized rendering. Prefix-level splits (88,572 unique MSP states) so held-out
points are genuinely unseen belief states (cleaner than a sequence-level split).
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

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
TOK_PER_STEP = 64 * 10
TRAIN_FRAC = 0.2          # paper: train on 20%, test on 80%
N_REPEAT = 50            # CV splits / shuffles for the MSE distribution
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


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["context_len"], ck["step"]


def get_XY(model, hmm, context_len, device):
    resid, _, belief, _, _ = U.collect_prefix_features_enumerated(model, hmm, context_len, device)
    seqs = list(resid.keys())
    X = np.stack([resid[s] for s in seqs])
    Y = np.stack([belief[s] for s in seqs])
    return X, Y


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def draw_triangle(ax):
    ax.plot(TRI[:, 0], TRI[:, 1], color="0.6", lw=1)
    ax.set_xlim(EXT[0], EXT[1])
    ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    rng = np.random.default_rng(SEED)

    # residual-stream MSE at each training checkpoint (panel D, training bars)
    train_mse = []
    X = Y = None
    for label, path in CKPTS:
        model, context_len, step = load(path, device)
        X, Y = get_XY(model, hmm, context_len, device)  # final loop iter -> final model's X,Y
        reg = LinearRegression().fit(X, Y)
        train_mse.append((label, mse(reg.predict(X), Y)))
        print(f"step {label}: residual MSE = {train_mse[-1][1]:.5f}")

    N = X.shape[0]
    rgb = np.clip(Y, 0, 1)

    # --- B: cross-validation on the final model (prefix-level split) ---
    perm = rng.permutation(N)
    ntr = int(TRAIN_FRAC * N)
    tr, te = perm[:ntr], perm[ntr:]
    reg_cv = LinearRegression().fit(X[tr], Y[tr])
    cv_pred = reg_cv.predict(X[te])
    cv_xy = U.simplex_to_xy(cv_pred)
    cv_img = U.rasterize_simplex(cv_xy, rgb[te], px=2)
    cv_mse_point = mse(cv_pred, Y[te])

    # --- C: shuffle control on the final model ---
    Ysh = Y[rng.permutation(N)]
    sh_pred = LinearRegression().fit(X, Ysh).predict(X)
    sh_xy = U.simplex_to_xy(sh_pred)
    sh_img = U.rasterize_simplex(sh_xy, rgb, px=2)           # fixed frame -> shows the collapse
    sh_inset = U.rasterize_simplex(sh_xy, rgb, px=3, auto=True)  # zoomed inset

    # --- D: CV / shuffle MSE distributions on the final model ---
    cv_errs, sh_errs = [], []
    for _ in range(N_REPEAT):
        p = rng.permutation(N)
        a, b = p[:ntr], p[ntr:]
        cv_errs.append(mse(LinearRegression().fit(X[a], Y[a]).predict(X[b]), Y[b]))
        sh_errs.append(mse(LinearRegression().fit(X, Y[rng.permutation(N)]).predict(X), Y))
    cv_mse, sh_mse = float(np.mean(cv_errs)), float(np.mean(sh_errs))
    print(f"cross-val MSE = {cv_mse:.5f}   shuffle MSE = {sh_mse:.5f}")

    # ------------------------------- plot ------------------------------- #
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    axB, axC, axD = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    axB.imshow(cv_img, extent=EXT, origin="lower")
    draw_triangle(axB)
    axB.set_title(f"Cross Validation\n(held-out MSE {cv_mse_point:.4f})")

    axC.imshow(sh_img, extent=EXT, origin="lower")
    draw_triangle(axC)
    axC.set_title("Shuffle Control")
    axins = inset_axes(axC, width="42%", height="42%", loc="upper right")
    axins.imshow(sh_inset, origin="lower")
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("0.5")

    labels = [f"step {l}" for l, _ in train_mse] + ["Cross Val", "Shuffle"]
    vals = [m for _, m in train_mse] + [cv_mse, sh_mse]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ypos = np.arange(len(vals))[::-1]
    axD.barh(ypos, vals, color=colors)
    for y, v in zip(ypos, vals):
        axD.text(v + sh_mse * 0.01, y, f"{v:.4f}", va="center", fontsize=9)
    axD.set_yticks(ypos)
    axD.set_yticklabels(labels)
    axD.set_xlabel("Mean Squared Error")
    axD.set_title("Residual stream vs. controls")
    axD.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Belief-state geometry is nontrivial  (Mess3, supervised)", fontsize=14)
    out = FIG_DIR / "fig07_supervised_controls_mess3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
