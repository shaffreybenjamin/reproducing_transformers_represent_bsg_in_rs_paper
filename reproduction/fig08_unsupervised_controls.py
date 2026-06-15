"""Figure 6 B/C/D — supervised vs. unsupervised controls on the RAW ACTIVATIONS.

Head-to-head, as fair as the two methods allow:
  - SAME prefix train/test split for both, SAME held-out test set, SAME rendering.
  - Supervised  : 64->3 linear map fit on TRAIN belief labels, applied to held-out
                  activations (residual-stream decode).
  - Unsupervised: 64->3 subspace fit on TRAIN dynamics (CCA->ALS), applied to the
                  SAME held-out activations (train-fit plane + affine for display).

  B Cross-Validation : held-out activations -> fractal persists for both.
  C Shuffle Control  : labels (sup) / dynamics (unsup) shuffled, refit -> collapse.
  D MSE bars         : belief-reconstruction MSE at four training checkpoints, plus
                       cross-validation and shuffle, with a broken x-axis (paper-style)
                       so the huge shuffle bar doesn't shrink the rest.

Irreducible asymmetry: an OOM transition needs the parent + all 3 children in-train,
so a prefix split gives the unsupervised method ~frac^4 of transitions; train=0.8
keeps ~12k while both rows share the same 20% held-out set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import unsupervised_belief_oom as U

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
LATENT_D = 3
TRAIN_FRAC = 0.8
SEED = 0

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
CKPTS = [
    ("step 100", MODEL_DIR / "progression" / "step_100.pt"),
    ("step 1,000", MODEL_DIR / "progression" / "step_1000.pt"),
    ("step 5,000", MODEL_DIR / "progression" / "step_5000.pt"),
    ("step 1,000,000", MODEL_DIR / "mess3_transformer.pt"),
]
TRI = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
EXT = (-0.05, 1.05, -0.05, np.sqrt(3) / 2 + 0.05)
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"]); cfg.device = device
    model = HookedTransformer(cfg); model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["context_len"]


def recover_basis(resid, soft, pp, max_len, vocab, d=LATENT_D):
    rows, X, P, Yc, Wt = U.build_transitions(resid, soft, pp, max_len, vocab)
    basis = U.predictive_cca(X, P, Yc, Wt)[0][:, :d]
    try:
        basis = U.als_refine_basis(X, P, Yc, Wt, basis, d)
    except Exception:
        pass
    return basis, len(rows)


def fit_plane(pts):
    mu = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    return mu, vt[:2].T


def fit_affine(src, dst, w):
    Xb = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(Xb * sw, dst * sw, rcond=None)
    return M


def apply_affine(src, M):
    return np.concatenate([src, np.ones((len(src), 1))], axis=1) @ M


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def decode_mse(feat_fit, y_fit, feat_eval, y_eval):
    return mse(LinearRegression().fit(feat_fit, y_fit).predict(feat_eval), y_eval)


def draw_triangle(ax, title):
    ax.plot(TRI[:, 0], TRI[:, 1], color="0.6", lw=1)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)


def broken_barh(axL, axR, vals, labels, wratio=3.0):
    """Horizontal bar chart with a broken x-axis: small bars on axL (fine scale),
    the outlier (shuffle) bar reaching into axR past a gap."""
    y = np.arange(len(vals))[::-1]
    small = vals[:-1]
    xmaxL = max(small) * 1.25
    big = vals[-1]
    xloR, xhiR = big * 0.985, big * 1.01
    for ax in (axL, axR):
        ax.barh(y, vals, color=BAR_COLORS[: len(vals)])
    axL.set_xlim(0, xmaxL)
    axR.set_xlim(xloR, xhiR)
    axL.set_yticks(y); axL.set_yticklabels(labels)
    axR.set_yticks([])
    axL.spines[["top", "right"]].set_visible(False)
    axR.spines[["top", "left"]].set_visible(False)
    axR.tick_params(left=False)
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
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size
    rng = np.random.default_rng(SEED)

    # recovery belief-MSE at each training checkpoint (both methods)
    sup_train, uns_train = [], []
    fin = None
    for label, path in CKPTS:
        model, context_len = load(path, device)
        resid, soft, belief, pp, max_len = U.collect_prefix_features_enumerated(model, hmm, context_len, device)
        seqs = [s for s in resid if s in belief]
        Y = np.array([belief[s] for s in seqs]); Xact = np.stack([resid[s] for s in seqs])
        sup_train.append(mse(LinearRegression().fit(Xact, Y).predict(Xact), Y))
        basis, _ = recover_basis(resid, soft, pp, max_len, vocab)
        uns_train.append(decode_mse(Xact @ basis, Y, Xact @ basis, Y))
        print(f"{label}: sup MSE={sup_train[-1]:.4f}  uns MSE={uns_train[-1]:.4f}")
        fin = (resid, soft, belief, pp, max_len, seqs, Y, Xact)

    resid, soft, belief, pp, max_len, seqs, Y, Xact = fin
    w = np.array([pp[s] for s in seqs]); rgb = np.clip(Y, 0, 1)
    is_tr = rng.random(len(seqs)) < TRAIN_FRAC
    tr = np.where(is_tr)[0]; te = np.where(~is_tr)[0]
    tr_keys = [seqs[i] for i in tr]
    print(f"split: train={len(tr)} test={len(te)}")

    # supervised CV + shuffle (final model)
    sup_cv = LinearRegression().fit(Xact[tr], Y[tr])
    sup_cv_pred = sup_cv.predict(Xact[te]); sup_cv_mse = mse(sup_cv_pred, Y[te])
    sup_sh_pred = LinearRegression().fit(Xact, Y[rng.permutation(len(seqs))]).predict(Xact)
    sup_sh_mse = mse(sup_sh_pred, Y)
    sup_cv_xy = U.simplex_to_xy(sup_cv_pred); sup_sh_xy = U.simplex_to_xy(sup_sh_pred)

    # unsupervised CV + shuffle (final model)
    basis_tr, n_tr = recover_basis({k: resid[k] for k in tr_keys}, {k: soft[k] for k in tr_keys},
                                   {k: pp[k] for k in tr_keys}, max_len, vocab)
    mu, comp = fit_plane(Xact[tr] @ basis_tr)
    xy_tr = (Xact[tr] @ basis_tr - mu) @ comp
    M = fit_affine(xy_tr, U.simplex_to_xy(Y[tr]), w[tr])
    unsup_cv_xy = apply_affine((Xact[te] @ basis_tr - mu) @ comp, M)
    unsup_cv_mse = decode_mse(Xact[tr] @ basis_tr, Y[tr], Xact[te] @ basis_tr, Y[te])
    _, Xt, Pt, Yct, Wtt = U.build_transitions(resid, soft, pp, max_len, vocab)
    perm = rng.permutation(Xt.shape[0])
    basis_sh = U.predictive_cca(Xt, Pt[perm], Yct[perm], Wtt)[0][:, :LATENT_D]
    try:
        basis_sh = U.als_refine_basis(Xt, Pt[perm], Yct[perm], Wtt, basis_sh, LATENT_D)
    except Exception:
        pass
    mu_s, comp_s = fit_plane(Xact @ basis_sh)
    xy_s = (Xact @ basis_sh - mu_s) @ comp_s
    unsup_sh_xy = apply_affine(xy_s, fit_affine(xy_s, U.simplex_to_xy(Y), w))
    unsup_sh_mse = decode_mse(Xact @ basis_sh, Y, Xact @ basis_sh, Y)
    print(f"sup  cv={sup_cv_mse:.4f} shuf={sup_sh_mse:.4f}")
    print(f"uns  cv={unsup_cv_mse:.4f} shuf={unsup_sh_mse:.4f}  (train transitions={n_tr})")

    # ---------------- plot ----------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.4], hspace=0.25, wspace=0.3)
    bar_labels = [l for l, _ in CKPTS] + ["Cross Val", "Shuffle"]
    rows = [
        ("Supervised\n(belief decode)", sup_cv_xy, rgb[te], sup_cv_mse, sup_sh_xy, rgb, sup_sh_mse,
         sup_train + [sup_cv_mse, sup_sh_mse]),
        ("Unsupervised\n(raw activations)", unsup_cv_xy, rgb[te], unsup_cv_mse, unsup_sh_xy, rgb, unsup_sh_mse,
         uns_train + [unsup_cv_mse, unsup_sh_mse]),
    ]
    for r, (rlabel, cvxy, cvc, cvm, shxy, shc, shm, vals) in enumerate(rows):
        axCV = fig.add_subplot(gs[r, 0]); axSh = fig.add_subplot(gs[r, 1])
        axCV.imshow(U.rasterize_simplex(cvxy, cvc, px=2), extent=EXT, origin="lower")
        draw_triangle(axCV, f"Cross Validation (MSE {cvm:.4f})")
        axSh.imshow(U.rasterize_simplex(shxy, shc, px=2), extent=EXT, origin="lower")
        draw_triangle(axSh, f"Shuffle Control (MSE {shm:.4f})")
        axCV.text(-0.16, 0.5, rlabel, transform=axCV.transAxes, rotation=90, va="center", ha="center", fontsize=12)
        sub = gs[r, 2].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
        broken_barh(fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), vals, bar_labels)

    fig.suptitle("Belief-state geometry is nontrivial — supervised vs. unsupervised, raw activations  (Mess3)\n"
                 f"shared 80/20 prefix split, same {len(te)}-prefix held-out set", fontsize=13)
    out = FIG_DIR / "fig08_unsupervised_controls_mess3.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
