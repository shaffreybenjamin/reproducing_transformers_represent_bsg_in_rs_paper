"""Figure 6 B/C/D — supervised vs. unsupervised controls on the RAW ACTIVATIONS.

Head-to-head, as fair as the two methods allow:
  - SAME prefix train/test split for both, SAME held-out test set, SAME rendering.
  - Supervised: a 64->3 linear map fit on TRAIN belief labels, applied to held-out
    activations (the residual-stream decode).
  - Unsupervised: a 64->3 subspace fit on TRAIN dynamics (CCA->ALS), applied to the
    SAME held-out activations (projected; a train-fit plane + affine gauge for
    display, never peeking at the test set).

Both rows therefore answer the identical question on identical held-out
activations: does the train-fit map place unseen prefixes on the fractal?

  B Cross-Validation : held-out activations -> fractal persists for both.
  C Shuffle Control  : labels (sup) / dynamics (unsup) shuffled, refit -> the map
                       can no longer place points correctly.
  D belief MSE       : recovery / cross-val / shuffle, same belief-reconstruction
                       metric for both.

The one irreducible asymmetry: an OOM transition needs the parent + all 3 children
in-train, so a prefix split gives the unsupervised method ~frac^4 of transitions.
We use train=0.8 so it still gets ~12k transitions while both rows share the same
20% (17,714-prefix) held-out set.
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
CKPT = MODEL_DIR / "mess3_transformer.pt"
TRI = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
EXT = (-0.05, 1.05, -0.05, np.sqrt(3) / 2 + 0.05)


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
    reg = LinearRegression().fit(feat_fit, y_fit)
    return mse(reg.predict(feat_eval), y_eval)


def draw_triangle(ax, title):
    ax.plot(TRI[:, 0], TRI[:, 1], color="0.6", lw=1)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size
    rng = np.random.default_rng(SEED)
    model, context_len = load(CKPT, device)

    resid, soft, belief, pp, max_len = U.collect_prefix_features_enumerated(model, hmm, context_len, device)
    seqs = [s for s in resid if s in belief]
    Y = np.array([belief[s] for s in seqs])
    Xact = np.stack([resid[s] for s in seqs])
    w = np.array([pp[s] for s in seqs])
    rgb = np.clip(Y, 0, 1)

    # shared prefix split
    is_tr = rng.random(len(seqs)) < TRAIN_FRAC
    tr = np.where(is_tr)[0]; te = np.where(~is_tr)[0]
    tr_keys = [seqs[i] for i in tr]
    print(f"split: train prefixes={len(tr)}  held-out test prefixes={len(te)}")

    # ===================== SUPERVISED (decode) =====================
    sup_full = LinearRegression().fit(Xact, Y)
    sup_rec_mse = mse(sup_full.predict(Xact), Y)
    sup_cv = LinearRegression().fit(Xact[tr], Y[tr])
    sup_cv_pred = sup_cv.predict(Xact[te])
    sup_cv_mse = mse(sup_cv_pred, Y[te])
    Ysh = Y[rng.permutation(len(seqs))]
    sup_sh_pred = LinearRegression().fit(Xact, Ysh).predict(Xact)
    sup_sh_mse = mse(sup_sh_pred, Y)
    sup_cv_xy = U.simplex_to_xy(sup_cv_pred)
    sup_sh_xy = U.simplex_to_xy(sup_sh_pred)

    # ===================== UNSUPERVISED (raw-activation projection) =====================
    basis_full, n_full = recover_basis(resid, soft, pp, max_len, vocab)
    unsup_rec_mse = decode_mse(Xact @ basis_full, Y, Xact @ basis_full, Y)
    basis_tr, n_tr = recover_basis({k: resid[k] for k in tr_keys}, {k: soft[k] for k in tr_keys},
                                   {k: pp[k] for k in tr_keys}, max_len, vocab)
    # CV visual: train-fit plane + affine (no test peeking), applied to held-out activations
    proj_tr = Xact[tr] @ basis_tr
    mu, comp = fit_plane(proj_tr)
    xy_tr = (proj_tr - mu) @ comp
    M = fit_affine(xy_tr, U.simplex_to_xy(Y[tr]), w[tr])
    unsup_cv_xy = apply_affine((Xact[te] @ basis_tr - mu) @ comp, M)
    unsup_cv_mse = decode_mse(Xact[tr] @ basis_tr, Y[tr], Xact[te] @ basis_tr, Y[te])
    print(f"unsup transitions: full={n_full}  train={n_tr}")
    # Shuffle: break the dynamics, refit subspace, project all activations
    _, Xt, Pt, Yct, Wtt = U.build_transitions(resid, soft, pp, max_len, vocab)
    perm = rng.permutation(Xt.shape[0])
    basis_sh = U.predictive_cca(Xt, Pt[perm], Yct[perm], Wtt)[0][:, :LATENT_D]
    try:
        basis_sh = U.als_refine_basis(Xt, Pt[perm], Yct[perm], Wtt, basis_sh, LATENT_D)
    except Exception:
        pass
    proj_sh = Xact @ basis_sh
    mu_s, comp_s = fit_plane(proj_sh)
    xy_s = (proj_sh - mu_s) @ comp_s
    Ms = fit_affine(xy_s, U.simplex_to_xy(Y), w)
    unsup_sh_xy = apply_affine(xy_s, Ms)
    unsup_sh_mse = decode_mse(Xact @ basis_sh, Y, Xact @ basis_sh, Y)

    print(f"belief MSE  sup: rec={sup_rec_mse:.4f} cv={sup_cv_mse:.4f} shuf={sup_sh_mse:.4f}")
    print(f"belief MSE  uns: rec={unsup_rec_mse:.4f} cv={unsup_cv_mse:.4f} shuf={unsup_sh_mse:.4f}")

    # ===================== plot =====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(U.rasterize_simplex(sup_cv_xy, rgb[te], px=2), extent=EXT, origin="lower")
    draw_triangle(axes[0, 0], f"Cross Validation (MSE {sup_cv_mse:.4f})")
    axes[0, 1].imshow(U.rasterize_simplex(sup_sh_xy, rgb, px=2), extent=EXT, origin="lower")
    draw_triangle(axes[0, 1], f"Shuffle Control (MSE {sup_sh_mse:.4f})")
    axes[1, 0].imshow(U.rasterize_simplex(unsup_cv_xy, rgb[te], px=2), extent=EXT, origin="lower")
    draw_triangle(axes[1, 0], f"Cross Validation (MSE {unsup_cv_mse:.4f})")
    axes[1, 1].imshow(U.rasterize_simplex(unsup_sh_xy, rgb, px=2), extent=EXT, origin="lower")
    draw_triangle(axes[1, 1], f"Shuffle Control (MSE {unsup_sh_mse:.4f})")

    for r, (rec, cv, sh) in enumerate([(sup_rec_mse, sup_cv_mse, sup_sh_mse),
                                       (unsup_rec_mse, unsup_cv_mse, unsup_sh_mse)]):
        ax = axes[r, 2]
        labels = ["Recovery", "Cross Val", "Shuffle"]
        vals = [rec, cv, sh]
        ax.barh([2, 1, 0], vals, color=["#4C72B0", "#55A868", "#937860"])
        for y, v in zip([2, 1, 0], vals):
            ax.text(v + max(vals) * 0.01, y, f"{v:.4f}", va="center", fontsize=9)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels(labels)
        ax.set_xlabel("belief-reconstruction MSE")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("error")

    axes[0, 0].text(-0.18, 0.5, "Supervised\n(belief decode)", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=12)
    axes[1, 0].text(-0.18, 0.5, "Unsupervised\n(raw activations)", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=12)
    fig.suptitle("Belief-state geometry is nontrivial — supervised vs. unsupervised, raw activations  (Mess3)\n"
                 f"shared 80/20 prefix split, same {len(te)}-prefix held-out set", fontsize=13)
    out = FIG_DIR / "fig08_unsupervised_controls_mess3.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
