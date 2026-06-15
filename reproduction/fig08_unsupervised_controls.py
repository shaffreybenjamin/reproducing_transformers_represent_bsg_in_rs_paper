"""Figure 6 (B/C/D) analogue — UNSUPERVISED (OOM) controls.

The supervised controls attack a regression that targets belief. The OOM never
sees belief — it recovers a subspace + per-token operators from activations +
softmax — so the analogues attack the *dynamics*:

  B  Cross-validation : fit the CCA->ALS subspace on a TRAIN split of prefixes,
                        project the held-out prefixes' activations -> fractal
                        persists (subspace is global, not memorised).
  C  Shuffle control  : permute the parent->future pairing in the transitions
                        (breaks the dynamics) and refit. The decisive collapse is
                        in the OPERATORS: eig(A^x) no longer matches eig(T^x).
                        (The raw-activation panel only partly collapses because
                        activations stay belief-rich whatever subspace you pick —
                        which is exactly why the operator/one-step test is the
                        real control here.)
  D  One-step MSE     : the OOM's native reconstruction error (rescaled one-step
                        prediction), for each training checkpoint, cross-validation
                        and shuffle.

Same model/config/projection/rasterizer as fig06/fig07.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.transition_matrices import mess3
import unsupervised_belief_oom as U

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
LATENT_D = 3
# A transition needs parent + all 3 children in the train set, so a prefix split
# keeps only frac^4 of transitions. Use 0.85 so the train count (~15k transitions)
# is comparable to the supervised CV's training samples -> a fair held-out test.
TRAIN_FRAC = 0.85
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


def recover(resid, soft, prefix_prob, max_len, vocab, d=LATENT_D, als=True):
    """CCA(->ALS) basis + per-token operators from a (subset of) prefixes."""
    rows, X, P, Yc, Wt = U.build_transitions(resid, soft, prefix_prob, max_len, vocab)
    basis = U.predictive_cca(X, P, Yc, Wt)[0][:, :d]
    if als:
        try:
            basis = U.als_refine_basis(X, P, Yc, Wt, basis, d)
        except Exception as exc:
            print(f"  ALS skipped ({exc})")
    ops, _ = U.fit_operators(X @ basis, P, Yc @ basis, Wt)
    return basis, ops, (rows, X, P, Yc, Wt)


def onestep_mse(X, P, Yc, Wt, basis, ops):
    A, Bc = X @ basis, Yc @ basis
    num = den = 0.0
    for x in ops:
        r = A @ ops[x] - P[:, x, None] * Bc[:, x, :]
        num += float((Wt[:, None] * r ** 2).sum())
        den += float(Wt.sum()) * A.shape[1]
    return num / den


def sub(d, keys):
    return {k: d[k] for k in keys}


def draw_triangle(ax):
    ax.plot(TRI[:, 0], TRI[:, 1], color="0.6", lw=1)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect("equal"); ax.axis("off")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size
    rng = np.random.default_rng(SEED)
    tm = np.array(mess3(**MESS3_PARAMS))
    true_eig = np.sort(np.linalg.eigvals(tm[0]).real)

    # one-step MSE at each training checkpoint (panel D, training bars)
    train_mse = []
    final = None
    for label, path in CKPTS:
        model, context_len, step = load(path, device)
        resid, soft, belief, pp, max_len = U.collect_prefix_features_enumerated(model, hmm, context_len, device)
        basis, ops, (rows, X, P, Yc, Wt) = recover(resid, soft, pp, max_len, vocab)
        train_mse.append((label, onestep_mse(X, P, Yc, Wt, basis, ops)))
        print(f"step {label}: one-step MSE = {train_mse[-1][1]:.5f}")
        final = (resid, soft, belief, pp, max_len, basis, ops, X, P, Yc, Wt)

    resid, soft, belief, pp, max_len, basis, ops, X, P, Yc, Wt = final
    seqs = [s for s in resid if s in belief]
    true_b = np.array([belief[s] for s in seqs])
    true_xy = U.simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1)
    wcol = np.array([pp[s] for s in seqs])
    Xfull = np.stack([resid[s] for s in seqs])
    eig_rec = np.sort(np.linalg.eigvals(ops[0]).real)

    # --- B: cross-validation (subspace from train prefixes, project held-out) ---
    is_tr = rng.random(len(seqs)) < TRAIN_FRAC
    tr_keys = [s for s, t in zip(seqs, is_tr) if t]
    te_keys = [s for s, t in zip(seqs, is_tr) if not t]
    basis_cv, _, (rows_cv, *_rest) = recover(sub(resid, tr_keys), sub(soft, tr_keys), sub(pp, tr_keys), max_len, vocab)
    print(f"CV: train transitions = {len(rows_cv)}  (held-out prefixes = {len(te_keys)})")
    te_xy = U.simplex_to_xy(np.array([belief[s] for s in te_keys]))
    te_rgb = np.clip(np.array([belief[s] for s in te_keys]), 0, 1)
    te_proj = U.plane_coords(np.stack([resid[s] for s in te_keys]) @ basis_cv)
    te_aligned, _ = U.affine_align(te_proj, te_xy, np.array([pp[s] for s in te_keys]))
    cv_img = U.rasterize_simplex(te_aligned, te_rgb, px=2)

    # panel-D CV bar: proper held-out one-step MSE via a TRANSITION-level split
    # (fit subspace+operators on 80% of transitions, evaluate on the held-out 20%)
    tperm = rng.permutation(X.shape[0])
    ntr = int(0.8 * X.shape[0])
    ti, vi = tperm[:ntr], tperm[ntr:]
    basis_t = U.predictive_cca(X[ti], P[ti], Yc[ti], Wt[ti])[0][:, :LATENT_D]
    try:
        basis_t = U.als_refine_basis(X[ti], P[ti], Yc[ti], Wt[ti], basis_t, LATENT_D)
    except Exception:
        pass
    ops_t = U.fit_operators(X[ti] @ basis_t, P[ti], Yc[ti] @ basis_t, Wt[ti])[0]
    cv_mse = onestep_mse(X[vi], P[vi], Yc[vi], Wt[vi], basis_t, ops_t)

    # --- C: shuffle the parent->future pairing, refit ---
    perm = rng.permutation(X.shape[0])      # permute future rows w.r.t. past -> breaks dynamics
    basis_sh = U.predictive_cca(X, P[perm], Yc[perm], Wt)[0][:, :LATENT_D]
    try:
        basis_sh = U.als_refine_basis(X, P[perm], Yc[perm], Wt, basis_sh, LATENT_D)
    except Exception:
        pass
    ops_sh = U.fit_operators(X @ basis_sh, P[perm], Yc[perm] @ basis_sh, Wt)[0]
    eig_sh = np.sort(np.linalg.eigvals(ops_sh[0]).real)
    sh_proj = U.plane_coords(Xfull @ basis_sh)
    sh_aligned, _ = U.affine_align(sh_proj, true_xy, wcol)
    sh_img = U.rasterize_simplex(sh_aligned, color, px=2)
    sh_mse = onestep_mse(X, P, Yc, Wt, basis_sh, ops_sh)  # shuffle operators vs REAL dynamics
    print(f"one-step MSE  recovery={train_mse[-1][1]:.5f}  CV={cv_mse:.5f}  shuffle={sh_mse:.5f}")
    print(f"eig: true={np.round(true_eig,3)}  recovery={np.round(eig_rec,3)}  shuffle={np.round(eig_sh,3)}")

    # ------------------------------- plot ------------------------------- #
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    axB, axC, axD = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    axB.imshow(cv_img, extent=EXT, origin="lower"); draw_triangle(axB)
    axB.set_title("Cross Validation\n(held-out prefixes, raw activations)")

    axC.imshow(sh_img, extent=EXT, origin="lower"); draw_triangle(axC)
    axC.set_title("Shuffle Control\n(broken dynamics)")
    axC.text(0.5, -0.02,
             f"eig(A$^x$): true {np.round(true_eig,2)}\nrecovery {np.round(eig_rec,2)}  vs  shuffle {np.round(eig_sh,2)}",
             ha="center", va="top", transform=axC.transAxes, fontsize=8)

    labels = [f"step {l}" for l, _ in train_mse] + ["Cross Val", "Shuffle"]
    vals = [m for _, m in train_mse] + [cv_mse, sh_mse]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ypos = np.arange(len(vals))[::-1]
    axD.barh(ypos, vals, color=colors)
    for y, v in zip(ypos, vals):
        axD.text(v + max(vals) * 0.01, y, f"{v:.4f}", va="center", fontsize=9)
    axD.set_yticks(ypos); axD.set_yticklabels(labels)
    axD.set_xlabel("one-step prediction MSE (rescaled)")
    axD.set_title("OOM recovery vs. controls")
    axD.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Belief-state geometry is nontrivial  (Mess3, unsupervised OOM)", fontsize=14)
    out = FIG_DIR / "fig08_unsupervised_controls_mess3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
