"""Principled RRXOR comparison using Stage 1 CCA/RRR subspace.

Identical to fig4_rrxor_principled.py but uses the Stage 1 regularized CCA/RRR
(Ledoit-Wolf + h=3) subspace instead of spectral-OOM.

Panels: Ground truth | Supervised (384-D) | Stage 1 CCA/RRR (5-D)

Requires: stage1_cca_output.pkl from running fig14_cca_rrr_estimator_twostage.py
"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10

MODEL_DIR = Path(__file__).parent.parent / "models"
FIG_DIR = Path(__file__).parent.parent / "figures"
D = 5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)

    # ---- Load Stage 1 subspace ----
    stage1_file = Path(__file__).parent / "stage1_cca_output.pkl"
    if not stage1_file.exists():
        print(f"ERROR: {stage1_file} not found. Run fig14_cca_rrr_estimator_twostage.py first.")
        return

    with open(stage1_file, "rb") as f:
        stage1_results = pickle.load(f)

    B_stage1 = stage1_results["B_stage1"]  # (256, 10)
    B = B_stage1[:, :D]  # Take top 5 dimensions
    print(f"Loaded Stage 1 subspace: {B.shape}")

    # ---- fig10's exact data: every (input, position) of the length-10 enumeration ----
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    concat = np.concatenate([acts[h] for h in hooks], axis=-1)  # (N, 10, 384)
    Xf = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, B36.shape[1])
    fidx = idx.reshape(-1)
    print(f"(input,position) points: {len(Xf)}   states: {len(B36)}")

    # ---- affine readouts (label-fit, identical convention) ----
    counts = np.bincount(fidx, minlength=len(B36))
    sw = 1.0 / np.clip(counts[fidx], 1, None)
    sw = sw / sw.mean()
    sup = LinearRegression().fit(Xf, Y, sample_weight=sw)
    r2_sup = sup.score(Xf, Y, sample_weight=sw)
    pred_sup = sup.predict(Xf)
    S = Xf @ B
    uns = LinearRegression().fit(S, Y, sample_weight=sw)
    r2_uns = uns.score(S, Y, sample_weight=sw)
    pred_uns = uns.predict(S)
    print(f"per-position decode R^2 (per-state-equal wt):  "
          f"supervised(384-D)={r2_sup:.3f}   stage1(5-D)={r2_uns:.3f}")

    # state-COM geometry R^2
    def com_r2(pred):
        com = np.array([pred[fidx == s].mean(0) for s in range(len(B36)) if (fidx == s).any()])
        bel = np.array([Y[fidx == s][0] for s in range(len(B36)) if (fidx == s).any()])
        return LinearRegression().fit(com, bel).score(com, bel)

    com_sup = com_r2(Xf)
    com_stage1 = com_r2(S)
    print(f"state-COM geometry R^2:  supervised={com_sup:.3f}   stage1={com_stage1:.3f}")

    # ---- identical panel-B PCA (fit on the 36 ground-truth beliefs) ----
    pca = PCA(n_components=4).fit(B36)
    tgt = pca.transform(B36)[:, [0, 2]]
    xy_sup = pca.transform(pred_sup)[:, [0, 2]]
    xy_stage1 = pca.transform(pred_uns)[:, [0, 2]]
    colors = np.array(F10.distinct_colors(len(B36)))

    def draw(ax, xy, title, scatter=True):
        if scatter:
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[fidx], alpha=0.25, edgecolors="none")
            for s in range(len(B36)):
                m = fidx == s
                if m.any():
                    ax.scatter(*xy[m].mean(0), s=90, c=[colors[s]], edgecolors="black",
                               linewidths=0.6, zorder=3)
            ax.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 3")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max()
            pad = 0.05 * (hi - lo)
            setl(lo - pad, hi + pad)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
    draw(ax[1], xy_sup, f"Supervised: full residual stream (384-D)\ndecode R$^2$={r2_sup:.2f} (per-state-equal wt)")
    draw(ax[2], xy_stage1, f"Stage 1 CCA/RRR (Ledoit-Wolf, h=3): 5-D subspace\ndecode R$^2$={r2_uns:.2f} (per-state-equal wt)")

    out = FIG_DIR / "fig4_rrxor_principled_stage1.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
