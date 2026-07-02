"""fig4-style RRXOR belief-geometry figure for the NEW model (Adam, ctx14) WITH multistep ALS.

Identical construction to fig4_rrxor_principled_adam_ctx14.py -- same per-(input, position) data,
same per-state-equal-weighted affine readout, same PC1/PC3 display -- but the unsupervised
subspace is built from the MULTISTEP-ALS-refined operators (depth 5, als_max_order 3,
als_n_iter 30), the canonical fig4 ALS config, instead of the plain single-pass operators.

Panel 2 (supervised): full concatenated residual stream -> belief.
Panel 3 (unsupervised): recovered obs-OOM 5-D subspace (multistep ALS) -> belief.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

MODEL_DIR = Path(__file__).parent.parent / "models"          # NEW Adam/ctx14 models
FIG_DIR = Path(__file__).parent.parent / "figures"
CTX = 14                                                      # new model context window
D = 5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    F14.MAX_LEN = CTX          # prefix enumeration length for the subspace fit
    F10.N_CTX = CTX            # per-(input,position) enumeration length for scoring
    F14.MODEL_DIR = MODEL_DIR
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
    assert model.cfg.n_ctx == CTX, f"model n_ctx={model.cfg.n_ctx}, expected {CTX}"

    # ---- unsupervised subspace B: multistep-ALS spectral-OOM (depth 5, order 3) ----
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    resid, soft, belief_u, _ = F14._collect(model, T, pi, device)
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    pw = F14.analytic_prefix_probs(resid, T, pi)
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, 2, depth=5, wmap=pw,
                                                  use_multistep_als=True, als_max_order=3,
                                                  als_n_iter=30)
    B = Uobs[:, :D]

    # ---- fig10 data: every (input, position) of the length-CTX enumeration ----
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    concat = np.concatenate([acts[h] for h in hooks], axis=-1)
    Xf = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, B36.shape[1])
    fidx = idx.reshape(-1)
    print(f"(input,position) points: {len(Xf)}   states present: {len(np.unique(fidx))}/{len(B36)}")

    # ---- per-state-equal weighting (both panels) ----
    counts = np.bincount(fidx, minlength=len(B36))
    sw = 1.0 / np.clip(counts[fidx], 1, None)
    sw = sw / sw.mean()
    sup = LinearRegression().fit(Xf, Y, sample_weight=sw)
    r2_sup = sup.score(Xf, Y, sample_weight=sw); pred_sup = sup.predict(Xf)
    S = Xf @ B
    uns = LinearRegression().fit(S, Y, sample_weight=sw)
    r2_uns = uns.score(S, Y, sample_weight=sw); pred_uns = uns.predict(S)
    print(f"per-position decode R^2 (per-state-equal wt): supervised={r2_sup:.3f}  unsupervised(5-D,ALS)={r2_uns:.3f}")

    def com_r2(pred):
        present = [s for s in range(len(B36)) if (fidx == s).any()]
        com = np.array([pred[fidx == s].mean(0) for s in present])
        bel = np.array([Y[fidx == s][0] for s in present])
        return LinearRegression().fit(com, bel).score(com, bel)
    print(f"state-COM geometry R^2: supervised={com_r2(Xf):.3f}  unsupervised={com_r2(S):.3f}")

    # ---- identical panel-B PCA (fit on the 36 ground-truth beliefs) ----
    pca = PCA(n_components=4).fit(B36)
    tgt = pca.transform(B36)[:, [0, 2]]
    xy_sup = pca.transform(pred_sup)[:, [0, 2]]
    xy_uns = pca.transform(pred_uns)[:, [0, 2]]
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
        ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo); setl(lo - pad, hi + pad)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.9))
    fig.suptitle("RRXOR belief geometry -- NEW model (Adam, context 14) + multistep ALS (order 3, depth 5)",
                 fontsize=14, y=1.02)
    draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
    draw(ax[1], xy_sup, f"Supervised: full residual stream\ndecode R$^2$={r2_sup:.2f} (per-state-equal wt)")
    draw(ax[2], xy_uns, f"Unsupervised: spectral-OOM subspace + multistep ALS (5-D)\ndecode R$^2$={r2_uns:.2f} (per-state-equal wt)")
    out = FIG_DIR / "fig4_rrxor_principled_als_adam_ctx14.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
