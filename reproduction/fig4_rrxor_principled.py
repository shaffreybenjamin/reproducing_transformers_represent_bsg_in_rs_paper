"""Principled apples-to-apples RRXOR belief-geometry comparison (Fig 7C style).

Everything is held identical to the SUPERVISED fig10 pipeline -- the same per-(input,
position) set from the length-10 enumeration, the same panel-B PCA (fit on the 36
ground-truth beliefs, PC1/PC3), the same per-position scatter + per-state COM display,
and the same label-fit affine readout to the simplex. The ONLY thing that changes is
what feeds that affine map:

  panel 2 (supervised)   : full concatenated residual stream (384-D)  -> belief
  panel 3 (unsupervised) : the recovered OBS-OOM 5-D subspace          -> belief

So the figure isolates exactly "does the unsupervised 5-D subspace retain the belief
geometry the full activation has." The affine placement uses ground truth in BOTH (it
only demonstrates the recovered geometry lands on the simplex); the unsupervised content
is the subspace B, found from activations + model softmax + tree alone.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

MODEL_DIR = Path(__file__).parent / "models"
FIG_DIR = Path(__file__).parent / "figures"
D = 5


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)

    # ---- unsupervised subspace B (OBS-OOM), found target-free on unique prefixes ----
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
    pw = F14.analytic_prefix_probs(resid, T, pi)             # P(w) sample weights
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, 2, wmap=pw)
    B = Uobs[:, :D]                                           # (384, 5) unsupervised subspace

    # ---- fig10's exact data: every (input, position) of the length-10 enumeration ----
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    concat = np.concatenate([acts[h] for h in hooks], axis=-1)        # (N, 10, 384)
    Xf = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, B36.shape[1])
    fidx = idx.reshape(-1)
    print(f"(input,position) points: {len(Xf)}   states: {len(B36)}")

    # ---- affine readouts (label-fit, identical convention) ----
    # PER-STATE-EQUAL weighting (applied to BOTH panels): the geometry is the SET of 36
    # belief states, so each state should count equally, not be weighted by how often the
    # process visits it. Occurrence weighting (fig10/fig24) is central-heavy (shallow
    # prefixes sit near the root) and makes the limited 5-D map compress the apex states;
    # equal-per-state weighting removes that bias while staying matched across panels.
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
          f"supervised(384-D)={r2_sup:.3f}   unsupervised(5-D)={r2_uns:.3f}")
    # state-COM geometry R^2 (the state-level number), same weighting-agnostic COMs
    def com_r2(pred):
        com = np.array([pred[fidx == s].mean(0) for s in range(len(B36)) if (fidx == s).any()])
        bel = np.array([Y[fidx == s][0] for s in range(len(B36)) if (fidx == s).any()])
        return LinearRegression().fit(com, bel).score(com, bel)
    print(f"state-COM geometry R^2:  supervised={com_r2(Xf):.3f}   unsupervised={com_r2(S):.3f}")

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

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
    draw(ax[1], xy_sup, f"Supervised: full residual stream (384-D)\ndecode R$^2$={r2_sup:.2f} (per-state-equal wt)")
    draw(ax[2], xy_uns, f"Unsupervised: recovered spectral-OOM subspace (5-D)\ndecode R$^2$={r2_uns:.2f} (per-state-equal wt)")
    out = FIG_DIR / "fig26_rrxor_principled.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
