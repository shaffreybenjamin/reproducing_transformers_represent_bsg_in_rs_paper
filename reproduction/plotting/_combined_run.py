"""Run the estimator on RRXOR (ctx-10 model) and output the observability singular spectrum + figure.

DEFAULT = multistep ALS (order 3) for the activation-space transition operators + the exact 1-step
RIDGE readout. To experiment, flip the single toggle below:
    READOUT = "ridge"      # default: 1-step softmax readout (best at ctx-10)
    READOUT = "cca"        # replace C with the sampled CCA/RRR multistep readout
    READOUT = "ridge+cca"  # concatenate both (NB: not monotone-safe -- can displace good dirs;
                           #   prefer a held-out selector for "never hurt")
(The ALS operators always use the EXACT enumeration transitions; sampling is only used to build the
CCA readout, and only when READOUT != "ridge".)
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

READOUT = "ridge"          # <-- toggle: "ridge" (default) | "cca" | "ridge+cca"
HORIZON = 4                # CCA readout future-window horizon (ignored when READOUT == "ridge")
ALS_ORDER, DEPTH = 3, 6

t0 = time.time(); tic = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
FIG_DIR = Path(__file__).parent.parent / "figures"
device, d = "cpu", 5

model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
V = T.shape[0]
hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
Dd = len(hooks) * model.cfg.d_model

resid, soft, _, _ = F14._collect(model, T, pi, device)
reach = {}
for w in resid:
    ok, pre = True, ()
    for t in w:
        if pre in soft and soft[pre][t] <= F14.EPS:
            ok = False; break
        pre = pre + (t,)
    reach[w] = ok
pw = F14.analytic_prefix_probs(resid, T, pi)
tic("collected enumeration")

XF = None
if READOUT in ("cca", "ridge+cca"):
    XF = F14.sample_readout_features(model, T, pi, hooks, V, horizon=HORIZON, n_seq=60000)
    tic(f"sampled readout features X{XF[0].shape} F{XF[1].shape}")
tic(f"fitting order-{ALS_ORDER} ALS operators + '{READOUT}' readout, depth {DEPTH} ...")

rows, A, P, Gs, U, sv = F14.observable_subspace(
    resid, soft, reach, V, depth=DEPTH, wmap=pw,
    use_multistep_als=True, als_max_order=ALS_ORDER, als_n_iter=30,
    readout=READOUT, cca_horizon=HORIZON, cca_sampled_data=XF)
tic("observability matrix built")

print("\n=== observability matrix singular values (normalised sigma_i / sigma_1) ===")
print(np.round(sv[:14] / sv[0], 4))
ratios = (sv[:10] / sv[0])[:-1] / np.clip((sv[:10] / sv[0])[1:], 1e-12, None)
print("consecutive ratios sigma_i/sigma_{i+1}:", "  ".join(f"[{i+1}->{i+2}]{r:.2f}" for i, r in enumerate(ratios)))
print(f"elbow (largest drop) -> d_hat = {int(np.argmax(ratios))+1}")

# decode on the enumeration (fig4-identical)
B36, index = F10.msp_states(); es, bel, idx = F10.enumerate_inputs(F10.N_CTX, index); ea = F10.collect_activations(model, es, device, hooks)
Xf = np.concatenate([ea[h] for h in hooks], -1).reshape(-1, Dd); Yb = bel.reshape(-1, B36.shape[1]); fidx = idx.reshape(-1)
cnt = np.bincount(fidx, minlength=len(B36)); wgt = 1.0 / np.clip(cnt[fidx], 1, None); wgt /= wgt.mean()
def com_r2(S):
    c = np.array([S[fidx == k].mean(0) for k in range(len(B36)) if (fidx == k).any()])
    b = np.array([Yb[fidx == k][0] for k in range(len(B36)) if (fidx == k).any()])
    return LinearRegression().fit(c, b).score(c, b)
S = Xf @ U[:, :d]
reg = LinearRegression().fit(S, Yb, sample_weight=wgt); pp = reg.score(S, Yb, sample_weight=wgt); scom = com_r2(S)
sup = LinearRegression().fit(Xf, Yb, sample_weight=wgt)
print(f"\nsupervised      per-pos {sup.score(Xf,Yb,sample_weight=wgt):.3f}  state-COM {com_r2(Xf):.3f}")
print(f"estimator ({READOUT})  per-pos {pp:.3f}  state-COM {scom:.3f}   (obs-only OOM ref 0.661/0.911)")

# figure
pca = PCA(4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]; colors = np.array(F10.distinct_colors(len(B36)))
def draw(ax, xy, title, scatter=True):
    if scatter:
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[fidx], alpha=.25, edgecolors="none")
        for k in range(len(B36)):
            mk = fidx == k
            if mk.any(): ax.scatter(*xy[mk].mean(0), s=90, c=[colors[k]], edgecolors="black", linewidths=.6, zorder=3)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    else: ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=.5)
    ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
    for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])): lo, hi = v.min(), v.max(); p = .05 * (hi - lo); setl(lo - p, hi + p)
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
draw(ax[1], pca.transform(sup.predict(Xf))[:, [0, 2]], f"Supervised: full residual stream\ndecode R$^2$={sup.score(Xf,Yb,sample_weight=wgt):.2f}")
draw(ax[2], pca.transform(reg.predict(S))[:, [0, 2]], f"ALS ops + {READOUT} readout (5-D, L={DEPTH})\nper-pos R$^2$={pp:.2f}  state-COM={scom:.2f}")
out = FIG_DIR / f"rrxor_estimator_{READOUT.replace('+', '_')}.png"
fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}"); tic("done")
