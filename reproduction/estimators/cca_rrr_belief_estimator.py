"""Sampling CCA/RRR belief-subspace estimator for RRXOR (consolidated).

Recovers the transformer's belief-state geometry WITHOUT labels by canonical-correlation analysis
between residual-stream activations (the "past") and the empirically-observed future-token window
(the H-gram). Sampling many stochastic futures per prefix -- instead of one exact enumerated future
distribution -- keeps the future covariance full-rank, so the true canonical structure emerges
(the enumeration version overfits with spurious cc->1). This is the mature harness distilled from
plotting/_cca_figure.py / _cca_sampling.py and the fig14_observable_oom.py CCA path.

Pipeline (per model, per horizon H):
  1. Sample N sequences of length L=n_ctx from the true RRXOR process, run them through the model,
     collect concatenated residual-stream activations a(w) and forward-filter the TRUE beliefs b(w)
     (labels used ONLY to score, never to fit the subspace).
  2. For each position t (1..L-H), pair a(w_t) with the one-hot future H-gram feature.
  3. Whitened CCA of (activation, future H-gram) -> canonical correlations cc + subspace B = top-d
     activation canonical directions.
  4. Score B three ways: consistent held-out per-pos R^2 (own-distribution, the honest score),
     enumeration per-pos R^2 and state-COM R^2 (fig4 36-state diamond, comparable to the
     operator-composition methods), plus decode-vs-d and the largest-log-gap elbow on cc.

Outputs (per model): H-sweep figure, canonical-correlation-spectrum+elbow figure, and the 3-panel
belief-geometry figure at the best horizon. With >1 model, also a head-to-head comparison figure.

Examples
--------
  # final model, default horizons 4/6/8
  python cca_rrr_belief_estimator.py

  # a training checkpoint
  python cca_rrr_belief_estimator.py --models ckpt10k=rrxor_checkpoints/rrxor_epoch_10000.pt

  # head-to-head (reproduces the 10k-vs-20k comparison)
  python cca_rrr_belief_estimator.py \
      --models 10k=rrxor_checkpoints/rrxor_epoch_10000.pt "20k=rrxor_transformer.pt"
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

_THIS = Path(__file__).resolve().parent
REPRO = _THIS.parent
for _p in (REPRO / "plotting", REPRO / "estimators"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import fig10_rrxor_representation as F10          # model loader + enumeration + belief states
from spectral_oom_refined import detect_elbow     # largest-log-gap model-order selector
from simplexity.generative_processes.transition_matrices import rrxor

MODEL_DIR = REPRO / "models"
FIG_DIR = REPRO / "figures"
# reference d=5 numbers from the operator-composition estimators on the ctx14 model (for context lines)
REF_PLAIN_OBS = 0.749       # plain obs-OOM (spectral_oom_refined baseline)
REF_ALS_BEST = 0.804        # best multistep ALS (order 2, depth 10)

_t0 = time.time()
def tic(m): print(f"[{time.time()-_t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# process sampling + belief filtering + future features
# ---------------------------------------------------------------------------

def sample_process(T, pi, N, L, seed):
    """Sample N length-L token sequences from the true process (transition tensor T, start pi)."""
    rng = np.random.default_rng(seed); V, ns = T.shape[0], T.shape[1]
    st = rng.choice(ns, N, p=pi); s = np.empty((N, L), np.int64)
    for t in range(L):
        Px = np.stack([T[x][st].sum(1) for x in range(V)], 1); Px = np.clip(Px, 1e-12, None); Px /= Px.sum(1, keepdims=True)
        x = (rng.random(N)[:, None] < np.cumsum(Px, 1)).argmax(1); s[:, t] = x
        tr = T[x, st] / T[x, st].sum(1, keepdims=True); st = (rng.random(N)[:, None] < np.cumsum(tr, 1)).argmax(1)
    return s


def beliefs_for(seqs, T, pi):
    """Forward-filter the TRUE belief b(w) after each token (used only to score)."""
    Nq, Lq = seqs.shape; ns = T.shape[1]; B = np.empty((Nq, Lq, ns), np.float32); b = np.tile(pi, (Nq, 1))
    for t in range(Lq):
        nb = np.einsum('ni,nij->nj', b, T[seqs[:, t]]); nb /= nb.sum(1, keepdims=True); B[:, t] = nb; b = nb
    return B


def future_feature(fut, H, V):
    """One-hot encode the length-H future window as [1-gram | 2-gram | ... | H-gram]."""
    n = len(fut); fe = []
    for k in range(1, H + 1):
        idx = np.zeros(n, np.int64)
        for j in range(k):
            idx = idx * V + fut[:, j]
        oh = np.zeros((n, V ** k), np.float32); oh[np.arange(n), idx] = 1.0; fe.append(oh)
    return np.hstack(fe)


# ---------------------------------------------------------------------------
# activation collection + evaluation scaffolding
# ---------------------------------------------------------------------------

def collect_acts(model, seqs, hooks, bs=8192):
    out = []
    for i in range(0, len(seqs), bs):
        with torch.no_grad():
            _, c = model.run_with_cache(torch.from_numpy(seqs[i:i + bs]), names_filter=lambda n: n in hooks)
        out.append(np.concatenate([c[h].numpy() for h in hooks], -1).astype(np.float32))
    return np.concatenate(out, 0)


class EnumEval:
    """fig4 36-state-diamond enumeration eval set (model-independent sequences)."""
    def __init__(self, n_ctx):
        F10.N_CTX = n_ctx
        self.B36, index = F10.msp_states()
        self.seqs, ebel, eidx = F10.enumerate_inputs(n_ctx, index)
        self.Yb = ebel.reshape(-1, self.B36.shape[1]); self.fidx = eidx.reshape(-1)
        cnt = np.bincount(self.fidx, minlength=len(self.B36))
        self.wgt = 1.0 / np.clip(cnt[self.fidx], 1, None); self.wgt /= self.wgt.mean()

    def com_r2(self, S):
        com = np.array([S[self.fidx == k].mean(0) for k in range(len(self.B36)) if (self.fidx == k).any()])
        bl = np.array([self.Yb[self.fidx == k][0] for k in range(len(self.B36)) if (self.fidx == k).any()])
        return LinearRegression().fit(com, bl).score(com, bl)


def fit_cca(acts, seqs, bel, Xf, ev, H, L, V, d, d_grid, ridge_x=1e-1, ridge_f=1e-3):
    """Whitened CCA(activation, future H-gram) on the train half; score on held-out + enumeration.

    Returns dict with cc (canonical corrs), B (D,d subspace), reg (enum decode map), cons/pp/sc
    (held-out per-pos / enum per-pos / enum state-COM at dim d), dec (enum per-pos vs d), elbow.
    """
    N = len(acts); Dd = acts.shape[-1]; trm = np.arange(N) % 2 == 0
    Xtr, Ftr, Ytr, Xte, Yte = [], [], [], [], []
    for t in range(L - H):
        Xtr.append(acts[trm, t]); Ftr.append(future_feature(seqs[trm, t + 1:t + 1 + H], H, V)); Ytr.append(bel[trm, t])
        Xte.append(acts[~trm, t]); Yte.append(bel[~trm, t])
    Xtr = np.vstack(Xtr); Ftr = np.vstack(Ftr); Ytr = np.vstack(Ytr); Xte = np.vstack(Xte); Yte = np.vstack(Yte)
    Xc = (Xtr - Xtr.mean(0)).astype(np.float64); Fc = (Ftr - Ftr.mean(0)).astype(np.float64); n = len(Xtr)
    ex, Ux = np.linalg.eigh(Xc.T @ Xc / n + ridge_x * np.eye(Dd)); Wx = Ux @ np.diag(ex ** -0.5) @ Ux.T
    ef, Uf = np.linalg.eigh(Fc.T @ Fc / n + ridge_f * np.eye(Ftr.shape[1])); Wf = Uf @ np.diag(np.clip(ef, 1e-12, None) ** -0.5) @ Uf.T
    U2, cc, _ = np.linalg.svd(Wx @ (Xc.T @ Fc / n) @ Wf, full_matrices=False)
    B = Wx @ U2[:, :d]
    cons = LinearRegression().fit(Xtr @ B, Ytr).score(Xte @ B, Yte)
    reg = LinearRegression().fit(Xf @ B, ev.Yb, sample_weight=ev.wgt)
    pp = reg.score(Xf @ B, ev.Yb, sample_weight=ev.wgt); sc = ev.com_r2(Xf @ B)
    dec = []
    for dd in d_grid:
        Bd = Wx @ U2[:, :dd]
        dec.append((dd, LinearRegression().fit(Xf @ Bd, ev.Yb, sample_weight=ev.wgt).score(Xf @ Bd, ev.Yb, sample_weight=ev.wgt)))
    return dict(cc=cc, B=B, reg=reg, cons=cons, pp=pp, sc=sc, dec=dec,
                elbow=detect_elbow(cc, k_max=min(15, len(cc))))


def eval_model(model, sampled, ev, horizons, d, d_grid):
    """Run the CCA estimator over all horizons for one model; return per-H results + supervised refs."""
    seqs, bel = sampled
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    L = model.cfg.n_ctx; V = model.cfg.d_vocab
    acts = collect_acts(model, seqs, hooks)
    Xf = np.concatenate([F10.collect_activations(model, ev.seqs, "cpu", hooks)[h] for h in hooks], -1).reshape(-1, acts.shape[-1])
    sup = LinearRegression().fit(Xf, ev.Yb, sample_weight=ev.wgt)
    sup_pp = sup.score(Xf, ev.Yb, sample_weight=ev.wgt); sup_com = ev.com_r2(Xf)
    res = {H: fit_cca(acts, seqs, bel, Xf, ev, H, L, V, d, d_grid) for H in horizons}
    return dict(Xf=Xf, sup=sup, sup_pp=sup_pp, sup_com=sup_com, res=res)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _draw_simplex(ax, xy, colors, fidx, tgt, title, scatter=True):
    if scatter:
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[fidx], alpha=.25, edgecolors="none")
        for k in range(len(tgt)):
            mk = fidx == k
            if mk.any():
                ax.scatter(*xy[mk].mean(0), s=90, c=[colors[k]], edgecolors="black", linewidths=.6, zorder=3)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=.5)
    ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
        lo, hi = v.min(), v.max(); p = .05 * (hi - lo); setl(lo - p, hi + p)


def fig_geometry(out, label, M, ev, best_H, N):
    B = M["res"][best_H]["B"]; reg = M["res"][best_H]["reg"]
    pp = M["res"][best_H]["pp"]; sc = M["res"][best_H]["sc"]
    pca = PCA(4).fit(ev.B36); tgt = pca.transform(ev.B36)[:, [0, 2]]
    colors = np.array(F10.distinct_colors(len(ev.B36)))
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.9))
    fig.suptitle(f"RRXOR belief geometry -- {label}, sampling CCA/RRR (H={best_H}, N={N})", fontsize=14, y=1.02)
    _draw_simplex(ax[0], tgt, colors, ev.fidx, tgt, "Ground truth belief geometry", scatter=False)
    _draw_simplex(ax[1], pca.transform(M["sup"].predict(M["Xf"]))[:, [0, 2]], colors, ev.fidx, tgt,
                  f"Supervised: full residual stream\nper-pos R$^2$={M['sup_pp']:.2f}")
    _draw_simplex(ax[2], pca.transform(reg.predict(M["Xf"] @ B))[:, [0, 2]], colors, ev.fidx, tgt,
                  f"Sampling-CCA subspace (5-D, H={best_H})\nper-pos R$^2$={pp:.2f}  state-COM={sc:.2f}")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"saved -> {out}")


def fig_hsweep(out, M, horizons, sup_pp):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(horizons, [M["res"][H]["pp"] for H in horizons], "o-", lw=1.8, ms=6, label="enum per-pos R$^2$")
    ax.plot(horizons, [M["res"][H]["sc"] for H in horizons], "s-", lw=1.8, ms=6, label="enum state-COM R$^2$")
    ax.plot(horizons, [M["res"][H]["cons"] for H in horizons], "^-", lw=1.8, ms=6, label="held-out per-pos R$^2$ (own-dist)")
    ax.axhline(sup_pp, color="k", ls="--", lw=1, alpha=0.6, label=f"supervised per-pos = {sup_pp:.2f}")
    ax.axhline(REF_ALS_BEST, color="tab:orange", ls=":", lw=1.4, alpha=0.8, label=f"best ALS = {REF_ALS_BEST}")
    ax.axhline(REF_PLAIN_OBS, color="tab:blue", ls=":", lw=1.2, alpha=0.7, label=f"plain obs-OOM = {REF_PLAIN_OBS}")
    ax.set_xlabel("future horizon H"); ax.set_ylabel(r"belief R$^2$ (d=5)")
    ax.set_title("RRXOR: sampling CCA/RRR vs horizon H")
    ax.set_ylim(0, 1.02); ax.set_xticks(horizons); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"saved -> {out}")


def fig_spectrum(out, M, horizons, sup_pp, true_dim):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5)); cmap = plt.get_cmap("viridis")
    for i, H in enumerate(horizons):
        r = M["res"][H]; cc = r["cc"]; col = cmap(i / max(1, len(horizons) - 1)); m = min(12, len(cc))
        ax0.plot(range(1, m + 1), cc[:m], "o-", color=col, lw=1.7, ms=5, label=f"H={H} (elbow d={r['elbow']})")
        ax1.plot([dd for dd, _ in r["dec"]], [v for _, v in r["dec"]], "o-", color=col, lw=1.7, ms=5, label=f"H={H}")
    ax0.axvline(true_dim, color="red", ls="--", lw=1.5, alpha=0.7, label=f"n states = {true_dim}")
    ax0.set_xlabel("canonical mode index"); ax0.set_ylabel("canonical correlation")
    ax0.set_title("CCA/RRR canonical-correlation spectrum"); ax0.set_ylim(0, 1.0); ax0.grid(alpha=0.3); ax0.legend(fontsize=8, loc="upper right")
    ax1.axhline(sup_pp, color="k", ls="--", lw=1, alpha=0.6, label=f"supervised = {sup_pp:.2f}")
    ax1.axvline(true_dim, color="red", ls="--", lw=1.2, alpha=0.6)
    ax1.set_xlabel("subspace dim d"); ax1.set_ylabel(r"enum per-pos decode R$^2$")
    ax1.set_title("CCA/RRR decode saturation vs d"); ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"saved -> {out}")


def fig_compare(out, models, ev, cmp_H):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5)); cmap = plt.get_cmap("tab10")
    for i, (label, M) in enumerate(models.items()):
        r = M["res"][cmp_H]; cc = r["cc"]; col = cmap(i); m = min(12, len(cc))
        ax0.plot(range(1, m + 1), cc[:m], "o-", color=col, lw=1.7, ms=5, label=f"{label} (elbow d={r['elbow']})")
        ax1.plot([dd for dd, _ in r["dec"]], [v for _, v in r["dec"]], "o-", color=col, lw=1.7, ms=5, label=f"{label}")
        ax1.axhline(M["sup_pp"], color=col, ls="--", lw=1, alpha=0.5)
    ax0.set_xlabel("canonical mode index"); ax0.set_ylabel("canonical correlation")
    ax0.set_title(f"CCA/RRR spectrum, model comparison (H={cmp_H})"); ax0.set_ylim(0, 1.0); ax0.grid(alpha=0.3); ax0.legend(fontsize=9, loc="upper right")
    ax1.set_xlabel("subspace dim d"); ax1.set_ylabel(r"enum per-pos decode R$^2$")
    ax1.set_title(f"belief recovery, model comparison (H={cmp_H})\n(dashed = each model's supervised ceiling)")
    ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"saved -> {out}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _resolve(path):
    for cand in (Path(path), REPRO / path, MODEL_DIR / path):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"model not found: {path} (tried cwd, {REPRO}, {MODEL_DIR})")


def _parse_models(specs):
    out = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            path = label; label = Path(path).stem
        out[label] = _resolve(path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rrxor=rrxor_transformer.pt"],
                    help="one or more LABEL=PATH (or bare PATH). Paths resolve vs cwd, repro root, or models/.")
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--n-samples", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=5, help="belief subspace dim used for the headline/geometry")
    ap.add_argument("--d-grid", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--compare-H", type=int, default=6, help="horizon used for the multi-model comparison figure")
    ap.add_argument("--tag", default="", help="suffix appended to every output filename")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models_spec = _parse_models(args.models)
    device = "cpu"
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    V, ns = T.shape[0], T.shape[1]

    # peek n_ctx from the first model to size the sampling length + enumeration
    first = F10.load_model(next(iter(models_spec.values())), device)
    L = first.cfg.n_ctx
    ev = EnumEval(L)
    seqs = sample_process(T, pi, args.n_samples, L, args.seed); bel = beliefs_for(seqs, T, pi)
    sampled = (seqs, bel)
    tic(f"sampled N={args.n_samples}, L={L}; enumeration set {ev.Yb.shape[0]} pts")

    models = {}
    for label, path in models_spec.items():
        model = F10.load_model(path, device)
        assert model.cfg.n_ctx == L, f"{label}: n_ctx {model.cfg.n_ctx} != {L} (mixed context windows unsupported in one run)"
        M = eval_model(model, sampled, ev, args.horizons, args.dim, args.d_grid)
        models[label] = M
        tic(f"[{label}] supervised per-pos {M['sup_pp']:.3f} state-COM {M['sup_com']:.3f}")
        print(f"\n### {label}   (supervised per-pos {M['sup_pp']:.3f}, state-COM {M['sup_com']:.3f}) ###")
        print("   H | cc[:6]                               | held-out | enum-pp | enum-COM | elbow")
        for H in args.horizons:
            r = M["res"][H]
            print(f"  H{H} | {np.round(r['cc'][:6], 3)} |  {r['cons']:.3f}  |  {r['pp']:.3f}  |  {r['sc']:.3f}  |  d={r['elbow']}")
        bH = max(args.horizons, key=lambda H: M["res"][H]["pp"])
        print(f"   best H by enum-pp: H={bH} -> enum-pp {M['res'][bH]['pp']:.3f}  state-COM {M['res'][bH]['sc']:.3f}  held-out {M['res'][bH]['cons']:.3f}")

        safe = label.replace(" ", "_").replace("(", "").replace(")", "")
        fig_hsweep(FIG_DIR / f"cca_rrr_hsweep_{safe}{args.tag}.png", M, args.horizons, M["sup_pp"])
        fig_spectrum(FIG_DIR / f"cca_rrr_spectrum_{safe}{args.tag}.png", M, args.horizons, M["sup_pp"], ns)
        fig_geometry(FIG_DIR / f"cca_rrr_geometry_{safe}{args.tag}.png", label, M, ev, bH, args.n_samples)

    if len(models) > 1:
        labels = "_vs_".join(k.replace(" ", "_").replace("(", "").replace(")", "") for k in models)
        cmp_H = args.compare_H if args.compare_H in args.horizons else args.horizons[0]
        fig_compare(FIG_DIR / f"cca_rrr_compare_{labels}{args.tag}.png", models, ev, cmp_H)

    tic("done")


if __name__ == "__main__":
    main()
