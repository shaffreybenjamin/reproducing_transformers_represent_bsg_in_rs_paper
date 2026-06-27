"""Figure 7 (unsupervised OOM analogue) for RRXOR -- a NEGATIVE result with mechanism.

We run the same target-free OOM that recovered Mess3's belief geometry (R^2~0.998)
on the concatenation of RRXOR's per-layer activations, using only activations + the
model's softmax. RRXOR is not full-support, so we enumerate ALL token strings (every
prefix then has both children); the forbidden child is handled by the rescaling
itself -- its target P(x|w)*a(wx) ~ 0 (correct: the true operator maps a forbidden
belief to 0) -- so no process knowledge beyond vocab + the model's softmax is used.

FINDING: the unsupervised OOM CANNOT recover RRXOR's belief geometry. Even in the
most generous setting (uniform reachability weights, PCA pre-reduction, best d), the
belief decoded from the unsupervised subspace plateaus at R^2 ~ 0.41, far below the
supervised residual-stream concat ceiling (0.86) and Mess3's 0.998. Three compounding,
quantified reasons -- the unsupervised mirror of the paper's central RRXOR point
(belief geometry is distributed and weakly linear):
  1. RRXOR's activations are near-deterministic given the prefix, so the CCA
     canonical correlations all saturate at 1.0 -- the predictive-correlation
     ranking that isolates belief on Mess3 carries no signal here.
  2. Belief is spread across 100+ activation dimensions (PCA-120 decode 0.71) -- the
     activation-space echo of the paper's "spread across layers" (panel E).
  3. RRXOR's determinism gives a tiny effective sample size (~77 under probability
     weighting): too few belief states to estimate a high-dim subspace target-free.
Supervised regression still reaches 0.86 because it has belief targets, all 384 dims
and 1046 equally-weighted prefixes; the unsupervised estimator has none of these.

The figure documents this: belief-decode R^2 vs subspace dimension (plateaus low),
plus the decoded "geometry" (a blob, not the 36-state structure).
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.transition_matrices import rrxor
import unsupervised_belief_oom as U
import fig09_rrxor_ground_truth as B  # distinct_colors only

P1 = P2 = 0.5
MAX_LEN = 10
D_MAX = 8
PRE_PCA_K = 20         # PCA pre-reduction before the dynamics CCA (K << effective sample size ~77)
SEED = 0
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"

T = np.array(rrxor(P1, P2))
NS = T.shape[1]
STATIONARY = np.array([2, 1, 1, 1, 1]) / 6.0


def ndist(b):
    return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])


def upd(b, x):
    nb = b @ T[x]
    return nb / nb.sum()


def belief_prob(w):
    b, p = STATIONARY, 1.0
    for x in w:
        d = ndist(b)
        p *= d[x]
        if d[x] < 1e-12:
            return None, 0.0
        b = upd(b, x)
    return b, p


def msp_index():
    beliefs = {tuple(np.round(STATIONARY, 5)): STATIONARY}
    fr = [STATIONARY]
    for _ in range(20):
        nx = []
        for b in fr:
            d = ndist(b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                b2 = upd(b, x); k = tuple(np.round(b2, 5))
                if k not in beliefs:
                    beliefs[k] = b2; nx.append(b2)
        fr = nx
    keys = list(beliefs.keys())
    return np.array([beliefs[k] for k in keys]), {k: i for i, k in enumerate(keys)}


def load_model(device):
    ck = torch.load(MODEL_DIR / "rrxor_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"]); cfg.device = device
    m = HookedTransformer(cfg); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m


def collect(model, device):
    """All token strings (len 1..MAX_LEN): concat last-pos activation, softmax, belief, P(w)."""
    nL = model.cfg.n_layers
    hooks = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(nL)] + ["ln_final.hook_normalized"]
    resid, soft, belief, pp = {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product([0, 1], repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, s in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in s)
                resid[w] = concat[j]; soft[w] = sm[j]
                b, p = belief_prob(w)
                belief[w] = b if b is not None else np.full(NS, np.nan)
                pp[w] = p
    return resid, soft, belief, pp


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B36, index = msp_index()
    model = load_model(device)
    resid, soft, belief, pp = collect(model, device)
    nvalid = sum(p > 1e-12 for p in pp.values())
    print(f"strings: {len(resid)}  valid prefixes: {nvalid}  concat dim: {len(next(iter(resid.values())))}")

    # supervised linear ceiling: how linearly is belief encoded in the concat at all?
    from sklearn.linear_model import LinearRegression
    vkeys = [w for w in resid if pp[w] > 1e-12 and np.isfinite(belief[w]).all()]
    ceil = LinearRegression().fit(np.stack([resid[w] for w in vkeys]),
                                  np.stack([belief[w] for w in vkeys])).score(
                                  np.stack([resid[w] for w in vkeys]),
                                  np.stack([belief[w] for w in vkeys]))
    print(f"supervised linear ceiling (concat->belief R^2) = {ceil:.4f}")

    from sklearn.decomposition import PCA
    rows, X, P, Yc, Wt = U.build_transitions(resid, soft, pp, MAX_LEN, 2)
    ess = (Wt.sum() ** 2) / (Wt ** 2).sum()
    Wu = (Wt > 1e-12).astype(float)   # uniform (reachability) weights: prob-weighting starves the estimator
    print(f"transition rows={len(rows)}  effective sample size: prob-weighted={ess:.0f}  uniform={int(Wu.sum())}")

    # Best-conditioned unsupervised subspace: PCA pre-reduction (K << ESS) then dynamics CCA,
    # uniform weights. This is the most generous setting; it still fails (see decode curve).
    Xb = np.stack([resid[w] for w in vkeys]); Yb = np.stack([belief[w] for w in vkeys])
    Bpca = PCA(n_components=PRE_PCA_K).fit(Xb).components_.T
    dirs, sv = U.predictive_cca(X @ Bpca, P, Yc @ Bpca, Wu)
    print(f"CCA canonical correlations (top5) = {np.round(sv[:5], 3)}  (all ->1 == degenerate, near-deterministic)")

    dd = [3, 4, 5, 6, 8, 10]
    decode = [LinearRegression().fit(Xb @ (Bpca @ dirs[:, :d]), Yb).score(Xb @ (Bpca @ dirs[:, :d]), Yb) for d in dd]
    print("belief-decode R^2 vs d (unsupervised subspace):", {d: round(r, 3) for d, r in zip(dd, decode)})
    best = int(np.argmax(decode)); d_best = dd[best]
    basis = Bpca @ dirs[:, :d_best]
    print(f"BEST unsupervised belief-decode = {decode[best]:.3f} at d={d_best}   (supervised ceiling = {ceil:.3f})")

    ops, _ = U.fit_operators(X @ basis, P, Yc @ basis, Wu)
    print("eigenvalue check  eig(A^x) vs eig(T^x):")
    for x in range(2):
        ev = np.linalg.eigvals(ops[x]); ev_a = np.sort(ev[np.argsort(-np.abs(ev))][:NS].real)
        print(f"  token {x}: eig(T^x)={np.round(np.sort(np.linalg.eigvals(T[x]).real),3)}  eig(A^x)={np.round(ev_a,3)}")

    # decoded-belief geometry from the unsupervised subspace (a blob, not the 36-state structure)
    reg = LinearRegression().fit(Xb @ basis, Yb)
    pcaB = PCA(n_components=4).fit(B36)
    tgt_xy = pcaB.transform(B36)[:, [0, 2]]
    xy = pcaB.transform(reg.predict(Xb @ basis))[:, [0, 2]]
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in vkeys])
    colors = np.array(B.distinct_colors(len(B36)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax0.plot(dd, decode, "ko-", lw=1.5, label="RRXOR unsupervised OOM subspace")
    ax0.axhline(ceil, color="r", ls="--", lw=1, label=f"supervised ceiling (concat) = {ceil:.2f}")
    ax0.axhline(0.998, color="green", ls=":", lw=1.2, label="Mess3 unsupervised OOM = 0.998")
    ax0.set_xlabel("subspace dimension d"); ax0.set_ylabel("belief-decode R$^2$")
    ax0.set_title(f"Unsupervised OOM cannot capture RRXOR belief\n(ESS={ess:.0f}; canonical corr$\\to$1 == near-deterministic)")
    ax0.set_ylim(0, 1.02); ax0.legend(fontsize=8, loc="center right"); ax0.grid(alpha=0.3)

    ax1.scatter(xy[:, 0], xy[:, 1], s=3, c=colors[idx], alpha=0.25, edgecolors="none")
    for s in range(len(B36)):
        m = idx == s
        if m.any():
            ax1.scatter(*xy[m].mean(0), s=70, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
    ax1.scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    ax1.set_title(f"Belief decoded from unsupervised subspace (d={d_best})\n"
                  f"decode R$^2$ = {decode[best]:.2f}  (rings = ground truth; structure NOT recovered)")
    ax1.set_xlabel("PCA 1"); ax1.set_ylabel("PCA 3"); ax1.set_aspect("equal")
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig13_rrxor_unsupervised_oom.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
