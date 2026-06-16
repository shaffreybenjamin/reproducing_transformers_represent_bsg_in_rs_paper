"""Reduced-rank / spectral OOM for RRXOR: input-whitened activations, UN-whitened
DIRECT multi-step observable future, belief subspace = top-d left singular vectors,
rank read from the spectrum.

This is the "spectral OOM adapted for noisy feature inputs": whiten the activation
side (removes within-state-spread nuisance variance) but NOT the future side (keeps
the magnitude spectrum -> rank readable, no CCA degeneracy); the future is built from
DIRECT descendant softmaxes (one forward pass each -> no operator compounding, no
chain-rule products).

Future phi(w) = [ P(.|w), P(.|w.p) for all paths p up to horizon h ]. Fit uses prefixes
with len <= n_ctx-h (so descendants fit the context); the subspace is then EVALUATED on
ALL reachable prefixes / all 35 belief states (apples-to-apples with fig17/fig18).
Compare COM-geometry vs predictive CCA 0.57 and observability OOM 0.68.
"""

import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9

FIG_DIR = Path(__file__).parent / "figures"
VOCAB = 2


def inv_sqrt(C, ridge):
    C = C + ridge * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    return (vecs / np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    T = np.array(rrxor(0.5, 0.5)); pi = np.array([2, 1, 1, 1, 1]) / 6.0
    model = F14._load("rrxor_transformer.pt", "cpu")
    resid, soft, belief, pp = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    A_all = np.stack([resid[w] for w in allw]); Yb = np.stack([belief[w] for w in allw])
    B36, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    groups = defaultdict(list)
    for i, s in enumerate(idx):
        groups[s].append(i)

    def com_geom(B):
        s = A_all @ B
        com = np.array([s[g].mean(0) for g in groups.values()])
        bel = np.array([Yb[g[0]] for g in groups.values()])
        return LinearRegression().fit(com, bel).score(com, bel)

    def per_prefix(B):
        s = A_all @ B
        return LinearRegression().fit(s, Yb).score(s, Yb)

    paths = {h: [p for L in range(h + 1) for p in itertools.product(range(VOCAB), repeat=L)]
             for h in range(0, 6)}

    def phi(w, h):
        return np.concatenate([soft[w + p] for p in paths[h]])

    print("horizon sweep (d=5, ridge=0.1):  [spectral OOM / reduced-rank regression]")
    best = None
    for h in [1, 2, 3, 4, 5]:
        fitw = [w for w in allw if len(w) <= F14.MAX_LEN - h]
        Af = np.stack([resid[w] for w in fitw])
        Phi = np.stack([phi(w, h) for w in fitw])
        Winv = inv_sqrt(Af.T @ Af / len(Af), 0.1)
        M = (Af @ Winv).T @ Phi / len(Af)
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        B = Winv @ U[:, :5]
        cg, pp_ = com_geom(B), per_prefix(B)
        ncov = len({idx[allw.index(w)] for w in fitw})
        print(f"  h={h}: fit_prefixes={len(fitw):4d} states_covered={ncov}/35  "
              f"COM-geom={cg:.3f}  per-prefix={pp_:.3f}  spectrum={np.round(S[:7]/S[0],2)}")
        if best is None or cg > best[0]:
            best = (cg, pp_, h, B)

    cg, pp_, h, B = best
    print(f"\nBEST: h={h}  COM-geometry R^2={cg:.3f}  per-prefix={pp_:.3f}  "
          f"(vs CCA 0.57, observability 0.68, supervised 1.00)")

    # geometry figure (GT | RRR-decoded), panel-B PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]
    reg = LinearRegression().fit(A_all @ B, Yb)
    xy = pca.transform(reg.predict(A_all @ B))[:, [0, 2]]
    colors = np.array(B9.distinct_colors(len(B36)))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5.6))
    a0.scatter(tgt[:, 0], tgt[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    a0.set_title("Ground truth  (PCA 1 / PCA 3)")
    for s, g in groups.items():
        a1.scatter(xy[g, 0], xy[g, 1], s=3, c=[colors[s]], alpha=0.25, edgecolors="none")
        a1.scatter(*xy[g].mean(0), s=80, c=[colors[s]], edgecolors="black", linewidths=0.5, zorder=3)
    a1.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    a1.set_title(f"Reduced-rank / spectral OOM (h={h})\nCOM-geometry R$^2$={cg:.2f}")
    for ax in (a0, a1):
        ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.08 * (hi - lo); setl(lo - pad, hi + pad)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3")
    out = FIG_DIR / "fig19_rrr_spectral_rrxor.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
