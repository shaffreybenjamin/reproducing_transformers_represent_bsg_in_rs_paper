"""Noise-aware OOM: recover RRXOR belief geometry by treating within-state activation
spread as MEASUREMENT NOISE on the latent state, and removing it with the dynamics.

The geometry is provably, perfectly linearly present (35 state centres-of-mass ->
belief = R^2 1.000); the only job is to find the right ~5-D subspace. The plain
observable-anchored OOM (fig14) finds a wrong subspace (~0.48) because the
per-prefix within-state spread biases its regressions (errors-in-variables) and the
magnitude-ranked SVD mixes noise into the top directions.

Two improvements, both kept linear (so they carry to quantum/post-quantum):
  (1) EM rollout-denoising: alternate  fit operators -> roll each prefix's state out
      purely from the dynamics (merges same-state prefixes, cancels measurement noise)
      -> refit the activation->state subspace to the denoised states.
  (2) Noise-whitened subspace: re-estimate the subspace in the metric of the
      reconstruction-residual (within-state-noise) covariance, so high-noise directions
      are down-weighted and weakly-observable belief modes surface.

Scored against ground truth ONLY (per-prefix decode + the 35-COM geometry R^2); the
subspace itself is found from activations + model softmax + tree structure alone.

FINDING (negative): neither (1) nor (2) beats the baseline observability OOM.
  * Baseline observability OOM already recovers RRXOR at COM-geometry R^2 = 0.685
    (much better than the 0.48 PER-PREFIX number -- averaging to COMs cancels the
    within-state spread, as expected). Mess3 baseline = 0.997.
  * (1) EM rollout-denoising is UNSTABLE: rolling each state out through imperfect
    operators compounds error; the COM-geometry oscillates and ends BELOW baseline
    (RRXOR 0.685->0.43, and it even wrecks Mess3 0.997->0.49).
  * (2) Static noise-whitening by the operator-consistency residual covariance has
    NO effect (RRXOR stays 0.685).
Why: at the COM level the within-state spread is ALREADY averaged out, so the gap
from 0.685 to the 1.000 ceiling is NOT within-state noise -- it is the
weakly-observable belief modes. (1)/(2) target within-state noise, the wrong
constraint. Proof: the SUPERVISED subspace gives COM = 1.000; the observability
subspace gives 0.685; the difference is exactly the modes the dynamics can't surface
without compounding (operator-iteration noise or softmax-product error). The only
route that touched those modes was DIRECT multi-node descendant observables (~0.93
in-sample) -- but that is limited by the model's context window (long horizon -> few,
short prefixes -> overfit), so not a clean win either.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.transition_matrices import mess3, rrxor
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import fig09_rrxor_ground_truth as B9

D_RR = 5
D_M3 = 3
N_ITER = 10
RIDGE = 1e-2
FIG_DIR = Path(__file__).parent / "figures"


def stabilize(M):
    """Clip operator eigenvalue magnitudes to <=1 so the rollout can't blow up."""
    try:
        w, V = np.linalg.eig(M)
        w = w / np.maximum(np.abs(w), 1.0)
        return (V @ np.diag(w) @ np.linalg.pinv(V)).real
    except np.linalg.LinAlgError:
        return M


def fit_ops_eval(s, P, resid, rows, vocab, B):
    ops = {}
    for x in range(vocab):
        m = P[:, x] > F14.EPS
        sc = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]]) @ B
        ops[x] = np.linalg.lstsq(s[m], P[m, x][:, None] * sc, rcond=None)[0]
    e = np.linalg.lstsq(s, np.ones(len(s)), rcond=None)[0]
    return ops, e


def inv_sqrt(C, reg=1e-2):
    C = C + reg * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    return (vecs / np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def em_denoise(rows, A, P, resid, vocab, B0, score, n_iter=N_ITER, whiten=True):
    B = B0.copy()
    hist = [score(B)]
    for _ in range(n_iter):
        s = A @ B
        ops, e = fit_ops_eval(s, P, resid, rows, vocab, B)
        sops = {x: stabilize(ops[x]) for x in ops}
        proj = {w: s[i] for i, w in enumerate(rows)}
        sh = U.rollout_states(proj, sops, e, F14.MAX_LEN, vocab)           # (1) dynamics denoise
        s_hat = np.array([sh.get(w, s[i]) for i, w in enumerate(rows)])
        bad = ~np.isfinite(s_hat).all(1)                                   # rollout can divide by ~0
        s_hat[bad] = s[bad]
        Psi = Ridge(alpha=RIDGE, fit_intercept=False).fit(s_hat, A).coef_.T  # state->activation
        noise = A - s_hat @ Psi
        Winv = inv_sqrt(noise.T @ noise / len(A)) if whiten else np.eye(A.shape[1])  # (2) whiten
        Bw = Ridge(alpha=RIDGE, fit_intercept=False).fit(A @ Winv, s_hat).coef_.T
        B, _, _ = np.linalg.svd(Winv @ Bw, full_matrices=False)
        B = B[:, :B0.shape[1]]
        hist.append(score(B))
    return B, hist


def prepare(ckpt, T, pi):
    device = "cpu"
    m = F14._load(ckpt, device)
    resid, soft, belief, pp = F14._collect(m, T, pi, device)
    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    return resid, soft, belief, reach


def scorers(rows, A, belief):
    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    keys = [tuple(np.round(belief[w], 5)) for w in rows]
    groups = {}
    for i, k in zip(range(len(rows)), keys):
        if np.isfinite(belief[rows[i]]).all():
            groups.setdefault(k, []).append(i)

    def decode(B):
        s = A @ B
        return LinearRegression().fit(s[fin], Yb[fin]).score(s[fin], Yb[fin])

    def com_geom(B):
        s = A @ B
        com = np.array([s[ix].mean(0) for ix in groups.values()])
        bel = np.array([Yb[ix[0]] for ix in groups.values()])
        return LinearRegression().fit(com, bel).score(com, bel), com, bel

    return decode, com_geom, Yb, fin, groups


def run(name, ckpt, T, pi, d):
    resid, soft, belief, reach = prepare(ckpt, T, pi)
    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, T.shape[0])
    decode, com_geom, Yb, fin, groups = scorers(rows, A, belief)
    print(f"\n=== {name} (d={d}) ===  rows={len(rows)}")
    print(f"  baseline (observability SVD): per-prefix decode={decode(Uobs[:,:d]):.3f}  "
          f"COM-geometry R^2={com_geom(Uobs[:,:d])[0]:.3f}")
    B, hist = em_denoise(rows, A, P, resid, T.shape[0], Uobs[:, :d],
                         score=lambda B: com_geom(B)[0])
    print("  COM-geometry R^2 over EM iters:", [round(h, 3) for h in hist])
    print(f"  AFTER EM: per-prefix decode={decode(B):.3f}  COM-geometry R^2={com_geom(B)[0]:.3f}")
    return dict(name=name, rows=rows, A=A, belief=belief, B=B, base_B=Uobs[:, :d], hist=hist,
                base=com_geom(Uobs[:, :d])[0], decode_after=decode(B), com_after=com_geom(B)[0])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    R = run("RRXOR", "rrxor_transformer.pt", np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0, D_RR)
    M = run("Mess3", "mess3_transformer.pt", np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0, D_M3)

    # ---- plot: (left) RRXOR COM-geometry vs EM iter; (right) recovered RRXOR geometry ----
    B36, index = F13.msp_index()
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]
    rows, A, belief = R["rows"], R["A"], R["belief"]
    Bplot = R["base_B"]                                            # baseline subspace is best; EM degrades it
    fmask = np.array([np.isfinite(belief[w]).all() for w in rows])
    rowsf = [w for w, f in zip(rows, fmask) if f]
    s = A[fmask] @ Bplot
    Yf = np.stack([belief[w] for w in rowsf])
    com_reg = LinearRegression().fit(s, Yf)                        # affine to belief, then panel-B PCA
    xy = pca.transform(com_reg.predict(s))[:, [0, 2]]
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in rowsf])
    colors = np.array(B9.distinct_colors(len(B36)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.4))
    ax0.plot(range(len(R["hist"])), R["hist"], "ko-", label="RRXOR")
    ax0.plot(range(len(M["hist"])), M["hist"], "r^-", label="Mess3")
    ax0.axhline(1.0, color="0.7", ls=":", lw=1)
    ax0.set_xlabel("EM iteration"); ax0.set_ylabel("COM-geometry R$^2$")
    ax0.set_title("Improvements (1)+(2) do NOT help: EM degrades it\n(iter 0 = baseline observability SVD = best)")
    ax0.set_ylim(0, 1.05); ax0.legend(fontsize=9); ax0.grid(alpha=.3)

    for sidx in range(len(B36)):
        mm = idx == sidx
        if mm.any():
            ax1.scatter(xy[mm, 0], xy[mm, 1], s=3, c=[colors[sidx]], alpha=0.25, edgecolors="none")
            ax1.scatter(*xy[mm].mean(0), s=80, c=[colors[sidx]], edgecolors="black", linewidths=0.5, zorder=3)
    ax1.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
    ax1.set_title(f"Baseline unsupervised RRXOR geometry\nCOM-geometry R$^2$={R['base']:.2f} (ceiling 1.00)")
    ax1.set_xlabel("PCA 1"); ax1.set_ylabel("PCA 3"); ax1.set_aspect("equal")
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig15_oom_denoised.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
