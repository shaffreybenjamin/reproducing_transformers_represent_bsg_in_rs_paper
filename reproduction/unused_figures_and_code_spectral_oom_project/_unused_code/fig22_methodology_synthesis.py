"""Synthesis figure for the two open questions about the unsupervised methodology.

Panel A (Q1): the apparent "two different winners" is a confound. With future-
  construction held fixed, the WHITENING choice (CCA / RRR / PLS) is a wash on BOTH
  processes -> the discriminator is not the algorithm.
Panel B (Q1): the real lever is the observable-future HORIZON. Mess3 (strongly
  observable) saturates at h=1; RRXOR (weakly observable) needs h~3 to climb to ~0.65.
  So there is ONE method -- input-whitened reduced-rank regression onto a horizon-h
  future, h chosen by held-out fit -- that is best (or tied-best) on both.
Panel C (Q2): operator-rollout state spread vs depth. Mess3 converges onto its fractal
  attractor (stable, healthy spread); RRXOR destabilises (orders-of-magnitude, erratic).
Panel D (Q2): why -- the true belief-update operators. Mess3's are diagonalisable strict
  contractions (attractor = belief fractal); RRXOR's are nilpotent/defective (T^1
  nilpotent, T^0 a triple eigenvalue), so the belief states are transients of a
  defective semigroup, not an attractor -- unlearnable-by-rollout for ANY estimator.
"""
from collections import defaultdict
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor, mess3
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import reproduction.estimators.unsupervised_belief_oom as U
from diag_unification import collect, build, subspace_ab, score, whiten_power

FIG_DIR = Path(__file__).parent / "figures"


def whitening_bars(name, T):
    resid, soft, belief, reach = collect(name)[1:]
    vocab, d = T.shape[0], T.shape[1]
    rows, X, P, Yc, Wt, allw, Aall, Yb, proj = build(resid, soft, belief, reach, vocab, 30)
    groups = None
    if name == "RRXOR":
        _, index = F13.msp_index()
        idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
        groups = defaultdict(list)
        for i, s in enumerate(idx):
            groups[s].append(i)
    out = {}
    for tag, (a, b) in [("CCA", (0.5, 0.5)), ("RRR", (0.5, 0.0)), ("PLS", (0.0, 0.0))]:
        B, _ = subspace_ab(X, P, Yc, Wt, a, b, d)
        per, com = score(Aall, Yb, B, groups)
        out[tag] = com if com is not None else per
    return out


def horizon_curve(name, T):
    resid, soft, belief, reach = collect(name)[1:]
    vocab, d = T.shape[0], T.shape[1]
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    Aall = np.stack([resid[w] for w in allw])
    groups = None
    if name == "RRXOR":
        _, index = F13.msp_index()
        idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
        groups = defaultdict(list)
        for i, s in enumerate(idx):
            groups[s].append(i)
    curve = []
    for h in [1, 2, 3, 4]:
        paths = [p for L in range(h + 1) for p in itertools.product(range(vocab), repeat=L)]
        fitw = [w for w in allw if len(w) <= F14.MAX_LEN - h and all((w + p) in soft for p in paths)]
        if len(fitw) < 2 * d:
            continue
        Af = np.stack([resid[w] for w in fitw])
        Phi = np.stack([np.concatenate([soft[w + p] for p in paths]) for w in fitw])
        Winv = whiten_power(Af.T @ Af / len(Af), 0.5, 1e-1)
        M = (Af @ Winv).T @ Phi / len(Af)
        Ud = np.linalg.svd(M, full_matrices=False)[0]
        B = Winv @ Ud[:, :d]
        per, com = score(Aall, Yb, B, groups)
        curve.append((h, com if com is not None else per))
    return curve


def rollout_spread(name, T):
    resid, soft, belief, reach = collect(name)[1:]
    vocab, d = T.shape[0], T.shape[1]
    rows, X, P, Yc, Wt, allw, Aall, Yb, proj = build(resid, soft, belief, reach, vocab, 30)
    B, _ = subspace_ab(X, P, Yc, Wt, 0.5, 0.0, d)
    ops, _ = U.fit_operators(X @ B, P, Yc @ B, Wt)
    projB = {w: proj[w] @ B for w in proj if reach[w]}
    e, _ = U.recover_eval_functional(np.stack([projB[w] for w in projB]))
    s0 = np.mean([projB[s] for s in projB if len(s) == 1], 0); s0 = s0 / (s0 @ e)
    states, frontier, spread = {(): s0}, [()], []
    for depth in range(1, F14.MAX_LEN + 1):
        nxt = []
        for w in frontier:
            for x in range(vocab):
                c = w + (x,)
                if c not in projB:
                    continue
                sc = states[w] @ ops[x]; den = sc @ e
                states[c] = sc / den if abs(den) > 1e-12 else sc
                nxt.append(c)
        frontier = nxt
        if len(frontier) >= 2:
            pts = np.stack([states[w] for w in frontier])
            spread.append((depth, float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())))
    raw = float(np.linalg.norm((Aall @ B) - (Aall @ B).mean(0), axis=1).mean())
    return spread, raw


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Tr, Tm = np.array(rrxor(0.5, 0.5)), np.array(mess3(x=0.05, a=0.85))

    barM, barR = whitening_bars("Mess3", Tm), whitening_bars("RRXOR", Tr)
    curveM, curveR = horizon_curve("Mess3", Tm), horizon_curve("RRXOR", Tr)
    sprM, rawM = rollout_spread("Mess3", Tm)
    sprR, rawR = rollout_spread("RRXOR", Tr)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # A: whitening bars
    a = ax[0, 0]
    keys = ["CCA", "RRR", "PLS"]
    xb = np.arange(3)
    a.bar(xb - 0.18, [barM[k] for k in keys], 0.36, label="Mess3 (decode R$^2$)", color="#c0392b")
    a.bar(xb + 0.18, [barR[k] for k in keys], 0.36, label="RRXOR (COM-geom R$^2$)", color="#2c3e50")
    a.set_xticks(xb); a.set_xticklabels(["CCA\n(in+out)", "RRR\n(input)", "PLS\n(none)"])
    a.set_ylim(0, 1.05); a.set_ylabel("recovery R$^2$")
    a.set_title("A. Whitening choice is a WASH\n(future + pre-PCA fixed -> methods tie on each process)")
    a.legend(fontsize=8); a.grid(axis="y", alpha=.3)

    # B: horizon curve
    b = ax[0, 1]
    b.plot([h for h, _ in curveM], [r for _, r in curveM], "o-", color="#c0392b",
           label="Mess3 (decode R$^2$) - saturates at h=1")
    b.plot([h for h, _ in curveR], [r for _, r in curveR], "s-", color="#2c3e50",
           label="RRXOR (COM-geom R$^2$) - climbs with h")
    b.set_xlabel("observable-future horizon h"); b.set_ylabel("recovery R$^2$")
    b.set_xticks([1, 2, 3, 4]); b.set_ylim(0, 1.05)
    b.set_title("B. The real lever is HORIZON (= observability)\none input-whitened method, h by held-out fit")
    b.legend(fontsize=8); b.grid(alpha=.3)

    # C: rollout spread vs depth
    c = ax[1, 0]
    c.semilogy([d for d, _ in sprM], [s for _, s in sprM], "o-", color="#c0392b", label="Mess3 rollout")
    c.semilogy([d for d, _ in sprR], [s for _, s in sprR], "s-", color="#2c3e50", label="RRXOR rollout")
    c.axhline(rawM, color="#c0392b", ls="--", lw=1, alpha=.6, label="Mess3 raw-state spread")
    c.axhline(rawR, color="#2c3e50", ls="--", lw=1, alpha=.6, label="RRXOR raw-state spread")
    c.set_xlabel("rollout depth (operator compositions)"); c.set_ylabel("state spread (log)")
    c.set_title("C. Operator rollout: Mess3 -> attractor (stable)\nRRXOR -> destabilises (erratic, orders of magnitude)")
    c.legend(fontsize=8); c.grid(alpha=.3)

    # D: operator eigenvalues in complex plane
    d = ax[1, 1]
    th = np.linspace(0, 2 * np.pi, 200)
    d.plot(np.cos(th), np.sin(th), color="0.7", lw=1)
    for x in range(Tm.shape[0]):
        ev = np.linalg.eigvals(Tm[x])
        d.scatter(ev.real, ev.imag, c="#c0392b", s=70, marker="o",
                  label="Mess3 T$^x$ (contractions)" if x == 0 else None, zorder=3)
    for x in range(Tr.shape[0]):
        ev = np.linalg.eigvals(Tr[x])
        d.scatter(ev.real, ev.imag, c="#2c3e50", s=70, marker="x",
                  label="RRXOR T$^x$ (nilpotent/defective)" if x == 0 else None, zorder=3)
    d.axhline(0, color="0.85", lw=.8); d.axvline(0, color="0.85", lw=.8)
    d.set_aspect("equal"); d.set_xlabel("Re"); d.set_ylabel("Im")
    d.set_title("D. WHY: true belief operators\nMess3 diagonalisable contractions; RRXOR nilpotent (T$^1$) / defective (T$^0$)")
    d.legend(fontsize=8, loc="upper left"); d.grid(alpha=.2)

    fig.tight_layout()
    out = FIG_DIR / "fig22_methodology_synthesis.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")
    print("bars Mess3:", {k: round(v, 3) for k, v in barM.items()})
    print("bars RRXOR:", {k: round(v, 3) for k, v in barR.items()})
    print("horizon Mess3:", [(h, round(r, 3)) for h, r in curveM])
    print("horizon RRXOR:", [(h, round(r, 3)) for h, r in curveR])


if __name__ == "__main__":
    main()
