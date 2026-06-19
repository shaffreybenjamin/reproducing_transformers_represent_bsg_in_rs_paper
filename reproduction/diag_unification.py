"""Diagnostic for the two open questions in RESULTS_SUMMARY part B:

Q1. "Two different methods win in the two DGPs" -- is there really no single best?
    We run ONE apples-to-apples harness on BOTH Mess3 and RRXOR that varies only the
    *whitening exponents* (alpha on past, beta on future) of the same past x future SVD:
        M = Sxx^{-alpha}  Sxf  Sff^{-beta}
      (alpha,beta) = (0.5,0.5)  -> CCA            (double whiten)
      (alpha,beta) = (0.5,0.0)  -> reduced-rank   (input whiten only)
      (alpha,beta) = (0.0,0.0)  -> PLS / raw cross-cov (no whiten)
    Same past X=a(w), same future F=concat_x P(x|w) a(wx), same eval metric, same
    pre-PCA. If input-whiten >= the others in BOTH processes, there IS a single best
    method and CCA's extra output-whitening is just a fragile special case.

Q2. "Operator rollout collapses on RRXOR with either method." We dissect WHY:
    print eig(true T^x) vs eig(learned A^x), and measure the spread (mean pairwise
    distance) of the rolled-out states as a function of rollout depth. Collapse =
    contraction to a shared fixed point (spread -> 0), which is correct for Mess3's
    fractal attractor but wrong for RRXOR's finite synchronizing tree.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor, mess3
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import unsupervised_belief_oom as U


def whiten_power(C, alpha, ridge=1e-2):
    """Ridge-regularised C^{-alpha}.  alpha=0 -> identity (no whitening)."""
    C = C + ridge * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    val, vec = np.linalg.eigh(C)
    val = np.clip(val, 1e-12, None)
    return (vec * val ** (-alpha)) @ vec.T


def subspace_ab(X, P, Yc, Wt, alpha, beta, d, ridge=1e-2):
    """top-d past-side directions of  Sxx^{-alpha} Sxf Sff^{-beta}  + the singular spectrum."""
    N, m = X.shape
    F = (P[:, :, None] * Yc).reshape(N, -1)
    w = Wt / Wt.sum()
    Sxx = (X * w[:, None]).T @ X
    Sff = (F * w[:, None]).T @ F
    Sxf = (X * w[:, None]).T @ F
    Wx, Wf = whiten_power(Sxx, alpha, ridge), whiten_power(Sff, beta, ridge)
    M = Wx @ Sxf @ Wf
    Ud, s, _ = np.linalg.svd(M, full_matrices=False)
    return Wx @ Ud[:, :d], s


def collect(name):
    if name == "RRXOR":
        T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
        ck = "rrxor_transformer.pt"
    else:
        T, pi = np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0
        ck = "mess3_transformer.pt"
    model = F14._load(ck, "cpu")
    resid, soft, belief, _ = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    return T, resid, soft, belief, reach


def build(resid, soft, belief, reach, vocab, pre_pca):
    """rows with all reachable children; past X, softmax P, children Yc, weights Wt,
    plus the all-prefix evaluation arrays."""
    rows = [w for w in resid if reach[w] and len(w) < F14.MAX_LEN
            and all((w + (x,)) in resid and reach[w + (x,)]
                    for x in range(vocab) if soft[w][x] > F14.EPS)
            and all((w + (x,)) in resid for x in range(vocab))]
    Xr = np.stack([resid[w] for w in rows])
    # pre-PCA (shared across all whitenings, so it is not what differs)
    mu = Xr.mean(0)
    Bpca = np.linalg.svd(Xr - mu, full_matrices=False)[2][:pre_pca].T
    proj = {w: (resid[w] - mu) @ Bpca for w in resid}
    X = np.stack([proj[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    Yc = np.stack([[proj[w + (x,)] for x in range(vocab)] for w in rows])
    Wt = np.ones(len(rows))
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Aall = np.stack([proj[w] for w in allw])
    Yb = np.stack([belief[w] for w in allw])
    return rows, X, P, Yc, Wt, allw, Aall, Yb, proj


def score(Aall, Yb, B, groups=None):
    s = Aall @ B
    per = LinearRegression().fit(s, Yb).score(s, Yb)
    com = None
    if groups is not None:
        cm = np.array([s[g].mean(0) for g in groups.values()])
        bl = np.array([Yb[g[0]] for g in groups.values()])
        com = LinearRegression().fit(cm, bl).score(cm, bl)
    return per, com


def q1():
    print("=" * 78)
    print("Q1  apples-to-apples whitening sweep (same SVD, only alpha/beta differ)")
    print("=" * 78)
    for name in ["Mess3", "RRXOR"]:
        T, resid, soft, belief, reach = collect(name)
        vocab = T.shape[0]
        d = T.shape[1]                      # true belief dim (3 / 5)
        pre_pca = 30
        rows, X, P, Yc, Wt, allw, Aall, Yb, proj = build(
            resid, soft, belief, reach, vocab, pre_pca)
        groups = None
        if name == "RRXOR":
            _, index = F13.msp_index()
            idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
            groups = defaultdict(list)
            for i, s in enumerate(idx):
                groups[s].append(i)
        print(f"\n--- {name}: rows={len(rows)} eval_prefixes={len(allw)} "
              f"states={len(groups) if groups else 'cont'} pre_pca={pre_pca} d={d} ---")
        for tag, (a, b) in [("CCA   (in+out whiten)", (0.5, 0.5)),
                            ("RRR   (input whiten )", (0.5, 0.0)),
                            ("PLS   (no whiten    )", (0.0, 0.0))]:
            B, s = subspace_ab(X, P, Yc, Wt, a, b, d)
            per, com = score(Aall, Yb, B, groups)
            sp = np.round(s[:6] / s[0], 3)
            comstr = f" COM-geom={com:.3f}" if com is not None else ""
            print(f"  {tag}: per-prefix decode R^2={per:.3f}{comstr}   topSV(rel)={sp}")


def q1b():
    """Disentangle whitening from FUTURE CONSTRUCTION. Fix whitening = input-only
    (RRR) and sweep the observable-future horizon h:  phi(w) = [P(.|wp) : |p|<=h].
    Hypothesis: Mess3 saturates at h=1 (strongly observable); RRXOR keeps improving
    with h until the n_ctx coverage wall -- i.e. the real lever is horizon, not the
    CCA-vs-observability label."""
    import itertools
    print("\n" + "=" * 78)
    print("Q1b  same method (input-whitened RRR), sweep observable-future HORIZON h")
    print("=" * 78)
    for name in ["Mess3", "RRXOR"]:
        T, resid, soft, belief, reach = collect(name)
        vocab, d = T.shape[0], T.shape[1]
        groups = None
        allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
        Yb = np.stack([belief[w] for w in allw])
        if name == "RRXOR":
            _, index = F13.msp_index()
            idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
            groups = defaultdict(list)
            for i, s in enumerate(idx):
                groups[s].append(i)
        Aall = np.stack([resid[w] for w in allw])
        print(f"\n--- {name}: eval_prefixes={len(allw)} "
              f"states={len(groups) if groups else 'cont'} d={d} ---")
        for h in [1, 2, 3, 4]:
            paths = [p for L in range(h + 1) for p in itertools.product(range(vocab), repeat=L)]
            def phi(w):
                return np.concatenate([soft[w + p] for p in paths if all(
                    (w + p[:k + 1]) in soft for k in range(len(p)))] or [soft[w]])
            # fit rows: prefixes whose full horizon-h descendants exist
            fitw = [w for w in allw if len(w) <= F14.MAX_LEN - h
                    and all((w + p) in soft for p in paths)]
            if len(fitw) < 2 * d:
                print(f"  h={h}: too few fit prefixes ({len(fitw)})"); continue
            Af = np.stack([resid[w] for w in fitw])
            Phi = np.stack([phi(w) for w in fitw])
            Winv = whiten_power(Af.T @ Af / len(Af), 0.5, 1e-1)
            M = (Af @ Winv).T @ Phi / len(Af)
            Ud = np.linalg.svd(M, full_matrices=False)[0]
            B = Winv @ Ud[:, :d]
            per, com = score(Aall, Yb, B, groups)
            ncov = len({idx[allw.index(w)] for w in fitw}) if groups else "-"
            comstr = f" COM-geom={com:.3f}" if com is not None else ""
            print(f"  h={h}: fit={len(fitw):5d} states_cov={ncov}  "
                  f"per-prefix R^2={per:.3f}{comstr}")


def q2():
    print("\n" + "=" * 78)
    print("Q2  operator-rollout collapse anatomy (RRXOR vs Mess3)")
    print("=" * 78)
    for name in ["Mess3", "RRXOR"]:
        T, resid, soft, belief, reach = collect(name)
        vocab = T.shape[0]
        d = T.shape[1]
        print(f"\n--- {name} ---")
        # true belief-update operators (unnormalised) eigenvalues
        for x in range(vocab):
            ev = np.sort(np.abs(np.linalg.eigvals(T[x])))[::-1]
            print(f"  eig|T^{x}| = {np.round(ev, 3)}  (spectral radius {ev[0]:.3f})")

        # learn operators with the input-whitened (RRR) subspace = the best method
        rows, X, P, Yc, Wt, allw, Aall, Yb, proj = build(
            resid, soft, belief, reach, vocab, 30)
        B, _ = subspace_ab(X, P, Yc, Wt, 0.5, 0.0, d)
        A, Bc = X @ B, Yc @ B
        ops, _ = U.fit_operators(A, P, Bc, Wt)
        for x in range(vocab):
            ev = np.sort(np.abs(np.linalg.eigvals(ops[x])))[::-1]
            print(f"  eig|A^{x}| (learned) = {np.round(ev, 3)}")

        # roll out and measure spread vs depth (IN the d-dim subspace B)
        proj_d = {w: proj[w] @ B for w in proj if reach[w]}
        e, _ = U.recover_eval_functional(np.stack([proj_d[w] for w in proj_d]))
        # seed
        init = [proj_d[s] for s in proj_d if len(s) == 1]
        s0 = np.mean(init, 0); s0 = s0 / (s0 @ e)
        states, frontier = {(): s0}, [()]
        spreads = []
        for depth in range(1, F14.MAX_LEN + 1):
            nxt = []
            for w in frontier:
                sw = states[w]
                for x in range(vocab):
                    c = w + (x,)
                    if c not in proj_d:
                        continue
                    sc = sw @ ops[x]; den = sc @ e
                    states[c] = sc / den if abs(den) > 1e-12 else sc
                    nxt.append(c)
            frontier = nxt
            if len(frontier) >= 2:
                pts = np.stack([states[w] for w in frontier])
                spreads.append((depth, len(frontier), float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())))
        print("  rolled-state spread (mean dist to centroid) vs depth:")
        for dep, n, sp in spreads:
            print(f"    depth {dep:2d}: n={n:4d}  spread={sp:.4f}")
        # compare to the spread of the RAW activation states in the SAME d-dim subspace
        raw_pts = Aall @ B
        raw_spread = float(np.linalg.norm(raw_pts - raw_pts.mean(0), axis=1).mean())
        print(f"  RAW activation-state spread (reference): {raw_spread:.4f}")


if __name__ == "__main__":
    q1()
    q1b()
    q2()
