"""CORRECTED CCA ceiling: canonical correlation between BELIEF b(w) (what the activation encodes)
and the future H-gram -- NOT state<->future (my earlier pre-check error). This equals the past<->future
Hankel and is bounded by observability + by how synchronized the available (short) prefixes are.

Computed exactly from T^x for prefixes up to length pmax and future H-gram. pmax = 10-H is the
realistic ctx-10 regime; larger pmax = the 'longer context' scenario. Shows whether the empirical
~0.45 is the true ceiling and whether longer prefixes (longer context) would raise it.
"""
import time
from itertools import product as iproduct
import numpy as np
from simplexity.generative_processes.transition_matrices import rrxor

t0 = time.time()
T = np.array(rrxor(0.5, 0.5)); V, ns = T.shape[0], T.shape[1]
pi = np.array([2, 1, 1, 1, 1]) / 6.0
ones = np.ones(ns)

def beliefs_upto(pmax):
    """forward-filter beliefs b(w) and P(w) for all reachable w, 1<=|w|<=pmax."""
    out = []  # (b, Pw)
    frontier = [(pi.copy(), 1.0)]
    cur_words = [()]
    items = {(): (pi.copy(), 1.0)}
    for L in range(1, pmax + 1):
        nxt = {}
        for w, (b, p) in list(items.items()):
            if len(w) != L - 1:
                continue
            for x in range(V):
                nb = b @ T[x]; s = nb.sum()
                if s < 1e-12:
                    continue
                nxt[w + (x,)] = (nb / s, p * s)
        items.update(nxt)
    return [(b, p) for w, (b, p) in items.items() if 1 <= len(w) <= pmax]

def futurevecs(H):
    grams = list(iproduct(range(V), repeat=H))
    M = np.empty((len(grams), ns))
    for i, v in enumerate(grams):
        vec = ones.copy()
        for x in reversed(v):
            vec = T[x] @ vec
        M[i] = vec                       # P(v|state) = (T^v 1)_s
    return M                             # (2^H, ns)

def psqrt(A, tol=1e-11):
    e, U = np.linalg.eigh(A)
    return U @ np.diag(np.where(e > tol, e ** -0.5, 0.0)) @ U.T

def belief_future_cancorr(pmax, H):
    data = beliefs_upto(pmax)
    Bel = np.array([b for b, _ in data])          # (nw, ns) beliefs
    Pw = np.array([p for _, p in data]); Pw = Pw / Pw.sum()
    FV = futurevecs(H)                            # (2^H, ns): P(gram|state)
    Pfut = Bel @ FV.T                             # (nw, 2^H): P(gram|w) = b(w).FV
    bbar = Pw @ Bel; fbar = Pw @ Pfut
    Bc = Bel - bbar;
    VarB = (Pw[:, None] * Bc).T @ Bc
    # future-onehot: within-prefix second moment E[oo^T|w]=diag(Pfut(w)); marginal Pmarg=fbar
    VarF = np.diag(fbar) - np.outer(fbar, fbar)
    CovBF = (Pw[:, None] * Bc).T @ (Pfut - fbar)  # (ns, 2^H)
    M = psqrt(VarB) @ CovBF @ psqrt(VarF)
    cc = np.clip(np.sort(np.linalg.svd(M, compute_uv=False))[::-1][:ns - 1], 0, 1)
    return cc

print("Canonical correlation between BELIEF and future H-gram (exact). weakest = 5th-mode CCA ceiling.\n")
for H in [4, 5, 6]:
    print(f"H={H}:")
    for pmax in [10 - H, 8, 12, 16, 20]:
        cc = belief_future_cancorr(pmax, H)
        tag = "  <- ctx-10 realistic" if pmax == 10 - H else ""
        print(f"   prefixes len<= {pmax:>2}: cc = {np.round(cc,3)}{tag}")
print(f"\n[{time.time()-t0:.1f}s]  (empirical sampling-CCA gave ~0.45 cc, decode 0.46-0.59 < obs-only 0.66)")
