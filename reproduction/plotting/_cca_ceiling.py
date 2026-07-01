"""CCA ceiling pre-check (exact, from known T^x; no model, no sampling).

The user's CCA/RRR estimator regresses, in activation space:
  observability: activation(=state) <-> FUTURE window   -> ceiling = canonical corrs(state, future)
  controllability: PAST window <-> activation(=state)    -> ceiling = canonical corrs(state, past)
                                                          = obs-CCA of the TIME-REVERSED process.
Canonical correlation (whitened) != observability magnitude ratio sigma5/sigma1 (=0.15). This computes
the actual canonical-correlation spectra so we know the true ceiling each side can reach with infinite
clean data -- and whether the strongly-controllable past side has headroom above the weak future side.
"""
import time
from itertools import product as iproduct
import numpy as np
from simplexity.generative_processes.transition_matrices import rrxor

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
T = np.array(rrxor(0.5, 0.5)); V, ns = T.shape[0], T.shape[1]
pi = np.array([2, 1, 1, 1, 1]) / 6.0

# time-reversed OOM operators: Ttil^x = diag(pi)^-1 (T^x)^T diag(pi)  (past <-> reversed future)
Trev = np.array([np.diag(1 / pi) @ T[x].T @ np.diag(pi) for x in range(V)])

def psqrt_pinv(Acov, tol=1e-11):
    e, U = np.linalg.eigh(Acov)
    inv = np.where(e > tol, e ** -0.5, 0.0)
    return U @ np.diag(inv) @ U.T

def state_window_cancorr(ops, pi, H):
    """Canonical correlations between hidden state S (5 cats, ~pi) and a length-<=H token window
    generated forward by `ops`. P(v|s) = (op^v 1)_s. Returns sorted canonical correlations."""
    ones = np.ones(ns)
    words = [()]
    for k in range(1, H + 1):
        words += list(iproduct(range(V), repeat=k))
    Pcond = np.empty((len(words), ns))
    for i, v in enumerate(words):
        vec = ones.copy()
        for x in reversed(v):
            vec = ops[x] @ vec                       # op^{v1}..op^{vk} 1
        Pcond[i] = vec                               # P(v|s) for each state s
    Pv = Pcond @ pi                                  # P(v) = sum_s pi_s P(v|s)
    Joint = pi[None, :] * Pcond                      # (nv, 5): P(s,v)
    Cov = Joint.T - np.outer(pi, Pv)                 # (5, nv): Cov(1_s, 1_v)
    VarS = np.diag(pi) - np.outer(pi, pi)
    VarF = np.diag(Pv) - np.outer(Pv, Pv)
    M = psqrt_pinv(VarS) @ Cov @ psqrt_pinv(VarF)
    cc = np.linalg.svd(M, compute_uv=False)
    return np.clip(np.sort(cc)[::-1][:ns - 1], 0, 1)  # up to ns-1=4 nonzero canonical corrs

print("Canonical correlations between hidden STATE and a length-H window (exact).")
print("  weakest of the 4 = the 5th-mode recoverability ceiling for a CCA/RRR method.\n")
print(f"{'H':>3} | {'OBSERVABILITY  state<->future (cc1..cc4)':<44} | {'CONTROLLABILITY  state<->past (cc1..cc4)':<44}")
for H in range(1, 11):
    co = state_window_cancorr(T, pi, H)
    cc = state_window_cancorr(Trev, pi, H)
    so = "  ".join(f"{x:.3f}" for x in co)
    sc = "  ".join(f"{x:.3f}" for x in cc)
    print(f"{H:>3} | {so:<44} | {sc:<44}")
tic("done")
print("\nReference: observability MAGNITUDE ratio sigma5/sigma1 = 0.15 (saturated). Estimator noise floor ~0.34.")
print("Read: the SMALLEST column on each side is the weak-mode ceiling for that CCA factor.")
