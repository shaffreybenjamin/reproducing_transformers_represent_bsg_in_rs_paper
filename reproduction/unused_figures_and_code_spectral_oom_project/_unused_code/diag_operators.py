"""Confirm the RRXOR operator pathology behind the rollout collapse (Q2)."""
import numpy as np
from simplexity.generative_processes.transition_matrices import rrxor, mess3

np.set_printoptions(precision=3, suppress=True)
for name, T, pi in [("RRXOR", np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0),
                    ("Mess3", np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0)]:
    print(f"\n================  {name}  ================")
    V, S = T.shape[0], T.shape[1]
    Tsum = T.sum(0)
    print("sum_x T^x row sums (stochastic?):", np.round(Tsum.sum(1), 3))
    for x in range(V):
        r = np.linalg.matrix_rank(T[x], tol=1e-9)
        ev = np.linalg.eigvals(T[x])
        nil = np.allclose(ev, 0)
        print(f"  T^{x}: rank={r}/{S}  nilpotent={nil}  |eig|={np.round(np.sort(np.abs(ev))[::-1],3)}")
    # is the stationary belief recoverable by rollout? check the projective update
    # denominator b T^x 1 along a few random walks -> does it hit ~0 (blow-up)?
    def upd(b, x):
        nb = b @ T[x]; s = nb.sum()
        return (nb / s if s > 1e-12 else nb), s
    rng = np.random.default_rng(0)
    mins = []
    for _ in range(2000):
        b = pi.copy()
        for _ in range(10):
            # sample an allowed token
            d = np.array([(b @ T[x]).sum() for x in range(V)])
            x = rng.choice(V, p=d / d.sum())
            b, s = upd(b, x)
            mins.append(s)
    mins = np.array(mins)
    print(f"  projective-update normalizer P(x|w): min={mins.min():.3e}  "
          f"frac<1e-3={(mins<1e-3).mean():.3f}  (near-zero => rollout renorm blows up)")
