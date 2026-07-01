"""sigma_5-vs-horizon pre-check for RRXOR (capstone of the unsupervised-recovery investigation).

TRUE observability spectrum from the known T^x (exact arithmetic, process property, no model):
columns of O_H = {T^w 1 : |w| <= H} (5-vectors); sigma_i(O_H) via the Gramian
G_H = sum_{|w|<=H} (T^w 1)(T^w 1)^T accumulated by levels. sigma_5/sigma_1 = inherent strength
of the weak 5th belief mode at horizon H (unweighted, and P(w)-weighted to match the estimator).

Two numbers (per the review):
 (1) exact: smallest H with sigma_5 > 0  -> minimal horizon for ANY signal, and the SATURATION value.
 (2) noise floor: the estimator's empirical floor (activation observability sigma_6/sigma_1 at ctx10);
     does true sigma_5 ever clear it? If sigma_5 saturates BELOW the floor -> quantified dead end at this
     model quality; longer context can only help by LOWERING the floor (cleaner belief encoding).
"""
import time
from pathlib import Path
import numpy as np
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
T = np.array(rrxor(0.5, 0.5)); V, ns = T.shape[0], T.shape[1]
pi = np.array([2, 1, 1, 1, 1]) / 6.0
ones = np.ones(ns)

# ---- TRUE observability sigma_i(H) via level-accumulated Gramian (unweighted + P(w)-weighted) ----
def specs(Hmax):
    cur = {(): ones.copy()}
    Gu = np.outer(ones, ones)                       # H=0 term
    Gw = float(pi @ ones) * np.outer(ones, ones)
    out = []
    for H in range(1, Hmax + 1):
        nxt = {}; Su = np.zeros((ns, ns)); Sw = np.zeros((ns, ns))
        for w, v in cur.items():
            for x in range(V):
                vx = T[x] @ v
                nxt[(x,) + w] = vx
                Su += np.outer(vx, vx)
                Sw += float(pi @ vx) * np.outer(vx, vx)
        Gu += Su; Gw += Sw; cur = nxt
        eu = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(Gu))[::-1], 0, None))
        ew = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(Gw))[::-1], 0, None))
        out.append((H, eu[4] / eu[0], eu[3] / eu[0], ew[4] / ew[0]))
    return out

print("TRUE RRXOR observability (exact arithmetic):")
print(f"{'H':>3}{'s5/s1 (unwt)':>14}{'s4/s1 (unwt)':>14}{'s5/s1 (Pw-wt)':>15}")
rows = specs(14)
for H, s5u, s4u, s5w in rows:
    print(f"{H:>3}{s5u:>14.4f}{s4u:>14.4f}{s5w:>15.5f}")
sat_u = rows[-1][1]; sat_w = rows[-1][3]
first_nz = next(H for H, s5u, _, _ in rows if s5u > 1e-9)
tic("true spectrum done")

# ---- estimator empirical noise floor: activation observability spectrum at ctx10, L=6 ----
SCR = Path("/tmp/claude-1000/-home-ben-mathematics-thesis-project-reproducing-transformers-represent-bsg-in-rs-paper/1bd34b5c-84ca-4b50-b39b-bbdaf648abc2/scratchpad")
z = np.load(SCR / "rrxor_oom_order3_ops.npz"); C, Gs = z["C"], [g for g in z["Gs"]]
cols, fr = [C], [C]
for _ in range(5):
    nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
sv = np.linalg.svd(np.hstack(cols), compute_uv=False); sv = sv / sv[0]
floor = sv[5]                                       # sigma_6/sigma_1 = top "should-be-zero" noise dir
print(f"\nActivation observability spectrum (ctx10, L=6): {np.round(sv[:8],3)}")
print(f"  noise floor (sigma_6/sigma_1) = {floor:.3f};  activation sigma_5/sigma_1 = {sv[4]:.3f}")

# supervised within-state non-belief variance (the model's intrinsic encoding floor)
print("\n================  VERDICT  ================")
print(f"(1) EXACT: true sigma_5 first nonzero at H = {first_nz}; SATURATES at "
      f"~{sat_u:.3f} (unwt) / ~{sat_w:.4f} (P(w)-wt) by H~9-12.")
print(f"(2) FLOOR: estimator noise floor ~{floor:.3f}.  true sigma_5 (signal) "
      f"{'<' if sat_u < floor else '>'} floor  ->  "
      f"{'SUB-FLOOR: longer horizon cannot surface it; signal is saturated below the floor.' if sat_u < floor else 'clears floor at the saturating H.'}")
tic("done")
