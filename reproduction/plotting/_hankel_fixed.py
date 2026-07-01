"""Corrected two-factor test addressing the review:
 (fix 1) build the ACTUAL forward-seeded controllability factor R = [a0, a0 Gx, a0 Gx Gy, ...]
         (writeup's dual, reusing the trusted forward operators -- no backward fit, no C~).
 (fix 2) SCALE-MATCH the union before SVD: combine ORTHONORMAL bases of col(O) and row(R), so
         neither block dominates by column-norm (the bug that made [O|R~] collapse to backward-only).
Reports obs-only / controllability-only / scale-matched unions, the belief-energy captured by R's
spectrum, and (control) the backward factor under a scale-matched union to confirm fix 2."""
import time
from itertools import product as iproduct
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
EPS = F14.EPS
SCR = Path("/tmp/claude-1000/-home-ben-mathematics-thesis-project-reproducing-transformers-represent-bsg-in-rs-paper/1bd34b5c-84ca-4b50-b39b-bbdaf648abc2/scratchpad")
FCACHE = SCR / "rrxor_oom_order3_ops.npz"; BCACHE = SCR / "rrxor_backward_order3_ops.npz"
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
device, d, L = "cpu", 5, 6

model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
V = T.shape[0]
z = np.load(FCACHE); C, Gs = z["C"], [g for g in z["Gs"]]
resid, soft, _, _ = F14._collect(model, T, pi, device)
reach = {}
for w in resid:
    ok, pre = True, ()
    for t in w:
        if pre in soft and soft[pre][t] <= EPS:
            ok = False; break
        pre = pre + (t,)
    reach[w] = ok
pw = F14.analytic_prefix_probs(resid, T, pi)

# forward observability O (col space = belief)
cols, fr = [C], [C]
for _ in range(L - 1):
    nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
O = np.hstack(cols); U_O, sO, _ = np.linalg.svd(O, full_matrices=False)

# a0 = P(w)-weighted mean activation (= stationary root b0*Psi by tower property)
ws = np.array([pw[w] for w in resid]); As = np.stack([resid[w] for w in resid])
a0 = (ws[:, None] * As).sum(0) / ws.sum()

# forward-seeded controllability R = [a0; a0 Gx; a0 Gx Gy; ...] over ALL words to depth L (rows)
rowsR = [a0.copy()]; frontier = [a0.copy()]
for _ in range(L):
    nxt = [f @ Gx for Gx in Gs for f in frontier]; rowsR += nxt; frontier = nxt
R = np.stack(rowsR)                                   # (nR, D); row space = reachable belief subspace
U_Rr, sR, Vt_R = np.linalg.svd(R, full_matrices=False)
V_R = Vt_R.T                                          # (D, r); columns = controllability belief dirs
tic(f"O {O.shape}  R {R.shape}")

# backward factor (cached) for the scale-matched control test
B = [b for b in np.load(BCACHE)["B"]]
rc = [w for w in resid if reach[w]]
from sklearn.linear_model import Ridge
Ac = np.array([resid[w] for w in rc]); last = np.array([w[-1] for w in rc])
swc = np.array([max(pw.get(w, 0.0), 1e-12) for w in rc])
Ctil = Ridge(alpha=F14.RIDGE, fit_intercept=False).fit(Ac, np.eye(V)[last], sample_weight=swc).coef_.T
cols, fr = [Ctil], [Ctil]
for _ in range(L - 1):
    nxt = [Bx @ f for Bx in B for f in fr]; cols += nxt; fr = nxt
U_Rb = np.linalg.svd(np.hstack(cols), full_matrices=False)[0]

# decode pipeline
B36, index = F10.msp_states()
seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
acts = F10.collect_activations(model, seqs, device, hooks)
Xf = np.concatenate([acts[h] for h in hooks], axis=-1).reshape(-1, len(hooks) * model.cfg.d_model)
Yb = beliefs.reshape(-1, B36.shape[1]); fidx = idx.reshape(-1)
counts = np.bincount(fidx, minlength=len(B36)); wgt = 1.0 / np.clip(counts[fidx], 1, None); wgt /= wgt.mean()
def com_r2(S):
    com = np.array([S[fidx == k].mean(0) for k in range(len(B36)) if (fidx == k).any()])
    bel = np.array([Yb[fidx == k][0] for k in range(len(B36)) if (fidx == k).any()])
    return LinearRegression().fit(com, bel).score(com, bel)
def decode(Bm):
    S = Xf @ Bm[:, :d]
    return LinearRegression().fit(S, Yb, sample_weight=wgt).score(S, Yb, sample_weight=wgt), com_r2(S)

# belief-energy of forward-controllability spectrum: how much of belief (U_O[:,:5]) is in R's top-K?
print("\nR (forward controllability) sing spectrum:", np.round(sR[:8] / sR[0], 3))
print("belief-subspace energy captured by R's top-K right-sing-vecs:",
      {K: round(float(np.sum((V_R[:, :K].T @ U_O[:, :5]) ** 2) / 5.0), 3) for K in [5, 10, 20, 40]})

def ortho_union(Ua, Ub, K):
    M = np.hstack([Ua[:, :K], Ub[:, :K]])             # scale-matched: both orthonormal
    return np.linalg.svd(M, full_matrices=False)[0]

print(f"\n{'method':<40}{'per-pos':>9}{'state-COM':>11}")
print(f"{'observability only U_O[:,:5]':<40}{decode(U_O)[0]:>9.3f}{decode(U_O)[1]:>11.3f}")
print(f"{'controllability only (fwd-seeded)':<40}{decode(V_R)[0]:>9.3f}{decode(V_R)[1]:>11.3f}")
for K in [5, 8, 12]:
    Uh = ortho_union(U_O, V_R, K)
    print(f"{f'scale-matched union(O, R_fwd) K={K}':<40}{decode(Uh)[0]:>9.3f}{decode(Uh)[1]:>11.3f}")
# column-normalized hstack union (the review's fix-1 literal form)
def colnorm(Mx): return Mx / (np.linalg.norm(Mx, axis=0, keepdims=True) + 1e-12)
Uh_cn = np.linalg.svd(np.hstack([colnorm(O), colnorm(R.T)]), full_matrices=False)[0]
print(f"{'col-normalized hstack union(O,R_fwd)':<40}{decode(Uh_cn)[0]:>9.3f}{decode(Uh_cn)[1]:>11.3f}")
# control: backward factor under scale-matched union (does fix-2 rescue it from 0.139?)
for K in [5, 8]:
    Uh = ortho_union(U_O, U_Rb, K)
    print(f"{f'scale-matched union(O, R_bwd) K={K}':<40}{decode(Uh)[0]:>9.3f}{decode(Uh)[1]:>11.3f}")
tic("done")
