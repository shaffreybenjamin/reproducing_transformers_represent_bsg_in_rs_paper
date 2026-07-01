"""Adjudicator for the two-factor question: does the forward-seeded controllability factor R
contain belief that observability misses, or not -- at ANY depth K?

  - belief-energy of R's top-K directions vs the TRUE (supervised) belief subspace B_true
    (not U_O, to avoid circularity), with the random-subspace baseline K/D for reference.
  - ORACLE decode: rank R's directions by belief-relevance (using labels) and take the best 5 --
    the BEST any controllability-based subspace could possibly do. If even the oracle is low,
    belief is not in R; if it's high, belief is in R but unfindable without labels.
"""
import time
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
FCACHE = SCR / "rrxor_oom_order3_ops.npz"
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
device, d, L = "cpu", 5, 6

model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
V = T.shape[0]
z = np.load(FCACHE); C, Gs = z["C"], [g for g in z["Gs"]]
resid, soft, _, _ = F14._collect(model, T, pi, device)
pw = F14.analytic_prefix_probs(resid, T, pi)
Ddim = C.shape[0]

cols, fr = [C], [C]
for _ in range(L - 1):
    nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
O = np.hstack(cols); U_O = np.linalg.svd(O, full_matrices=False)[0]

ws = np.array([pw[w] for w in resid]); As = np.stack([resid[w] for w in resid])
a0 = (ws[:, None] * As).sum(0) / ws.sum()
rowsR = [a0.copy()]; frontier = [a0.copy()]
for _ in range(L):
    nxt = [f @ Gx for Gx in Gs for f in frontier]; rowsR += nxt; frontier = nxt
R = np.stack(rowsR)
V_R = np.linalg.svd(R, full_matrices=False)[2].T          # (D, r) controllability row-space dirs
rankR = V_R.shape[1]
tic(f"O {O.shape}  R {R.shape}  rank(R)={rankR}")

# decode pipeline + TRUE belief subspace from supervised regression
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

# TRUE belief subspace = orthonormal row space of supervised coef (decodes ~0.91 by construction)
sup = LinearRegression().fit(Xf, Yb, sample_weight=wgt)
B_true = np.linalg.qr(sup.coef_.T)[0][:, :d]              # (D, 5) orthonormal
print(f"\nsanity: supervised B_true decodes -> per-pos {decode(B_true)[0]:.3f}  state-COM {decode(B_true)[1]:.3f}")
print(f"        obs-only U_O[:,:5] decodes -> per-pos {decode(U_O)[0]:.3f}  state-COM {decode(U_O)[1]:.3f}")

print("\nbelief-energy of R's top-K directions vs TRUE belief (random baseline = K/D):")
print(f"{'K':>5}{'energy_vs_Btrue':>17}{'random K/D':>13}{'energy_vs_U_O':>15}")
for K in [5, 10, 20, 40, 80, rankR]:
    e_true = float(np.sum((V_R[:, :K].T @ B_true) ** 2) / d)
    e_uo = float(np.sum((V_R[:, :K].T @ U_O[:, :d]) ** 2) / d)
    print(f"{K:>5}{e_true:>17.3f}{K / Ddim:>13.3f}{e_uo:>15.3f}")

# ORACLE: project B_true onto the FULL row space of R; how much is captured, and does it decode?
P_R = V_R @ V_R.T
captured = float(np.sum((P_R @ B_true) ** 2) / d)
B_oracle = np.linalg.qr(P_R @ B_true)[0][:, :d]
opp, oc = decode(B_oracle)
print(f"\nORACLE (best controllability can do, using labels to pick directions):")
print(f"  fraction of true belief inside row(R): {captured:.3f}   (random {rankR}-dim subspace would give {rankR/Ddim:.3f})")
print(f"  oracle decode from that belief-aligned part of R: per-pos {opp:.3f}  state-COM {oc:.3f}")
tic("done")
