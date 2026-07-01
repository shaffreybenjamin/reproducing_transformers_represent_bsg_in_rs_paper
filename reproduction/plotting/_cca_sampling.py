"""Sampling-based CCA/RRR estimator for RRXOR belief recovery.

Escapes the enumeration sparsity that killed the earlier CCA: instead of one exact future
distribution per prefix (rank-deficient future covariance -> spurious cc=1.0 overfit), we SAMPLE
many stochastic futures per prefix (full-rank, accurately estimated future covariance), so the true
canonical structure emerges. Pre-check said the weak mode's canonical correlation -> 1.0 by H=6, far
above the spurious floor; prediction: decode approaches supervised ~0.91 (vs obs-only 0.66/0.91).

Pipeline: sample length-10 sequences from the TRUE process -> run through the EXISTING model ->
collect residual-stream activations + the observed future-token windows -> CCA(activation, future
window) -> top-5 canonical directions = belief subspace -> decode on the fig4 enumeration.
"""
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
device, d, L = "cpu", 5, 10
N_SEQ = 60000


def sample_rrxor(T, pi, N, L, seed=0):
    rng = np.random.default_rng(seed)
    V, ns = T.shape[0], T.shape[1]
    states = rng.choice(ns, size=N, p=pi)
    seqs = np.empty((N, L), dtype=np.int64)
    ar = np.arange(N)
    for t in range(L):
        Px = np.stack([T[x][states].sum(1) for x in range(V)], axis=1)       # (N,V) P(x|s)
        Px = np.clip(Px, 1e-12, None); Px /= Px.sum(1, keepdims=True)
        x = (rng.random(N)[:, None] < np.cumsum(Px, 1)).argmax(1)
        seqs[:, t] = x
        trans = T[x, states] / T[x, states].sum(1, keepdims=True)            # (N,ns) next-state
        states = (rng.random(N)[:, None] < np.cumsum(trans, 1)).argmax(1)
    return seqs


def collect_resid(model, seqs, hooks, device, bs=8192):
    out = []
    for i in range(0, len(seqs), bs):
        inp = torch.from_numpy(seqs[i:i + bs]).to(device)
        with torch.no_grad():
            _, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
        out.append(np.concatenate([c[h].cpu().numpy() for h in hooks], axis=-1))   # (b,L,D)
    return np.concatenate(out, 0)


def future_feature(fut, H, V):
    N = len(fut); feats = []
    for k in range(1, H + 1):
        idx = np.zeros(N, dtype=np.int64)
        for j in range(k):
            idx = idx * V + fut[:, j]
        oh = np.zeros((N, V ** k)); oh[np.arange(N), idx] = 1.0
        feats.append(oh)
    return np.hstack(feats)


def cca(X, F, lamx, lamf, d):
    Xc = X - X.mean(0); Fc = F - F.mean(0); N = len(X)
    Cxx = Xc.T @ Xc / N + lamx * np.eye(X.shape[1])
    Cff = Fc.T @ Fc / N + lamf * np.eye(F.shape[1])
    Cxf = Xc.T @ Fc / N
    ex, Ux = np.linalg.eigh(Cxx); Wx = Ux @ np.diag(ex ** -0.5) @ Ux.T
    ef, Uf = np.linalg.eigh(Cff); Wf = Uf @ np.diag(np.clip(ef, 1e-12, None) ** -0.5) @ Uf.T
    U, s, _ = np.linalg.svd(Wx @ Cxf @ Wf, full_matrices=False)
    return Wx @ U[:, :d], s


def main():
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    V = T.shape[0]
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    Dd = len(hooks) * model.cfg.d_model

    tic(f"sampling {N_SEQ} sequences + collecting activations")
    seqs = sample_rrxor(T, pi, N_SEQ, L)
    acts = collect_resid(model, seqs, hooks, device)                          # (N,L,D)
    tic(f"acts {acts.shape}; distinct length-4 prefixes sampled: "
        f"{len(set(map(tuple, seqs[:, :4].tolist())))}")

    # ---- decode pipeline (fig4-identical enumeration eval) ----
    B36, index = F10.msp_states()
    eseqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    eacts = F10.collect_activations(model, eseqs, device, hooks)
    Xf = np.concatenate([eacts[h] for h in hooks], axis=-1).reshape(-1, Dd)
    Yb = beliefs.reshape(-1, B36.shape[1]); fidx = idx.reshape(-1)
    counts = np.bincount(fidx, minlength=len(B36)); wgt = 1.0 / np.clip(counts[fidx], 1, None); wgt /= wgt.mean()
    def com_r2(S):
        com = np.array([S[fidx == k].mean(0) for k in range(len(B36)) if (fidx == k).any()])
        bel = np.array([Yb[fidx == k][0] for k in range(len(B36)) if (fidx == k).any()])
        return LinearRegression().fit(com, bel).score(com, bel)
    def decode(B):
        S = Xf @ B[:, :d]
        return LinearRegression().fit(S, Yb, sample_weight=wgt).score(S, Yb, sample_weight=wgt), com_r2(S)
    sup = LinearRegression().fit(Xf, Yb, sample_weight=wgt)
    print(f"\nreference: supervised {sup.score(Xf,Yb,sample_weight=wgt):.3f} (state-COM {com_r2(Xf):.3f}); "
          f"obs-only OOM = 0.661 / 0.911")

    print(f"\n{'method':<34}{'per-pos':>9}{'state-COM':>11}   canon-corr[:6]")
    for H in [4, 5, 6]:
        Xs, Fs = [], []
        for t in range(L - H):
            Xs.append(acts[:, t, :])
            Fs.append(future_feature(seqs[:, t + 1:t + 1 + H], H, V))
        X = np.vstack(Xs); F = np.vstack(Fs)
        B, s = cca(X, F, lamx=1e-1, lamf=1e-3, d=d)
        r2pp, r2c = decode(B)
        print(f"{f'sampling-CCA H={H} (n={len(X)})':<34}{r2pp:>9.3f}{r2c:>11.3f}   {np.round(s[:6],3)}", flush=True)
        tic(f"  done H={H}")


if __name__ == "__main__":
    main()
