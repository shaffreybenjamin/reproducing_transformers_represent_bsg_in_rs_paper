"""Controllability-side recovery using LONG synchronized prefixes (the half I missed).

RRXOR synchronises: a long prefix's last-k tokens determine the belief state. So unlike
observability (forced onto short prefixes <=10-H), the controllability factor can use prefixes up
to length 10. Two estimators of the belief subspace from past-determined activation structure:
  (A) controllability-CCA: CCA(one-hot last-k-gram, activation), top-5 activation canonical dirs.
  (B) gram-mean SVD (RRR): cluster long-prefix activations by last-k-gram, SVD the gram-mean
      activations (the span of the synchronized belief states). top-5.
Sampled from the true process through the EXISTING ctx-10 model. Compare to obs-only 0.66/0.91,
supervised 0.91/1.00.
"""
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
device, d, L, N_SEQ = "cpu", 5, 10, 80000


def sample_rrxor(T, pi, N, L, seed=0):
    rng = np.random.default_rng(seed); V, ns = T.shape[0], T.shape[1]
    states = rng.choice(ns, size=N, p=pi); seqs = np.empty((N, L), dtype=np.int64)
    for t in range(L):
        Px = np.stack([T[x][states].sum(1) for x in range(V)], 1); Px = np.clip(Px, 1e-12, None); Px /= Px.sum(1, keepdims=True)
        x = (rng.random(N)[:, None] < np.cumsum(Px, 1)).argmax(1); seqs[:, t] = x
        tr = T[x, states] / T[x, states].sum(1, keepdims=True)
        states = (rng.random(N)[:, None] < np.cumsum(tr, 1)).argmax(1)
    return seqs


def collect_resid(model, seqs, hooks, device, bs=8192):
    out = []
    for i in range(0, len(seqs), bs):
        inp = torch.from_numpy(seqs[i:i + bs]).to(device)
        with torch.no_grad():
            _, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
        out.append(np.concatenate([c[h].cpu().numpy() for h in hooks], -1))
    return np.concatenate(out, 0)


def gram_ids(toks, V):
    idx = np.zeros(len(toks), dtype=np.int64)
    for j in range(toks.shape[1]):
        idx = idx * V + toks[:, j]
    return idx


def main():
    model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    V = T.shape[0]
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    Dd = len(hooks) * model.cfg.d_model
    tic(f"sampling {N_SEQ} seqs + activations")
    seqs = sample_rrxor(T, pi, N_SEQ, L); acts = collect_resid(model, seqs, hooks, device)

    # decode pipeline (fig4 enumeration)
    B36, index = F10.msp_states()
    eseqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    eacts = F10.collect_activations(model, eseqs, device, hooks)
    Xf = np.concatenate([eacts[h] for h in hooks], -1).reshape(-1, Dd)
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
    tic(f"refs: supervised {sup.score(Xf,Yb,sample_weight=wgt):.3f}/{com_r2(Xf):.3f}   obs-only 0.661/0.911")
    print(f"\n{'method':<40}{'per-pos':>9}{'state-COM':>11}   detail")

    for k in [3, 4, 5, 6]:
        # gather (last-k-gram, activation) over positions with >=k past tokens (prefix len k..10)
        Xs, gids = [], []
        for t in range(k - 1, L):
            Xs.append(acts[:, t, :]); gids.append(gram_ids(seqs[:, t - k + 1:t + 1], V))
        X = np.vstack(Xs); g = np.concatenate(gids); ng = V ** k

        # (B) gram-mean SVD (RRR): span of synchronized belief-state activations
        sums = np.zeros((ng, Dd)); cnt = np.zeros(ng)
        np.add.at(sums, g, X); np.add.at(cnt, g, 1)
        m = cnt > 0; gm = sums[m] / cnt[m][:, None]; w = cnt[m] / cnt[m].sum()
        mean = (w[:, None] * gm).sum(0)
        Xc = (gm - mean) * np.sqrt(w)[:, None]
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        rpp, rc = decode(Vt[:d].T)
        sv = np.linalg.svd(Xc, full_matrices=False)[1]
        print(f"{f'gram-mean SVD  k={k}':<40}{rpp:>9.3f}{rc:>11.3f}   sv5/sv1={sv[4]/sv[0]:.3f}", flush=True)

        # (A) controllability-CCA: CCA(one-hot gram, activation)
        F = np.zeros((len(g), ng)); F[np.arange(len(g)), g] = 1.0
        Xc2 = X - X.mean(0); Fc = F - F.mean(0); n = len(X)
        Cxx = Xc2.T @ Xc2 / n + 1e-1 * np.eye(Dd)
        Cff = Fc.T @ Fc / n + 1e-3 * np.eye(ng)
        ex, Ux = np.linalg.eigh(Cxx); Wx = Ux @ np.diag(ex ** -0.5) @ Ux.T
        ef, Uf = np.linalg.eigh(Cff); Wf = Uf @ np.diag(np.clip(ef, 1e-12, None) ** -0.5) @ Uf.T
        U2, s2, _ = np.linalg.svd(Wx @ (Xc2.T @ Fc / n) @ Wf, full_matrices=False)
        rpp, rc = decode(Wx @ U2[:, :d])
        print(f"{f'controllability-CCA  k={k}':<40}{rpp:>9.3f}{rc:>11.3f}   cc[:6]={np.round(s2[:6],3)}", flush=True)
        tic(f"  done k={k}")


if __name__ == "__main__":
    main()
