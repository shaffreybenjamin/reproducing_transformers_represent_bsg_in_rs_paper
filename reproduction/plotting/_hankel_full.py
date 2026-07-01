"""Full two-factor estimator: forward observability O (cached order-3 ALS, L=6) + a CLEAN,
regression-based controllability factor = BACKWARD observability R~.

Backward operators B_x: regress child a(wx) -> parent a(w), refined by the SAME multistep
ALS (order 3) as forward (Sylvester-CG, no rescale; target is the actual ancestor activation).
Backward readout C~: regress a(w) -> onehot(last token). Then R~ = [C~, B_x C~, ...] to L=6.
Both factors are regression-based (project out non-belief variance); the backward factor uses
the PAST, which a synchronising process determines strongly -> independent vote for the weak
forward-observable mode.

Compares obs-only vs full Hankel (union of col(O), col(R~)); prints singular values; plots both.
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
EPS, RIDGE = F14.EPS, F14.RIDGE
SCR = Path("/tmp/claude-1000/-home-ben-mathematics-thesis-project-reproducing-transformers-represent-bsg-in-rs-paper/1bd34b5c-84ca-4b50-b39b-bbdaf648abc2/scratchpad")
FCACHE = SCR / "rrxor_oom_order3_ops.npz"          # forward C, Gs (cached)
BCACHE = SCR / "rrxor_backward_order3_ops.npz"     # backward B (cache after first fit)
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
FIG_DIR = Path(__file__).parent.parent / "figures"
device, d, L = "cpu", 5, 6


def fit_backward_als(resid, reach, pw, vocab, max_order=3, n_iter=30, lambda_decay=0.5,
                     ridge=RIDGE, verbose=True):
    """Backward operators B_x by multistep ALS: a(w x1..xk) B_{xk}..B_{x1} ~= a(w).
    Mirror of forward fit_operators_multistep_als (Sylvester-CG) with no rescale; design is
    the descendant activation, target the ancestor activation, P(w x..)-weighted."""
    D = next(iter(resid.values())).shape[0]
    anc = [w for w in resid if reach[w]]
    # one-step init: ridge child->parent
    B = []
    for x in range(vocab):
        Ac, Ap, ww = [], [], []
        for w in anc:
            wx = w + (x,)
            if wx in resid and reach[wx]:
                Ac.append(resid[wx]); Ap.append(resid[w]); ww.append(max(pw.get(wx, 0.0), 1e-12))
        B.append(Ridge(alpha=ridge, fit_intercept=False).fit(np.array(Ac), np.array(Ap),
                                                             sample_weight=np.array(ww)).coef_.T)
    # precompute operator-free moments per token-pattern
    pats = []
    for k in range(1, max_order + 1):
        ow = lambda_decay ** (k - 1)
        for s in iproduct(range(vocab), repeat=k):
            Ad, Ap, wt = [], [], []
            for w in anc:
                cur, ok = w, True
                for t in s:
                    nxt = cur + (t,)
                    if nxt not in resid or not reach[nxt]:
                        ok = False; break
                    cur = nxt
                if ok:
                    Ad.append(resid[cur]); Ap.append(resid[w]); wt.append(max(pw.get(cur, 0.0), 1e-12))
            if not Ad:
                continue
            Ad = np.array(Ad); Ap = np.array(Ap); wt = np.array(wt)[:, None]
            pats.append(dict(s=s, k=k, ow=ow, Sxx=Ad.T @ (wt * Ad), Sxy=Ad.T @ (wt * Ap)))

    def bw_left(s, p):                 # operators applied first (right of activation): B[s_{k-1}]..B[s_{p+1}]
        M = None
        for q in range(len(s) - 1, p, -1):
            M = B[s[q]] if M is None else M @ B[s[q]]
        return M

    def bw_right(s, p):                # trailing operators: B[s_{p-1}]..B[s_0]
        M = None
        for q in range(p - 1, -1, -1):
            M = B[s[q]] if M is None else M @ B[s[q]]
        return M

    def solve(j):
        PI = np.zeros((D, D)); RHS = np.zeros((D, D)); syl = []
        for pdj in pats:
            s = pdj["s"]
            for p in range(pdj["k"]):
                if s[p] != j:
                    continue
                Lm, Mr = bw_left(s, p), bw_right(s, p)
                Sxx = pdj["Sxx"] if Lm is None else Lm.T @ pdj["Sxx"] @ Lm
                Sxy = pdj["Sxy"] if Lm is None else Lm.T @ pdj["Sxy"]
                Sxx, Sxy = pdj["ow"] * Sxx, pdj["ow"] * Sxy
                if Mr is None:
                    PI += Sxx; RHS += Sxy
                else:
                    syl.append((Sxx, Mr @ Mr.T)); RHS += Sxy @ Mr.T
        if not PI.any() and not syl:
            return B[j]
        def apply(X):
            Y = PI @ X + ridge * X
            for Pt, Q in syl:
                Y = Y + Pt @ X @ Q
            return Y
        return F14._cg_matrix(apply, RHS, B[j].copy(), 300, 1e-9)

    for it in range(n_iter):
        delta = 0.0
        for j in range(vocab):
            Bn = solve(j); delta = max(delta, float(np.linalg.norm(Bn - B[j]))); B[j] = Bn
        if verbose and (it == 0 or (it + 1) % 5 == 0 or it == n_iter - 1):
            tic(f"  [bwd-ALS {it+1:2d}/{n_iter}] delta={delta:.2e}")
        if delta < 1e-6:
            break
    return B


def main():
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
    tic("loaded forward ops; collected activations")

    # forward observability O (L=6)
    cols, fr = [C], [C]
    for _ in range(L - 1):
        nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
    O = np.hstack(cols)
    U_O, sO, _ = np.linalg.svd(O, full_matrices=False)

    # backward operators (cache after first fit)
    if BCACHE.exists():
        B = [b for b in np.load(BCACHE)["B"]]
        tic("loaded cached backward operators")
    else:
        tic("fitting backward order-3 ALS (once)...")
        B = fit_backward_als(resid, reach, pw, V, max_order=3, n_iter=30, lambda_decay=0.5)
        np.savez(BCACHE, B=np.stack(B))
        tic("backward operators fit + cached")

    # backward readout C~ : a(w) -> onehot(last token); backward observability R~ (L=6)
    rc = [w for w in resid if reach[w]]
    Ac = np.array([resid[w] for w in rc]); last = np.array([w[-1] for w in rc])
    swc = np.array([max(pw.get(w, 0.0), 1e-12) for w in rc])
    Ctil = Ridge(alpha=RIDGE, fit_intercept=False).fit(Ac, np.eye(V)[last], sample_weight=swc).coef_.T
    cols, fr = [Ctil], [Ctil]
    for _ in range(L - 1):
        nxt = [Bx @ f for Bx in B for f in fr]; cols += nxt; fr = nxt
    Rt = np.hstack(cols)
    U_R, sR, _ = np.linalg.svd(Rt, full_matrices=False)

    # combined factors
    U_H, sH, _ = np.linalg.svd(np.hstack([O, Rt]), full_matrices=False)   # union (full Hankel)

    def nspec(s, n=10): return np.round(s[:n] / s[0], 4)
    def elbow(s):
        r = s[:min(11, len(s))] / s[0]
        return max([(i + 1, r[i] / r[i + 1]) for i in range(len(r) - 1)], key=lambda t: t[1])[0]
    print("\n--- singular spectra (sigma_i/sigma_1) ---")
    print("observability O   :", nspec(sO), " elbow d_hat =", elbow(sO))
    print("backward R~       :", nspec(sR), " elbow d_hat =", elbow(sR))
    print("full Hankel [O|R~]:", nspec(sH), " elbow d_hat =", elbow(sH))

    # supervised decode pipeline (fig4-identical)
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    Xf = np.concatenate([acts[h] for h in hooks], axis=-1).reshape(-1, len(hooks) * model.cfg.d_model)
    Y = beliefs.reshape(-1, B36.shape[1]); fidx = idx.reshape(-1)
    counts = np.bincount(fidx, minlength=len(B36)); wgt = 1.0 / np.clip(counts[fidx], 1, None); wgt /= wgt.mean()
    def com_r2(S):
        com = np.array([S[fidx == k].mean(0) for k in range(len(B36)) if (fidx == k).any()])
        bel = np.array([Y[fidx == k][0] for k in range(len(B36)) if (fidx == k).any()])
        return LinearRegression().fit(com, bel).score(com, bel)
    def decode(Bm):
        S = Xf @ Bm[:, :d]
        return LinearRegression().fit(S, Y, sample_weight=wgt), \
               LinearRegression().fit(S, Y, sample_weight=wgt).score(S, Y, sample_weight=wgt), com_r2(S)

    print(f"\n{'method':<26}{'per-pos R2':>12}{'state-COM R2':>14}")
    results = {}
    for name, Bm in [("observability only", U_O), ("backward only", U_R), ("full Hankel [O|R~]", U_H)]:
        reg, r2pp, r2com = decode(Bm)
        results[name] = (Bm, reg, r2pp, r2com)
        print(f"{name:<26}{r2pp:>12.3f}{r2com:>14.3f}")

    # plots: obs-only and full Hankel
    pca = PCA(n_components=4).fit(B36); tgt = pca.transform(B36)[:, [0, 2]]
    colors = np.array(F10.distinct_colors(len(B36)))
    def draw(ax, xy, title, scatter=True):
        if scatter:
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors[fidx], alpha=0.25, edgecolors="none")
            for k in range(len(B36)):
                m = fidx == k
                if m.any():
                    ax.scatter(*xy[m].mean(0), s=90, c=[colors[k]], edgecolors="black", linewidths=0.6, zorder=3)
            ax.scatter(tgt[:, 0], tgt[:, 1], s=150, facecolors="none", edgecolors="0.6", linewidths=1.0)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
        ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        for setl, v in ((ax.set_xlim, tgt[:, 0]), (ax.set_ylim, tgt[:, 1])):
            lo, hi = v.min(), v.max(); pad = 0.05 * (hi - lo); setl(lo - pad, hi + pad)
    sup = LinearRegression().fit(Xf, Y, sample_weight=wgt); r2_sup = sup.score(Xf, Y, sample_weight=wgt)
    for tag, key, fname in [("Observability only", "observability only", "rrxor_obsonly_L6.png"),
                            ("Full Hankel (obs + backward controllability)", "full Hankel [O|R~]", "rrxor_fullhankel_L6.png")]:
        Bm, reg, r2pp, r2com = results[key]
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
        draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
        draw(ax[1], pca.transform(sup.predict(Xf))[:, [0, 2]],
             f"Supervised: full residual stream\ndecode R$^2$={r2_sup:.2f}")
        draw(ax[2], pca.transform(reg.predict(Xf @ Bm[:, :d]))[:, [0, 2]],
             f"{tag} (5-D, L={L})\ndecode R$^2$={r2pp:.2f}  state-COM={r2com:.2f}")
        out = FIG_DIR / fname
        fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out}")
    tic("done")


if __name__ == "__main__":
    main()
