"""Operator-consistency refit (writeup Appendix 'Operator Rollouts', the B_op ALS).

Start from the observability subspace and rotate it, within the top-K observability directions,
toward the d-dim subspace the rescaled-transition operators map into themselves. Concretely:
find R (K x d, Stiefel) and small operators M_x (d x d) minimizing

   J = sum_{w,x} P(w) || s(w) R M_x  -  P(x|w) s(wx) R ||^2 ,   s(w) = a(w) U_K

by ALS (closed-form M_x; Riemannian-gradient/QR step on R). B_op = U_K R. Init R = [I_d;0] so
B_op starts EXACTLY at U_O[:,:d] (obs-only) and can only re-select operator-consistent directions
from ranks d+1..K -- it can promote a dynamics-consistent weak mode over operator-noise directions.
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0 = time.time()
def tic(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
EPS = F14.EPS
FCACHE = Path("/tmp/claude-1000/-home-ben-mathematics-thesis-project-reproducing-transformers-represent-bsg-in-rs-paper/1bd34b5c-84ca-4b50-b39b-bbdaf648abc2/scratchpad/rrxor_oom_order3_ops.npz")
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
FIG_DIR = Path(__file__).parent.parent / "figures"
device, d, L = "cpu", 5, 6


def bop_refit(data, U_K, d, n_outer=60, inner=8, verbose=True):
    """data: list per token x of (Xp, Xc, c, pw) already projected to U_K (N_x x K).
    Returns R (K x d), M_x list, J trace."""
    K = U_K.shape[1]
    R = np.eye(K)[:, :d].copy()

    def fit_M(R):
        Ms = []
        for (Xp, Xc, c, pw) in data:
            sp, sc = Xp @ R, Xc @ R                      # (N,d)
            W = pw[:, None]
            A = sp.T @ (W * sp)                          # (d,d)
            Bm = sp.T @ (W * (c[:, None] * sc))          # (d,d)
            Ms.append(np.linalg.solve(A + 1e-9 * np.eye(d), Bm))
        return Ms

    def J_and_grad(R, Ms):
        J = 0.0; G = np.zeros_like(R)
        for (Xp, Xc, c, pw), M in zip(data, Ms):
            sp, sc = Xp @ R, Xc @ R
            res = sp @ M - c[:, None] * sc               # (N,d)
            W = pw[:, None]
            J += float(np.sum(W * res * res))
            G += Xp.T @ (W * res) @ M.T - Xc.T @ (W * (c[:, None] * res))
        return J, G

    Ms = fit_M(R)
    J0, _ = J_and_grad(R, Ms)
    Jtr = [J0]
    for it in range(n_outer):
        Ms = fit_M(R)
        for _ in range(inner):
            J, G = J_and_grad(R, Ms)
            Gt = G - R @ (R.T @ G + G.T @ R) / 2.0       # Stiefel tangent projection
            eta = 1.0
            while eta > 1e-12:
                Rn, _ = np.linalg.qr(R - eta * Gt)       # QR retraction
                Jn, _ = J_and_grad(Rn, Ms)
                if Jn < J:
                    R = Rn; break
                eta *= 0.5
            else:
                break
        Jtr.append(J_and_grad(R, fit_M(R))[0])
        if verbose and (it == 0 or (it + 1) % 15 == 0):
            tic(f"  [B_op K={K} it {it+1}] J={Jtr[-1]:.4e}")
        if len(Jtr) > 2 and abs(Jtr[-2] - Jtr[-1]) < 1e-10 * Jtr[0]:
            break
    return R, fit_M(R), Jtr


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
    cols, fr = [C], [C]
    for _ in range(L - 1):
        nxt = [Gx @ f for Gx in Gs for f in fr]; cols += nxt; fr = nxt
    O = np.hstack(cols); U_O, sO, _ = np.linalg.svd(O, full_matrices=False)
    tic("forward O + U_O ready")
    print("observability spectrum sigma_i/sigma_1:", np.round(sO[:10] / sO[0], 4))

    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                 for x in range(V) if soft[w][x] > EPS)]

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
        reg = LinearRegression().fit(S, Y, sample_weight=wgt)
        return reg, reg.score(S, Y, sample_weight=wgt), com_r2(S)

    print(f"\n{'method':<22}{'per-pos R2':>12}{'state-COM R2':>14}")
    reg0, r0pp, r0c = decode(U_O)
    print(f"{'observability only':<22}{r0pp:>12.3f}{r0c:>14.3f}")

    best = None
    for Kdim in [8, 12, 20, 30]:
        U_K = U_O[:, :Kdim]
        data = []
        for x in range(V):
            m = [w for w in rows if soft[w][x] > EPS]
            Xp = np.stack([resid[w] for w in m]) @ U_K
            Xc = np.stack([resid[w + (x,)] for w in m]) @ U_K
            cc = np.array([soft[w][x] for w in m]); ww = np.array([pw[w] for w in m])
            data.append((Xp, Xc, cc, ww))
        R, Ms, Jtr = bop_refit(data, U_K, d, verbose=False)
        B_op = U_K @ R
        reg, rpp, rc = decode(B_op)
        print(f"{f'B_op refit  K={Kdim:2d}':<22}{rpp:>12.3f}{rc:>14.3f}   J:{Jtr[0]:.2e}->{Jtr[-1]:.2e}")
        if best is None or rc > best[0]:
            best = (rc, rpp, B_op, reg, Kdim, Ms)

    rc, rpp, B_op, reg, Kbest, Ms = best
    print(f"\nbest B_op: K={Kbest}  per-pos={rpp:.3f}  state-COM={rc:.3f}")
    print("operator-consistency check  eig(M_x) vs eig(T^x) [real parts, sorted]:")
    for x in range(V):
        et = np.sort(np.linalg.eigvals(T[x]).real)
        em = np.sort(np.linalg.eigvals(Ms[x]).real)
        print(f"  token {x}: eig(T)={np.round(et,3)}   eig(M)={np.round(em,3)}")

    # plot best B_op vs ground truth + supervised
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
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    draw(ax[0], tgt, "Ground truth belief geometry", scatter=False)
    draw(ax[1], pca.transform(sup.predict(Xf))[:, [0, 2]], f"Supervised: full residual stream\ndecode R$^2$={r2_sup:.2f}")
    draw(ax[2], pca.transform(reg.predict(Xf @ B_op[:, :d]))[:, [0, 2]],
         f"Operator-consistency refit $B_{{op}}$ (5-D, L={L}, K={Kbest})\ndecode R$^2$={rpp:.2f}  state-COM={rc:.2f}")
    out = FIG_DIR / "rrxor_bop_refit_L6.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")
    tic("done")


if __name__ == "__main__":
    main()
