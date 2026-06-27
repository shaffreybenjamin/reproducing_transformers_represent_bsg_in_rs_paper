"""Observable-anchored OOM: recover belief geometry from activations WITHOUT the
CCA-correlation step that degenerates on (near-)deterministic models, and WITHOUT
any DGP knowledge (model softmax + activations + the prefix tree only).

Pipeline (all ridge regressions -- no whitening, no destructive PCA):
  1. Observable anchor      C  : a(w) -> P(.|w)            (1-step softmax; seed + eval functional)
  2. Rescaled operators     G_x: a(w) -> P(x|w) a(wx)      (activation-space, one P factor: no compounding)
  3. Observability closure       grow span{C} under {G_x} until rank saturates -> latent dim d
  4. Reduce, fit A_x, eval e; validate (decode R^2, eig(A_x) vs eig(T_x), geometry)

DGP-free: reachability and weights come from the MODEL softmax, d is ESTIMATED (not
set to 5). Ground-truth belief is used ONLY to score, never to fit. The construction
assumes only LINEARITY (linear state / operators / observable), so it carries over to
quantum / post-quantum predictive representations unchanged.

FINDINGS (this is a correct method, validated on Mess3; RRXOR is intrinsically hard):
  * Mess3 (control): clean rank drop in the observability spectrum at d=3, belief-decode
    R^2 = 0.997 (= supervised), and eig(A_x) MATCHES eig(T_x). So the estimator works,
    and it is NOT the broken CCA from fig13.
  * RRXOR: decode only ~0.48 at d=5 / 0.61 at d=10, vs supervised ceiling 0.89; the
    spectrum has no clean rank. The earlier suspects are ruled out -- ESS was a weighting
    artifact, the clock is excluded by the P-anchored relation, and CCA is gone.
  * The real cause: RRXOR's belief has WEAKLY-OBSERVABLE modes (directions that barely
    change the predicted future -- the multi-step version of panel D's 0.33). Surfacing
    them unsupervised needs deep multi-step propagation, which compounds: iterating the
    estimated operators makes decode DROP with depth (0.53 -> 0.35 for depth 2 -> 8),
    because operator-estimation noise grows faster than the weak belief signal. The
    softmax-product route compounds chain-rule error instead -- same wall.
  * Supervised regression sidesteps all of this by using belief LABELS directly, so it
    never has to surface the weak modes from observations. The supervised/unsupervised
    gap (0.89 vs ~0.5) is exactly the "value of the labels": near zero for Mess3
    (strongly observable belief), large for RRXOR (weakly observable belief).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.transition_matrices import mess3, rrxor

MAX_LEN = 10
EPS = 1e-3            # model-softmax reachability threshold
RIDGE = 1e-2
OUT_DIR = Path(__file__).parent.parent  # go up to reproduction/ from plotting/
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def observable_subspace(resid, soft, reach, vocab, depth=3, wmap=None, use_multistep_als=True,
                       als_max_order=2, als_n_iter=20, als_lambda_decay=0.5):
    """Observability matrix O = [C, G_x C, G_x G_y C, ...]; its SVD gives candidate
    belief directions ordered by strength. Columns of O are the rescaled multi-step
    observables P(x..|w)*P(.|w x..), each LINEAR in belief, spanning belief as depth grows.

    wmap (optional): {prefix: P(w)} sample weights for the C/G_x ridge regressions, so the
    operators -- and hence the subspace -- are estimated toward high-probability prefixes
    (the post-quantum paper's P(w)-weighting). None => uniform (backwards-compatible).

    use_multistep_als (default True): refine operators using multi-step ALS after initial fit.
    als_max_order, als_n_iter: parameters for the ALS refinement."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows]); P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T   # (D, V)
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else sw[m]
        Gs.append(Ridge(alpha=RIDGE, fit_intercept=False).fit(A[m], tgt, sample_weight=swm).coef_.T)

    # Optionally refine operators using multi-step ALS
    if use_multistep_als:
        Gs = fit_operators_multistep_als(rows, A, P, resid, soft, vocab, Gs,
                                        max_order=als_max_order, n_iter=als_n_iter,
                                        lambda_decay=als_lambda_decay, sw=sw)

    cols, frontier = [C], [C]                                              # observability matrix
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols += nxt; frontier = nxt
    O = np.hstack(cols)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)
    return rows, A, P, Gs, U, sv


def analytic_prefix_probs(resid, T, pi):
    """{prefix: P(w)} via the belief-update product, for transition tensor T and start pi.
    Forbidden prefixes get 0. Used as the P(w) sample-weights for the subspace/operator fits."""
    out = {}
    for w in resid:
        b, p = np.array(pi, dtype=float), 1.0
        for x in w:
            d = np.array([(b @ T[i]).sum() for i in range(T.shape[0])])
            if d[x] < 1e-12:
                p = 0.0; break
            p *= float(d[x]); b = b @ T[x] / d[x]
        out[w] = p
    return out


def fit_operators_multistep_als(rows, A, P, resid, soft, vocab, Gs_init, max_order=3,
                               ridge_base=RIDGE, lambda_decay=0.5, n_iter=50, tol=1e-6, sw=None):
    """Refine operators {A_x} using multi-step ALS (alternating least squares).

    Starts from one-step ridge operators Gs_init and iteratively improves them by fitting
    against multi-step targets up to order max_order. Each ALS iteration holds all but one
    operator fixed and solves a linear ridge regression for that operator.

    Key implementation detail: for higher-order terms where j is not at position 1, the design
    matrix must be the composition of preceding operators (e.g., A @ Gs[y] for j at position 2),
    not the actual intermediate activations. This ensures the problem remains linear in A_j.

    Args:
        rows: list of prefix tuples (must match order of A and P)
        A: (n_rows, D) activation matrix
        P: (n_rows, vocab) softmax matrix
        resid: {prefix -> activation} dict
        soft: {prefix -> softmax logits} dict
        vocab: vocabulary size
        Gs_init: list of vocab initial (D, D) operator matrices
        max_order: maximum step order K to include (1, 2, ..., max_order)
        ridge_base: base ridge lambda for order-1 terms
        lambda_decay: multiplicative decay of ridge lambda per order (default 0.5)
        n_iter: max ALS iterations (default 50)
        tol: convergence tolerance on operator change (Frobenius norm)
        sw: optional (n_rows,) sample weights P(w); None => uniform

    Returns:
        Gs_refined: list of refined operators (same shape as Gs_init)
    """
    Gs = [G.copy() for G in Gs_init]
    n_rows = len(rows)
    D = Gs[0].shape[0]

    # Pre-compute one-step targets for diagnostics
    one_step_targets = {}  # {x: (X_1, Y_1, W_1)}
    for x in range(vocab):
        X_1 = []
        Y_1 = []
        W_1 = []
        for i, w in enumerate(rows):
            if soft[w][x] > EPS:
                wx = w + (x,)
                if wx in resid:
                    X_1.append(A[i])
                    Y_1.append(soft[w][x] * resid[wx])
                    W_1.append(sw[i] if sw is not None else 1.0)
        if X_1:
            one_step_targets[x] = (np.array(X_1), np.array(Y_1), np.array(W_1))

    def compute_one_step_residual():
        """Compute weighted MSE for one-step predictions."""
        total_mse = 0.0
        total_weight = 0.0
        for x in range(vocab):
            if x in one_step_targets:
                X_1, Y_1, W_1 = one_step_targets[x]
                pred = X_1 @ Gs[x]
                mse = np.sum(W_1[:, None] * (pred - Y_1) ** 2)
                total_mse += mse
                total_weight += W_1.sum()
        return total_mse / max(total_weight, 1e-12) if total_weight > 0 else np.nan

    # Record initial one-step quality
    initial_one_step_mse = compute_one_step_residual()
    print(f"  [ALS init] one-step MSE={initial_one_step_mse:.3e}")

    converged = False
    for ais_iter in range(n_iter):
        max_delta = 0.0

        # Update each operator in turn
        for j in range(vocab):
            # Collect all design matrices and targets involving operator j across all orders
            X_blocks = []
            Y_blocks = []
            W_blocks = []

            # ===== ORDER 1: a(w) A_j ≈ P(j|w) a(wj) =====
            X_1 = []
            Y_1 = []
            W_1 = []
            for i, w in enumerate(rows):
                if soft[w][j] > EPS:
                    wj = w + (j,)
                    if wj in resid:
                        X_1.append(A[i])
                        Y_1.append(soft[w][j] * resid[wj])
                        W_1.append(sw[i] if sw is not None else 1.0)

            if X_1:
                X_blocks.append(np.array(X_1))
                Y_blocks.append(np.array(Y_1))
                W_blocks.append(np.array(W_1))

            # ===== ORDER 2 =====
            if max_order >= 2:
                # j at position 1: a(w) A_j A_y ≈ P(jy|w) a(wjy)
                for y in range(vocab):
                    X_2 = []
                    Y_2 = []
                    W_2 = []
                    for i, w in enumerate(rows):
                        if (soft[w][j] > EPS and w + (j,) in resid and w + (j,) in soft and
                            soft[w + (j,)][y] > EPS and w + (j, y) in resid):
                            rescale = soft[w][j] * soft[w + (j,)][y]
                            X_2.append(A[i])
                            Y_2.append(rescale * resid[w + (j, y)])
                            W_2.append(sw[i] if sw is not None else 1.0)

                    if X_2:
                        X_blocks.append(np.array(X_2))
                        Y_blocks.append(np.array(Y_2))
                        W_blocks.append(np.array(W_2))

                # j at position 2: a(w) A_y A_j ≈ P(yj|w) a(wyj)
                # CORRECT: design matrix is A @ A_y (operator composition), not a(wy)
                for y in range(vocab):
                    X_2 = []
                    Y_2 = []
                    W_2 = []
                    for i, w in enumerate(rows):
                        if (soft[w][y] > EPS and w + (y,) in resid and w + (y,) in soft and
                            soft[w + (y,)][j] > EPS and w + (y, j) in resid):
                            rescale = soft[w][y] * soft[w + (y,)][j]
                            X_2.append(A[i] @ Gs[y])  # FIXED: operator composition, not actual activation
                            Y_2.append(rescale * resid[w + (y, j)])
                            W_2.append(sw[i] if sw is not None else 1.0)

                    if X_2:
                        X_blocks.append(np.array(X_2))
                        Y_blocks.append(np.array(Y_2))
                        W_blocks.append(np.array(W_2))

            # ===== ORDER 3 =====
            if max_order >= 3:
                # j at position 1: a(w) A_j A_y A_z ≈ P(jyz|w) a(wjyz)
                for y in range(vocab):
                    for z in range(vocab):
                        X_3 = []
                        Y_3 = []
                        W_3 = []
                        for i, w in enumerate(rows):
                            if (soft[w][j] > EPS and w + (j,) in resid and w + (j,) in soft and
                                soft[w + (j,)][y] > EPS and w + (j, y) in resid and w + (j, y) in soft and
                                soft[w + (j, y)][z] > EPS and w + (j, y, z) in resid):
                                rescale = soft[w][j] * soft[w + (j,)][y] * soft[w + (j, y)][z]
                                X_3.append(A[i])
                                Y_3.append(rescale * resid[w + (j, y, z)])
                                W_3.append(sw[i] if sw is not None else 1.0)

                        if X_3:
                            X_blocks.append(np.array(X_3))
                            Y_blocks.append(np.array(Y_3))
                            W_blocks.append(np.array(W_3))

                # j at position 2: a(w) A_y A_j A_z ≈ P(yjz|w) a(wyjz)
                # CORRECT: design matrix is A @ A_y
                for y in range(vocab):
                    for z in range(vocab):
                        X_3 = []
                        Y_3 = []
                        W_3 = []
                        for i, w in enumerate(rows):
                            if (soft[w][y] > EPS and w + (y,) in resid and w + (y,) in soft and
                                soft[w + (y,)][j] > EPS and w + (y, j) in resid and w + (y, j) in soft and
                                soft[w + (y, j)][z] > EPS and w + (y, j, z) in resid):
                                rescale = soft[w][y] * soft[w + (y,)][j] * soft[w + (y, j)][z]
                                X_3.append(A[i] @ Gs[y])  # FIXED: operator composition
                                Y_3.append(rescale * resid[w + (y, j, z)])
                                W_3.append(sw[i] if sw is not None else 1.0)

                        if X_3:
                            X_blocks.append(np.array(X_3))
                            Y_blocks.append(np.array(Y_3))
                            W_blocks.append(np.array(W_3))

                # j at position 3: a(w) A_y A_z A_j ≈ P(yzj|w) a(wyzj)
                # CORRECT: design matrix is A @ A_y @ A_z
                for y in range(vocab):
                    for z in range(vocab):
                        X_3 = []
                        Y_3 = []
                        W_3 = []
                        for i, w in enumerate(rows):
                            if (soft[w][y] > EPS and w + (y,) in resid and w + (y,) in soft and
                                soft[w + (y,)][z] > EPS and w + (y, z) in resid and w + (y, z) in soft and
                                soft[w + (y, z)][j] > EPS and w + (y, z, j) in resid):
                                rescale = soft[w][y] * soft[w + (y,)][z] * soft[w + (y, z)][j]
                                X_3.append(A[i] @ Gs[y] @ Gs[z])  # FIXED: full operator composition
                                Y_3.append(rescale * resid[w + (y, z, j)])
                                W_3.append(sw[i] if sw is not None else 1.0)

                        if X_3:
                            X_blocks.append(np.array(X_3))
                            Y_blocks.append(np.array(Y_3))
                            W_blocks.append(np.array(W_3))

            # Assemble and solve the ridge regression for operator j
            if X_blocks:
                X_all = np.vstack(X_blocks)
                Y_all = np.vstack(Y_blocks)
                W_all = np.concatenate(W_blocks)

                ridge = Ridge(alpha=ridge_base, fit_intercept=False)
                ridge.fit(X_all, Y_all, sample_weight=W_all)
                G_new = ridge.coef_.T

                # Track convergence
                delta = np.linalg.norm(G_new - Gs[j], 'fro')
                max_delta = max(max_delta, delta)
                Gs[j] = G_new

        # Compute and monitor one-step residual
        one_step_mse = compute_one_step_residual()
        mse_ratio = one_step_mse / initial_one_step_mse if initial_one_step_mse > 0 else 1.0

        # Diagnostic output
        if (ais_iter + 1) % 5 == 0 or ais_iter == 0 or ais_iter == n_iter - 1:
            print(f"    [ALS iter {ais_iter+1:2d}/{n_iter}] delta={max_delta:.2e}  one-step MSE={one_step_mse:.3e} (ratio={mse_ratio:.3f})")

        # Warn if one-step quality is degrading significantly
        if mse_ratio > 1.5:
            print(f"    WARNING: one-step MSE degraded to {mse_ratio:.1f}× initial; consider reducing lambda_decay")

        if max_delta < tol:
            converged = True
            print(f"  [multistep-ALS] converged after {ais_iter + 1} iterations (delta={max_delta:.2e})")
            break

    if not converged:
        print(f"  [multistep-ALS] did not converge after {n_iter} iters (final delta: {max_delta:.2e}, one-step MSE ratio: {mse_ratio:.3f})")

    return Gs


def fit_at_dim(rows, A, P, resid, vocab, B):
    """Project to subspace B (D x d), fit operators A_x and eval functional e."""
    s_all = A @ B
    ops = {}
    for x in range(vocab):
        m = np.array([P[i, x] > EPS for i in range(len(rows))])
        sw = s_all[m]
        sc = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]]) @ B
        ops[x] = np.linalg.lstsq(sw, P[m, x][:, None] * sc, rcond=None)[0]
    e = np.linalg.lstsq(s_all, np.ones(len(s_all)), rcond=None)[0]
    return s_all, ops, e


def run(name, ckpt, proc_T, stationary, use_multistep_als=False, als_max_order=2, als_n_iter=15, als_lambda_decay=0.2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load(ckpt, device)  # resid_post concat, softmax, analytic belief for SCORING only
    resid, soft, belief, pp = _collect(m, proc_T, stationary, device)
    # model-determined reachability (DGP-free): node reachable if every step's softmax > EPS
    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:   # first token has no context -> always reachable
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    vocab = proc_T.shape[0]
    print(f"\n=== {name} ===  reachable nodes: {sum(reach.values())}/{len(reach)}")

    rows, A, P, Gs, U, sv = observable_subspace(resid, soft, reach, vocab, use_multistep_als=use_multistep_als,
                                               als_max_order=als_max_order, als_n_iter=als_n_iter,
                                               als_lambda_decay=als_lambda_decay)
    print(f"  rows={len(rows)}  D={A.shape[1]}  observability singular spectrum:\n   {np.round(sv[:12] / sv[0], 3)}")
    if use_multistep_als:
        print(f"  [multistep-ALS] applied (order up to {als_max_order}, {als_n_iter} iters)")

    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    decs = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :d])
        dec = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
        decs.append((d, dec))
    print("  belief-decode R^2 vs d:", {d: round(r, 3) for d, r in decs})

    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])
    dstar = proc_T.shape[1]                                   # report eig/geometry at the true belief dim
    s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :dstar])
    reg = LinearRegression().fit(s_all[fin], Yb[fin]); decode = reg.score(s_all[fin], Yb[fin])
    print(f"  supervised ceiling={ceiling:.3f};  at d={dstar}: decode R^2={decode:.3f}; eig(A^x) vs eig(T^x):")
    for x in range(vocab):
        evt = np.sort(np.linalg.eigvals(proc_T[x]).real)
        eva = np.sort(np.linalg.eigvals(ops[x]).real)
        print(f"    token {x}: eig(T)={np.round(evt,3)}  eig(A)={np.round(eva,3)}")
    return dict(resid=resid, soft=soft, belief=belief, rows=rows, A=A, P=P, Gs=Gs, U=U,
                fin=fin, Yb=Yb, reg=reg, dstar=dstar, decode=decode, decs=decs, ceiling=ceiling, vocab=vocab)


def _load(ckpt, device):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    c = torch.load(MODEL_DIR / ckpt, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(c["cfg"]); cfg.device = device
    mm = HookedTransformer(cfg); mm.load_state_dict(c["state_dict"]); mm.to(device).eval()
    return mm


def _collect(model, T, stationary, device):
    import itertools
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = T.shape[1]

    def ndist(b): return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])
    def upd(b, x): nb = b @ T[x]; return nb / nb.sum()
    def bp(w):
        b, p = stationary, 1.0
        for x in w:
            dd = ndist(b); p *= dd[x]
            if dd[x] < 1e-12: return None
            b = upd(b, x)
        return b

    resid, soft, belief, pp = {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]; soft[w] = sm[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan); pp[w] = 0.0
    return resid, soft, belief, pp


def depth_curve(res, depths=(2, 3, 4, 5, 6, 8)):
    """RRXOR decode (at d=5) vs observability depth -- iterating the estimated operators."""
    A, P, fin, Yb = res["A"], res["P"], res["fin"], res["Yb"]
    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P).coef_.T
    out = []
    for depth in depths:
        cols, fr = [C], [C]
        for _ in range(depth - 1):
            nxt = [Gx @ f for Gx in res["Gs"] for f in fr]; cols += nxt; fr = nxt
        U, _, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
        s = A @ U[:, :5]
        out.append((depth, LinearRegression().fit(s[fin], Yb[fin]).score(s[fin], Yb[fin])))
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    R = run("RRXOR", "rrxor_transformer.pt", np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0,
            use_multistep_als=True, als_max_order=2, als_n_iter=30)
    M = run("Mess3", "mess3_transformer.pt", np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0,
            use_multistep_als=True, als_max_order=2, als_n_iter=30)
    dc = depth_curve(R)
    print("RRXOR decode (d=5) vs observability depth:", {d: round(r, 3) for d, r in dc})

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    for res, nm, col in [(R, "RRXOR", "black"), (M, "Mess3", "red")]:
        ds = [d for d, _ in res["decs"]]; rr = [r for _, r in res["decs"]]
        ax0.plot(ds, rr, "o-", color=col, label=f"{nm} (observable OOM)")
        ax0.axhline(res["ceiling"], color=col, ls="--", lw=1, alpha=0.6,
                    label=f"{nm} supervised ceiling = {res['ceiling']:.2f}")
    ax0.set_xlabel("subspace dim d"); ax0.set_ylabel("belief-decode R$^2$")
    ax0.set_title("Observable-anchored OOM (DGP-free):\nrecovers Mess3, NOT RRXOR")
    ax0.set_ylim(0, 1.02); ax0.legend(fontsize=8, loc="center right"); ax0.grid(alpha=.3)

    dd = [d for d, _ in dc]; rr = [r for _, r in dc]
    ax1.plot(dd, rr, "ko-")
    ax1.set_xlabel("observability depth (operator iterations)")
    ax1.set_ylabel("RRXOR belief-decode R$^2$ (d=5)")
    ax1.set_title("RRXOR: deeper observability -> WORSE\n(operator-iteration noise compounds past the weak signal)")
    ax1.set_ylim(0, 0.6); ax1.grid(alpha=.3)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig14_observable_oom.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
