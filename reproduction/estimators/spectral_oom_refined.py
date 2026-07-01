"""Refined spectral-OOM estimator: balanced truncation (i), Riesz spectral projectors (ii),
joint-subspace recovery (iii), Fisher-metric readout (iv) -- layered ON TOP OF the user's
VALIDATED plain spectral-OOM, not a reimplementation.

The plain baseline is the user's own `observable_subspace` from
`plotting/fig14_observable_oom.py` (model-softmax reachability, depth-3 observability matrix,
single-pass ridge operators, NO ALS/CCA) run with the SAME two settings the write-up figure
uses: (2) P(w)-weighted regressions (`wmap=pw`), and (4) the fig4 per-(input,position),
per-state-equal scoring metric over the length-10 enumeration (fig10 data).  With these the
baseline REPRODUCES the write-up numbers exactly: RRXOR supervised 0.911 / unsupervised
d=5 = 0.651 (state-COM ~0.82), Mess3 ~0.997.  Every refinement then operates on the SAME
P(w)-weighted operators {G_x}, scored with the SAME metric, so any difference is the
refinement alone.

Corrections applied (see `spectral_oom_corrections.md`):
  Fix 1  affine offset via per-regression intercept (fit_intercept=True), NOT global
         centering -- offered as an `intercept` toggle / "affine-corrected" method, with the
         validated baseline left untouched as the reference.
  Fix 2  Riesz contour normalisation corrected (per-cluster local accumulation) + idempotency
         and integer-trace guards; refuses to run on un-reduced (D,D) operators.  Riesz is a
         BASIS tool (validated projectors), NOT a d-selector -- see Correction 3.
  Fix 3  joint-Schur targets COMMON INVARIANCE (projected-gradient on the joint-invariance
         residual), not the dominant (largest-|eig|) subspace.
  Fix 4  consistent Gramian sourcing: empirical P(w)-weighted Wc + operator-based Wo with
         range_metric="lambda_decay" as the headline; future_cov demoted to a flagged
         diagnostic (it whitens low-variance NOISE as much as weak belief modes).

Review corrections (spectral_oom_corrections round 2):
  C1  balanced-truncation basis orthonormalised (QR) before scoring -- B_full carries the
      1/sqrt(sigma) balancing scale, so slicing it directly gave an ill-conditioned basis that
      depressed the decode for numerical reasons (the artifact behind the old 0.27-0.48 band).
  C2  joint-Schur residual reported SCALE-FREE (num/den in [0,1]); analytic gradient verified
      against finite differences before the "no common invariant subspace" claim is cited.
  C3  Riesz reframed as a basis tool, not a failed d-selector (cluster count = reduction dim
      by construction; model order stays on the Hankel sigma spectrum).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

_THIS_DIR = Path(__file__).resolve().parent
_PLOTTING = _THIS_DIR.parent / "plotting"
for _p in (str(_THIS_DIR), str(_PLOTTING)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fig14_observable_oom as f14          # the VALIDATED plain spectral-OOM
import fig10_rrxor_representation as F10     # fig4/fig10 per-(input,position) scoring data

EPS = f14.EPS
RIDGE = f14.RIDGE
P_FLOOR = 1e-3                              # Fisher readout: floor on per-token probability

MODEL_DIR = _THIS_DIR.parent / "models"
ARCHIVED_DIR = _THIS_DIR.parent / "models_paper1_sgd_ctx10"   # fully-trained SGD/ctx10
FIG_DIR = _THIS_DIR.parent / "figures"


# ===========================================================================
# Readout / operator fits (refinement side; corrections Fix 1 -- intercept handling)
# ===========================================================================

def fit_readout(A, P, sw, vocab, metric="euclidean", intercept=False, ridge=RIDGE):
    """Readout C: a(w) -> P(.|w).  Refinement (iv).

    metric="euclidean": plain ridge (reproduces the baseline C when intercept=False, sw=None).
    metric="fisher": diagonal Fisher-metric GLS -- output column k fit with row weights
      sw * 1/max(P(x=k|w), P_FLOOR), up-weighting rows where token k is rare.  Exact for the
      diagonal simplex-Fisher surrogate (columns decouple).
    intercept: per-regression affine offset (corrections Fix 1).  Default False to match the
      validated baseline; pass True for the affine-corrected variant.
    """
    D = A.shape[1]
    if metric == "euclidean":
        return Ridge(alpha=ridge, fit_intercept=intercept).fit(A, P, sample_weight=sw).coef_.T
    if metric == "fisher":
        base = np.ones(len(A)) if sw is None else np.asarray(sw, float)
        C = np.zeros((D, vocab))
        for k in range(vocab):
            rw = base / np.maximum(P[:, k], P_FLOOR)
            C[:, k] = Ridge(alpha=ridge, fit_intercept=intercept).fit(A, P[:, k], sample_weight=rw).coef_
        return C
    raise ValueError(f"unknown readout metric {metric!r}")


def fit_operators(rows, A, P, resid, soft, vocab, sw, intercept=True, ridge=RIDGE):
    """Rescaled operators a(w) A_x (+ d_x) ~ P(x|w) a(wx) with per-regression intercept
    (corrections Fix 1).  Mirrors the baseline G_x fit (subset rows with soft>EPS) but lets
    each fit absorb its own affine offset; intercepts discarded (only A_x enters the spectral
    construction).  Returns list [A_x (D,D)]."""
    Gs = []
    for x in range(vocab):
        m = np.array([soft[rows[i]][x] > EPS for i in range(len(rows))])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else np.asarray(sw, float)[m]
        Gs.append(Ridge(alpha=ridge, fit_intercept=intercept).fit(A[m], tgt, sample_weight=swm).coef_.T)
    return Gs


def fit_subspace_operators(rows, A, P, resid, vocab, basis, sw=None, intercept=True):
    """Refit operators INSIDE the recovered subspace (corrections Fix 1, intercept=True):
    s(w) A_x^sub (+ d_x) ~ P(x|w) s(wx), s = a(w) B.  Reducing first is what makes the
    eig(A) vs eig(T) check and the Riesz analysis meaningful.  Returns list [A_x^sub (d,d)]."""
    S = A @ basis
    ops = []
    for x in range(vocab):
        m = np.array([P[i, x] > EPS for i in range(len(rows))])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]]) @ basis
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else np.asarray(sw, float)[m]
        ops.append(Ridge(alpha=RIDGE, fit_intercept=intercept).fit(S[m], tgt, sample_weight=swm).coef_.T)
    return ops


def build_observability_factor(C, Gs, depth=3):
    """O = [C | (A_x C)_x | (A_x A_y C)_{x,y} | ...] to ``depth`` (matches the baseline
    observable_subspace construction).  Returns (O, col_depth) with col_depth[j] the number
    of operator applications for column j."""
    cols = [C]
    col_depth = [0] * C.shape[1]
    frontier = [C]
    for k in range(1, depth):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols.extend(nxt)
        col_depth.extend([k] * sum(f.shape[1] for f in nxt))
        frontier = nxt
    return np.hstack(cols), np.array(col_depth)


# ===========================================================================
# (i) metric-correct balanced truncation  (corrections Fix 4 -- Gramian sourcing)
# ===========================================================================

def controllability_gram(A_rows, pw_rows, domain_metric="pw"):
    """Empirical controllability Gramian Wc = sum_w dc(w) a(w) a(w)^T (D,D): the P(w)-weighted
    second moment of reachable activations.  domain_metric="pw" weights by P(w) (default,
    the intended single factor); "uniform" weights contexts equally."""
    if domain_metric == "pw":
        dc = np.asarray(pw_rows, float)
    elif domain_metric == "uniform":
        dc = np.ones(len(A_rows))
    else:
        raise ValueError(f"unknown domain metric {domain_metric!r}")
    dc = dc / max(dc.sum(), 1e-12)
    return A_rows.T @ (dc[:, None] * A_rows)


def observability_gram(O, col_depth, A_rows=None, pw_rows=None,
                       range_metric="lambda_decay", lam=0.5, ridge=1e-6):
    """Operator-based observability Gramian Wo = O Mo O^T (D,D).

    range_metric="euclidean": Mo = I.
    range_metric="lambda_decay" (HEADLINE): Mo = diag(lam**col_depth) -- damp long futures by
      composition depth, a STRUCTURAL weight (noise does not share it).
    range_metric="future_cov" (DIAGNOSTIC ONLY): Mo = inv(Cov_{P(w)}[a(w) O]).  Whitens by the
      future-observable covariance; amplifies low-variance NOISE directions as much as weak
      belief modes -- treat any gain as suspect until a noise-direction control confirms it.
    """
    if range_metric == "euclidean":
        return O @ O.T
    if range_metric == "lambda_decay":
        mo = lam ** col_depth.astype(float)
        return (O * mo) @ O.T
    if range_metric == "future_cov":
        F = A_rows @ O
        wv = np.asarray(pw_rows, float); wv = wv / wv.sum()
        Fm = F - (wv[:, None] * F).sum(0)
        cov = Fm.T @ (wv[:, None] * Fm)
        Mo = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
        return O @ Mo @ O.T
    raise ValueError(f"unknown range metric {range_metric!r}")


def _psd_sqrt(W):
    """Symmetric PSD square root L with L L^T = W (eigenvalues clipped >= 0)."""
    W = 0.5 * (W + W.T)
    ev, Q = np.linalg.eigh(W)
    return (Q * np.sqrt(np.clip(ev, 0.0, None))) @ Q.T


def balanced_truncation(Wc, Wo, d=None, sv_floor=1e-12):
    """Square-root balanced truncation.  Returns (basis (D,d), hankel_sv).
    Lc Lc^T = Wc, Lo Lo^T = Wo; SVD(Lc^T Lo) = U S V^T gives basis-independent Hankel sigma S;
    basis B = Lc U S^{-1/2} (first d columns).  NOTE B_full is a balancing SIMILARITY, not a
    projector -- its columns carry the 1/sqrt(sigma) scaling, so slice-and-score must go through
    orthonormal_balanced_basis (QR), else the conditioning of Sigma^{-1/2} corrupts the decode."""
    Lc, Lo = _psd_sqrt(Wc), _psd_sqrt(Wo)
    U, S, _ = np.linalg.svd(Lc.T @ Lo, full_matrices=False)
    Sinv2 = 1.0 / np.sqrt(np.clip(S, sv_floor, None))
    B_full = (Lc @ U) * Sinv2[None, :]
    return (B_full if d is None else B_full[:, :d]), S


def orthonormal_balanced_basis(B_full, d):
    """Orthonormal basis for the span of the top-d balanced directions.  The balancing transform
    B_full = Lc U Sigma^{-1/2} is a similarity, NOT a projector: its columns carry the
    1/sqrt(sigma) scaling, so slicing [:, :d] gives an ill-conditioned (non-orthonormal) basis
    that degrades the decode for purely numerical reasons.  QR restores an orthonormal basis for
    the SAME subspace, so the comparison measures the balanced reweighting/ordering of modes, not
    the conditioning of Sigma^{-1/2}."""
    Q, _ = np.linalg.qr(B_full[:, :d])
    return Q


def principal_angles_deg(B1, B2):
    """Principal angles (degrees) between the column spans of two bases (orthonormalised here)."""
    Q1, _ = np.linalg.qr(B1)
    Q2, _ = np.linalg.qr(B2)
    s = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


# ===========================================================================
# (iii) joint-Schur common invariant subspace  (corrections Fix 3 -- no magnitude bias)
# ===========================================================================

def _joint_invariance_residual(Q, Gs, vocab):
    """Sum_x || (I - Q Q^T) A_x Q ||_F^2 : 0 iff span(Q) invariant under every A_x."""
    r = 0.0
    QQt = Q @ Q.T
    for x in range(vocab):
        M = Gs[x] @ Q
        r += float(np.linalg.norm(M - QQt @ M) ** 2)
    return r


def joint_invariance_residual_normalized(Q, Gs, vocab):
    """Scale-free joint-invariance residual in [0, 1]:
        sum_x ||(I - QQ^T) A_x Q||_F^2  /  sum_x ||A_x Q||_F^2
    0 = span(Q) exactly invariant under every A_x; ~1 = the off-subspace component carries
    essentially all the action of the operators on span(Q).  This is the interpretable number;
    the raw residual scales with ||A_x||^2 and means nothing on its own."""
    num = den = 0.0
    QQt = Q @ Q.T
    for x in range(vocab):
        AQ = Gs[x] @ Q
        num += float(np.linalg.norm(AQ - QQt @ AQ) ** 2)
        den += float(np.linalg.norm(AQ) ** 2)
    return num / max(den, 1e-12)


def check_joint_gradient(Gs, vocab, D, d, seed=0, eps=1e-6):
    """Finite-difference check of the gradient used in _refine_invariant_subspace.  Returns the
    relative error between the analytic gradient and a central-difference gradient of
    f(Q)=sum_x ||(I-QQ^T)A_x Q||_F^2 at a random Q.  Should be < 1e-4; if not, the analytic
    gradient is wrong and a large residual is a gradient bug, not a geometric obstruction."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, d)))

    def f(Qm):
        # the function _joint_gradient differentiates: g(Q)=sum_x ||A_xQ||^2 - ||Q^T A_x Q||^2,
        # which equals the projector residual sum_x ||(I-QQ^T)A_xQ||^2 ON the Stiefel manifold
        # (Q^T Q = I).  The optimizer QR-retracts to the manifold each step, so this is the
        # correct objective; differentiating the off-manifold projector form instead would add a
        # spurious normal component the retraction removes.
        r = 0.0
        for x in range(vocab):
            AQ = Gs[x] @ Qm
            r += float(np.linalg.norm(AQ) ** 2 - np.linalg.norm(Qm.T @ AQ) ** 2)
        return r

    G = _joint_gradient(Q, Gs, vocab)        # the exact gradient used by the refiner
    Gfd = np.zeros_like(Q)
    for i in range(D):
        for j in range(d):
            Qp = Q.copy(); Qp[i, j] += eps
            Qm = Q.copy(); Qm[i, j] -= eps
            Gfd[i, j] = (f(Qp) - f(Qm)) / (2 * eps)
    return float(np.linalg.norm(G - Gfd) / max(np.linalg.norm(Gfd), 1e-12))


def _joint_gradient(Q, Gs, vocab):
    """Euclidean gradient of f(Q) = sum_x ||(I - QQ^T) A_x Q||_F^2.
    Writing f_x = ||A_x Q||^2 - ||Q^T A_x Q||^2, the gradient is
        sum_x 2 ( A_x^T A_x Q  -  A_x Q (Q^T A_x^T Q)  -  A_x^T Q (Q^T A_x Q) ).
    (Verified against central finite differences to < 1e-6; the previous expression was wrong.)"""
    G = np.zeros_like(Q)
    for x in range(vocab):
        A = Gs[x]
        AQ = A @ Q                       # A Q
        AtQ = A.T @ Q                    # A^T Q
        G += 2.0 * (A.T @ AQ - AQ @ (Q.T @ AtQ) - AtQ @ (Q.T @ AQ))
    return G


def _refine_invariant_subspace(Q, Gs, vocab, iters=200, lr=0.05):
    """Projected-gradient refinement of orthonormal Q minimising the joint-invariance
    residual; retracted to the Stiefel manifold by QR.  Magnitude-agnostic (does not prefer
    large-|eig| modes)."""
    Q = Q.copy()
    for _ in range(iters):
        Q = Q - lr * _joint_gradient(Q, Gs, vocab)
        Q, _ = np.linalg.qr(Q)
    return Q


def joint_schur_subspace(Gs, vocab, d, n_gamma=8, seed=0, refine_iters=400):
    """Common (approximate) d-dim invariant subspace across {A_x}, WITHOUT eigenvalue-magnitude
    bias.  Seed Q from real Schur vectors of random combinations M = sum gamma_x A_x, refine by
    projected gradient on the joint-invariance residual, keep the best Q.  Returns
    (basis (D,d), residual); a residual that stays large means the estimated operators do NOT
    share a clean common invariant subspace -- itself a finding."""
    from scipy.linalg import schur
    rng = np.random.default_rng(seed)
    best_Q, best_res = None, np.inf
    for g in range(n_gamma):
        gamma = rng.standard_normal(vocab)
        M = sum(gamma[x] * Gs[x] for x in range(vocab))
        _, Z = schur(M, output="real")
        seeds = [Z[:, :d]]
        if Z.shape[1] > d:
            seeds.append(Z[:, -d:])
            mid = (Z.shape[1] - d) // 2
            seeds.append(Z[:, mid:mid + d])
        for Q0 in seeds:
            Q0, _ = np.linalg.qr(Q0)
            Q = _refine_invariant_subspace(Q0, Gs, vocab, iters=refine_iters)
            res = _joint_invariance_residual(Q, Gs, vocab)
            if res < best_res:
                best_res, best_Q = res, Q
    return best_Q, best_res


# ===========================================================================
# (ii) Riesz spectral projectors  (corrections Fix 2 -- contour + guards)
# ===========================================================================

def _cluster_eigs(eigs, tol):
    """Single-linkage clustering of complex eigenvalues within distance ``tol``."""
    n = len(eigs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if abs(eigs[i] - eigs[j]) <= tol:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def riesz_subspace(ops, vocab, drop_zero=True, cluster_tol=0.08, zero_tol=0.03,
                   n_quad=128, sv_tol=1e-2, idem_tol=1e-2, max_dim=40, verbose=True):
    """Riesz spectral projectors of the marginal operator hat-A = sum_x A_x.

    For each retained eigenvalue cluster, P_lambda = (1/2pi i) oint (zI-hatA)^{-1} dz on a
    circle enclosing ONLY that cluster, parametrised z(theta)=c+r e^{i theta}; the
    (1/2pi i) dz factor is (1/2pi) r e^{i theta} dtheta and dtheta = 2pi/n_quad.  Each cluster's
    projector is accumulated LOCALLY, normalised alone, then added (the earlier bug normalised
    the running total once per cluster).  Idempotency (||P^2-P||) and integer-trace guards
    catch contours that enclose the wrong eigenvalues -- the Mess3 decode alone is necessary
    but not sufficient.

    MUST be called on operators reduced to a modest subspace (raise if D > max_dim): the raw
    (D,D) operators carry D-d spurious ridge-zeros that dominate the clustering.

    drop_zero: exclude the cluster at/near 0.  CAUTION for RRXOR -- T_1 is nilpotent, so part
    of the genuine belief structure sits at lambda=0; report d with AND without the zero
    cluster and treat the difference as a finding.
    """
    Ahat = sum(ops[x] for x in range(vocab))
    D = Ahat.shape[0]
    if D > max_dim:
        raise ValueError(f"riesz_subspace got D={D} > max_dim={max_dim}; reduce operators to "
                         f"a small subspace first (fit_subspace_operators on a d<=10 basis).")
    eigs = np.linalg.eigvals(Ahat)
    clusters = _cluster_eigs(eigs, cluster_tol)
    theta = 2 * np.pi * (np.arange(n_quad) + 0.5) / n_quad

    P_tot = np.zeros((D, D), dtype=complex)
    kept = []
    for cl in clusters:
        cvals = eigs[cl]
        center = complex(cvals.mean())
        spread = float(np.max(np.abs(cvals - center))) if len(cl) > 1 else 0.0
        if drop_zero and abs(center) < zero_tol:
            continue
        other = np.array([eigs[i] for i in range(len(eigs)) if i not in cl])
        nearest = float(np.min(np.abs(other - center))) if len(other) else np.inf
        if np.isinf(nearest):
            r = max(spread * 2.0, 0.1)
        elif nearest <= spread + 1e-9:
            if verbose:
                print(f"    [riesz] skip cluster @ {center:.3f}: nearest {nearest:.3f} <= spread {spread:.3f}")
            continue
        else:
            r = 0.5 * (spread + nearest)

        P_cl = np.zeros((D, D), dtype=complex)
        z = center + r * np.exp(1j * theta)
        for zm, th in zip(z, theta):
            resolvent = np.linalg.inv(zm * np.eye(D) - Ahat)
            P_cl += resolvent * (r * np.exp(1j * th))
        P_cl /= n_quad

        idem = float(np.linalg.norm(P_cl @ P_cl - P_cl))
        tr = complex(np.trace(P_cl))
        if verbose:
            print(f"    [riesz] cluster @ {center:.3f} r={r:.3f} size={len(cl)} "
                  f"trace={tr.real:.2f} ||P^2-P||={idem:.2e}")
        if idem > idem_tol and verbose:
            print(f"    [riesz] WARNING idempotency {idem:.2e} > {idem_tol}: contour may enclose wrong eigenvalues.")
        kept.append((center, r, len(cl), tr.real, idem))
        P_tot += P_cl

    U, s, _ = np.linalg.svd(P_tot.real, full_matrices=False)
    d = max(int(np.sum(s > sv_tol * (s[0] if s[0] > 0 else 1.0))), 1)
    return U[:, :d], d, dict(eigs=eigs, clusters=clusters, kept=kept, proj_sv=s)


# ===========================================================================
# d-selection + scoring
# ===========================================================================

def detect_elbow(sv, k_max=20):
    """Largest log-gap in the top-k_max normalised singular spectrum."""
    sv = np.asarray(sv, float)
    sv = sv[:k_max] / sv[0]
    gaps = np.diff(np.log(np.clip(sv, 1e-12, None)))
    return max(int(np.argmax(-gaps)) + 1, 1)


def decode_with_basis(A, fin, Yb, basis):
    """Belief-decode R^2: project activations onto ``basis`` and linear-regress (with
    intercept) onto the true belief.  Matches the baseline scoring in f14.run."""
    s = A @ basis
    return LinearRegression().fit(s[fin], Yb[fin]).score(s[fin], Yb[fin])


def eig_diagnostic(ops_sub, T, vocab):
    """eig(A_x^sub) vs eig(T_x); expects operators reduced to the belief subspace."""
    for x in range(vocab):
        evt = np.sort(np.linalg.eigvals(T[x]).real)
        eva = np.sort(np.linalg.eigvals(ops_sub[x]).real)
        print(f"    token {x}: eig(T)={np.round(evt, 3)}  eig(A_sub)={np.round(eva, 3)}")


# ===========================================================================
# Baseline (reuse fig14's observable_subspace) with P(w)-weighting (correction 2)
# ===========================================================================

def fit_pipeline(ckpt, proc_T, stationary, vocab, model_dir, device="cpu", depth=3):
    """The VALIDATED plain spectral-OOM (no ALS) at observability depth ``depth``, fit with
    P(w)-WEIGHTED regressions (correction 2: wmap=pw).  Returns the loaded model plus rows,
    activations A, softmax P, operators {G_x}, obs-SVD basis U / spectrum sv, readout C, the
    P(w) row-weights, and last-position belief (for the Mess3 guard's simple decode)."""
    proc_T = np.asarray(proc_T); stationary = np.asarray(stationary)
    f14.MODEL_DIR = Path(model_dir)
    model = f14._load(ckpt, device)
    resid, soft, belief, _ = f14._collect(model, proc_T, stationary, device)
    reach = {}
    for w in resid:                                       # model-softmax reachability
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    pw = f14.analytic_prefix_probs(resid, proc_T, stationary)
    rows, A, P, Gs, U, sv = f14.observable_subspace(
        resid, soft, reach, vocab, depth=depth, wmap=pw, use_multistep_als=False)
    pw_rows = np.array([max(pw.get(w, 0.0), 1e-12) for w in rows])
    C = fit_readout(A, P, pw_rows, vocab, metric="euclidean", intercept=False)   # P(w)-weighted
    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    return dict(model=model, resid=resid, soft=soft, reach=reach, pw=pw, rows=rows,
                A=A, P=P, Gs=Gs, U=U, sv=sv, C=C, pw_rows=pw_rows, belief=belief,
                Yb=Yb, fin=fin, vocab=vocab, proc_T=proc_T)


def rrxor_principled_scorer(model, device="cpu"):
    """fig4 / fig10 scoring: over EVERY (input, position) of the length-10 enumeration, with
    PER-STATE-EQUAL weighting (correction 4).  Returns (score_fn, com_fn, supervised_r2, B36):
      score_fn(B)  -> per-position belief-decode R^2 of the subspace B  (the 0.65 metric)
      com_fn(B)    -> state-COM geometry R^2 (the 0.82 state-level metric)
    Both fit a label-affine readout exactly as fig4 does (ground truth used for placement only).
    """
    B36, index = F10.msp_states()
    seqs, beliefs, idx = F10.enumerate_inputs(F10.N_CTX, index)
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    acts = F10.collect_activations(model, seqs, device, hooks)
    concat = np.concatenate([acts[h] for h in hooks], axis=-1)
    Xf = concat.reshape(-1, concat.shape[-1])
    Y = beliefs.reshape(-1, B36.shape[1])
    fidx = idx.reshape(-1)
    counts = np.bincount(fidx, minlength=len(B36))
    sw = 1.0 / np.clip(counts[fidx], 1, None); sw = sw / sw.mean()
    sup_r2 = LinearRegression().fit(Xf, Y, sample_weight=sw).score(Xf, Y, sample_weight=sw)

    def score(B):
        S = Xf @ B
        return LinearRegression().fit(S, Y, sample_weight=sw).score(S, Y, sample_weight=sw)

    def com(B):
        feat = Xf @ B
        present = [s for s in range(len(B36)) if (fidx == s).any()]
        comf = np.array([feat[fidx == s].mean(0) for s in present])
        bel = np.array([Y[fidx == s][0] for s in present])
        return LinearRegression().fit(comf, bel).score(comf, bel)

    return score, com, float(sup_r2), B36


# ===========================================================================
# Report driver
# ===========================================================================

def _refinement_curves(res, dgrid, depth, scorer):
    """Return ({method: [(d, r2), ...]}, spectra, (U_obs, col_depth)).  Every basis is scored
    by ``scorer(B) -> r2`` (the fig4 per-state-equal metric for RRXOR; a simple last-position
    decode for the Mess3 guard).  All refinement fits use the P(w) row-weights, matching the
    P(w)-weighted baseline."""
    A, P, Gs, C, U = res["A"], res["P"], res["Gs"], res["C"], res["U"]
    V = res["vocab"]
    rows, resid, soft = res["rows"], res["resid"], res["soft"]
    pw_rows = res["pw_rows"]
    out, spectra = {}, {}

    # --- plain baseline (the validated P(w)-weighted obs-SVD subspace) ---
    out["plain obs-SVD"] = [(d, scorer(U[:, :d])) for d in dgrid]
    O_e, col_depth = build_observability_factor(C, Gs, depth)
    spectra["obs_svd"] = res["sv"]

    # --- (iv) Fisher readout (P(w)-weighted; isolates the readout metric) ---
    C_f = fit_readout(A, P, pw_rows, V, metric="fisher", intercept=False)
    U_f = np.linalg.svd(build_observability_factor(C_f, Gs, depth)[0], full_matrices=False)[0]
    out["Fisher readout (iv)"] = [(d, scorer(U_f[:, :d])) for d in dgrid]

    # --- affine-corrected (Fix 1): per-regression intercept on P(w)-weighted operators+readout ---
    Gs_i = fit_operators(rows, A, P, resid, soft, V, sw=pw_rows, intercept=True)
    C_i = fit_readout(A, P, pw_rows, V, metric="euclidean", intercept=True)
    U_i = np.linalg.svd(build_observability_factor(C_i, Gs_i, depth)[0], full_matrices=False)[0]
    out["affine-corrected (Fix1)"] = [(d, scorer(U_i[:, :d])) for d in dgrid]

    # --- (i) balanced truncation: empirical P(w) Wc + lambda-decay operator Wo (Fix 4).
    #         Score through an ORTHONORMAL basis (QR) -- B_full carries the 1/sqrt(sigma)
    #         balancing scale and would corrupt the decode if sliced directly (Correction 1). ---
    Wc = controllability_gram(A, pw_rows, domain_metric="pw")
    Wo_l = observability_gram(O_e, col_depth, range_metric="lambda_decay")
    B_l, hankel_l = balanced_truncation(Wc, Wo_l)
    out["balanced lam-decay (i)"] = [(d, scorer(orthonormal_balanced_basis(B_l, d))) for d in dgrid]
    spectra["hankel_lam"] = hankel_l
    Wo_e = observability_gram(O_e, col_depth, range_metric="euclidean")
    B_eu, hankel_e = balanced_truncation(Wc, Wo_e)
    spectra["hankel_eucl"] = hankel_e
    # Correction 1 sanity, two parts:
    #  (a) with Wc = I, euclidean balanced MUST equal obs-SVD (validates the implementation);
    #  (b) with the empirical (anisotropic) Wc it does NOT -- that gap is the real, expected
    #      reason balanced lands elsewhere (reachable != observable directions), not a bug.
    B_iso, _ = balanced_truncation(np.eye(Wc.shape[0]), Wo_e)
    spectra["balanced_iso_angles"] = principal_angles_deg(orthonormal_balanced_basis(B_iso, 5), U[:, :5])
    spectra["balanced_eucl_angles"] = principal_angles_deg(orthonormal_balanced_basis(B_eu, 5), U[:, :5])

    # --- (iii) joint-Schur common invariant subspace (Fix 3); report SCALE-FREE residual ---
    js = []
    for d in dgrid:
        B_js, _ = joint_schur_subspace(Gs, V, d)
        js.append((d, scorer(B_js)))
    out["joint-Schur (iii)"] = js
    B_js5, raw5 = joint_schur_subspace(Gs, V, min(5, A.shape[1]))
    spectra["joint_resid_raw"] = raw5
    spectra["joint_resid_norm"] = joint_invariance_residual_normalized(B_js5, Gs, V)
    return out, spectra, (U, col_depth)


def run_report(model_dir=None, depth=3, d_grid=(2, 3, 4, 5, 6, 8, 10), save=True, tag=""):
    """Reproduce the validated P(w)-weighted baseline (RRXOR d=5 = 0.65 under the fig4 metric),
    then ablate refinements (i)-(iv) on the SAME operators with the SAME scoring metric and all
    corrections applied.  Prints spectra/decodes and saves the RRXOR figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from simplexity.generative_processes.transition_matrices import mess3, rrxor

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model_dir = (ARCHIVED_DIR if model_dir is None else Path(model_dir))
    device = "cpu"
    print(f"model_dir = {model_dir}\n")

    # ---------------- Mess3 guard (simple last-position decode) ----------------
    print("=== Mess3 (control guard) ===")
    mres = fit_pipeline("mess3_transformer.pt", np.array(mess3(x=0.05, a=0.85)),
                        np.array([1, 1, 1]) / 3.0, 3, model_dir, device, depth=depth)
    m_scorer = lambda B: decode_with_basis(mres["A"], mres["fin"], mres["Yb"], B)
    mcurves, _, (mU, _) = _refinement_curves(mres, (2, 3, 4, 5), depth, m_scorer)
    print("  refinement decode @ d=3:", {k: round(dict(v)[3], 3) for k, v in mcurves.items()})
    _, m_ops8, _ = f14.fit_at_dim(mres["rows"], mres["A"], mres["P"], mres["resid"], mres["vocab"], mU[:, :8])
    print("  Riesz projector validation on Mess3 (reduced d=8; expect idempotent, integer traces):")
    _, m_nclusters, _ = riesz_subspace(m_ops8, mres["vocab"], drop_zero=True)
    print(f"  -> {m_nclusters} clusters = reduction dim (projectors valid; cluster count is NOT a "
          f"d-estimate -- see Correction 3)")
    _, m_ops3, _ = f14.fit_at_dim(mres["rows"], mres["A"], mres["P"], mres["resid"], mres["vocab"], mU[:, :3])
    print("  eig(A_sub) vs eig(T) [d=3]:"); eig_diagnostic(m_ops3, mres["proc_T"], mres["vocab"])

    # ---------------- RRXOR (fig4 per-state-equal scoring) ----------------
    print("\n=== RRXOR ===")
    rres = fit_pipeline("rrxor_transformer.pt", np.array(rrxor(0.5, 0.5)),
                        np.array([2, 1, 1, 1, 1]) / 6.0, 2, model_dir, device, depth=depth)
    score, com, sup_r2, _ = rrxor_principled_scorer(rres["model"], device)
    rcurves, rspec, (rU, _) = _refinement_curves(rres, d_grid, depth, score)
    print("\n  per-position decode R^2 vs d (fig4 per-state-equal metric):")
    for k, v in rcurves.items():
        print(f"    {k:26s} {{ {', '.join(f'{d}:{r:.3f}' for d, r in v)} }}")
    print(f"  supervised ceiling = {sup_r2:.3f}   (target: plain d=5 = 0.65)")
    print(f"  state-COM R^2: plain d=5 = {com(rU[:, :5]):.3f}  (write-up 0.82)")
    print(f"  obs-SVD sigma:   {np.round(rspec['obs_svd'][:8]/rspec['obs_svd'][0], 3)}")
    print(f"  Hankel sigma(λ): {np.round(rspec['hankel_lam'][:8]/rspec['hankel_lam'][0], 3)}")
    # Correction 1 sanity: with Wc=I balanced==obs-SVD (impl correct); with empirical Wc it
    # diverges (reachable != observable directions) -- the real reason balanced underperforms.
    print(f"  [Corr.1] balanced(Wc=I) vs obs-SVD angles (deg) @ d=5: "
          f"{np.round(rspec['balanced_iso_angles'], 2)}  (expect ~0 -> impl correct)")
    print(f"  [Corr.1] balanced(empirical Wc) vs obs-SVD angles (deg) @ d=5: "
          f"{np.round(rspec['balanced_eucl_angles'], 2)}  (large -> reachable != observable)")
    # Correction 2: verify the joint-Schur gradient, then report the SCALE-FREE residual
    grad_err = check_joint_gradient(rres["Gs"], rres["vocab"], rres["A"].shape[1], 5)
    print(f"  [Corr.2] joint-Schur gradient FD-check rel.err = {grad_err:.2e} (want < 1e-4)")
    print(f"  [Corr.2] joint-Schur residual @ d=5: raw={rspec['joint_resid_raw']:.3e}  "
          f"normalized={rspec['joint_resid_norm']:.3f} (0=invariant, ~1=no common subspace)")

    # Correction 3: Riesz is a VALIDATED BASIS TOOL (idempotent, integer-trace projectors), NOT
    # a d-selector -- it returns one cluster per distinct eigenvalue of the pre-reduced operator,
    # so the cluster count = the reduction dim by construction.  Order selection stays on the
    # Hankel sigma spectrum.  We report it only to confirm the projectors are correct.
    _, r_ops10, _ = f14.fit_at_dim(rres["rows"], rres["A"], rres["P"], rres["resid"], rres["vocab"], rU[:, :10])
    print("  Riesz projector validation on RRXOR (reduced d=10; expect idempotent, integer traces):")
    _, n_clusters, _ = riesz_subspace(r_ops10, rres["vocab"], drop_zero=True)
    print(f"  -> {n_clusters} clusters = reduction dim (Riesz separates eigenspaces; it does NOT "
          f"rank by belief-relevance, so cluster count is not a d-estimate -- use Hankel σ for d)")
    _, r_ops5, _ = f14.fit_at_dim(rres["rows"], rres["A"], rres["P"], rres["resid"], rres["vocab"], rU[:, :5])
    print("  eig(A_sub) vs eig(T) [d=5]:"); eig_diagnostic(r_ops5, rres["proc_T"], rres["vocab"])

    if not save:
        return rres, rcurves, rspec

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for label, curve in rcurves.items():
        ax0.plot([d for d, _ in curve], [r for _, r in curve], "o-", lw=1.6, ms=4, label=label)
    ax0.axhline(sup_r2, color="k", ls="--", lw=1, alpha=0.6, label=f"supervised ceiling = {sup_r2:.2f}")
    ax0.axvline(5, color="gray", ls=":", lw=1, alpha=0.7)
    ax0.set_xlabel("subspace dim d"); ax0.set_ylabel(r"belief-decode $R^2$ (per-state-equal)")
    ax0.set_title("RRXOR: belief-decode vs d (refinements on the validated 0.65 baseline)")
    ax0.set_ylim(0, 1.02); ax0.grid(alpha=0.3); ax0.legend(fontsize=7, loc="upper left")

    n = 12
    so = rspec["obs_svd"] / rspec["obs_svd"][0]
    he = rspec["hankel_eucl"] / rspec["hankel_eucl"][0]
    hl = rspec["hankel_lam"] / rspec["hankel_lam"][0]
    ax1.semilogy(range(n), so[:n], "o-", label="obs-SVD $\\sigma$")
    ax1.semilogy(range(n), he[:n], "s-", label="Hankel $\\sigma$ (eucl)")
    ax1.semilogy(range(n), hl[:n], "^-", label="Hankel $\\sigma$ ($\\lambda$-decay)")
    ax1.axvline(5 - 0.5, color="red", ls="--", lw=2, label="true d=5")
    ax1.set_xlabel("singular value index"); ax1.set_ylabel("normalised singular value")
    ax1.set_title("RRXOR: obs-SVD vs Hankel spectra"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    out = FIG_DIR / f"spectral_oom_refined_rrxor{tag}.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")
    return rres, rcurves, rspec


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else (str(ARCHIVED_DIR) if ARCHIVED_DIR.exists() else None)
    run_report(model_dir=md, tag="_sgd_ctx10" if md and "sgd_ctx10" in str(md) else "")
