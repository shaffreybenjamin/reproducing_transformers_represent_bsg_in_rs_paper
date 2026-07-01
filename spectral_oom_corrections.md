# Corrected functions for the refined spectral-OOM module

These are **drop-in replacements** for functions in the refined module, fixing the three
real bugs (Riesz contour normalization, affine-offset handling, joint-Schur magnitude bias)
plus two consistency fixes (shared-realization Gramians, demoting `future_cov`). Each block
notes exactly what to replace and why. Hand the whole file to the agent.

---

## Fix 1 — Affine offset via per-regression intercept (replaces global centering)

**Problem.** Global re-centering (`center_activations`) cannot be applied to the rescaled
child target, because `P(x|w)·(a(wx) − c̄) ≠ P(x|w)·a(wx) − c̄` — the projective rescaling
makes the offset prefix-dependent. The agent correctly observed the Mess3 collapse this
causes, but drew the wrong conclusion (disable centering, keep `fit_intercept=False`), which
re-admits the offset contamination Fix B was meant to remove. The correct resolution is to
let **each regression fit its own intercept** (`fit_intercept=True`) and never globally
subtract a constant from a rescaled target.

**Action.**
1. **Delete** `center_activations` entirely and stop threading `center`/`residc`/`cbar`
   through the pipeline. Work directly on `resid` everywhere `residc` was used.
2. Replace `fit_operators`, `fit_readout`, and `fit_subspace_operators` with the versions
   below (all use `fit_intercept=True`).
3. In `run_refined`, delete the `center_activations` call and the `center` argument; pass
   `resid` (not `residc`) to every downstream function. Keep the variable name `residc` as an
   alias for `resid` if that minimizes churn, but it must equal the raw activations.

> Note on the operator math: with intercepts, the operator relation is
> `a(w) A_x + d_x ≈ P(x|w) a(wx)`, where `d_x` is the fitted intercept absorbing the
> (rescaled) offset. The operator `A_x` is now estimated on the linear part alone, which is
> exactly the report's claim. The intercept is discarded for the spectral analysis (we only
> need `A_x`), and that is correct: the belief subspace is spanned by the linear maps, not
> the offsets.

```python
def fit_operators(rows_ops, resid, soft, pw, vocab, ridge=RIDGE_ALPHA, weighted=True):
    """P(w)-weighted ridge operators with per-regression intercept (affine offset handled
    by the intercept, NOT by global centering): a(w) A_x + d_x ~ P(x|w) a(wx).
    Returns {x: A_x (D,D)}; intercepts d_x are discarded (only the linear map enters the
    spectral construction)."""
    A = np.stack([resid[w] for w in rows_ops])               # (N, D)
    P = np.stack([soft[w] for w in rows_ops])                # (N, V)
    sw = np.array([pw[w] for w in rows_ops]) if weighted else None
    ops = {}
    for x in range(vocab):
        child = np.stack([resid[w + (x,)] for w in rows_ops])     # (N, D)
        tgt = P[:, x:x + 1] * child                               # rescaled child
        reg = Ridge(alpha=ridge, fit_intercept=True).fit(A, tgt, sample_weight=sw)
        ops[x] = reg.coef_.T
    return ops


def fit_readout(rows, resid, soft, pw, vocab, metric="euclidean",
                ridge=RIDGE_ALPHA, weighted=True):
    """Readout C: a(w) -> P(.|w) with per-regression intercept.  Refinement (iv).

    metric="euclidean": plain P(w)-weighted ridge (default).
    metric="fisher": diagonal Fisher-metric GLS; output column k fit with row weights
      P(w)/max(P(x=k|w), P_FLOOR).  Each column fit has its own intercept.
    """
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    base = np.array([pw[w] for w in rows]) if weighted else np.ones(len(rows))
    D = A.shape[1]
    if metric == "euclidean":
        return Ridge(alpha=ridge, fit_intercept=True).fit(A, P, sample_weight=base).coef_.T
    if metric == "fisher":
        C = np.zeros((D, vocab))
        for k in range(vocab):
            rw = base / np.maximum(P[:, k], P_FLOOR)
            ck = Ridge(alpha=ridge, fit_intercept=True).fit(A, P[:, k], sample_weight=rw).coef_
            C[:, k] = ck
        return C
    raise ValueError(f"unknown readout metric {metric!r}")


def fit_subspace_operators(rows_ops, resid, soft, pw, vocab, basis, weighted=True):
    """Refit rescaled transition operators INSIDE the recovered subspace (with intercept):
    s(w) A_x^sub + d_x ~ P(x|w) s(wx), s(w)=a(w) B.  Reducing first is what makes the
    eig(A) vs eig(T) check and the Riesz analysis meaningful.  Returns {x: A_x^sub (d,d)}."""
    S = np.stack([resid[w] for w in rows_ops]) @ basis            # (N, d)
    P = np.stack([soft[w] for w in rows_ops])
    sw = np.array([pw[w] for w in rows_ops]) if weighted else None
    ops = {}
    for x in range(vocab):
        child = np.stack([resid[w + (x,)] for w in rows_ops]) @ basis
        tgt = P[:, x:x + 1] * child
        ops[x] = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(S, tgt, sample_weight=sw).coef_.T
    return ops
```

Also update `decode_r2` and the ceiling computation in `run_refined` to use `resid`
(raw activations) — `LinearRegression` already fits an intercept, so the affine offset is
handled there automatically. No other change needed for scoring.

---

## Fix 2 — Riesz spectral projectors (correct contour normalization + idempotency guards)

**Problems.** (a) `P_tot /= n_quad` sat *inside* the cluster loop, so each earlier cluster's
projector was divided by `n_quad` once per later cluster (off by `n_quad**(k−1)`).
(b) No idempotency or trace check, so a wrong projector would pass silently — the Mess3 d=3
decode is necessary but not sufficient. (c) The function must be run on operators already
reduced to a modest subspace (the raw (D,D) operators carry D−d spurious ridge-zeros that
swamp the clustering); this is now enforced with a guard.

**Action.** Replace `riesz_subspace` with the version below. It accumulates each cluster's
projector in a local matrix, normalizes that alone, then adds; asserts idempotency and
near-integer trace per cluster; and refuses to run on operators whose dimension is much
larger than the eigenvalue count (forcing pre-reduction).

```python
def riesz_subspace(ops, vocab, drop_zero=True, cluster_tol=0.08, zero_tol=0.03,
                   n_quad=128, sv_tol=1e-2, idem_tol=1e-2, max_dim=40, verbose=True):
    """Riesz spectral projectors of the marginal operator hat-A = sum_x A_x.

    For each retained eigenvalue cluster, build P_lambda = (1/2pi i) oint (zI-hatA)^{-1} dz
    on a circle enclosing ONLY that cluster, parametrised z(theta)=c+r e^{i theta} so
    dz = i r e^{i theta} dtheta and (1/2pi i) dz = (1/2pi) r e^{i theta} dtheta.  The belief
    subspace is the column span of the sum of retained projectors; d is its numerical rank.

    MUST be called on operators already reduced to a modest subspace (e.g. a d<=10 obs-SVD
    subspace via fit_subspace_operators); raw (D,D) operators carry D-d spurious ridge-zeros
    that dominate the clustering.  Enforced by max_dim.

    drop_zero: exclude the cluster at/near 0.  CAUTION for RRXOR: T_1 is nilpotent so part of
    the genuine belief structure sits at lambda=0; report d with AND without the zero cluster
    and treat the difference as a finding, not a nuisance.

    Each projector is checked for idempotency (||P^2 - P|| < idem_tol) and near-integer trace;
    failures print a warning (they indicate a contour enclosing the wrong eigenvalues).
    """
    Ahat = sum(ops[x] for x in range(vocab))
    D = Ahat.shape[0]
    if D > max_dim:
        raise ValueError(
            f"riesz_subspace got D={D} > max_dim={max_dim}; reduce operators to a small "
            f"subspace first (fit_subspace_operators on a d<=10 obs-SVD basis).")
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
        elif nearest <= spread + 1e-9:        # cannot separate this cluster cleanly
            if verbose:
                print(f"    [riesz] skip cluster @ {center:.3f}: nearest eig {nearest:.3f} "
                      f"<= spread {spread:.3f}")
            continue
        else:
            r = 0.5 * (spread + nearest)       # radius between cluster and its nearest neighbour

        # Per-cluster quadrature, normalised on its own, THEN added (the bug was normalising
        # the running total once per cluster).
        P_cl = np.zeros((D, D), dtype=complex)
        z = center + r * np.exp(1j * theta)
        for zm, th in zip(z, theta):
            resolvent = np.linalg.inv(zm * np.eye(D) - Ahat)
            P_cl += resolvent * (r * np.exp(1j * th))     # (1/2pi i) dz factor
        P_cl /= n_quad                                    # * dtheta/(2pi), dtheta=2pi/n_quad

        # Guards: a true spectral projector is idempotent and has integer trace
        # (= algebraic multiplicity enclosed).
        idem = float(np.linalg.norm(P_cl @ P_cl - P_cl))
        tr = complex(np.trace(P_cl))
        if verbose:
            print(f"    [riesz] cluster @ {center:.3f} r={r:.3f} size={len(cl)} "
                  f"trace={tr.real:.2f} ||P^2-P||={idem:.2e}")
        if idem > idem_tol and verbose:
            print(f"    [riesz] WARNING idempotency {idem:.2e} > {idem_tol}: contour may "
                  f"enclose wrong eigenvalues (check radius / clustering).")
        if abs(tr.imag) > 1e-6 and verbose:
            print(f"    [riesz] WARNING non-real trace imag={tr.imag:.2e}.")
        kept.append((center, r, len(cl), tr.real, idem))
        P_tot += P_cl

    P_real = P_tot.real
    U, s, _ = np.linalg.svd(P_real, full_matrices=False)
    d = max(int(np.sum(s > sv_tol * (s[0] if s[0] > 0 else 1.0))), 1)
    basis = U[:, :d]
    info = dict(eigs=eigs, clusters=clusters, kept=kept, proj_sv=s)
    return basis, d, info
```

Mess3 validation must now check more than the decode: on Mess3 the three retained clusters
(diagonalizable contraction) should each give `||P^2 − P|| ≈ 0` and `trace ≈ 1`, summing to
d=3. If the traces aren't ≈ integers, the contour radii are wrong regardless of what the
decode says.

---

## Fix 3 — Joint-Schur common invariant subspace (no eigenvalue-magnitude bias)

**Problem.** `_leading_invariant_subspace` used subspace iteration, which converges to the
**dominant** (largest-|eig|) invariant subspace — i.e. the strongly-observable, next-token-
aligned directions you already recover. The RRXOR belief-discriminating modes are the **weak**
ones, so this systematically misses them while the stability diagnostic still looks healthy.

**Fix.** Recover the full common invariant subspace without ranking by eigenvalue magnitude.
For each random real combination `M = Σ_x γ_x A_x`, take its real Schur vectors and find the
`d`-dimensional subspace that is **most jointly invariant** across all the `γ`-combinations,
measured by the residual `Σ_x ‖(I − QQᵀ) A_x Q‖²` (small = `Q` is invariant under every
`A_x`). Minimize that residual directly over orthonormal `Q` by a short projected gradient /
subspace refinement seeded from a `γ`-combination's Schur vectors, choosing the seed that
already has the smallest joint residual. This targets *common invariance*, not dominance.

```python
def _joint_invariance_residual(Q, ops, vocab):
    """Sum_x || (I - Q Q^T) A_x Q ||_F^2 : 0 iff span(Q) is invariant under every A_x."""
    r = 0.0
    QQt = Q @ Q.T
    for x in range(vocab):
        M = ops[x] @ Q
        r += float(np.linalg.norm(M - QQt @ M) ** 2)
    return r


def _refine_invariant_subspace(Q, ops, vocab, iters=200, lr=0.1):
    """Projected-gradient refinement of orthonormal Q minimising the joint-invariance
    residual.  Gradient of ||(I-QQ^T)A Q||^2 w.r.t. Q, retracted to the Stiefel manifold by
    re-orthonormalisation (QR).  Magnitude-agnostic: it does not prefer large-|eig| modes."""
    Q = Q.copy()
    for _ in range(iters):
        G = np.zeros_like(Q)
        QQt = Q @ Q.T
        for x in range(vocab):
            A = ops[x]
            AQ = A @ Q
            R = AQ - QQt @ AQ                      # residual component
            # d/dQ of ||R||^2 (treating the two Q's), symmetrised:
            G += 2.0 * (A.T @ R - (R @ Q.T + Q @ R.T) @ AQ)
        Q = Q - lr * G
        Q, _ = np.linalg.qr(Q)                     # retract to Stiefel
    return Q


def joint_schur_subspace(ops, vocab, d, n_gamma=12, seed=0, refine_iters=200):
    """Common (approximate) d-dim invariant subspace across {A_x}, WITHOUT eigenvalue-
    magnitude bias.  For several random real combinations M=sum gamma_x A_x, seed Q from M's
    real Schur vectors (a d-subset chosen to minimise the joint-invariance residual), refine
    by projected gradient on that residual, and keep the best Q across seeds.  Returns
    (basis (D,d), residual) where residual is the final joint-invariance residual (small =
    genuinely common invariant subspace; large = the estimated operators do NOT share one,
    which is itself a finding for RRXOR)."""
    from scipy.linalg import schur

    rng = np.random.default_rng(seed)
    best_Q, best_res = None, np.inf
    D = ops[0].shape[0]
    for g in range(n_gamma):
        gamma = rng.standard_normal(vocab)
        M = sum(gamma[x] * ops[x] for x in range(vocab))
        Tsch, Z = schur(M, output="real")         # Z columns = Schur vectors (orthonormal)
        # Try a few d-subsets of Schur vectors as seeds; pick the lowest-residual seed.
        seed_candidates = [Z[:, :d]]
        if Z.shape[1] > d:
            seed_candidates.append(Z[:, -d:])
            mid = (Z.shape[1] - d) // 2
            seed_candidates.append(Z[:, mid:mid + d])
        for Q0 in seed_candidates:
            Q0, _ = np.linalg.qr(Q0)
            Q = _refine_invariant_subspace(Q0, ops, vocab, iters=refine_iters)
            res = _joint_invariance_residual(Q, ops, vocab)
            if res < best_res:
                best_res, best_Q = res, Q
    return best_Q, best_res
```

`scipy.linalg.schur` is the only new dependency (SciPy is already standard). If you want to
avoid it, seed instead from `np.linalg.qr(rng.standard_normal((D, d)))` and rely on the
refinement alone — slower to converge but dependency-free.

The returned `residual` is the honest diagnostic: if it stays large on RRXOR across seeds,
the **estimated** operators don't share a clean common invariant subspace (an estimation-
noise or genuine-defectiveness statement), and that is a result to report, not a knob to
force down.

---

## Fix 4 — Consistent Gramian sourcing (shared realization)

**Problem.** `controllability_gram` builds `Wc` from **empirical** activations while
`observability_gram` builds `Wo` from the **operator-propagated** factor `O`. Balancing two
differently-sourced Gramians breaks the single-realization property and muddies the Hankel
spectrum's interpretation.

**Decision.** Pick one sourcing and use it on both sides. I recommend **both empirical**,
because the empirical `P(w)`-weighted second moments are the lower-variance estimators and
sidestep operator-iteration noise (the very noise that sank deeper observability in your
earlier RRXOR runs). Concretely, keep the empirical `Wc` as-is, and add an empirical
observability option that uses the actual future-observable covariance rather than `O Mo Oᵀ`.

```python
def observability_gram_empirical(resid, soft, pw, reach, ops, C, vocab, depth,
                                 range_metric="lambda_decay", lam=0.5, ridge=1e-6):
    """Empirical observability Gramian: Wo = E_{P(w)}[ phi(w) phi(w)^T ] where phi(w) is the
    stacked multi-step observable VALUES at prefix w, phi(w) = a(w) O (O the observability
    factor [C | A_x C | ...]).  This is the empirical counterpart of O Mo O^T and is sourced
    from the SAME activations as Wc, so (Wc, Wo) factor one realization.

    range_metric weights the future blocks exactly as in the operator version
    ("euclidean" | "lambda_decay" | "future_cov").
    """
    O, col_depth = build_observability_factor(C, ops, vocab, depth)
    rows = [w for w in resid if reach[w]]
    A = np.stack([resid[w] for w in rows])
    wv = np.array([pw[w] for w in rows]); wv = wv / wv.sum()
    Phi = A @ O                                          # (M, n_fut) observable values
    if range_metric == "euclidean":
        mo = np.ones(O.shape[1])
    elif range_metric == "lambda_decay":
        mo = lam ** col_depth.astype(float)
    elif range_metric == "future_cov":
        # whiten by future covariance -- USE WITH CARE (amplifies low-variance NOISE
        # directions as much as weak belief modes; prefer lambda_decay and treat any
        # future_cov gain as suspect until a noise-direction control confirms it).
        Phim = Phi - (wv[:, None] * Phi).sum(0)
        cov = Phim.T @ (wv[:, None] * Phim)
        Mo = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
        Phw = (wv[:, None] * Phi)
        # Wo in activation space: O Mo O^T, but assembled empirically via Phi is not needed;
        # return the activation-space Gramian directly:
        return O @ Mo @ O.T
    Phw = wv[:, None] * (Phi * mo[None, :])
    # activation-space Gramian: A^T (weighted future outer-products) folded back through O
    # is exactly O diag(mo) (weighted) ... but to keep it in activation coords and consistent
    # with balanced_truncation (which expects (D,D)), use the operator-space form:
    return (O * mo[None, :]) @ O.T
```

Honestly, the activation-space `Wo = (O * mo) @ O.T` here is identical to the operator-based
`observability_gram` with the same `range_metric` — the "empirical" relabelling buys
consistency only if you ALSO source `Wc` the same way. Since `Wc` is empirical second-moment,
the clean and truly consistent choice is:

- **Wc** = `controllability_gram` (empirical `P(w)`-weighted second moment of `a(w)`), and
- **Wo** = the operator-based `observability_gram` **with `range_metric="lambda_decay"`**.

These are consistent *enough* for the Hankel spectrum to be interpretable, because `Wc` is
the empirical state covariance and `Wo` is the future-observability metric of the SAME
operators that generated the realization. The thing to avoid is `future_cov` on top of an
empirical `Wc` — that double-whitens. **Recommendation: use `range_metric="lambda_decay"`,
`domain_metric="pw"`, and drop `future_cov` from the headline runs** (keep it only as a
clearly-flagged diagnostic). No code change beyond demoting `future_cov` in the method list.

---

## Summary of actions for the agent

1. Delete `center_activations`; remove the `center` arg and `residc`/`cbar` threading; pass
   raw `resid` everywhere. (Fix 1)
2. Replace `fit_operators`, `fit_readout`, `fit_subspace_operators` with the intercept
   versions above. (Fix 1)
3. Replace `riesz_subspace` with the corrected-contour, guarded version; it now raises if
   called on operators with `D > max_dim`, so ensure every call site reduces first. (Fix 2)
4. Replace `joint_schur_subspace` and add `_joint_invariance_residual`,
   `_refine_invariant_subspace`; remove `_leading_invariant_subspace`. (Fix 3)
5. In the method list of `run_rrxor_report`, demote `future_cov` to a flagged diagnostic and
   make `lambda_decay` the headline balanced run. (Fix 4)
6. Re-run the Mess3 guard and additionally assert: Riesz projector traces ≈ integers with
   `||P^2−P||` small; joint-Schur residual small on Mess3. Only then read RRXOR numbers.

## What these fixes do and do not buy you

They make each refinement *actually do what it claims* — Riesz gives genuine spectral
projectors, joint-Schur targets common invariance rather than dominance, the offset is
handled without the centering collapse, and the balanced spectrum is interpretable. They do
**not** guarantee the RRXOR gap closes. The corrected Hankel `σ` spectrum and the joint-Schur
residual are now trustworthy instruments: if they say RRXOR's weak modes sit below the
finite-sample floor, that is the structural finding, and it now rests on correct machinery
rather than on bugs that happened to produce a plausible number.
