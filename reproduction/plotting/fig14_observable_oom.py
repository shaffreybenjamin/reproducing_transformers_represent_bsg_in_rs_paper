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


def _sample_process(T, pi, N, L, seed=0):
    """Sample N length-L token sequences from the true OOM (T substochastic operators, pi start)."""
    rng = np.random.default_rng(seed); V, ns = T.shape[0], T.shape[1]
    st = rng.choice(ns, N, p=pi); s = np.empty((N, L), np.int64)
    for t in range(L):
        Px = np.stack([T[x][st].sum(1) for x in range(V)], 1); Px = np.clip(Px, 1e-12, None); Px /= Px.sum(1, keepdims=True)
        s[:, t] = (rng.random(N)[:, None] < np.cumsum(Px, 1)).argmax(1)
        x = s[:, t]; tr = T[x, st] / T[x, st].sum(1, keepdims=True)
        st = (rng.random(N)[:, None] < np.cumsum(tr, 1)).argmax(1)
    return s


def _collect_resid_seqs(model, seqs, hooks, device, bs=8192):
    out = []
    for i in range(0, len(seqs), bs):
        inp = torch.from_numpy(seqs[i:i + bs]).to(device)
        with torch.no_grad():
            _, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
        out.append(np.concatenate([c[h].cpu().numpy() for h in hooks], -1))
    return np.concatenate(out, 0)


def _future_onehot(fut, H, V):
    n = len(fut); fe = []
    for k in range(1, H + 1):
        idx = np.zeros(n, np.int64)
        for j in range(k):
            idx = idx * V + fut[:, j]
        oh = np.zeros((n, V ** k)); oh[np.arange(n), idx] = 1.0; fe.append(oh)
    return np.hstack(fe)


def sample_readout_features(model, T, pi, hooks, vocab, horizon, n_seq=60000, device="cpu", seed=0):
    """Build (X, F) for the SAMPLED multistep readout: draw n_seq sequences from the true process,
    run them through the model, and pair each position's activation with the ONE-HOT of its observed
    future window (k-grams up to `horizon`). Many stochastic futures per prefix -> full-rank future
    covariance (fixes the enumeration overfitting). Returns (X (n,D), F (n,nF))."""
    L = model.cfg.n_ctx
    seqs = _sample_process(np.asarray(T), np.asarray(pi), n_seq, L, seed)
    acts = _collect_resid_seqs(model, seqs, hooks, device)
    Xs, Fs = [], []
    for t in range(L - horizon):
        Xs.append(acts[:, t, :]); Fs.append(_future_onehot(seqs[:, t + 1:t + 1 + horizon], horizon, vocab))
    return np.vstack(Xs), np.vstack(Fs)


def cca_multistep_readout(rows, A, soft, vocab, sw=None, horizon=4, rank=None,
                          cc_threshold=0.05, ridge_a=1e-1, ridge_f=1e-3, sampled=None):
    """CCA/RRR MULTI-STEP readout -- a belief-richer drop-in replacement for the 1-step ridge C.

    The ridge C regresses a(w) -> next-token softmax, so it is blind to belief directions that
    barely move the *next* token (next-token degeneracy -- the RRXOR weak mode). This instead
    finds the activation-space directions whose CANONICAL CORRELATION with the MULTI-step future
    (k-grams up to `horizon`) is highest. CCA whitening ranks by correlation, not magnitude, so a
    weakly-but-consistently predictive mode survives.

    sampled (preferred): (X, F) from sample_readout_features -- many stochastic futures per prefix
    => full-rank future covariance, no overfitting. If None, falls back to ENUMERATION (only rows
    with len(w)+horizon <= MAX_LEN; few long-future rows -> overfits at short context).
    `rank` = readout directions to keep; None => gap-based elbow on the canonical spectrum (>= vocab).
    Returns (C (D, rank), canonical_correlations); C is used exactly like the ridge C downstream."""
    if sampled is not None:
        Asub, F = sampled
        wv = np.ones(len(Asub), float)
        D = Asub.shape[1]
    else:
        from itertools import product as iproduct
        grams = [s for k in range(1, horizon + 1) for s in iproduct(range(vocab), repeat=k)]
        keep = [i for i, w in enumerate(rows) if len(w) + horizon <= MAX_LEN]
        if not keep:
            raise ValueError(f"cca readout: no rows with horizon {horizon} available (MAX_LEN={MAX_LEN})")
        F = np.zeros((len(keep), len(grams)))
        for r, i in enumerate(keep):
            w = rows[i]
            for j, s in enumerate(grams):
                p, cur = 1.0, w
                for t in s:
                    if soft[cur][t] <= EPS:
                        p = 0.0; break
                    p *= soft[cur][t]; cur = cur + (t,)
                F[r, j] = p
        Asub = A[keep]
        wv = np.ones(len(keep)) if sw is None else np.asarray(sw, float)[keep]
        D = A.shape[1]
    wv = wv / wv.sum()
    Am = Asub - (wv[:, None] * Asub).sum(0)
    Fm = F - (wv[:, None] * F).sum(0)
    D = A.shape[1]
    Cxx = (wv[:, None] * Am).T @ Am + ridge_a * np.eye(D)
    Cff = (wv[:, None] * Fm).T @ Fm + ridge_f * np.eye(F.shape[1])
    Cxf = (wv[:, None] * Am).T @ Fm
    ex, Ux = np.linalg.eigh(Cxx); Wx = Ux @ np.diag(ex ** -0.5) @ Ux.T
    ef, Uf = np.linalg.eigh(Cff); Wf = Uf @ np.diag(np.clip(ef, 1e-12, None) ** -0.5) @ Uf.T
    U, s, _ = np.linalg.svd(Wx @ Cxf @ Wf, full_matrices=False)
    if rank is None:
        # gap-based elbow on the canonical-correlation spectrum (the CCA d-selection), floor = vocab;
        # ignore directions already below cc_threshold so the gap is found among real modes.
        sr = s[:max(2, int(np.sum(s > cc_threshold)) + 1)]
        ratios = sr[:-1] / np.clip(sr[1:], 1e-12, None)
        rank = int(max(vocab, np.argmax(ratios) + 1)) if len(ratios) else vocab
    rank = min(rank, len(s))
    C = Wx @ U[:, :rank]
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-12)   # unit columns (scale-match G_x C)
    return C, s


def observable_subspace(resid, soft, reach, vocab, depth=3, wmap=None, use_multistep_als=True,
                       als_max_order=2, als_n_iter=20, als_lambda_decay=0.5,
                       readout="ridge", cca_horizon=4, cca_rank=None, cca_sampled_data=None):
    """Observability matrix O = [C, G_x C, G_x G_y C, ...]; its SVD gives candidate
    belief directions ordered by strength. Columns of O are the rescaled multi-step
    observables P(x..|w)*P(.|w x..), each LINEAR in belief, spanning belief as depth grows.

    wmap (optional): {prefix: P(w)} sample weights for the C/G_x ridge regressions, so the
    operators -- and hence the subspace -- are estimated toward high-probability prefixes
    (the post-quantum paper's P(w)-weighting). None => uniform (backwards-compatible).

    use_multistep_als (default True): refine operators using multi-step ALS after initial fit.
    als_max_order, als_n_iter: parameters for the ALS refinement.

    readout: "ridge" (default, exact 1-step softmax readout), "cca" (replace C with the CCA/RRR
    MULTI-step readout, which sees belief modes the next-token readout misses), or "ridge+cca"
    (concatenate both, scale-matched: the exact 1-step readout for the strongly-observable modes
    PLUS the CCA directions for the weak ones -- monotone-safe, can't hurt). Operators and the
    rest of the pipeline are identical for all three."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows]); P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T   # (D, V) 1-step readout
    if readout in ("cca", "ridge+cca"):
        C_cca, cca_cc = cca_multistep_readout(rows, A, soft, vocab, sw=sw, horizon=cca_horizon,
                                              rank=cca_rank, sampled=cca_sampled_data)
        src = "sampled" if cca_sampled_data is not None else "enumeration"
        print(f"  [cca-readout/{src}] horizon={cca_horizon} rank={C_cca.shape[1]} canonical-corr={np.round(cca_cc[:8],3)}")
        if readout == "cca":
            C = C_cca
        else:                         # ridge+cca: exact 1-step readout (scale-matched) + CCA directions
            Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-12)
            C = np.hstack([Cn, C_cca])
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


def observable_subspace_logit(resid, logit, reach, vocab, depth=3, wmap=None,
                              rescale=False, soft=None, intercept=True):
    """LOGIT ("energy-based" / EHMM) spectral-OOM -- the sister estimator to observable_subspace
    for networks whose residual stream is linear in the *logit*-based predictive vector rather
    than the *probability*-based belief (cf. the EHMM of Poncini's "computational mechanics for
    logits", and the fact that a transformer's logits are a LINEAR readout of the residual stream
    while its softmax is not).

    Two coherent changes vs the softmax estimator, both dictated by the EHMM update
    eta(wx)=eta(w)H^x (eq. 22 there), which is linear with NO normaliser:
      (1) readout C regresses a(w) -> LOGITS z(.|w)  (linear in eta), not the softmax;
      (2) operators are fit PLAIN, a(w) A_x ~ a(wx), with NO P(x|w) rescaling, because there is
          no projective normaliser to clear (set rescale=True, soft=... to recover the HMM-style
          rescaled target as a diagnostic).
    `intercept` absorbs the affine offset c per-regression (recommended; True). Everything
    downstream -- the observability stacking [C | A_x C | ...] and its SVD -- is IDENTICAL to
    observable_subspace, so the two are drop-in comparable. Returns (rows, A, Z, Gs, U, sv) with
    Z the stacked logits (the analogue of P)."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab))]
    A = np.stack([resid[w] for w in rows])
    Z = np.stack([logit[w] for w in rows])                      # (N, V) raw logits
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C = Ridge(alpha=RIDGE, fit_intercept=intercept).fit(A, Z, sample_weight=sw).coef_.T   # (D, V) logit readout
    Gs = []
    for x in range(vocab):
        child = np.stack([resid[w + (x,)] for w in rows])
        if rescale:
            if soft is None:
                raise ValueError("rescale=True needs soft=... (the model softmax) for the P(x|w) factor")
            P = np.stack([soft[w] for w in rows])
            tgt = P[:, x:x + 1] * child                         # HMM-style rescaled target (diagnostic)
        else:
            tgt = child                                         # EHMM: plain next-step activation
        Gs.append(Ridge(alpha=RIDGE, fit_intercept=intercept).fit(A, tgt, sample_weight=sw).coef_.T)

    cols, frontier = [C], [C]                                   # identical observability stacking
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols += nxt; frontier = nxt
    O = np.hstack(cols)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)
    return rows, A, Z, Gs, U, sv


def transition_linearity_diagnostic(resid, soft, reach, vocab, seed=0, train_frac=0.7, p_floor=1e-2):
    """Fair, same-target test of which update law the residual stream obeys (HMM vs EHMM),
    target-free (activations + model softmax only). Predict the SAME next activation a(wx) two
    ways and compare HELD-OUT R^2 on the identical target:
      EHMM:  a_hat(wx) = a(w) A^plain_x           (A^plain fit on  a(w) -> a(wx))
      HMM:   a_hat(wx) = a(w) A^resc_x / P(x|w)    (A^resc fit on  a(w) -> P(x|w) a(wx), then /P)
    Higher R^2 => that process class better describes the representation. Scored only on reachable
    transitions with P(x|w) > p_floor (the HMM inversion divides by P, ill-posed as P->0). Returns
    (r2_ehmm, r2_hmm)."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab))]
    A = np.stack([resid[w] for w in rows])
    perm = np.random.default_rng(seed).permutation(len(rows))
    ntr = int(train_frac * len(rows)); tr, te = perm[:ntr], perm[ntr:]
    num_e = den = num_h = 0.0
    for x in range(vocab):
        child = np.stack([resid[w + (x,)] for w in rows])
        P = np.array([soft[w][x] for w in rows])
        keep = P[te] > p_floor
        if keep.sum() == 0:
            continue
        Ae = Ridge(alpha=RIDGE, fit_intercept=True).fit(A[tr], child[tr])
        pe = Ae.predict(A[te][keep])
        Ah = Ridge(alpha=RIDGE, fit_intercept=True).fit(A[tr], (P[:, None] * child)[tr])
        ph = Ah.predict(A[te][keep]) / P[te][keep][:, None]
        y = child[te][keep]; sstot = ((y - y.mean(0)) ** 2).sum()
        num_e += ((y - pe) ** 2).sum(); num_h += ((y - ph) ** 2).sum(); den += sstot
    return 1.0 - num_e / den, 1.0 - num_h / den


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


def _cg_matrix(apply, B, X0, iters, tol):
    """Conjugate gradients for a symmetric-PD matrix operator  apply(X) = B,  with the
    unknown X a (D, D) matrix and the Frobenius inner product. Used to solve the summed
    -Sylvester normal equations  Σ_t P_t X Q_t + λ X = B  (each P_t, Q_t PSD => apply SPD)."""
    X = X0
    R = B - apply(X)
    Pp = R.copy()
    rs = float(np.sum(R * R))
    b2 = float(np.sum(B * B)) + 1e-30
    for _ in range(iters):
        if rs / b2 < tol * tol:
            break
        Ap = apply(Pp)
        denom = float(np.sum(Pp * Ap))
        if denom <= 1e-30:
            break
        alpha = rs / denom
        X = X + alpha * Pp
        R = R - alpha * Ap
        rs_new = float(np.sum(R * R))
        Pp = R + (rs_new / rs) * Pp
        rs = rs_new
    return X


def fit_operators_multistep_als(rows, A, P, resid, soft, vocab, Gs_init, max_order=2,
                                ridge_base=RIDGE, lambda_decay=0.5, n_iter=20, tol=1e-6, sw=None,
                                cg_iters=300, cg_tol=1e-9, verbose=True):
    r"""Refine operators {A_x} by minimising the *shared-operator* multi-step loss

        L = Sum_k  lam_k Sum_{x_1..x_k}  || a(w) A_{x_1}..A_{x_k}
                                            - P(x_1..x_k|w) a(w x_1..x_k) ||^2_{P(w)}

    over all words up to order K = max_order, with the SAME operators appearing in every
    term. The order-1 term anchors each operator's one-step duty; the higher-order terms
    constrain the products, which is where weakly-observable belief directions live (the
    multi-step version of RRXOR's next-token degeneracy). lam_k = lambda_decay**(k-1) is
    the order-k loss weight (decayed in k because high-order words are rare / noise-dominated),
    sw are the P(w) sample weights (None => uniform), ridge_base is the per-solve ridge.

    Optimised by alternating least squares (ALS): hold every operator but A_j fixed and
    solve the resulting linear sub-problem for A_j, starting from the one-step ridge fit.

    Why the previous implementation was wrong, and what is fixed here
    ----------------------------------------------------------------
    When A_j sits at a *non-terminal* position p of a length-k word x_1..x_k its term is

        ell(w) . A_j . M  ~=  y ,     ell(w) = a(w) prod_{q<p} A_{x_q},
                                      M       = prod_{q>p} A_{x_q}  (fixed),
                                      y       = P(x_1..x_k|w) a(w x_1..x_k).

    This is linear in A_j but it is a *Sylvester* least-squares problem, whose normal
    equations are

        Sum_t  P_t A_j Q_t  +  lam A_j  =  Sum_t R_t ,
        P_t = ell^T W ell,   Q_t = M M^T,   R_t = ell^T W y M^T.

    The earlier code instead stacked (a(w), y) and called an ordinary ridge fit
    a(w) A_j ~= y, dropping the trailing factor M entirely -- correct ONLY when A_j is the
    rightmost factor (M = I, so Q_t = I and the equation collapses to ordinary ridge). For
    every interior position that corrupts the operator. Here we keep the full M and solve
    the summed-Sylvester normal equations directly with matrix conjugate gradients, so every
    position -- and repeated occurrences of A_j within one word, e.g. (j, j) -- is handled
    correctly (each occurrence contributes one term, the other occurrence held at its
    current value, the usual ALS linearisation).

    Efficiency: the operator-free per-pattern moments  Sxx = a(w)^T W a(w)  and
    Sxy = a(w)^T W y  are precomputed ONCE per token-pattern; every ALS sweep then only
    forms small (D, D) operator products  L^T Sxx L  and  Sxy M^T, so per-sweep cost is
    independent of the number of prefixes.

    Returns the refined operator list (same shapes as Gs_init).
    """
    from itertools import product as iproduct

    Gs = [G.copy() for G in Gs_init]
    D = Gs[0].shape[0]
    w_all = np.ones(len(rows)) if sw is None else np.asarray(sw, dtype=float)

    # ---- one-time precompute: for every token pattern s (|s| = 1..K) collect the valid
    #      (parent, rescaled-child) pairs and reduce them to the fixed moments Sxx, Sxy.
    #      Validity / rescale / child are data-only (independent of the operators), so this
    #      is done once; the operators enter only through L, M products built per sweep. ----
    pats = []
    for k in range(1, max_order + 1):
        ow = lambda_decay ** (k - 1)                       # order-k loss weight
        for s in iproduct(range(vocab), repeat=k):
            idx, resc, child = [], [], []
            for i, w in enumerate(rows):
                cur, r, ok = w, 1.0, True
                for t in s:
                    pr = soft[cur][t] if cur in soft else 0.0
                    nxt = cur + (t,)
                    if pr <= EPS or nxt not in resid:      # chain-rule rescale + reachability
                        ok = False; break
                    r *= pr; cur = nxt
                if ok:
                    idx.append(i); resc.append(r); child.append(resid[cur])
            if not idx:
                continue
            idx = np.array(idx)
            wts = w_all[idx][:, None]
            par = A[idx]                                    # (n, D)  ell_0 = a(w)
            tgt = np.array(resc)[:, None] * np.stack(child)  # (n, D)  P(s|w) a(ws)
            Sxx = par.T @ (wts * par)                       # (D, D)  operator-free Gram
            Sxy = par.T @ (wts * tgt)                       # (D, D)  operator-free cross moment
            Syy = float(np.sum(wts * tgt * tgt))            # scalar (one-step diagnostic only)
            pats.append(dict(s=s, k=k, ow=ow, Sxx=Sxx, Sxy=Sxy, Syy=Syy, wsum=float(wts.sum())))

    one_step = {p["s"][0]: p for p in pats if p["k"] == 1}

    def left_prod(s, p):
        M = None
        for q in range(p):
            M = Gs[s[q]] if M is None else M @ Gs[s[q]]
        return M                                            # None == identity

    def right_prod(s, p):
        M = None
        for q in range(p + 1, len(s)):
            M = Gs[s[q]] if M is None else M @ Gs[s[q]]
        return M                                            # None == identity

    def one_step_mse():
        """Weighted one-step MSE  Sum w || a(w) A_x - P(x|w) a(wx) ||^2 / Sum w  -- must not
        regress as higher orders are added (the shared-operator anchor)."""
        num = den = 0.0
        for x, pd in one_step.items():
            G = Gs[x]
            num += float(np.trace(G.T @ pd["Sxx"] @ G) - 2.0 * np.trace(G.T @ pd["Sxy"]) + pd["Syy"])
            den += pd["wsum"]
        return num / max(den, 1e-12)

    def solve_operator(j):
        """ALS sub-solve for A_j: assemble Sum_t P_t A_j Q_t + ridge A_j = Sum_t R_t over
        every (pattern, position) in which token j appears, then solve by matrix-CG."""
        PI = np.zeros((D, D))      # accumulated P_t for the M = I (rightmost) terms
        RHS = np.zeros((D, D))     # accumulated Sum_t R_t
        syl = []                   # (P_t, Q_t) for the genuine Sylvester (M != I) terms
        for pd in pats:
            s = pd["s"]
            for p in range(pd["k"]):
                if s[p] != j:
                    continue
                L = left_prod(s, p)
                Mr = right_prod(s, p)
                Sxx = pd["Sxx"] if L is None else L.T @ pd["Sxx"] @ L
                Sxy = pd["Sxy"] if L is None else L.T @ pd["Sxy"]
                Sxx = pd["ow"] * Sxx
                Sxy = pd["ow"] * Sxy
                if Mr is None:                              # rightmost factor: ordinary ridge block
                    PI += Sxx
                    RHS += Sxy
                else:                                       # interior factor: Sylvester block
                    syl.append((Sxx, Mr @ Mr.T))
                    RHS += Sxy @ Mr.T
        if not PI.any() and not syl:
            return Gs[j]

        def apply(X):
            Y = PI @ X + ridge_base * X
            for Pt, Q in syl:
                Y = Y + Pt @ X @ Q
            return Y

        return _cg_matrix(apply, RHS, Gs[j].copy(), cg_iters, cg_tol)

    init_mse = one_step_mse()
    if verbose:
        print(f"  [multistep-ALS] K={max_order}  patterns={len(pats)}  init one-step MSE={init_mse:.3e}")

    converged = False
    delta = mse = np.nan
    for it in range(n_iter):
        delta = 0.0
        for j in range(vocab):
            Gnew = solve_operator(j)
            delta = max(delta, float(np.linalg.norm(Gnew - Gs[j])))
            Gs[j] = Gnew
        mse = one_step_mse()
        ratio = mse / max(init_mse, 1e-30)
        if verbose and (it == 0 or (it + 1) % 5 == 0 or it == n_iter - 1):
            print(f"    [ALS iter {it+1:2d}/{n_iter}] delta={delta:.2e}  one-step MSE={mse:.3e} (ratio={ratio:.3f})")
        if ratio > 1.5 and verbose:
            print(f"    WARNING: one-step MSE degraded to {ratio:.1f}x initial; consider raising lambda_decay")
        if delta < tol:
            converged = True
            if verbose:
                print(f"  [multistep-ALS] converged after {it+1} iters (delta={delta:.2e})")
            break

    if verbose and not converged:
        print(f"  [multistep-ALS] stopped at {n_iter} iters (delta={delta:.2e}, one-step MSE ratio={mse/max(init_mse,1e-30):.3f})")
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


def _collect_logits(model, T, stationary, device):
    """Same enumeration as _collect, but ALSO keeps the model's raw pre-softmax LOGITS per
    prefix (needed by observable_subspace_logit; the per-prefix log-partition is lost in the
    softmax, so logits cannot be reconstructed from `soft`). Returns
    (resid, soft, logit, belief, pp)."""
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

    resid, soft, logit, belief, pp = {}, {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            lg = logits[:, -1, :].cpu().numpy()
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]; soft[w] = sm[j]; logit[w] = lg[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan); pp[w] = 0.0
    return resid, soft, logit, belief, pp


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
