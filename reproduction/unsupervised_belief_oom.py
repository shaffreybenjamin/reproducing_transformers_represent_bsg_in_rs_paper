"""Unsupervised activation-side OOM / DMD estimator for the Mess3 belief geometry.

Goal
----
Recover the belief-state geometry from a trained transformer's residual stream
*without ever regressing onto analytic belief targets*, and **without knowing
anything about the task or process** (no hand-supplied latent dimension, no
process-specific subspace). The only ingredients are (i) residual-stream
activations and (ii) the network's own softmax next-token probabilities.

The idea (rescaling trick)
--------------------------
The optimal belief update is projective:
    b(wx) = b(w) T^(x) / ( b(w) T^(x) 1 ),    where  b(w) T^(x) 1 = P(x|w).
Naive linear regression z(wx) ~ z(w) is therefore mis-specified. But the
normalizer P(x|w) is exactly what the transformer outputs as its softmax. So
multiply it through:
    z(w) A^(x)  ≈  P(x|w) z(wx)                              (now LINEAR)
and fit one operator A^(x) per token. Because activations are an (uncentered)
linear image of the belief, a(w) = b(w) Ψ, one has A^(x) = Ψ^{-1} T^(x) Ψ —
similar to the true per-token operator (HKZ Lemma 3: B_x = (U^T O) A_x (U^T O)^-1).
Hence eig(A^(x)) ≈ eig(T^(x)): a gauge-invariant validation needing no targets.

Two general fixes over the naive version (task/model-agnostic)
--------------------------------------------------------------
The naive estimator chose its working subspace by *maximum variance* (top-d SVD
of the activations) and scored rollout by *dividing* by P(x|w). Both are
task-agnostic to apply, but both are wrong in general:

(1) DYNAMICS-CONSISTENT SUBSPACE + ORDER SELECTION.
    Belief is a *low-variance* feature; the top variance directions of a
    residual stream are dominated by nuisance (position/norm) axes, and there is
    no clean singular-value gap to read the latent dimension from. Instead we
    pick the subspace by *predictive* structure, not variance:

      - Build past features X = a(w) and the stacked, rescaled future
        F = concat_x [ P(x|w) a(wx) ].  The operator law says F is a *linear*
        image of X, so the dynamically-relevant subspace is the one in which X
        predicts F. A (uncentered, ridged) canonical-correlation analysis
        between X and F WHITENS OUT raw variance and keeps only predictively
        shared directions — exactly HKZ/Balle's "SVD of the estimation matrix",
        moved onto activations. This is reduced-rank / subspace-identification
        (N4SID / two-stage-regression) and recovers the state subspace up to the
        gauge.
      - The latent dimension d (model order) is then chosen by HELD-OUT
        predictive consistency: sweep d, fit operators on a train split, score
        one-step prediction on a val split, and take the knee. This replaces the
        missing variance gap with a dynamics gap — the principled order-selection
        criterion when the activation spectrum has none. Nothing here uses the
        number of hidden states or any analytic target.

(2) CONSISTENT FIT/EVAL METRIC (no 1/P amplification).
    Operators are fit in the rescaled domain (minimize ||a(w)A_x - P a(wx)||).
    The honest one-step score is computed in the SAME domain and probability-
    weighted, so it does not divide by P(x|w) and blow up rare transitions. The
    old "divide by P" rollout R^2 is still reported for transparency, but it
    grades the operators on a 1/P-amplified scale the fit never optimized.

Data path (paper-faithful, general)
-----------------------------------
Following the Astera analysis code (`casper/analyses`: build_prefix_dataset +
weighted regression), we SAMPLE sequences, take every position, and deduplicate
by prefix; the row weight is the EMPIRICAL prefix frequency (occurrence count).
This uses no MSP enumeration and no analytic P(w) — only sampling — so it relies
only on knowledge available for any model. (Causal attention makes the per-prefix
activation exact, and the count is an unbiased estimate of P(w).)

What is and isn't unsupervised
------------------------------
- Subspace choice, order selection, operator fit, eigenvalue check, eval
  functional, one-step R^2: fully target-free (activations + softmax only).
- The final overlay on the canonical fractal, the RGB coloring, and the
  alignment-R^2 diagnostic use analytic beliefs *for display/scoring only*
  (the recovered positions/structure come entirely from the estimator).
"""

from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import grey_dilation
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)
from simplexity.generative_processes.transition_matrices import mess3

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
N_SEQUENCES = 60000     # sampled sequences (paper-style; replaces MSP enumeration)
BATCH = 2000
D_MAX = 8               # largest latent dimension considered in the order sweep
RIDGE = 1e-3            # relative ridge for the CCA whitening (numerical, generic)
VAL_FRAC = 0.3          # held-out fraction for dynamics-consistent order selection
KNEE_TOL = 0.01         # accept smallest d within this of the best held-out R^2
SEED = 0

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


# --------------------------------------------------------------------------- #
# Model + data
# --------------------------------------------------------------------------- #
def load_model(device):
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device  # checkpoint was trained on cuda; honor the local device (e.g. cpu)
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def collect_prefix_features_sampled(model, hmm, n_sequences, seq_len, batch, device):
    """Paper-style data path (Astera `casper/analyses`): SAMPLE sequences, take
    every position, and deduplicate by prefix. Because attention is causal, a
    position's activation / softmax / belief depend only on the prefix, so all
    occurrences of a prefix are identical and dedup is lossless. The row weight is
    the EMPIRICAL prefix frequency (occurrence count, normalised) — an unbiased
    estimate of P(w) obtained from sampling alone: no MSP enumeration and no
    analytic probabilities. Belief is recorded for display/scoring ONLY.

    Returns the same (resid, soft, belief, prefix_prob, max_len) interface as the
    enumerated collector, so the rest of the pipeline is unchanged.
    """
    layer = model.cfg.n_layers - 1
    init = jnp.array(hmm.initial_state)
    key = jax.random.PRNGKey(SEED)
    resid, soft, belief = {}, {}, {}
    count = defaultdict(int)
    done = 0
    while done < n_sequences:
        b = min(batch, n_sequences - done)
        key, bk = jax.random.split(key)
        gs = jnp.repeat(init[None, :], b, axis=0)
        data = generate_data_batch_with_full_history(gs, hmm, b, seq_len, bk, device=device)
        inputs = data["inputs"].long().to(device)
        beliefs = np.asarray(data["belief_states"])                      # (b, T, n_states)
        with torch.no_grad():
            logits, cache = model.run_with_cache(inputs, names_filter=f"blocks.{layer}.hook_resid_post")
        z = cache["resid_post", layer].cpu().numpy()                     # (b, T, d_model)
        p = torch.softmax(logits, dim=-1).cpu().numpy()                  # (b, T, vocab)
        toks = inputs.cpu().numpy()
        T = toks.shape[1]
        for si in range(b):
            pref = ()
            for pos in range(T):
                pref = pref + (int(toks[si, pos]),)
                count[pref] += 1
                if pref not in resid:                                    # first occurrence (exact)
                    resid[pref] = z[si, pos]
                    soft[pref] = p[si, pos]
                    belief[pref] = beliefs[si, pos]
        done += b
    total = sum(count.values())
    prefix_prob = {w: count[w] / total for w in resid}                   # empirical P(w)
    return resid, soft, belief, prefix_prob, seq_len


def build_transitions(resid, soft, prefix_prob, max_len, vocab):
    """Stack one row per prefix w that has *all* children enumerated.

    Returns
        X   (N, d_model)        past activation a(w), uncentered
        P   (N, vocab)          network softmax P(x|w)
        Yc  (N, vocab, d_model) child activations a(wx)
        Wt  (N,)                prefix probability P(w) (row weight)
    Requiring all children keeps the stacked future F = concat_x P(x|w) a(wx)
    well-defined; in Mess3 every internal MSP node has all 3 children.
    """
    rows = [
        w for w, _ in resid.items()
        if len(w) < max_len and all((w + (x,)) in resid for x in range(vocab))
    ]
    X = np.stack([resid[w] for w in rows])
    P = np.stack([[soft[w][x] for x in range(vocab)] for w in rows])
    Yc = np.stack([[resid[w + (x,)] for x in range(vocab)] for w in rows])
    Wt = np.array([prefix_prob[w] for w in rows])
    return rows, X, P, Yc, Wt


# --------------------------------------------------------------------------- #
# Dynamics-consistent subspace selection (the general fix)
# --------------------------------------------------------------------------- #
def _inv_sqrt(S):
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * vals ** -0.5) @ vecs.T


def predictive_cca(X, P, Yc, Wt, ridge=RIDGE):
    """Uncentered, probability-weighted CCA between past a(w) and rescaled future.

    F = concat_x [ P(x|w) a(wx) ]. We whiten both sides and SVD the cross-
    covariance; whitening divides out raw variance magnitude, so directions are
    ranked by *predictive correlation*, not variance. The canonical correlations
    (in [0,1]) reveal the latent dimension; their X-space directions are the
    candidate basis (ordered, so top-d gives the rank-d subspace).

    Uncentered (no mean subtraction) so the constant "belief sums to 1" direction
    is retained, keeping a(w)=b(w)Psi an exact *linear* (not affine) image.
    """
    N, m = X.shape
    F = (P[:, :, None] * Yc).reshape(N, -1)                     # (N, vocab*m)
    w = Wt / Wt.sum()
    Sxx = (X * w[:, None]).T @ X
    Sff = (F * w[:, None]).T @ F
    Sxf = (X * w[:, None]).T @ F
    Sxx += ridge * (np.trace(Sxx) / m) * np.eye(m)
    Sff += ridge * (np.trace(Sff) / F.shape[1]) * np.eye(F.shape[1])
    M = _inv_sqrt(Sxx) @ Sxf @ _inv_sqrt(Sff)                  # whitened cross-cov
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    dirs = _inv_sqrt(Sxx) @ U                                   # (m, m) canonical dirs (cols)
    # normalize each direction to unit L2 so projections are comparably scaled
    dirs = dirs / (np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-12)
    return dirs, s[:m]


def fit_operators(A, P, Bc, Wt):
    """Rescaled, probability-weighted least squares:  A_proj(w) Op ≈ P(x|w) B_proj(wx).

    A   (N, d)        projected past
    Bc  (N, vocab, d) projected children
    Returns {x: Op (d,d)} and the prob-weighted in-domain fit R^2 per token.
    """
    ops, fit_r2 = {}, {}
    sw = np.sqrt(np.clip(Wt, 1e-12, None))[:, None]
    for x in range(P.shape[1]):
        target = P[:, x, None] * Bc[:, x, :]
        Op, *_ = np.linalg.lstsq(A * sw, target * sw, rcond=None)
        ops[x] = Op
        pred = A @ Op
        wss = Wt[:, None]
        ss_res = (wss * (target - pred) ** 2).sum()
        ss_tot = (wss * (target - np.average(target, axis=0, weights=Wt)) ** 2).sum()
        fit_r2[x] = 1.0 - ss_res / ss_tot
    return ops, fit_r2


def fit_operators_iv(rows, resid, basis, P, Yc, Wt, mask=None, ridge=1e-6):
    """Errors-in-variables / two-stage (instrumental-variable) operator fit.

    The OLS fit regresses the target on the *noisy* regressor a(w); the
    activation is only an imperfect linear image of the true belief, so the
    regressor noise biases OLS toward zero (attenuation) and that bias compounds
    over depth in rollout. Two-stage least squares removes it:

      Stage 1 (denoise the regressor): predict a(w) from its HISTORY — the parent
        activation a(w[:-1]) — separately per last token (so the prediction is
        the clean one-step image b(w[:-1]) T^(last) Psi). The parent is a
        different forward pass, so its representation noise is independent of
        a(w)'s, which is exactly the instrument-validity condition.
      Stage 2: regress the rescaled target P(x|w) a(wx) on the *denoised* a_hat(w).

    This is the PSR two-stage regression / subspace-identification estimator and
    is statistically consistent (unlike OLS) under the EIV model. Fully
    target-free; needs only activations + softmax.
    """
    n = len(rows)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    vocab = P.shape[1]
    idx = np.array([i for i in range(n) if mask[i] and len(rows[i]) >= 2])
    Xw = np.stack([resid[rows[i]] for i in idx]) @ basis            # noisy regressor a(w)
    Zpar = np.stack([resid[rows[i][:-1]] for i in idx]) @ basis     # instrument a(parent)
    last = np.array([rows[i][-1] for i in idx])
    w = Wt[idx]
    sw = np.sqrt(np.clip(w, 1e-12, None))[:, None]
    d = basis.shape[1]
    rI = ridge * np.eye(d)

    # Stage 1: a_hat(w) = parent-predicted clean state, per last token.
    Xhat = np.zeros_like(Xw)
    for t in range(vocab):
        g = last == t
        if not g.any():
            continue
        Zg, Xg, swg = Zpar[g], Xw[g], sw[g]
        B1 = np.linalg.solve((Zg * swg).T @ (Zg * swg) + rI, (Zg * swg).T @ (Xg * swg))
        Xhat[g] = Zg @ B1

    # Stage 2: operators on the denoised regressor.
    ops = {}
    XtX = (Xhat * sw).T @ (Xhat * sw) + rI
    for x in range(vocab):
        Yx = P[idx, x, None] * (Yc[idx, x, :] @ basis)
        ops[x] = np.linalg.solve(XtX, (Xhat * sw).T @ (Yx * sw))
    return ops


def rescaled_r2(A, P, Bc, Wt, ops):
    """Consistent one-step score: in the *rescaled* domain, prob-weighted.

    Predict s_hat = A(w) Op_x, target s = P(x|w) B(wx). No division by P(x|w), so
    rare transitions are not amplified. This is the honest analogue of the old
    'divide-by-P' rollout R^2.
    """
    preds, targs, wts = [], [], []
    for x in ops:
        preds.append(A @ ops[x])
        targs.append(P[:, x, None] * Bc[:, x, :])
        wts.append(Wt)
    pred = np.concatenate(preds)
    targ = np.concatenate(targs)
    wt = np.concatenate(wts)[:, None]
    mu = np.average(targ, axis=0, weights=wt[:, 0])
    ss_res = (wt * (targ - pred) ** 2).sum()
    ss_tot = (wt * (targ - mu) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def select_order(X, P, Yc, Wt, d_max, val_frac, seed=SEED):
    """Pick the latent dimension d by held-out predictive consistency.

    Fit CCA basis + operators on a train split; score one-step (rescaled,
    prob-weighted) R^2 on a held-out split, for each candidate d. Return the
    full curve plus the chosen d (smallest within KNEE_TOL of the best — i.e.
    the dynamics 'knee'). Fully target-free.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    val = rng.random(N) < val_frac
    tr = ~val
    dirs_tr, cca_s = predictive_cca(X[tr], P[tr], Yc[tr], Wt[tr])

    curve = []
    for d in range(2, d_max + 1):
        basis = dirs_tr[:, :d]
        A_tr, Bc_tr = X[tr] @ basis, Yc[tr] @ basis
        ops, _ = fit_operators(A_tr, P[tr], Bc_tr, Wt[tr])
        A_va, Bc_va = X[val] @ basis, Yc[val] @ basis
        curve.append((d, rescaled_r2(A_va, P[val], Bc_va, Wt[val], ops)))

    best = max(r for _, r in curve)
    d_star = min(d for d, r in curve if r >= best - KNEE_TOL)
    return d_star, curve, cca_s


def _whiten(X, Wt, ridge=1e-6):
    """Return C^{-1/2} and C^{1/2} for the prob-weighted (uncentered) covariance."""
    w = Wt / Wt.sum()
    C = (X * w[:, None]).T @ X
    C += ridge * (np.trace(C) / X.shape[1]) * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return (vecs / np.sqrt(vals)) @ vecs.T, (vecs * np.sqrt(vals)) @ vecs.T


def _als_objective(Xt, P, Yt, Wt, U, ops):
    A = Xt @ U
    f = 0.0
    for k in ops:
        r = A @ ops[k] - P[:, k, None] * (Yt[:, k, :] @ U)
        f += (Wt * (r ** 2).sum(1)).sum()
    return f


def als_refine_basis(X, P, Yc, Wt, basis0, d, n_iter=40):
    """Bilinear ALS: re-orient the d-dim subspace to maximise operator consistency.

    Minimise  sum_w P(w) sum_x || a(w) A_x - P(x|w) a(wx) ||^2  jointly over the
    basis and the operators. The objective is *homogeneous* in the basis (scaling
    it scales the residual), so a free basis update collapses to zero; we instead
    constrain the projected states to unit covariance (B^T C_xx B = I) by working
    in whitened coordinates B = C^{-1/2} U with U on the Stiefel manifold
    (U^T U = I), and run projected-gradient with backtracking. Initialised at the
    CCA basis. Operators are refit in closed form each step (the ALS operator
    step). Returns the refined basis in raw coordinates.
    """
    Wmat, Chalf = _whiten(X, Wt)
    Xt, Yt = X @ Wmat, Yc @ Wmat                       # whitened past / children
    U, _ = np.linalg.qr(Chalf @ basis0)                # CCA basis -> orthonormal init
    U = U[:, :d]
    ops = fit_operators(Xt @ U, P, Yt @ U, Wt)[0]
    f = _als_objective(Xt, P, Yt, Wt, U, ops)
    lr = 1.0
    for _ in range(n_iter):
        A = Xt @ U
        grad = np.zeros_like(U)
        for k in ops:
            Rk = A @ ops[k] - P[:, k, None] * (Yt[:, k, :] @ U)
            wRk = Wt[:, None] * Rk
            grad += Xt.T @ wRk @ ops[k].T - Yt[:, k, :].T @ (P[:, k, None] * wRk)
        G = grad - U @ ((U.T @ grad + grad.T @ U) / 2)  # project to Stiefel tangent
        improved = False
        for _ls in range(6):
            Un, _ = np.linalg.qr(U - lr * G)
            Un = Un[:, :d]
            opsn = fit_operators(Xt @ Un, P, Yt @ Un, Wt)[0]
            fn = _als_objective(Xt, P, Yt, Wt, Un, opsn)
            if fn < f - 1e-9:
                U, ops, f, lr, improved = Un, opsn, fn, lr * 1.3, True
                break
            lr *= 0.5
        if not improved:
            break
    return Wmat @ U                                     # basis in raw coordinates


def recover_eval_functional(A):
    """Find e with a_proj(w) . e = 1 for all w (gauge image of the all-ones sum)."""
    ones = np.ones(A.shape[0])
    e, *_ = np.linalg.lstsq(A, ones, rcond=None)
    return e, float(np.sqrt(((A @ e - 1.0) ** 2).mean()))


# --------------------------------------------------------------------------- #
# Validation (target-free)
# --------------------------------------------------------------------------- #
def eigenvalue_comparison(ops, params):
    tm = np.array(mess3(**params))                      # (vocab, 3, 3)
    rows = []
    for x in sorted(ops):
        ev_op = np.sort_complex(np.linalg.eigvals(ops[x]))
        ev_true = np.sort_complex(np.linalg.eigvals(tm[x]))
        rows.append((x, ev_true, ev_op))
    return rows


def one_step_rollout_r2(model, hmm, basis, ops, context_len, device, n_seq=400):
    """LEGACY metric, kept for transparency: divide-by-P one-step prediction.

    Predict a_hat(wx) = a(w) A_x / P(x|w) and compare to a(wx). The 1/P factor
    amplifies low-probability transitions the fit deliberately down-weighted, so
    this understates operator quality; compare against the consistent rescaled R^2.
    """
    layer = model.cfg.n_layers - 1
    key = jax.random.PRNGKey(SEED)
    init = jnp.array(hmm.initial_state)
    gs = jnp.repeat(init[None, :], n_seq, axis=0)
    _, inputs, _ = generate_data_batch(gs, hmm, n_seq, context_len + 1, key, device=device)
    inputs = inputs.long().to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names_filter=f"blocks.{layer}.hook_resid_post")
    resid = cache["resid_post", layer].cpu().numpy()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    toks = inputs.cpu().numpy()
    A = resid @ basis

    preds, actuals = [], []
    for t in range(A.shape[1] - 1):
        x_next = toks[:, t + 1]
        p = probs[np.arange(len(toks)), t, x_next]
        for x in ops:
            m = x_next == x
            if not m.any():
                continue
            preds.append((A[m, t] @ ops[x]) / np.clip(p[m, None], 1e-8, None))
            actuals.append(A[m, t + 1])
    preds, actuals = np.concatenate(preds), np.concatenate(actuals)
    ss_res = ((actuals - preds) ** 2).sum()
    ss_tot = ((actuals - actuals.mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def rollout_states(proj_full, ops, e, max_len, vocab):
    """Denoised positions: regenerate each prefix purely from the recovered
    operators (an iterated function system of contraction maps), NOT from its
    raw activation. At each step  s(wx) = s(w) A_x, renormalized so s . e = 1.

    This is the target-free analogue of the supervised regression *prediction*:
    positions come entirely from the recovered dynamics, so per-point activation
    noise is gone and the result lives exactly on the eval-functional plane. The
    contraction (|eig|<1) makes the rollout self-correcting with depth.
    """
    init = [proj_full[s] for s in proj_full if len(s) == 1]
    s0 = np.mean(init, axis=0)
    s0 = s0 / (s0 @ e)
    states, frontier = {(): s0}, [()]
    for _ in range(max_len):
        nxt = []
        for w in frontier:
            sw = states[w]
            for x in range(vocab):
                child = w + (x,)
                if child not in proj_full:
                    continue
                sc = sw @ ops[x]
                denom = sc @ e
                states[child] = sc / denom if abs(denom) > 1e-12 else sc
                nxt.append(child)
        frontier = nxt
    states.pop((), None)
    return states


def plane_coords(points):
    P = np.asarray(points)
    Pc = P - P.mean(0)
    _, _, vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ vt[:2].T


def affine_align(src_xy, dst_xy, w=None):
    """Least-squares affine map src->dst (display overlay only). Returns aligned + R^2."""
    X = np.concatenate([src_xy, np.ones((len(src_xy), 1))], axis=1)
    if w is None:
        w = np.ones(len(src_xy))
    sw = np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(X * sw, dst_xy * sw, rcond=None)
    aligned = X @ M
    ss_res = (w[:, None] * (dst_xy - aligned) ** 2).sum()
    ss_tot = (w[:, None] * (dst_xy - np.average(dst_xy, axis=0, weights=w)) ** 2).sum()
    return aligned, 1.0 - ss_res / ss_tot


def belief_decode_r2(features, Y, w):
    """Prob-weighted R^2 of reconstructing the 3-D belief Y by a linear map from
    `features` (with intercept). The supervised baseline's metric; applied here
    identically to supervised and unsupervised representations for a fair compare.
    A k-D `features` gets only k+1 params/output, so a subspace of the activations
    can never beat the full-activation fit on this metric — a built-in leak check.
    """
    Xb = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Xb * sw, Y * sw, rcond=None)
    Yhat = Xb @ beta
    ss_res = (w[:, None] * (Y - Yhat) ** 2).sum()
    ss_tot = (w[:, None] * (Y - np.average(Y, axis=0, weights=w)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def simplex_to_xy(beliefs):
    # Paper's projection (simplexity casper/analyses): state-2=blue top apex,
    # state-0=red bottom-left, state-1=green bottom-right.
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def _disk(px):
    r = np.arange(-px, px + 1)
    yy, xx = np.meshgrid(r, r)
    return (xx ** 2 + yy ** 2) <= px ** 2


def rasterize_simplex(xy, color_rgb, px=2, W=1000, H=900):
    """Datashader-style render (matches the paper's tf.shade + tf.spread): bin
    points into a fixed WxH grid, colour each occupied cell by the mean RGB of
    its points, then dilate by `px`; empty cells stay white. Density is then
    independent of figure size/DPI (the cause of earlier 'sparse' looking plots).
    """
    x, y = xy[:, 0], xy[:, 1]
    x0, x1 = -0.05, 1.05
    y0, y1 = -0.05, np.sqrt(3) / 2 + 0.05
    ix = np.clip(((x - x0) / (x1 - x0) * W).astype(int), 0, W - 1)
    iy = np.clip(((y - y0) / (y1 - y0) * H).astype(int), 0, H - 1)
    sums = np.zeros((H, W, 3))
    cnt = np.zeros((H, W))
    np.add.at(sums, (iy, ix), color_rgb)
    np.add.at(cnt, (iy, ix), 1)
    occ = cnt > 0
    mean = np.zeros((H, W, 3))
    mean[occ] = sums[occ] / cnt[occ, None]
    fp = _disk(px)
    occ_d = grey_dilation(occ.astype(np.uint8), footprint=fp) > 0
    mean_d = np.stack([grey_dilation(mean[:, :, c], footprint=fp) for c in range(3)], axis=-1)
    img = np.ones((H, W, 3))
    img[occ_d] = np.clip(mean_d[occ_d], 0, 1)
    return img


# --------------------------------------------------------------------------- #
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size
    model, context_len = load_model(device)

    resid, soft, belief, prefix_prob, max_len = collect_prefix_features_sampled(
        model, hmm, N_SEQUENCES, context_len, BATCH, device
    )
    print(f"unique prefixes from {N_SEQUENCES} sampled sequences: {len(resid)}  (lengths 1..{max_len})")

    rows, X, P, Yc, Wt = build_transitions(resid, soft, prefix_prob, max_len, vocab)
    print(f"transition rows (prefixes with all children): {len(rows)}")

    # ---- dynamics-consistent order selection (no task knowledge) ----
    d_star, curve, cca_s = select_order(X, P, Yc, Wt, D_MAX, VAL_FRAC)
    print("\n=== DYNAMICS-CONSISTENT SUBSPACE SELECTION ===")
    print("CCA canonical correlations:", np.round(cca_s[:D_MAX], 3))
    print("held-out one-step R^2 vs latent dim d:")
    for d, r in curve:
        mark = "  <- chosen" if d == d_star else ""
        print(f"    d={d}:  R^2 = {r:.4f}{mark}")
    print(f"selected latent dimension d* = {d_star}")

    # ---- refit basis on all data at d*, then ALS orientation refinement ----
    dirs, _ = predictive_cca(X, P, Yc, Wt)

    # ALS is kept only if it beats CCA on a held-out split (never regresses).
    rng2 = np.random.default_rng(SEED + 2)
    val2 = rng2.random(len(rows)) < VAL_FRAC
    tr2 = ~val2
    B_cca_tr = predictive_cca(X[tr2], P[tr2], Yc[tr2], Wt[tr2])[0][:, :d_star]
    B_als_tr = als_refine_basis(X[tr2], P[tr2], Yc[tr2], Wt[tr2], B_cca_tr, d_star)

    def _ho(B):
        ot = fit_operators(X[tr2] @ B, P[tr2], Yc[tr2] @ B, Wt[tr2])[0]
        return rescaled_r2(X[val2] @ B, P[val2], Yc[val2] @ B, Wt[val2], ot)

    r2_cca_b, r2_als_b = _ho(B_cca_tr), _ho(B_als_tr)
    print("\n=== CCA vs ALS subspace — held-out one-step R^2 ===")
    print(f"  CCA: {r2_cca_b:.4f}   ALS: {r2_als_b:.4f}")
    if r2_als_b > r2_cca_b + 1e-4:
        basis = als_refine_basis(X, P, Yc, Wt, dirs[:, :d_star], d_star)
        print("  -> ALS basis kept (improves held-out prediction)")
    else:
        basis = dirs[:, :d_star]
        print("  -> CCA basis kept (ALS did not improve held-out prediction)")

    A, Bc = X @ basis, Yc @ basis
    ops_ols, fit_r2 = fit_operators(A, P, Bc, Wt)
    ops_iv = fit_operators_iv(rows, resid, basis, P, Yc, Wt)
    # The IV/two-stage fit was tested to remove OLS attenuation bias; empirically
    # it does NOT help here (activations are deterministic forward passes, so
    # there is no i.i.d. regressor noise to correct — see notes below), so OLS
    # remains the primary estimator and IV is kept only for the comparison.
    ops = ops_ols
    print("\nper-token operator in-domain fit R^2 (OLS):", {x: round(r, 4) for x, r in fit_r2.items()})

    e, e_rmse = recover_eval_functional(A)
    print(f"recovered eval functional rmse(a.e - 1) = {e_rmse:.2e}")

    # ---- held-out head-to-head: OLS vs IV (two-stage) ----
    rng = np.random.default_rng(SEED + 1)
    val = rng.random(len(rows)) < VAL_FRAC
    tr = ~val
    ho_ols, _ = fit_operators(A[tr], P[tr], Bc[tr], Wt[tr])
    ho_iv = fit_operators_iv(rows, resid, basis, P, Yc, Wt, mask=tr)
    r2_ho_ols = rescaled_r2(A[val], P[val], Bc[val], Wt[val], ho_ols)
    r2_ho_iv = rescaled_r2(A[val], P[val], Bc[val], Wt[val], ho_iv)
    print("\n=== OLS vs IV (two-stage) — held-out one-step R^2 (consistent) ===")
    print(f"  OLS:        {r2_ho_ols:.4f}")
    print(f"  IV/2-stage: {r2_ho_iv:.4f}")

    print("\n=== TARGET-FREE VALIDATION (primary: OLS operators) ===")
    if d_star == 3:
        for x, ev_true, ev_op in eigenvalue_comparison(ops, MESS3_PARAMS):
            print(f"token {x}: eig(T^x) = {np.round(ev_true, 3)}   eig(A^x) = {np.round(ev_op, 3)}")
    r2_consistent = rescaled_r2(A, P, Bc, Wt, ops)
    print(f"one-step R^2  (consistent, prob-weighted, no 1/P): {r2_consistent:.4f}")
    r2_legacy = one_step_rollout_r2(model, hmm, basis, ops, context_len, device)
    print(f"one-step R^2  (legacy divide-by-P, for comparison): {r2_legacy:.4f}")

    # ---- geometry recovery (display-only; positions are target-free) ----
    seqs = [s for s in resid if s in belief]
    true_b = np.array([belief[s] for s in seqs])
    true_xy = simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1)
    wcol = np.array([prefix_prob[s] for s in seqs])

    # (a) RAW: each prefix's actual activation, projected to the chosen subspace.
    act_xy = plane_coords(np.stack([resid[s] for s in seqs]) @ basis)
    act_aligned, raw_r2_w = affine_align(act_xy, true_xy, wcol)
    _, raw_r2_u = affine_align(act_xy, true_xy)

    # (b) DENOISED: positions regenerated by the recovered operators (analogue of
    #     the supervised regression *prediction*). Lives exactly on a plane.
    proj_full = {s: resid[s] @ basis for s in resid}
    roll = rollout_states(proj_full, ops, e, max_len, vocab)
    rseqs = [s for s in seqs if s in roll]
    rtrue_xy = simplex_to_xy(np.array([belief[s] for s in rseqs]))
    rcolor = np.clip(np.array([belief[s] for s in rseqs]), 0, 1)
    rwcol = np.array([prefix_prob[s] for s in rseqs])
    roll_xy = plane_coords([roll[s] for s in rseqs])
    roll_aligned, roll_r2_w = affine_align(roll_xy, rtrue_xy, rwcol)
    _, roll_r2_u = affine_align(roll_xy, rtrue_xy)

    # IV/2-stage rollout, same prefixes, for a direct OLS-vs-IV fine-detail comparison.
    roll_iv = rollout_states(proj_full, ops_iv, e, max_len, vocab)
    iv_xy = plane_coords([roll_iv[s] for s in rseqs])
    _, rolliv_r2_w = affine_align(iv_xy, rtrue_xy, rwcol)
    _, rolliv_r2_u = affine_align(iv_xy, rtrue_xy)

    print("\n=== RECOVERED-GEOMETRY vs ANALYTIC FRACTAL (display-only) ===")
    print("                                  weighted R^2 | unweighted R^2")
    print(f"  raw activations (subspace):       {raw_r2_w:.4f}     |   {raw_r2_u:.4f}")
    print(f"  operator rollout, OLS (primary):  {roll_r2_w:.4f}     |   {roll_r2_u:.4f}")
    print(f"  operator rollout, IV/2-stage:     {rolliv_r2_w:.4f}     |   {rolliv_r2_u:.4f}")
    print("  (weighted = prob-weighted toward the dense centre; unweighted gives the")
    print("   sparse deep-prefix filaments equal say, so it reflects fine-detail fidelity.)")

    # ---- FAIR COMMON METRIC: belief-reconstruction R^2, identical for all three ----
    # This is the metric the supervised baseline (fig03b) reports: fit a linear map
    # to the 3-D belief and score reconstruction. We apply the SAME metric to the
    # supervised features (full 64-D activations) and to the unsupervised products.
    # The map-fit uses ground truth for ALL THREE equally (that is what makes it a
    # fair comparison) — the recovery itself was still truth-free.
    Xfull = np.stack([resid[s] for s in seqs])              # supervised: 64-D activations
    r2_sup = belief_decode_r2(Xfull, true_b, wcol)          # replicates fig03b 0.997
    r2_sub = belief_decode_r2(Xfull @ basis, true_b, wcol)  # from recovered 3-D subspace
    roll_states = np.stack([roll[s] for s in rseqs])
    roll_belief = np.array([belief[s] for s in rseqs])
    r2_roll = belief_decode_r2(roll_states, roll_belief, rwcol)
    print("\n=== FAIR COMMON METRIC: belief-reconstruction R^2 (all map-fit to truth) ===")
    print(f"  supervised, full 64-D activations -> belief:   {r2_sup:.4f}")
    print(f"  unsup, recovered 3-D subspace     -> belief:   {r2_sub:.4f}  (must be <= supervised)")
    print(f"  unsup, operator-rollout state     -> belief:   {r2_roll:.4f}  (generative; certified by eig)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rasterize_simplex(true_xy, color, px=2), origin="lower")
    axes[0].set_title("Ground truth (analytic belief)")
    axes[1].imshow(rasterize_simplex(act_aligned, color, px=2), origin="lower")
    axes[1].set_title(
        "Unsupervised: raw activations\n"
        f"(d*={d_star}; align R^2 w={raw_r2_w:.3f} / u={raw_r2_u:.3f})"
    )
    axes[2].imshow(rasterize_simplex(roll_aligned, rcolor, px=2), origin="lower")
    axes[2].set_title(
        "Unsupervised: operator rollout (denoised)\n"
        f"(align R^2 w={roll_r2_w:.3f} / u={roll_r2_u:.3f})"
    )
    for ax in axes:
        ax.axis("off")
    out = FIG_DIR / "fig04_unsupervised_oom_mess3.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
