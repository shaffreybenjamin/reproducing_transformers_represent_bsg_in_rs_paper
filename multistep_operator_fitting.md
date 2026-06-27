# Multi-step operator fitting for the spectral-OOM estimator

## Goal

Improve the activation-space transition operators `{A_x}` in the spectral-OOM estimator by fitting them against **multi-step** transition targets, not just one-step pairs. The current estimator fits each `A_x` by an independent one-step ridge regression; the resulting operators compose poorly, so high-order observability blocks (where weakly-observable belief directions live) are swamped by compounded error. Fitting on multi-step targets directly should let those directions survive into the products and let longer stacking horizons `L` actually help.

## Background context (what already exists)

- Activations `a(w)` are gathered by enumerating all valid (nonzero-probability) sequences `w` from length 0 to the context window (length 10 for RRXOR), running each through the trained transformer, and reading the residual stream. So **multi-step ground truth already lives in the activations**: for any `w` and continuation `xy...`, both `a(w)` and `a(wxy...)` are present in the enumeration.
- The current one-step fit regresses, per token `x`, transition pairs `(a(w), a(wx))` rescaled by the next-token probability:
  `a(w) A_x ≈ P(x|w) a(wx)`, weighted by `P(w)`.
- `P(x|w)` is the network's own softmax (the projective normalizer that, after rescaling, linearizes the update). This rescaling is essential and must be preserved at every step-order.
- The readout `C` and the observability matrix construction are unchanged by this work — we are only improving how `{A_x}` are fit.

## The objective

Fit all operators jointly in a **single shared-operator loss** spanning step-orders `k = 1 … K`:

```
L = Σ_x  || a(w) A_x − P(x|w) a(wx) ||²_{P(w)}
  + λ₂ Σ_{x,y}  || a(w) A_x A_y − P(xy|w) a(wxy) ||²_{P(w)}
  + λ₃ Σ_{x,y,z} || a(w) A_x A_y A_z − P(xyz|w) a(wxyz) ||²_{P(w)}
  + …  (up to order K)
```

The **same** `A_x` appears in every term. The one-step term anchors each operator's marginal (one-step) quality; the higher-order terms constrain the products. Because operators are shared across terms, the optimizer cannot improve a product by drifting `A_x` away from its one-step duty without paying the one-step penalty. This is what prevents the multi-step targets from degrading the operators as one-step maps.

## How to optimize: ALS (alternating least squares)

The loss is non-convex once products appear (bilinear at order 2, multilinear beyond), so there is no closed-form joint solution. Use ALS:

1. **Initialize** all `A_x` from the existing one-step ridge solution (do NOT random-init).
2. **Sweep**: hold all operators *except one* (`A_j`) fixed. With the others fixed, every product term becomes **linear in `A_j`** (e.g. `A_x A_y` is linear in `A_x` when `A_y` is fixed). The per-operator subproblem is therefore an ordinary `P(w)`-weighted ridge regression with a closed-form solution — assemble the design matrix by stacking the contributions of `A_j` from every term/order in which it appears.
3. Repeat sweeps over all operators until the loss / operators converge.

ALS and "fit on multi-step targets" are the same idea: ALS is just how the non-convexity is handled. Each sub-step stays a linear ridge solve.

## Critical correctness requirements

- **Consistent rescaling across orders.** The k-step rescale is the product of conditionals via the chain rule:
  `P(xy|w) = P(x|w) · P(y|wx)`, `P(xyz|w) = P(x|w) · P(y|wx) · P(z|wxy)`, etc.
  Use the network's softmax for each conditional factor. If the rescaling is inconsistent across orders, the terms fight each other for the wrong reason and the fit is corrupted. Verify this rescale is correct before trusting any result.
- **Consistent `P(w)` weighting** across all orders, matching the existing one-step convention.
- **Ridge regularization** in each ALS sub-solve (carry over the existing `λ` choice as a starting point; the per-order `λ_k` below is separate).

## Practical notes / cautions

- **Decay `λ_k` in `k`.** High-order terms are built from rare, individually low-probability words (by `k≈10` on a length-10 enumeration, the discriminating words are very rare), so those targets are noise-dominated. Start with `λ_k` decreasing in `k` and tune. Do not assume going to the full context window (`K=10`) helps — it likely won't, due to sparsity.
- **Choose `K` from the noise floor, not the context window.** The useful horizon is the largest `k` at which the true high-order singular values still clear the estimation noise floor. If available, compute the *true* observability spectrum from the known transition operators (e.g. RRXOR's known `T^x`) at increasing horizon and check where the smallest relevant singular value (`σ₅` for RRXOR) drops below the achievable noise level. Use that to set `K` and to sanity-check whether the gain is reachable at all.
- **Non-convexity / local optima.** ALS converges to a local optimum; seeding from the one-step ridge solution matters. Monitor that the one-step residual does not blow up as higher-order terms are added — if it does, `λ_k` for high `k` is too large relative to `λ₁`.
- **Non-normal operators.** For RRXOR the true operators are non-normal (defective `T_0`, nilpotent `T_1`, `T_1^5 = 0`), so products have large transient behavior and one-step fits are a poor proxy for multi-step accuracy. This is exactly the regime the multi-step objective is meant to fix, but it also means products are sensitive to operator error — another reason to keep `K` modest and `λ_k` decaying.

## Scope boundary

- This improves operator **consistency**, not information content. It lets genuinely-present (weakly-observable) directions clear the noise floor; it cannot recover strictly-unobservable directions (those annihilated by operator chains of every length through the terminal normalizer). For a minimal presentation like RRXOR there should be no strictly-unobservable directions, so the gap is expected to be weak-observability (estimation), which this targets.

## Suggested validation

- Confirm one-step `R²` is preserved (or improved) after the multi-step fit — it must not regress.
- Compare the observability-matrix singular spectrum and the recovered-subspace `R²` (per-position and state-level) before vs. after, at several horizons `L`, to confirm that longer `L` now helps rather than hurts.
- Test first on a process with next-token degeneracy **and** sufficient data density (to isolate estimator quality from the RRXOR sparsity artifact) before drawing conclusions from RRXOR itself.
