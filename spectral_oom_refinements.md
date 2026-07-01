# Spectral-OOM refinements: metric-correct balanced truncation, joint-subspace recovery, Riesz projectors, Fisher readout

## Purpose and scope

This document instructs a coding agent to extend the existing **spectral-OOM** estimator with four functional-analytic refinements. These are **additive refinements**, not a rewrite. The existing pipeline stands:

1. Fit rescaled transition operators `A_x` in activation space (`a(w) A_x ≈ P(x|w) a(wx)`).
2. Fit the one-step readout `C` (`a(w) → P(·|w)`).
3. Assemble the observability factor `O = [C | A_x C | A_x A_y C | ...]` and/or the controllability factor `R` (reachable activations `P(w) a(w)`).
4. SVD to recover the belief subspace basis `B`, project activations, score.

The four refinements target one documented failure: on RRXOR the estimator is **biased toward the next-token observable `C`**, so weakly-observable belief modes (geometrically distinct states with near-identical next-token distributions) are under-recovered. Mess3 works already and must keep working — it is the **control / regression guard** for every change.

**Critical implementation constraint:** every refinement must be a **toggleable path that defaults to the existing plain behaviour**. The agent must be able to run each one in isolation and ablate it. Do **not** bundle all four behind a single flag. The whole point is to learn *which* refinement closes the RRXOR gap; bundling destroys that information.

**Out of scope / explicitly disregard:** the multistep-ALS operator refinement and the CCA/RRR multistep readout from the user's prior code. Those were prior attempts to fix the same problem and did not help. Build the refinements below on the **plain** spectral-OOM (single-pass ridge `A_x`, ridge `C`), not on top of ALS or CCA. The agent may keep ALS/CCA code in the file but must not invoke it in the new paths.

---

## Preliminary fixes (do these first — they are prerequisites, not optional)

Two inconsistencies in the current two-factor code will silently corrupt balanced truncation. Fix both before implementing anything else.

### Fix A — one shared operator set for both factors

In the current `run_hankel_spectral_oom`, the controllability factor `R` is built from `ops` (the operators from `fit_transition_operators`), while the observability factor `S` is built inside `build_observability_factor`, which **refits its own operators `Gs`**. `R` and `S` are therefore factors of two *different* realizations, so `H = R·S` is not a consistent factorization and balanced truncation on it is meaningless.

**Required change:** fit the operators `{A_x}` and the readout `C` **once**, and pass them into both factor builders. Concretely:

- Fit `{A_x}` once (plain ridge, `fit_intercept=True` — see Fix B).
- Fit `C` once (plain ridge or Fisher ridge — see refinement iv).
- `build_observability_factor` takes `C, {A_x}` as arguments and only *composes* them; it does not refit.
- `build_controllability_factor` takes `{A_x}, a_0` and only composes.

After this, `O` and `R` share `{A_x}`, and `H = R·S` is a single realization's Hankel factorization.

### Fix B — honour the affine offset `c`

The writeup states the encoding is **affine**, `a(w) = b(w)Ψ + c`, and that the offset is absorbed by fitting with an intercept / on `P(w)`-centered activations. But the current operator and `C` regressions use `fit_intercept=False`. That forces the operators to also fit the constant `c`, which contaminates them.

**Required change:** centre activations by the `P(w)`-weighted mean before all regressions (equivalently, fit with intercept), and keep the offset `c̄ = Σ_w P(w) a(w)` so it can be added back when reconstructing/scoring. Apply the **same** centering consistently to parents, children, and root `a_0`. This is a correctness fix that helps all processes; do it once, globally.

> Guard after Fixes A & B: re-run the existing Mess3 path. Belief-decode R² at d=3 must remain ≈ 0.997 and the eig(A_x) ≈ eig(T_x) check must still pass. If Mess3 regresses, stop and report — something in the shared-operator/centering refactor is wrong.

---

## Refinement (iv) — Fisher-metric readout `C`  *(do this first: cheapest, most diagnostic)*

### Why
`C` is currently a plain ridge regression of activations onto the next-token softmax row, scored in Euclidean output space. Belief states live in a probability simplex; the meaningful output geometry is Fisher–Rao, not ℓ². RRXOR's next-token-degenerate states are **ℓ²-close in output space but the belief distinction is real**. A Fisher-weighted readout weights each output direction by inverse next-token variance, separating distinctions the Euclidean readout merges.

### What to implement
Replace the plain ridge fit of `C` with a **generalized least squares** fit using the next-token Fisher information as the output-space weight.

For a categorical next-token distribution `p = P(·|w)` over `V` tokens, the Fisher metric on the simplex is `g(p) = diag(1/p) − 11ᵀ` (the multinomial Fisher information), or the simpler diagonal surrogate `diag(1/p)`. Use the diagonal surrogate first (cheaper, stable):

- Per row `w`, weight the output residual by `M_w = diag(1/ max(p_w, p_floor))`, with `p_floor` ≈ 1e-3 to avoid blowup on near-zero token probabilities.
- This is a per-row, per-output-coordinate reweighting. Solve the weighted normal equations:
  `C = argmin_C Σ_w sw_w · (a(w)C − p_w) M_w (a(w)C − p_w)ᵀ + ridge·‖C‖²`
  where `sw_w` is the existing `P(w)` sample weight.
- Because `M_w` is diagonal and varies per row, the clean implementation is a per-output-column weighted ridge is **not** correct (the weight couples columns only through `M_w`'s structure; with the diagonal surrogate columns decouple, so you *can* solve column-by-column with row-weights `sw_w / p_w[k]` for output column `k`). Implement the diagonal-surrogate version as `V` independent weighted ridge regressions, output column `k` using row weights `sw_w · (1 / max(p_w[k], p_floor))`. This is exact for the diagonal surrogate and trivial to code.

### Toggle
`readout_metric: "euclidean" | "fisher"`, default `"euclidean"`. Everything downstream (`O`, SVD, scoring) is unchanged — `C` is still `(D, V)`.

### Diagnostic value
Run RRXOR with `readout_metric="fisher"` and the existing Euclidean SVD pipeline (no other refinement). If belief-decode R² at d=5 rises meaningfully, the degeneracy is **metric-induced** (recoverable). If it barely moves, the weak modes are deeper than the output metric and the structural tools (i, iii) are needed. Either outcome is informative; report the delta.

---

## Refinement (i) — metric-correct balanced truncation  *(highest-value structural change)*

### Why
The current `O`-SVD ranks belief directions by **Euclidean** singular value, which conflates "large in activation norm" with "important for belief". Hankel singular values — obtained by balancing the **controllability** Gramian against the **observability** Gramian — weight each mode by *both* how reachable it is (the process visits those belief states) *and* how observable it is (it shows up in the future). A mode that is reachable but weakly observable through `C` (RRXOR's pathology) survives balancing because the controllability Gramian still sees it. Hankel singular values are also **basis-independent**, which fixes the no-clean-elbow problem: they are the honest, gauge-invariant measure of how much belief structure is *actually present*, separating "recoverable with better geometry" from "genuinely sub-floor".

Crucially, the balancing must be done in the **correct inner products**: the `P(w)`-weighted metric on the domain (contexts), and the Fisher/output-covariance metric on the range. Plain Euclidean balancing (what `compute_hankel_svd_factored` currently does) inherits the next-token bias and defeats the purpose.

### What to implement (square-root balanced truncation, weighted)

You already have the two factors. Make them metric-correct, then balance.

**Step 1 — build consistent, weighted factors** (after Fixes A & B):
- Observability factor in activation coords: `O` is `(D, n_fut)` — columns are the multi-step observables `A_{x_1}...A_{x_k} C`. (Note: the current `build_observability_factor` returns `S` shaped `(D, n_fut)` — good, it is already in activation space on the row index.)
- Controllability factor: `R` is `(n_past, D)` — rows are `P(w) a(w)` reached by each word. Its **activation-space content is the row space**, i.e. carried by the right singular vectors of `R`.

**Step 2 — define the two Gramians in activation space** (both `(D, D)`):
- Observability Gramian: `Wo = O Mo Oᵀ`, where `Mo` is the range (future-output) metric. Start with `Mo = I` (Euclidean future) for a first correct version, then upgrade to the **Fisher/future-covariance** weighting: weight each future column block by the inverse covariance of that future observable under `P(w)` (or, simplest upgrade, weight the length-k block by `λ^k` with `λ<1` to stop short futures from dominating — a cheap proxy for the output metric that already helps the next-token bias).
- Controllability Gramian: `Wc = Rᵀ Dc R`, where `Dc = diag(P(w))` is the **domain metric** (the `P(w)` weighting on contexts). Since `R`'s rows are already `P(w) a(w)` (un-normalised), confirm whether `P(w)` is applied once or twice and correct to exactly one factor of `P(w)` in the Gramian. `Wc` must be the `P(w)`-weighted second moment of reachable activations.

> Both Gramians are `(D, D)` and symmetric PSD. This is cheap regardless of how many pasts/futures there are.

**Step 3 — square-root balance:**
- Cholesky (or PSD square root via eigendecomposition) `Wc = Lc Lcᵀ` and `Wo = Lo Loᵀ`.
- SVD the cross matrix `Lcᵀ Lo = U Σ Vᵀ`. The diagonal of `Σ` are the **Hankel singular values** `σ_k` — basis-independent, the honest spectrum.
- Balancing/truncation basis (for projecting activations to the rank-d belief subspace):
  `B = Lc U Σ^{-1/2}` (take first `d` columns), with the dual `B⁺ = Σ^{-1/2} Vᵀ Loᵀ`.
  Project activations as `s(w) = (a(w) − c̄) B_d`. (Use the centered activation from Fix B.)

**Step 4 — select `d` from the Hankel singular values**, not the raw `O`-SVD. The Hankel `σ_k` are the gauge-invariant spectrum; the elbow here is meaningful where the raw `O`-spectrum's was not. Report the full `σ_k` vector for both processes.

### Toggle
`factorization: "obs_svd" | "balanced"`, default `"obs_svd"` (current behaviour). Under `"balanced"`, use the Gramian route above. Keep `range_metric: "euclidean" | "lambda_decay" | "fisher"` and `domain_metric: "uniform" | "pw"` as sub-options so the agent can ablate the *metric* separately from the *balancing*.

### Diagnostics
- Print Hankel `σ_k` for Mess3 (expect a clean drop at k=3, decode unchanged ≈ 0.997) and RRXOR (this spectrum is the headline result — how many modes are genuinely above the finite-sample floor).
- The split between "balanced + Euclidean metric" and "balanced + pw/Fisher metric" tells you how much of the RRXOR gap is *metric* vs *balancing*. Report both.

---

## Refinement (iii) — joint-subspace recovery of `{A_x}`  *(pools signal across symbols)*

### Why
The operators `A_x` are currently fit **independently** per symbol, then composed. But they share a common invariant subspace (the belief subspace, common to all of them because they share `Ψ`), and in the true model they are simultaneously (block-)triangularizable. For RRXOR this is decisive: `T_1` alone is **nilpotent** (rank-deficient, individually uninformative) while `T_0` is **defective**; jointly `{T_0, T_1}` span the dynamics. Recovering a **common invariant subspace** across the operator family pools signal across symbols and surfaces belief directions that any single operator annihilates or hides.

### What to implement — joint Schur / common invariant subspace
Do **not** attempt exact joint diagonalization (the operators are non-normal and defective; it will not exist). Use **joint (approximate) Schur** — a common orthonormal basis that makes all `{A_x}` as upper-triangular as possible simultaneously:

- Standard approach: find orthonormal `Q` minimizing the total below-diagonal mass `Σ_x ‖ offlower(Qᵀ A_x Q) ‖²`, via a Jacobi-style sweep of Givens rotations (the Sheffer / simultaneous-Schur algorithm), or initialise `Q` from the Schur vectors of a random real combination `Σ_x γ_x A_x` and refine.
- A cheap, robust first version: take `M = Σ_x γ_x A_x` for a few random weight vectors `γ`, compute the (real) Schur decomposition of each, and verify the leading invariant subspace is stable across `γ`. The stable common leading subspace of dimension `d` is the joint belief subspace. (Stability across `γ` is itself a diagnostic that you've found a genuine *common* invariant subspace rather than an artifact of one combination.)
- Project activations onto this joint subspace and score as usual.

### Interaction with (i)
(iii) produces a **subspace**; (i) produces a **subspace + a ranking (Hankel σ)**. Best combination: use the joint-Schur common invariant subspace to define the belief-carrying directions, then balance *within* it. But first implement and ablate (iii) standalone (replace the `O`-SVD subspace with the joint-Schur subspace, keep everything else), so its contribution is isolated.

### Toggle
`subspace_method: "obs_svd" | "joint_schur" | "balanced"`, default `"obs_svd"`.

### Diagnostic
On RRXOR, compare the joint-Schur subspace decode against the `O`-SVD decode at d=5. If joint-Schur wins, the independent per-symbol fit was discarding cross-symbol belief signal (the expected RRXOR story). On Mess3 it should match (strongly observable; independent fits already suffice).

---

## Refinement (ii) — Riesz spectral projectors  *(principled d-selection under defectiveness)*

### Why
For RRXOR the `O`-SVD has **no clean elbow** — currently `d=5` is set by hand from the known state count, which is not target-free. The reason there's no elbow: a defective operator's powers `A^k` carry polynomial×geometric envelopes `k^{j} λ^k`, so "modes" smear across singular-value scales. The fix is to separate by **generalized eigenspace (Jordan structure)** rather than by singular-value magnitude. The **Riesz projector** `P_λ = (1/2πi) ∮_{Γ_λ} (zI − A)^{-1} dz` around each eigenvalue `λ` extracts the spectral subspace belonging to that eigenvalue, robust to defectiveness in a way the SVD elbow is not. The belief subspace is then `⊕_λ range(P_λ)` over the eigenvalues you keep. This makes `d`-selection **target-free even for RRXOR**.

### What to implement
Operate on a single composite operator capturing the dynamics — use `Â = Σ_x A_x` (the marginal transition operator; its activation-space spectrum should approximate that of `Σ_x T_x`), or run per-operator and intersect. Then:

- Compute the eigenvalues of `Â`. Cluster them (defective/repeated eigenvalues cluster; nilpotent contributions sit near 0).
- For each retained eigenvalue cluster `λ` (drop the cluster at/near 0 if it corresponds to the nilpotent/non-belief part — **decision point, see below**), build the Riesz projector numerically via the resolvent contour integral: discretise a small circle `Γ_λ` enclosing only that cluster, `P_λ ≈ (1/2πi) Σ_m (z_m I − Â)^{-1} Δz_m` over quadrature points `z_m` on the circle.
- The belief subspace is the column span of `Σ_{retained λ} P_λ`; its dimension is `d` (read off, not set by hand).

**Decision point — which eigenvalue clusters are "belief":** the stochastic/contractive belief dynamics has eigenvalues with `|λ| ≤ 1`, with a leading `λ=1` (stationary) direction. The nilpotent part contributes `λ=0`. Whether the zero/near-zero cluster is belief-bearing or noise is exactly the question; report the spectrum and the decode R² **with and without** the near-zero cluster so the choice is empirical, not assumed. This is the honest version of "fix d=5".

### Toggle
`d_selection: "elbow" | "riesz"`, default `"elbow"`. Under `"riesz"`, `d` and the subspace both come from the projector sum.

### Caveats for the agent
- The contour must enclose **only** the target cluster. Eigenvalues of the *estimated* `Â` are perturbed by estimation noise, so use generous radii but check no two clusters' contours overlap; if they do, merge them and report the merged dimension.
- Numerical resolvent integration is sensitive near eigenvalues; keep quadrature points off the eigenvalues and use enough of them (e.g. 64–128 per circle). Validate on Mess3 first: its operators are **diagonalizable contractions**, so Riesz projectors must recover exactly the same d=3 subspace as the SVD — that's the correctness check before trusting it on RRXOR.

---

## Recommended sequence and ablation protocol

Implement and **ablate in this order**, re-checking Mess3 as a guard after each:

1. **Fixes A & B** (shared operators, affine centering). Guard: Mess3 decode ≈ 0.997 unchanged.
2. **(iv) Fisher readout** — cheapest, run standalone on RRXOR, record decode delta. Tells you if the gap is metric-induced.
3. **(i) Balanced truncation** — run with `domain_metric="pw"` and both `range_metric="euclidean"` and `"fisher"`. The Hankel σ spectrum for RRXOR is the key deliverable: it quantifies how much belief structure is genuinely above the finite-sample floor (recoverable) vs sub-floor (structural). This single number reframes the whole RRXOR result.
4. **(iii) Joint-Schur subspace** — standalone, compare to obs-SVD subspace on RRXOR.
5. **(ii) Riesz d-selection** — validate on Mess3 (must give d=3), then report RRXOR spectrum and the with/without-zero-cluster decode.
6. **Best combination:** joint-Schur subspace (iii) → balance within it (i) with Fisher readout (iv) and Fisher range metric → Riesz for the honest d. But only after each part is individually ablated.

### What to report per run (for every process, every toggle)
- Belief-decode R² vs d (the existing `decs` table).
- Supervised ceiling (unchanged baseline).
- The relevant spectrum: raw `O`-SVD σ, **Hankel σ** (refinement i), eigenvalues of `Â` (refinement ii).
- For RRXOR specifically: the decode at the true d=5 and the eig(A_x) vs eig(T_x) comparison.

### The honest framing to preserve in all outputs
None of these refinements manufactures signal that isn't there. If a belief mode is genuinely sub-floor in the (metric-correct) Hankel spectrum — its `σ_d` below the finite-sample noise level — no change of basis or metric recovers it, and that is a **structural** statement about RRXOR + this model, not an estimator deficiency. What the refinements *do* is ensure you are not **discarding** signal that *is* present by using Euclidean, single-operator, SVD-elbow machinery. The metric-correct Hankel singular values (i) are the instrument that tells you which regime you're in. Mess3 is the control proving the machinery is sound when belief is strongly observable.

---

## Engineering notes

- Keep all new paths in the same module; gate by the toggle arguments listed. Default every toggle to current behaviour so existing results reproduce bit-for-bit.
- The Gramians, joint-Schur, and Riesz projectors are all `(D, D)` operations — cheap. The cost is unchanged; you are not enumerating more prefixes.
- Do not reintroduce multistep-ALS or CCA/RRR in the new paths. If present in the file, leave them dormant.
- Add a single `compare_methods(process)` driver that runs all toggle combinations and prints one table (rows = methods, cols = decode@d=5 / Hankel-σ-tail / detected-d), so the ablation is one command.
- Validate numerically at each step against Mess3 before trusting any RRXOR number.
