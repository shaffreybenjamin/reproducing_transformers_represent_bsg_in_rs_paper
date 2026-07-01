# Results summary — reproduction + unsupervised-OOM contribution

Source-of-truth handoff for the thesis write-up. Covers (A) the supervised reproduction of
Shai et al., *"Transformers Represent Belief State Geometry in their Residual Stream"*
(arXiv:2405.15943), and (B) the novel **unsupervised belief-geometry recovery** contribution.

All code in `reproduction/`. Models: `models/mess3_transformer.pt`, `models/rrxor_transformer.pt`
(both HookedTransformer, paper App. A.6 config: d_model 64, 4 layers, 1 head, d_head 8, d_mlp 256,
ReLU, LayerNorm, SGD lr 0.01, batch 64, **1e6 steps**).

---

## A. Supervised reproduction (all reproduced; match the paper)

### Mess3 (x=0.05, a=0.85; 3 states, vocab 3)
- Model trains to myopic optimum: final loss ≈ **0.7935**.
- `fig03_residual_belief_geometry.py` — **Fig 5** residual belief geometry. Enumerate all 3^10
  length-10 paths + positions, `blocks.3.hook_resid_post`, unweighted LinearRegression onto belief,
  paper simplex projection, **datashader-style raster** (fixed grid + grey_dilation) — this rendering
  is the key to matching density (raw scatter looks misleadingly sparse).
- `fig06_training_progression.py` — **Fig 6 top** (blob→fractal emergence over checkpoints).
- `fig07/fig08` — Fig 6 B/C/D controls (supervised + unsupervised head-to-head).

### RRXOR (p1=p2=0.5; 5 states, vocab 2) — **Fig 7**
- Synchronizing process: MSP has exactly **36 belief states** (35 + root). Model loss ≈ **0.5788**
  ≈ myopic optimum **0.5782** (NOT the 0.4621 asymptotic entropy rate — the training loss averages
  short contexts).
- `fig09_rrxor_ground_truth.py` — **Fig 7B** ground truth. 36 beliefs → PCA → **PC1/PC3**
  (the SYMMETRIC pair; PC1/PC2 is triangular and WRONG, PC2 is asymmetric). Gives the 4-apex diamond.
- `fig10_rrxor_representation.py` — **Fig 7C** residual-stream representation. Now 2-panel
  (GT | representation with GT rings). **Activation = concatenated residual stream
  `blocks.k.hook_resid_post`** (the paper TEXT says "residual streams"; their released notebook
  used `ln1.hook_normalized`, but resid_post matches the text and decodes belief more cleanly:
  R²≈0.88 vs 0.81). Regress concat→belief, project predicted belief with panel-B PCA.
- `fig11_rrxor_layerwise_mse.py` — **Fig 7E** per-layer MSE: RRXOR geometry **spread across layers**
  (every single layer high, Concat lowest 0.009); Mess3 low everywhere (in every layer).
- `fig12_rrxor_distances.py` — **Fig 7D** distance preservation: RRXOR belief-dist R²=0.88 but
  next-token-dist only **0.33** (geometry ≠ next-token); Mess3 1.00/1.00.

### Key supervised fact (used throughout Part B)
**Belief geometry is PERFECTLY linearly present in the residual stream**: the 35 per-state
centres-of-mass → belief is **R² = 1.000**. Per-prefix held-out decode is only ~0.68 (limited by
**within-state activation spread** — non-belief variation across prefixes sharing a state), and
fig10's in-sample is ~0.88. These are three different quantities; the *geometry* (state-level) is 1.0.

---

## B. Unsupervised belief-geometry recovery (the contribution)

**Goal:** recover the belief geometry the network uses **from activations + the model's softmax +
the prefix-tree only** — no ground-truth belief labels, no DGP knowledge — so it can test the
linear-belief hypothesis and extend to non-classical (quantum/post-quantum) predictive states.

### B.1 Unifying framework — three methods are ONE SVD under different normalizations
All methods are an SVD of a **past × future** cross-matrix; **rank = belief dimension**, the
**activation-side singular vectors = the affine map activation→belief**, the shifted matrix gives
the operators, the constant direction the eval functional.

| method | matrix it SVDs | normalization | code |
|---|---|---|---|
| **Predictive CCA** | Σxx^−1/2 Σxy Σyy^−1/2 | whiten BOTH sides → canonical correlations | `unsupervised_belief_oom.py`, fig04/13/17 |
| **Reduced-rank / spectral** | Σxx^−1/2 Σxy | whiten INPUT only | `fig19_rrr_spectral.py` |
| **Observability OOM** | obs. matrix `[C, GₓC, …]` SVD | operator-based, input-whitened | `fig14_observable_oom.py`, fig16/18 |

CCA's double-whitening **degenerates on (near-)deterministic activations** (all canonical
correlations → 1, no rank info), especially with few samples; un-whitened/input-whitened variants
keep a decaying spectrum → more robust. All are DGP-free; **d is estimated from the spectrum**;
ground-truth belief is used ONLY to score, never to fit. Assumes only LINEARITY → carries over to
quantum/post-quantum (density-matrix / GPT) representations unchanged.

### B.2 Subspace vs operators — TWO distinct estimated objects
- **Subspace** `B` (d-dim basis): learned by the SVD/CCA. "Which activation directions carry belief."
- **Operators** `Aₓ` (d×d): a **separate least-squares regression** *within* `B`:
  fit `Aₓ` to `s(w)Aₓ ≈ P(x|w)s(wx)`, `s(w)=a(w)B`. (`fit_at_dim` / `fit_operators`.)
- **Two visualizations use different objects:**
  - *Raw subspace plot*: project each prefix's ACTUAL activation, `s(w)=a(w)B`. Uses `B` only →
    tests "is belief linearly PRESENT?"
  - *Operator rollout plot*: discard per-prefix activations; regenerate every state from the
    operators along the token tree, `s(wx)=s(w)Aₓ/normalizer` (activations seed only the root) →
    tests "did we LEARN the dynamics?"
- A good subspace does NOT guarantee good operators. The **rollout amplifies operator quality**
  (composes operators ~10 deep), so it's the cleanest diagnostic of whether operators are trustworthy.

### B.3 Headline results

**Mess3 (easy: richly sampled ~83k states, strongly observable, belief in every layer)**
- All methods recover it cleanly: **decode R² ≈ 0.997**, operator eigenvalues match `Tₓ`.
- `fig16_oom_mess3_geometry.py` renders the full fractal (single layer `blocks.3.hook_resid_post`,
  3 panels GT | raw | operator-rollout). Observability OOM is **on par with** the older CCA (fig04),
  NOT better — Mess3 is the regime where method choice barely matters.
- Operator rollout > raw here (fig04 panel 3) because the operators are clean.

**RRXOR (hard: 35 states, synchronizing, weakly observable, belief spread across layers)** —
COM-geometry R² (state-level), apples-to-apples on the concat:

| | raw subspace | operator rollout |
|---|---|---|
| Predictive CCA | 0.57 (fig17) | 0.37 collapse (fig20) |
| Reduced-rank/spectral | 0.65 (fig19) | — |
| Observability OOM | **0.68** (fig18) | 0.26 collapse (fig21) |
| *Supervised* | *1.00* | — |

- **Three independent methods converge to 0.57–0.68** (far below supervised 1.0) → the ceiling is an
  **information limit, not a method artifact**. The spectrum shows **no clean rank-5 elbow**.
- Among the methods, **observability OOM is best for RRXOR** (0.68) — opposite of Mess3.
- **Both operator rollouts COLLAPSE** for RRXOR (0.26–0.37 ≪ raw) — opposite of Mess3. RRXOR's
  operators are unlearnable-from-activations under every method, so the **raw subspace is the
  trustworthy representation** there.

### B.4 Why RRXOR is hard (diagnosis — NOT just "richness")
Three compounding PROCESS properties, only one of which is richness:
1. **High-dim**: belief spread across layers → needs the ~384-D concat (Mess3: a single 64-D layer).
2. **Coverage**: 35 distinct states (~600 prefixes) → ill-conditioned covariance (Mess3: ~83k states).
3. **Weak observability**: belief barely moves next-token (Fig 7D = 0.33; Mess3 ≈ 1.0). The weak
   belief modes are faint singular values; surfacing them needs deep horizon, which runs into the
   **n_ctx=10 coverage wall** (longer horizon ⇒ shorter prefixes ⇒ fewer states).

**Crucial for the thesis narrative:** the determinant is **process richness + observability**, NOT
classical-vs-post-quantum. Mess3 is classical and works; RRXOR is classical and fails. So the method
should transfer to **richly-sampled post-quantum processes** (the natural next experiment).

### B.5 Honest negative results (documented, do not over-claim)
- `fig15_oom_denoised.py`: EM rollout-denoising (1) and noise-whitening (2) **both fail** to beat the
  baseline observability OOM — the EM is unstable (compounds operator error, collapses), whitening
  has no effect (within-state noise isn't the binding constraint at COM level; weak observability is).
- Applying ALS to the observability subspace **collapses the rollout** (gauge/eval-functional
  instability) — not the fix.
- The geometry *displays* place points via a SUPERVISED affine readout (decode subspace→belief with
  labels) in the shared panel-B PCA frame; the SUBSPACE is unsupervised but the final placement uses
  labels (standard for these plots, but be transparent — the visual "fit" is somewhat flattered).

---

## Figure index
- fig03/03b/03c, fig06, fig07, fig08 — Mess3 supervised (Fig 5, Fig 6).
- fig09 — RRXOR GT (Fig 7B). fig10 — Fig 7C (2-panel). fig11 — Fig 7E. fig12 — Fig 7D.
- fig04 — Mess3 unsupervised (predictive CCA, 3 panels incl. crisp rollout).
- fig13 — RRXOR predictive-CCA negative result (decode curves).
- fig14 — observability OOM (decode-vs-d + eigenvalue validation; Mess3 0.997, RRXOR partial).
- fig16 — Mess3 observability OOM geometry (single layer, 3 panels).
- fig17 / fig18 — RRXOR raw recovery: CCA (0.57) / observability (0.68), GT | decoded.
- fig19 — RRXOR reduced-rank/spectral (0.65) + horizon sweep + spectrum.
- fig20 / fig21 — RRXOR operator rollout: CCA+ALS (0.37) / observability (0.26) — both collapse.
- (stale: `fig14_observable_oom_rrxor.png` is an orphan from an early fig14 version = observability,
  same as fig18.)

## Suggested next step
Run the full battery (raw subspace + operator rollout, both estimators) on a **richly-sampled
post-quantum process** from the Simplex line ("Neural networks leverage nominally quantum and
post-quantum representations"), where — unlike RRXOR — we'd expect BOTH subspace and operators to
come out clean, providing the positive post-quantum demonstration.

## Memory pointers (auto-memory dir)
`figure-reproduction-recipe.md`, `unsupervised-oom-subspace-fix.md`, `unsupervised-oom-rrxor-fails.md`,
`runpod-workflow.md`, `user-profile.md`.
