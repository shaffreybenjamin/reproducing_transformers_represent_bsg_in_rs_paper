# Handoff context — belief-state-geometry reproduction + unsupervised recovery

Context dump for continuing the analysis / LaTeX write-up in a new conversation.
Written 2026-06-17. **This supersedes `RESULTS_SUMMARY.md`** for everything below (that file
predates the P(w)/ALS/split/figure-restructure decisions and is now partly stale).

---

## 1. Project & goal
- Maths thesis. Reproducing Shai et al., *"Transformers Represent Belief State Geometry in their
  Residual Stream"* (arXiv:2405.15943), **and** a novel **unsupervised** recovery of that geometry
  from activations + the model's own softmax + the prefix tree (no belief labels, no DGP knowledge).
- Deliverable in progress: `reproduction/writeup.tex` (~LaTeX report; Section A / Section B / Appendix).

## 2. Environment & how to run
- Work dir: `/home/ben/mathematics_thesis_project/reproducing_transformers_represent_bsg_in_rs_paper/reproduction`
- Python venv (CPU): `/home/ben/mathematics_thesis_project/reproducing_transformers_represent_bsg_in_rs_paper/.venv/bin/python`
  (run figure scripts from the `reproduction/` dir; they import each other by module name).
- Models train on RunPod GPU; figures regenerate locally on CPU. **Mess3 figures are slow**
  (~88,572 enumerated prefixes; ~1–2 min/forward-pass sweep). RRXOR is fast (~1046 prefixes).
- Models: `models/mess3_transformer.pt`, `models/rrxor_transformer.pt`,
  `models/progression/step_{100,1000,5000}.pt` (training checkpoints; final = the committed model).
- Arch (paper App. A.6): d_model 64, 4 layers, 1 head, d_mlp 256, ReLU, LayerNorm, SGD, 1e6 steps.
- **No LaTeX engine installed locally.** Compile `writeup.tex` on Overleaf, or install
  `texlive-latex-recommended/-extra/-fonts-recommended latexmk` + the VSCode "LaTeX Workshop" extension.

## 3. Processes
- **Mess3** (x=0.05, a=0.85): 3 states, vocab 3. MSP = dense Sierpinski-like fractal (continuum).
  **Strongly observable.** Trains to myopic optimum loss ≈ 0.7935.
- **RRXOR** (p1=p2=0.5): 5 states, vocab 2. MSP = **36 discrete belief states** (the symmetric
  four-apex "diamond", shown in the **PC1/PC3** plane). Synchronizing, **weakly observable**.
  Loss ≈ 0.5788 (≈ myopic optimum, not the 0.4621 entropy rate).

## 4. The unsupervised estimator (final adopted method)
Observability-OOM (Jaeger 2000; Thon–Jaeger 2015; HKZ spectral HMMs; PSRs):
- Observable anchor `C`: ridge-regress `a(w) → P(·|w)` (the model softmax).
- Rescaled operators `G_x`: ridge-regress `a(w) → P(x|w) a(wx)` (the rescaling trick linearises the
  projective belief update `b(wx)=b(w)T^x / P(x|w)`).
- Observability matrix `O=[C, G_xC, G_xG_yC, …]`; **SVD → belief subspace** `B` (rank = belief dim).
- Fit operators `A_x` in `B`, eval functional `e` (`a(w)B·e=1`); validate `eig(A_x)≈eig(T^x)`.

**ADOPTED RECIPE, consistent across both processes:**
`P(w)-weighted observability subspace` + `P(w)-weighted operators` + `ALS`.
- `F14.observable_subspace(resid, soft, reach, vocab, wmap=P(w))` — `wmap` weights the C/G_x ridge
  regressions by prefix probability. `F14.analytic_prefix_probs(resid, T, pi)` builds P(w) for RRXOR;
  Mess3 uses the analytic `prefix_prob` from `collect_prefix_features_enumerated`.
- **P(w) weighting comes from the "prefix dedup + P(w)" convention of the post-quantum/Simplex paper**
  (causal attention ⇒ a prefix's activation is identical at every occurrence → keep one, weight by
  summed probability). Already implemented in our collectors.
- **ALS** (`als_refine_basis`, alternating least squares re-orienting the subspace for operator
  consistency) is **kept by user preference but is nearly inert** — see §5.

## 5. Key findings (the scientific content)
- **Mess3 (easy):** recovered cleanly. Subspace decode R²≈0.998 (≈ supervised), `eig(A_x)≈eig(T^x)`,
  operator rollout regenerates the full fractal (R²≈1.00). Strongly observable ⇒ labels add ~nothing.
- **RRXOR (hard):** subspace recovers the diamond but diffusely. Per-(input,position) decode ≈0.65
  (vs supervised 0.91); **state-COM geometry R²≈0.82** with P(w) subspace (vs supervised 1.00).
  Unique-prefix / unweighted variants give ≈0.66–0.68. The gap is the **"value of the labels"**.
- **Determinant of difficulty = process OBSERVABILITY, not classical-vs-post-quantum.** Mess3 is
  classical and works; RRXOR is classical and is hard. (⇒ natural next experiment: a richly-sampled
  **post-quantum** process, where both subspace and operators should come out clean.)
- **Oracle control:** PCA-5 of the *true* 36 state-COMs decodes belief at only **R²=0.345** ⇒ belief
  is a **LOW-VARIANCE feature even among the state centres**; only *predictive* (not variance-based)
  subspaces work. Random-subspace control: apparent gains from higher latent dim are mostly
  regression overfitting on 36 COMs, not real recovery.
- **Operator ROLLOUT (regenerate every state from the learned operators, seeded at the root):**
  - Mess3: works (R²≈1.00) — its true operators are diagonalisable strict contractions whose
    attractor **is** the belief fractal; rollout self-corrects/denoises.
  - RRXOR: **collapses** (doesn't reach the outer apexes) for **every** subspace, supervised or
    unsupervised. **Intrinsic:** RRXOR's true operators are non-diagonalisable — `eig|T¹|=[0,0,0,0,0]`
    (rank-4 **nilpotent**), `eig|T⁰|=[0.63,0.63,0.63,0,0]` (**defective**, triple eigenvalue). The 36
    states are transients of a defective semigroup, not a contraction attractor. So for RRXOR the
    **raw subspace is the trustworthy representation; the rollout is not.** (Supervised-subspace
    rollout also collapses: 0.48 from raw 0.89 — confirms it's the process, not the estimator.)
- **The "two winners" puzzle is resolved:** whitening choice (CCA double-whiten vs RRR input-whiten
  vs none) is a near-wash; the real lever is **observation horizon/depth = observability**. A deep
  observable future (observability-OOM) is best-or-tied on both processes.
- **ALS is nearly inert (a 2×2 on the Mess3 rollout):** uniform+noALS 0.745, uniform+ALS 0.745,
  prob+noALS 0.996, prob+ALS 1.000. ⇒ **probability weighting** is what makes the Mess3 rollout crisp
  (it aligns the operator fit with the dense region the rollout traverses); ALS adds ~0.004 and is
  neutral-to-slightly-harmful on RRXOR. Kept in the recipe by user choice; don't over-credit it.

## 6. Visualisation & methodology decisions (important, settled)
- **Honest comparison convention:** identical pipeline, identical PCA, identical label-fit placement;
  the **only** thing that differs between supervised and unsupervised panels is the **affine-map
  input** — full residual stream vs the recovered subspace. Placement onto the simplex uses a
  **label-fit readout in both panels (display only)**; the unsupervised content is the subspace.
- **fig26 (RRXOR geometry) uses per-state-equal DISPLAY weighting** (user-endorsed as the honest
  visualisation — occurrence/P(w) display is central-heavy and understates the apexes). The **P(w)
  weighting is in the subspace LEARNING, not the display.** State-COM number (0.82) is
  display-weighting-independent.
- **Cross-validation splits (fig08):** each method judged on the toughest split it supports.
  **Final: 30/70 for BOTH** (train 30%, test 70%). The unsupervised needs ≥~30% train because its
  training unit is a **transition** (a prefix + ALL its children), which survives a random prefix
  split at only ~`f^4` (f=0.2 → ~50 transitions, marginal; f=0.3 → ~240, fine; f=0.8 → ~12k).
  Supervised CV-MSE 0.0004, unsupervised CV-MSE 0.0014 (both hold out 70%, both collapse under
  shuffle: sup 0.11 / uns 0.075). The 30/70 split detail lives in the write-up **text**, not the
  figure heading.
- **Distance/COM:** "COM" = centre of mass = per-state mean of the representation over the many
  prefixes that map to that state (RRXOR: ~1046 prefixes → 36 states). Averaging cancels within-state
  spread so the state-level geometry is clean.

## 7. Write-up structure (`writeup.tex`, current)
**Section A — Supervised reproduction**
- Mess3: `fig03` (paper Fig 5 fractal); `fig33_mess3_fig6_supervised.png` (paper Fig 6 combined:
  training-progression row + Cross-Validation + Shuffle + MSE bars **with broken x-axis**).
- RRXOR: `fig10_rrxor_residual_representation.png` (paper Fig 7C, GT-left | representation-right —
  the old standalone GT `fig09` was DROPPED; its mention now points to fig10's left panel);
  `fig34_rrxor_diagnostics_supervised.png` (paper **Fig 7D+E combined**: per-activation layerwise
  belief-MSE + the two distance-preservation hexbins).

**Section B — Supervised vs. unsupervised recovery** (OOM theory; NO rollouts)
- Mess3: `fig28_mess3_principled.png` (GT|sup|unsup geometry); `fig06_training_progression_mess3.png`
  (emergence, sup top row / unsup bottom row); `fig08_unsupervised_controls_mess3.png` (CV + shuffle +
  MSE bars, sup top / unsup bottom).
- RRXOR: `fig26_rrxor_principled.png` (GT|sup|unsup geometry); `fig35_rrxor_diagnostics_unsupervised.png`
  (unsupervised counterpart of fig34 — layerwise + distance).

**Appendix — Operator rollouts** (GT | subspace | rollout)
- Mess3: `fig29_mess3_supervised_rollout.png` (sup subspace), `fig23_mess3_unified.png` (unsup subspace).
- RRXOR: `fig29_rrxor_supervised_rollout.png` (sup), `fig24_rrxor_unified.png` (unsup) — both show the
  rollout collapse.

All figures are produced by `fig*.py` scripts of the same number; `fig34_rrxor_diagnostics.py`
produces **both** fig34 (supervised) and fig35 (unsupervised).

## 8. OPEN ISSUE to resolve next — fig35 distance "reversal" (READ THIS CAREFULLY)
The unsupervised RRXOR **distance-preservation** panel shows the recovered representation preserving
**next-token distance (R²≈0.65) MORE than belief distance (R²≈0.50)** — the *reverse* of the
supervised (belief ≈0.78 > next-token ≈0.31; intrinsic next-token↔belief baseline = 0.235).

**FINAL CONCLUSION (after I flip-flopped during the chat — trust this version):**
- This is a **GENUINE finding, NOT an artifact.** The paper's analysis ("method a") is:
  regress activation→belief, take each state's **per-prefix predicted-belief COM**, then distances.
  The supervised fig34 uses method (a) and **matches the paper**. The **rigorous unsupervised analogue
  is the same method (a)** applied to the recovered subspace (decode subspace→belief per-prefix → COM
  → distances), which is what fig35 already does.
- Under that proper analogue the reversal is real: next-token R²≈0.65 ≫ intrinsic 0.235 ⇒ the recovered
  representation is **actively next-token-dominated**, because the observability subspace is **anchored
  on the next-token observable `C`** (the first block of the observability matrix). Belief survives only
  as a low-variance linear feature (consistent with the oracle=0.345 result).
- A "method b" fix I floated — re-fitting the decode on the 36 COMs — **removes the reversal but is NOT
  analogous** (it inserts a COM-level refit the supervised analysis doesn't have). **Do NOT apply it.**
- **Pending cosmetic tweak only:** `fig34_rrxor_diagnostics.py`'s `collect_layers` builds its "Concat"
  from `hook_embed + 4×resid_post + ln_final` (6 components), whereas fig26/`F14._collect` use the
  `resid_post` concat. Aligning fig35's distance concat to `resid_post` gives ≈0.51/0.65 — **reversal
  unchanged**. Worth doing for consistency, doesn't change the conclusion. (The write-up text already
  frames the reversal as the genuine "C-anchored / weak-observability" finding — keep that framing.)
- Diagnostics that established this: `diag_distance_check.py` (compares per-prefix→COM "method a" vs
  COM→decode "method b" on fig26's exact representation).

## 9. Other pending items
- **Citation stub:** `writeup.tex` has `\bibitem{riechers2025}` = "Neural networks leverage nominally
  quantum and post-quantum representations" marked **"(verify authors/venue)"** — fill in correct cite.
- Author/title block in `writeup.tex` is empty.
- `RESULTS_SUMMARY.md` is stale (pre-dates these decisions) — ignore or update.

## 10. Key files
- `writeup.tex` — the report.
- `unsupervised_belief_oom.py` — core lib: `predictive_cca`, `als_refine_basis`, `fit_operators`,
  `rollout_states`, `recover_eval_functional`, `belief_decode_r2`, `rasterize_simplex`,
  `simplex_to_xy`, `collect_prefix_features_enumerated`.
- `fig14_observable_oom.py` — observability-OOM: `observable_subspace(…, wmap=)`,
  `analytic_prefix_probs`, `fit_at_dim`, `_collect`, `_load`.
- Figure scripts: `fig03, fig06, fig08, fig10, fig23, fig24, fig26, fig28, fig29, fig33, fig34` (→34+35).
- Diagnostics: `diag_distance_check.py`, `diag_methods_full.py`, `diag_cv_splits.py`, `diag_mess3_2x2.py`,
  `diag_unification.py`, `diag_operators.py`, `diag_push_rrxor.py`, `diag_head2head.py`.

## 11. Headline numbers (quick reference)
| quantity | Mess3 | RRXOR |
|---|---|---|
| unsup subspace decode (state-COM) | 0.998 | ~0.82 (P(w)) / ~0.66 (unique) |
| supervised state-COM | 1.00 | 1.00 |
| operator rollout | 1.00 (works) | collapses (nilpotent ops) |
| oracle PCA-5 of true COMs | — | 0.345 |
| distance: belief / next-token (sup) | ~1 / low | 0.78 / 0.31 |
| distance: belief / next-token (unsup) | — | 0.50 / 0.65 (genuine reversal) |
| CV 30/70 held-out MSE (sup / unsup) | 0.0004 / 0.0014 | — |
