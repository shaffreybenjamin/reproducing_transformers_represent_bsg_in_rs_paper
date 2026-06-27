# Unused Code Archive

This folder contains all Python scripts that are not directly used to generate figures in the paper (writeup_v1.tex). These are:
- Diagnostic scripts that test specific hypotheses or design choices
- Alternative method implementations that didn't make it into the paper
- Ablation studies and comparisons
- Exploratory analysis

## Organization

### Diagnostic Scripts (13 files)
Systematic tests of specific design choices and hypotheses:

- **diag_push_rrxor.py** — **[HIGH VALUE]** Comprehensive exploration of 5 methods (M1–M5) to push RRXOR recovery past 0.635 ceiling: multi-horizon ridge RRR, predictive-future clustering, activation clustering, EM refinement, ragged-horizon RRR. Documents why the ceiling is fundamental. ~120 lines each.
- **diag_mess3_2x2.py** — Ablation grid: {uniform, prob-weight} × {no-ALS, ALS}. Simple justification of design choices.
- **diag_horizon_depth_rrxor.py** — Tests whether longer observability horizons (L=3..15) improve RRXOR. Answer: no.
- **diag_weighting_rrxor.py** — Tests whether P(w)-weighting is responsible for poor RRXOR recovery.
- **diag_methods_full.py** — Definitive method comparison: CCA+ALS vs OBS-OOM vs OBS+ALS.
- **diag_mess3_operators.py** — Verify Mess3 operators are diagonalizable strict contractions.
- **diag_operators.py** — Confirm RRXOR operator pathology (non-diagonalizable, nilpotent).
- **diag_rrxor_operators.py** — Verify RRXOR operators are non-diagonalizable.
- **diag_horizon_depth_rrxor_fig26match.py** — Horizon sweep with fig26's exact data pipeline.
- **diag_distance_check.py** — Distance reversal diagnostic.
- **diag_head2head.py** — CCA vs observability OOM comparison.
- **diag_cv_splits.py** — Cross-validation at different train fractions.
- **diag_unification.py** — Addresses two open questions.

### Alternative Methods & Variants (15 files)
Different unsupervised approaches tested. All are superseded by the observability-OOM method chosen for the paper:

- **fig15_oom_denoised.py** — **[VALUABLE NEGATIVE RESULT]** Tests two refinements (EM rollout-denoising, noise-whitening); documents why both fail. Clear mechanistic explanation of limitations.
- **fig13_rrxor_unsupervised_oom.py** — Predictive-CCA for RRXOR (0.57 COM-geom vs 0.68 observability).
- **fig19_rrr_spectral.py** — Reduced-rank/spectral OOM (0.65 vs 0.68 observability).
- **fig17_rrxor_method_comparison.py** — CCA vs observability comparison.
- **fig20_rrxor_operator_rollout.py** — CCA operator rollout (collapses to 0.37).
- **fig21_rrxor_observability_rollout.py** — Observability OOM operator rollout (collapses to 0.26).
- **fig25_rrxor_multihorizon.py** — Multi-horizon ridge RRR variant.
- **fig27_rrxor_fig14scheme.py** — Variant matching fig14 orphan scheme.
- **fig30_mess3_uniform_als.py** — Uniform weighting + ALS ablation.
- **fig31_no_als.py** — Operator rollout without ALS (tests importance of ALS).
- **fig32_subspace_pw.py** — Probability-weighted subspace variant.
- **fig22_methodology_synthesis.py** — Synthesis figure of two open questions.
- **fig03_residual_belief_geometry.py**, **fig03b_residual_belief_enumerated.py**, **fig03c_pca_improved.py** — Multiple versions of paper's Mess3 belief-geometry figure.

### Supervised Reproductions & Variants (4 files)
- **fig01_belief_simplex.py** — Mess3 ground truth simplex (not in paper).
- **fig07_supervised_controls.py** — Paper's supervised controls (variant of controls in fig08).
- **fig11_rrxor_layerwise_mse.py** — Layerwise MSE (not in final paper).
- **fig12_rrxor_distances.py** — Distance preservation (not in final paper).
- **fig33_mess3_fig6_supervised.py** — Paper's Fig 6 supervised (variant).

### Training & Checkpoints (1 file)
- **train_progression_checkpoints.py** — Generates training checkpoints for visualization (used by fig06).

### Documentation (1 file)
- **CHAT_HANDOFF.md** — Previous handoff notes.

## Decision Guide

**Definitely Keep:**
- diag_push_rrxor.py (systematic exploration of observability ceiling)
- fig15_oom_denoised.py (documents why noise-denoising fails)

**Probably Keep:**
- diag_mess3_2x2.py (justifies design choices)
- diag_horizon_depth_rrxor.py (tests horizon length)
- diag_methods_full.py (method comparison)

**Safe to Remove:**
- All fig13, fig17, fig19–21, fig25, fig27 (methods superseded by observability-OOM)
- fig30, fig31, fig32 (simple ablations; reasons documented elsewhere)
- fig01, fig03*, fig07, fig11, fig12, fig33 (variant reproductions)
- All diag_* except those above (intermediate analysis)
