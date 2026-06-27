# Unused Figures Archive

This folder contains all PNG figures that are **not** used in writeup_v1.tex. These were generated during exploration, as variants, or as outputs from diagnostic scripts.

Total: 36 unused figures.

## By Category

### Diagnostic/Horizon Tests (3 files)
- `diag_horizon_best_r2.png` — Horizon length sweep for RRXOR
- `diag_horizon_decode_vs_d.png` — Decode R² vs subspace dimension
- `diag_horizon_spectra.png` — Observability spectra at different horizons
- `diag_weighting_comparison.png` — P(w)-weighting diagnostic

### Alternative Method Outputs (12 files)
From methods that underperform observability-OOM:
- `fig04_unsupervised_oom_mess3.png` — CCA on Mess3
- `fig13_rrxor_unsupervised_oom.png` — CCA on RRXOR (decode curves)
- `fig14_observable_oom.png` — OOM decode-vs-d + eigenvalue check
- `fig14_observable_oom_rrxor.png` — Orphan from early fig14 version
- `fig15_oom_denoised.png` — Noise-aware OOM (negative result)
- `fig17_rrxor_cca_recovery.png` — CCA on RRXOR (0.57 vs 0.68 observability)
- `fig18_rrxor_observability_recovery.png` — Observability OOM on RRXOR (0.68)
- `fig19_rrr_spectral_rrxor.png` — Reduced-rank/spectral (0.65)
- `fig20_rrxor_operator_rollout.py` — CCA rollout (collapses)
- `fig21_rrxor_observability_rollout.py` — Observability rollout (collapses)
- `fig22_methodology_synthesis.png` — Open questions synthesis
- `fig25_rrxor_multihorizon.png` — Multi-horizon ridge RRR variant

### Mess3 Variants (6 files)
Different versions/snapshots of Mess3 figures:
- `fig01_belief_simplex_mess3.png` — Ground truth simplex
- `fig01_compare_params.png` — Parameter comparison
- `fig02_training_loss_mess3.png` — Training loss curve
- `fig03_residual_belief_geometry_mess3.png` — Belief geometry variant
- `fig03b_residual_belief_enumerated_mess3.png` — MSP enumeration variant
- `fig03c_pca_improved_mess3.png` — PCA refinement variant

### RRXOR Supervised Reproductions (4 files)
Not in final paper (supervised baselines only used for comparison):
- `fig09_rrxor_ground_truth.png` — Ground truth RRXOR (used by other scripts but not in paper)
- `fig10_rrxor_residual_representation.png` — Supervised representation (paper's Fig 7C baseline)
- `fig11_rrxor_layerwise_mse.png` — Layerwise MSE
- `fig12_rrxor_distances.png` — Distance preservation

### Supervised Controls & Variants (2 files)
- `fig07_supervised_controls_mess3.png` — Supervised controls variant
- `fig33_mess3_fig6_supervised.png` — Fig 6 supervised variant

### Ablations (6 files)
Design choice ablations:
- `fig30_mess3_uniform_als.png` — Uniform weighting + ALS
- `fig31_mess3_no_als.png` — No ALS (Mess3)
- `fig31_rrxor_no_als.png` — No ALS (RRXOR)
- `fig32_mess3_subspace_pw.png` — P(w)-weighted subspace (Mess3)
- `fig32_rrxor_subspace_pw.png` — P(w)-weighted subspace (RRXOR)
- `fig_rrxor_training_loss.png` — Training loss curve

### Other Variants (3 files)
- `fig27_rrxor_fig14scheme.png` — Variant matching fig14 orphan scheme
- `fig16_oom_mess3_geometry.png` — Mess3 OOM geometry (positive result, but not in final paper)

## Decision Guide

**Safe to Remove:**
- All diagnostic figures (diag_*.png) — summary of results documented elsewhere
- All alternative method figures (fig13–22) — observability-OOM chosen
- All Mess3 variants (fig01–03*) — one clean version in main figures
- All RRXOR supervised figures (fig09–12) — not in final writeup
- All ablation figures (fig30–32) — design choices justified in code

**Worth Keeping Locally for Reference:**
- `fig16_oom_mess3_geometry.png` — Nice clean positive result
- `fig15_oom_denoised.png` — Documents negative result mechanically

## Regenerating Figures

If you need to regenerate any of these:
1. See `_unused_code/README.md` for the corresponding script
2. Run the script from the main `reproduction/` directory
3. Figure files are written to `figures/` by default
