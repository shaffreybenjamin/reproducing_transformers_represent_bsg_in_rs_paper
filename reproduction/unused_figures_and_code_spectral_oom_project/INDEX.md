# Spectral OOM Project — Unused Code and Figures Archive

This folder contains all exploratory code, diagnostic scripts, alternative methods, and unused figures from the spectral OOM paper reproduction project.

**Created:** 2026-06-21  
**Total size:** ~30 MB

## 2026-06-30 batch — two-stage estimator & d_selection (retired threads)

A second archival pass. Threads confirmed finished and moved here:

- **Two-stage estimator** (superseded by the consistent-CCA approach):
  `_unused_code/two_stage/` — the `stage0/stage1/stage2` pipeline
  (`run_two_stage_estimator.py`, `stage1_general.py`, `stage1_cca_general.py`,
  `*_output.pkl`), the `fig*_principled_stage1.py` / `fig_stage0_gate` /
  `fig_stage2_cluster_separation` plot variants, and `fig14_cca_rrr_estimator_twostage.py`.
  Matching outputs in `_unused_figures/` (`*_principled_stage1.png`,
  `stage2_cluster_separation_results.png`). Docs in `_unused_docs/`
  (`TWO_STAGE_*`, `QUICK_START_TWO_STAGE.md`, `two_stage_estimator_spec.md`).
- **d_selection sweep** (model-order elbow study, conclusions folded into method):
  `_unused_code/d_selection/` — the `fig1_*_d_selection.py` plots plus the whole
  `d_selection/` and `d_selection_results/` directories. Doc: `IMPLEMENTATION_D_SELECTION.md`.
- **Stale reports & logs** → `_unused_docs/` (`RESULTS_SUMMARY.md`,
  `IMPLEMENTATION_REVIEW.md`, `IMPLEMENTATION_SUMMARY.md`, `RRXOR_OPERATOR_ANALYSIS.md`,
  `OPERATOR_COMPARISON_MESS3_VS_RRXOR.md`, `REPO_ORGANIZATION.md`, `compass_artifact_*.md`)
  and `_unused_logs_artifacts/` (empty/old run logs, orphaned `writeup.*` LaTeX build files
  from the retired `writeup.tex`).

**Deliberately KEPT in the live tree** (do not assume these are dead): the Hankel
two-factor estimator (`estimators/spectral_oom_hankel.py`, `run_hankel_*`, `fig1_*_hankel.py`),
`fig7_8_operator_rollouts_simplex.py`, all current `_*.py` CCA/Hankel exploration,
the 7-process `fig1_*_principled.py`, and the library modules imported by the writeup
generators (`fig09/fig10/fig13/fig14/fig16`, `fig8_10_unified_best`, `unsupervised_belief_oom`).

## Contents

### `_unused_code/` — 35 files
Exploratory scripts organized by category:

- **Diagnostic scripts** (13 files) — Tests of specific hypotheses and design choices
  - `diag_push_rrxor.py` **[HIGH VALUE]** — Comprehensive exploration of 5 methods to overcome RRXOR ceiling
  - `diag_mess3_2x2.py` — Ablation grid justifying design choices
  - `diag_horizon_depth_rrxor.py` — Horizon length sweep
  - And 10 more diagnostic scripts

- **Alternative methods** (15 files) — Methods tested but not used in final paper
  - `fig15_oom_denoised.py` **[VALUABLE]** — Documents why noise-denoising improvements fail
  - `fig13_rrxor_unsupervised_oom.py` — Predictive-CCA for RRXOR
  - `fig19_rrr_spectral.py`, `fig20`, `fig21` — Alternative OOM variants
  - And more

- **Supervised reproductions & variants** (4 files)
  - `fig01_belief_simplex.py`, `fig07_supervised_controls.py`, etc.

- **Earlier writeup drafts** (4 files)
  - `writeup.tex`, `writeup.pdf` — Earlier version
  - `writeup_methodology.tex`, `writeup_methodology.pdf` — Methodology draft

**See `_unused_code/README.md` for detailed descriptions and decision guide.**

### `_unused_figures/` — 36 PNG files
All figure outputs that were not used in writeup_v1.tex:

- Diagnostic outputs (3 files) — Results from horizon sweeps, weighting tests
- Alternative method outputs (12 files) — From superseded methods
- Mess3 variants (6 files) — Different versions of the same figure
- RRXOR supervised outputs (4 files) — Not in final paper
- Ablation outputs (6 files) — Design choice experiments
- Other variants (3 files)

**See `_unused_figures/README.md` for detailed descriptions and decision guide.**

## How to Use This Archive

### Review Offline (Recommended)
1. Download to your laptop:
   ```bash
   scp -r <user>@<server>:/path/to/unused_figures_and_code_spectral_oom_project ~/thesis-archive/
   ```

2. Open and review:
   - `_unused_code/README.md` — Decision guide for scripts
   - `_unused_figures/README.md` — Decision guide for figures
   - Individual scripts marked `[HIGH VALUE]` or `[VALUABLE]`

3. Decide what to keep/discard based on relevance to future work

### Delete From Repo (After Review)
Once you've reviewed locally and decided what's safe to delete:
```bash
rm -rf reproduction/unused_figures_and_code_spectral_oom_project/
```

### Selectively Restore
If you change your mind about a specific script or figure:
```bash
# Copy from local archive back to repo
cp ~/thesis-archive/unused_figures_and_code_spectral_oom_project/_unused_code/diag_push_rrxor.py reproduction/
```

## Key Findings Documented Here

### High-Value Exploratory Work
- **diag_push_rrxor.py** — Tests 5 independent approaches to overcome the 0.635 RRXOR ceiling. Documents why it's fundamental, not a method artifact.
- **fig15_oom_denoised.py** — Negative result that prevented wasted effort: noise-denoising doesn't help because within-state noise isn't the binding constraint.

### Methods That Didn't Make the Paper
- Predictive-CCA (0.57 RRXOR vs 0.68 observability-OOM)
- Reduced-rank/spectral (0.65 RRXOR)
- Various operator refinements and rollout variants

### Design Choices Justified
- ALS operator refinement (critical for Mess3; collapses RRXOR)
- Probability-weighted operator fitting (vs uniform weighting)
- Observability horizon length (L=3 is optimal for RRXOR)

## Questions?

Refer to:
- `_unused_code/README.md` for script details
- `_unused_figures/README.md` for figure catalog
- `RESULTS_SUMMARY.md` (in main reproduction dir) for methodology overview
