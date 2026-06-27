# Spectral OOM Project — Unused Code and Figures Archive

This folder contains all exploratory code, diagnostic scripts, alternative methods, and unused figures from the spectral OOM paper reproduction project.

**Created:** 2026-06-21  
**Total size:** ~30 MB

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
