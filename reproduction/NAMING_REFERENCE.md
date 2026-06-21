# Naming Reference: Figure Numbering

All figures and scripts are now numbered 1-10 in order of appearance in **writeup_v1.tex**.

## Paper Figures (in writeup order)

| # | Script | Figure | Content |
|---|--------|--------|---------|
| 1 | `fig1_mess3_principled.py` | `fig1_mess3_principled.png` | **Mess3**: ground truth, supervised, unsupervised |
| 2 | `fig2_training_progression.py` | `fig2_training_progression_mess3.png` | **Mess3** emergence over training |
| 3 | `fig3_unsupervised_controls.py` | `fig3_unsupervised_controls_mess3.png` | **Mess3** controls: cross-validation & shuffle |
| 4 | `fig4_rrxor_principled.py` | `fig4_rrxor_principled.png` | **RRXOR**: ground truth, supervised, unsupervised |
| 5 | `fig5_rrxor_diagnostics.py` | `fig5_rrxor_diagnostics_supervised.png` | **RRXOR** diagnostics: supervised |
| 6 | (from fig5) | `fig6_rrxor_diagnostics_unsupervised.png` | **RRXOR** diagnostics: unsupervised |
| 7 | `fig7_8_supervised_and_unsupervised_rollouts.py` | `fig7_mess3_supervised_rollout.png` | **Mess3** supervised subspace |
| 8 | `fig8_10_unified_best.py` | `fig8_mess3_unified.png` | **Mess3** spectral-OOM unified (rollout) |
| 9 | (from fig7) | `fig9_rrxor_supervised_rollout.png` | **RRXOR** supervised subspace |
| 10 | (from fig8) | `fig10_rrxor_unified.png` | **RRXOR** spectral-OOM unified (rollout) |

## Supporting Helper Scripts

These scripts provide utilities and support but aren't directly generating main paper figures:

- `fig09_rrxor_ground_truth.py` — Ground truth belief generation for RRXOR
- `fig10_rrxor_representation.py` — Supervised baseline representation (used by fig4)
- `fig13_rrxor_unsupervised_oom.py` — Predictive-CCA alternative method
- `fig14_observable_oom.py` — **CORE:** Observable-matrix subspace recovery (used by fig1-10)
- `fig16_oom_mess3_geometry.py` — Mess3 OOM geometry helper

## Core Methodology Modules

- `unsupervised_belief_oom.py` — **CORE:** Spectral-OOM methodology and utilities
- `train_mess3.py` — Train Mess3 transformer
- `train_rrxor.py` — Train RRXOR transformer

## Documentation

- `writeup_v1.tex` — The main thesis writeup
- `RESULTS_SUMMARY.md` — Comprehensive results and methodology summary
- `OPERATOR_COMPARISON_MESS3_VS_RRXOR.md` — Operator analysis
- `RRXOR_OPERATOR_ANALYSIS.md` — RRXOR operator properties
- `REPO_ORGANIZATION.md` — Repository structure guide
- `NAMING_REFERENCE.md` — This file

---

## Quick Reference: Running Scripts

```bash
# Generate all paper figures (in order)
python fig1_mess3_principled.py
python fig2_training_progression.py
python fig3_unsupervised_controls.py
python fig4_rrxor_principled.py
python fig5_rrxor_diagnostics.py
python fig7_8_supervised_and_unsupervised_rollouts.py
python fig8_10_unified_best.py

# Train models (optional, models already provided)
python train_mess3.py    # ~1 hour on GPU
python train_rrxor.py    # ~30 min on GPU

# Compile thesis
pdflatex writeup_v1.tex
```

---

**Last Updated:** 2026-06-21  
**Status:** Clean, organized, production-ready
