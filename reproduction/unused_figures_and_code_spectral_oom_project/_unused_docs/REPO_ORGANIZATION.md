# Repository Organization

**Last Updated:** 2026-06-21

This document describes the current structure of the reproduction folder after the cleanup.

## Current Structure

### Core Scripts (in main directory)
These 15 Python files are **required to generate the figures in writeup_v1.tex**:

```
fig06_training_progression.py       — Fig 6 (Mess3 training progression)
fig08_unsupervised_controls.py       — Fig 8 (Mess3 controls)
fig23_unified_best.py                — Fig 23 & 24 (unified best method)
fig26_rrxor_principled.py            — Fig 26 (RRXOR principled comparison)
fig28_mess3_principled.py            — Fig 28 (Mess3 principled comparison)
fig29_supervised_rollout.py          — Fig 29 (supervised rollouts)
fig34_rrxor_diagnostics.py           — Fig 34 & 35 (RRXOR diagnostics)
```

Supporting modules:
```
unsupervised_belief_oom.py           — Core unsupervised OOM methodology
fig14_observable_oom.py              — Observable matrix subspace recovery
fig09_rrxor_ground_truth.py          — RRXOR ground truth beliefs (helper)
fig10_rrxor_representation.py        — RRXOR supervised baseline (helper)
fig13_rrxor_unsupervised_oom.py      — RRXOR CCA method (helper)
fig16_oom_mess3_geometry.py          — Mess3 OOM geometry (helper)
```

Training scripts:
```
train_mess3.py                       — Train Mess3 transformer (1M steps)
train_rrxor.py                       — Train RRXOR transformer
```

### Core Figures (figures/)
10 PNG files **used in writeup_v1.tex**:

```
fig06_training_progression_mess3.png
fig08_unsupervised_controls_mess3.png
fig23_mess3_unified.png
fig24_rrxor_unified.png
fig26_rrxor_principled.png
fig28_mess3_principled.png
fig29_mess3_supervised_rollout.png
fig29_rrxor_supervised_rollout.png
fig34_rrxor_diagnostics_supervised.png
fig35_rrxor_diagnostics_unsupervised.png
```

### Documentation (in main directory)
```
RESULTS_SUMMARY.md                   — Comprehensive results handoff
OPERATOR_COMPARISON_MESS3_VS_RRXOR.md — Operator analysis (Mess3 vs RRXOR)
RRXOR_OPERATOR_ANALYSIS.md           — RRXOR operator verification
writeup_v1.tex                       — The thesis writeup
REPO_ORGANIZATION.md                 — This file
```

### Unused Code Archive (_unused_code/)
**33 Python files** not used in the final paper:
- 13 diagnostic scripts (explore design choices, test hypotheses)
- 15 alternative method implementations (methods that didn't make the paper)
- 4 supervised reproductions (variant versions)
- 1 training checkpoint script

See `_unused_code/README.md` for detailed descriptions and a guide to deciding what to keep.

### Unused Figures Archive (figures/_unused_figures/)
**36 PNG files** generated during exploration:
- 3 diagnostic outputs (horizon tests, weighting tests)
- 12 alternative method outputs (from superseded methods)
- 6 Mess3 variants (different versions of the same figure)
- 4 RRXOR supervised outputs (not in final paper)
- 2 supervised controls/variants
- 6 ablation outputs
- 3 other variants

See `figures/_unused_figures/README.md` for detailed descriptions.

## Recommended Next Steps

1. **Download the archived folders locally:**
   ```bash
   # Download to your laptop
   scp -r user@server:/path/to/reproduction/_unused_code ~/thesis-archive/
   scp -r user@server:/path/to/reproduction/figures/_unused_figures ~/thesis-archive/
   ```

2. **Review offline:**
   - Open `_unused_code/README.md` and `figures/_unused_figures/README.md`
   - Examine scripts from the [HIGH VALUE] and [VALUABLE] categories
   - Decide which exploratory ideas are worth keeping

3. **Delete from repo once you've decided:**
   ```bash
   # After reviewing locally and deciding what's safe to delete:
   rm -rf reproduction/_unused_code/
   rm -rf reproduction/figures/_unused_figures/
   ```

4. **If you change your mind later:**
   - The archives are preserved locally on your laptop
   - You can selectively restore interesting scripts/figures to the repo
   - Or run the original scripts again if you have the code

## Quick Scripts Reference

### To regenerate any figure from the paper:
```bash
cd reproduction
python fig28_mess3_principled.py      # Or any other fig*.py
```

### To retrain models:
```bash
cd reproduction
python train_mess3.py                  # ~1 hour on GPU
python train_rrxor.py                  # ~30 min on GPU
```

### To run core analysis:
```bash
python -c "import unsupervised_belief_oom as U; ..."
```

## File Sizes

Current repo state:
- Core .py files: ~200 KB
- Core .png figures: ~5 MB
- Unused code archive: ~250 KB
- Unused figures archive: ~30 MB

Total size of archived material: ~30 MB (can be deleted after review)

## Version Control

After deleting unused folders, commit the clean state:
```bash
git add -A
git commit -m "Archive unused exploratory code and figures to local directories"
git push
```

Your clean repo with just the paper-producing code and documentation is now ready for the next research phase.
