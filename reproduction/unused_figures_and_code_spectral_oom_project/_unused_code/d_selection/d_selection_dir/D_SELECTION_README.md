# D-Selection Implementation: Effective Rank + MI Saturation

## Overview

This implementation adds dimensionality selection (d-selection) methods to your existing spectral-OOM and CCA/RRR estimators. Two complementary methods are used:

1. **Effective Rank** — counts singular values needed to explain 95% and 99% of variance
2. **MI Saturation** — finds the dimensionality where mutual information with next-token distribution plateaus

These methods enable **unsupervised dimensionality selection** on the 7 known processes before applying to Pythia 70M.

## Architecture (Minimal Changes)

### New Files Created

```
reproduction/
├── d_selection.py                           # Core d-selection functions (NEW)
├── d_selection_pipeline.py                  # Unified pipeline for any process (NEW)
│
├── fig1_zero_one_random_d_selection.py      # D-selection for zero_one_random (NEW)
├── fig1_fern_d_selection.py                 # D-selection for fern (NEW)
├── fig1_strata_d_selection.py               # D-selection for strata (NEW)
├── fig1_wing_d_selection.py                 # D-selection for wing (NEW)
├── fig1_arch_d_selection.py                 # D-selection for arch (NEW)
├── fig1_mess3_d_selection.py                # D-selection for mess3 (NEW)
├── fig1_rrxor_d_selection.py                # D-selection for rrxor (NEW)
│
├── run_d_selection_all_7_processes.py       # Orchestrator: run all 7 (NEW)
│
├── unsupervised_belief_oom.py               # (existing — NO CHANGES)
├── fig14_observable_oom.py                  # (existing — NO CHANGES)
├── stage1_cca_general.py                    # (existing — NO CHANGES)
└── ...
```

### Key Design Decisions

1. **No modifications to existing code** — all new functionality in separate modules
2. **Reuse existing interfaces** — `observable_subspace()` and `direct_multistep_readouts()` already return singular values
3. **Pipeline pattern** — `d_selection_pipeline.py` orchestrates the workflow for any process
4. **Process-specific scripts** — 7 thin wrappers (`fig1_*_d_selection.py`) that call the pipeline

## Module Descriptions

### `d_selection.py`
Core d-selection methods (no dependencies on HMM or models).

**Main functions:**

```python
effective_rank(singular_values, variance_thresholds=(0.95, 0.99))
    → dict: {threshold: d}
    
    Returns dimensionality needed to explain given variance fractions.
    E.g., {"0.95": 3, "0.99": 5}
```

```python
mutual_information_saturation(A, softmax_targets, basis_U, max_d=None, 
                              plateau_threshold=0.05, ridge_alpha=1e-2)
    → (d_selected, mi_curve, explanation)
    
    Finds d where mutual information with next-token distribution plateaus.
    - Trains linear decoders for d=1..max_d
    - Identifies plateau as first d where improvement < 5% of max
```

```python
compare_d_selection(effective_rank_results, mi_d, true_d=None, verbose=False)
    → dict with comparison results and agreement metrics
```

### `d_selection_pipeline.py`
Unified pipeline for any HMM process.

**Main function:**

```python
run_d_selection_pipeline(model, hmm, context_len, device, process_name, 
                         true_d, max_d_test=50, verbose=True)
    → dict with all results
    
    Orchestrates:
    1. Collect prefix features (activations, softmax, beliefs)
    2. Run spectral-OOM (observable_subspace)
    3. Run CCA/RRR (direct_multistep_readouts + cca_rrr_subspace_ledoit)
    4. Apply d-selection methods to both
    5. Report and compare
```

Returns:
- `spectral_oom`: dict with ER95, ER99, MI saturation, singular values, basis
- `cca_rrr`: dict with same structure
- `ground_truth_beliefs`: for reference and future analysis

### `fig1_*_d_selection.py` (x7)
Process-specific entry points. Each follows identical pattern:

```python
# Load process-specific model
# Build HMM
# Call d_selection_pipeline
# Write results to d_selection_PROCESS_results.txt
```

Example usage:
```bash
python fig1_arch_d_selection.py
python fig1_mess3_d_selection.py
python fig1_rrxor_d_selection.py
```

### `run_d_selection_all_7_processes.py`
Orchestrator that runs all 7 processes sequentially and generates:
1. Individual result files (`d_selection_PROCESS_results.txt`)
2. Comprehensive report (`D_SELECTION_COMPREHENSIVE_REPORT.txt`)
3. Summary table for easy comparison

## Usage

### Run a single process:
```bash
cd reproduction
python fig1_arch_d_selection.py
```

Output:
- Console: verbose progress + summary
- File: `d_selection_arch_results.txt`

### Run all 7 processes:
```bash
cd reproduction
python run_d_selection_all_7_processes.py
```

Output:
- Console: progress for each process
- Individual files: `d_selection_PROCESS_results.txt` (one per process)
- Summary: `D_SELECTION_COMPREHENSIVE_REPORT.txt`

### Inspect results:
```bash
cat d_selection_mess3_results.txt
cat D_SELECTION_COMPREHENSIVE_REPORT.txt
```

## Expected Results

For reference, the true dimensionalities are:

| Process | True d | Vocab | Notes |
|---------|--------|-------|-------|
| zero_one_random | 3 | 2 | Simpler stochastic process |
| fern | 3 | 2 | Deterministic |
| strata | 3 | 2 | Strong structure |
| wing | 3 | 2 | Good recovery expected |
| arch | 4 | 3 | Tetrahedron |
| mess3 | 3 | 3 | Paper baseline, Sierpinski fractal |
| rrxor | 5 | 2 | Next-token degenerate, challenging |

### Success Criteria

✓ **Effective Rank + MI Saturation recover true d (±1) on all 7**
  - e.g., if true d=3, either method should select 2, 3, or 4
  
✓ **Agreement across methods on most processes**
  - Both ER95 and MI should pick same d (or differ by ±1)

✓ **Spectral-OOM vs CCA/RRR comparison**
  - Which method produces cleaner spectrum (sharper elbow)?
  - Which method's d-selection is more reliable?

## Workflow for Pythia Experiment

Once d-selection is validated on the 7 known processes:

1. Apply same pipeline to Pythia 70M
2. Use selected d for subsequent belief extraction
3. Validate extracted beliefs via mechanistic interpretability

The validated d-selection methods provide confidence that automatic dimensionality detection works before scaling to real models.

## Modification Summary

**Existing files touched:** NONE (zero modifications to existing code)

**New files:** 11
- Core modules: `d_selection.py`, `d_selection_pipeline.py`
- Process scripts: 7 × `fig1_*_d_selection.py`
- Orchestrator: `run_d_selection_all_7_processes.py`
- Documentation: `D_SELECTION_README.md` (this file)

**Total LOC added:** ~700 lines (minimal, focused)

**Interfaces reused:**
- `U.collect_prefix_features_enumerated()` — existing, unchanged
- `U.observable_subspace()` — existing, already returns sv
- `S1.direct_multistep_readouts()` — existing, already returns sv
- `S1.cca_rrr_subspace_ledoit()` — existing, unchanged

## Next Steps

1. Run `python run_d_selection_all_7_processes.py` to test on known processes
2. Review `D_SELECTION_COMPREHENSIVE_REPORT.txt` for method performance
3. If successful (methods recover true d), proceed to Pythia 70M experiment
4. Use selected d values as input to Pythia analysis

---

**Implementation date:** 2026-06-25  
**Contact:** For questions on d-selection methods, see `d_selection.py` docstrings.
