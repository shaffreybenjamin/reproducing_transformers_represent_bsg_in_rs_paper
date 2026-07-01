# D-Selection Implementation Summary

**Date:** 2026-06-25  
**Status:** Complete and ready for testing  
**Modified existing files:** 0  
**New files:** 11  

## What Was Implemented

Dimensionality selection methods (effective rank + MI saturation) integrated with existing spectral-OOM and CCA/RRR estimators, ready to test on 7 known processes before Pythia 70M experiment.

## Files Added

### Core Modules
1. **`d_selection.py`** (180 lines)
   - `effective_rank()` — compute d for given variance thresholds (95%, 99%)
   - `mutual_information_saturation()` — find d where MI plateaus
   - `compare_d_selection()` — compare methods and report agreement

2. **`d_selection_pipeline.py`** (240 lines)
   - `run_d_selection_pipeline()` — unified orchestrator for any HMM process
   - Runs spectral-OOM + CCA/RRR in sequence
   - Applies both d-selection methods
   - Returns comprehensive results dict

### Process-Specific Scripts (x7)
3. **`fig1_zero_one_random_d_selection.py`** — d=3, vocab=2
4. **`fig1_fern_d_selection.py`** — d=3, vocab=2
5. **`fig1_strata_d_selection.py`** — d=3, vocab=2
6. **`fig1_wing_d_selection.py`** — d=3, vocab=2
7. **`fig1_arch_d_selection.py`** — d=4, vocab=3
8. **`fig1_mess3_d_selection.py`** — d=3, vocab=3
9. **`fig1_rrxor_d_selection.py`** — d=5, vocab=2

Each script:
- Loads pre-trained model
- Builds HMM process
- Calls `run_d_selection_pipeline()`
- Writes results to `d_selection_PROCESS_results.txt`

### Orchestrator & Documentation
10. **`run_d_selection_all_7_processes.py`** (180 lines)
    - Runs all 7 processes sequentially
    - Generates comprehensive report
    - Creates summary table for comparison

11. **`D_SELECTION_README.md`** (Documentation)
    - Architecture overview
    - Usage instructions
    - Expected results + success criteria
    - Workflow for Pythia experiment

Plus this file: **`IMPLEMENTATION_D_SELECTION.md`**

## Integration with Existing Code

### Reused (No Changes)
```python
# From unsupervised_belief_oom.py
U.collect_prefix_features_enumerated()
U.observable_subspace()  # returns U, sv ← we use sv

# From stage1_cca_general.py
S1.direct_multistep_readouts()  # returns U, sv ← we use sv
S1.cca_rrr_subspace_ledoit()
```

### Pattern
```
collect features → run spectral-OOM → extract sv → d_selection
                ↓ run CCA/RRR → extract sv → d_selection
                ↓
                compare results → report
```

## How It Works

### Effective Rank
```
Given: singular values [σ₁, σ₂, ...]
Compute: cumulative variance = Σσᵢ²/Σσᵢ²
For threshold τ (0.95, 0.99):
  d_τ = arg min{d : cumvar_d ≥ τ}
```
Simple, fast, interpretable.

### MI Saturation
```
For d = 1, 2, 3, ..., max_d:
  1. Project activations: A_d = A @ U[:,:d]
  2. Train linear decoder: A_d → softmax
  3. Compute R² on training set (proxy for MI)
  
Find d where improvement < plateau_threshold × max_improvement
  Default threshold = 5%
```

Captures information content without unsupervised learning.

## Testing & Validation

### Quick Manual Test
```bash
cd reproduction
python fig1_mess3_d_selection.py
# Expected: effective rank ≈ 3, MI saturation ≈ 3
cat d_selection_mess3_results.txt
```

### Full Test Suite (7 processes)
```bash
python run_d_selection_all_7_processes.py
# Generates: D_SELECTION_COMPREHENSIVE_REPORT.txt
```

## Success Criteria

✓ Effective Rank ER95/ER99 recover true d (±1)  
✓ MI Saturation recovers true d (±1)  
✓ Both methods agree on most processes  
✓ Spectral-OOM vs CCA/RRR show different trade-offs  

## Next Phase: Pythia 70M

Once validated on 7 processes:
1. Apply `run_d_selection_pipeline()` to Pythia 70M
2. Use selected d for belief extraction
3. Validate extracted beliefs via:
   - Next-token prediction
   - Attention-belief alignment
   - Information content

## Code Quality

- **Lines of code:** ~700 total (new)
- **Modifications to existing:** 0
- **Test coverage:** 7 known processes with ground truth
- **Documentation:** Inline docstrings + README
- **Error handling:** Try-catch in MI saturation with fallback

## Future Enhancements (Post-Validation)

1. **TDA (Topological Data Analysis)** — detect intrinsic manifold dimensionality
2. **Cross-validation R²** — replace training R² in MI saturation
3. **Spectrum visualization** — plot singular values with d selections marked
4. **Batch processing** — for Pythia at scale (now linear, could optimize)

---

## Files Ready for Commit

All 11 new files are:
- ✓ Complete and self-contained
- ✓ Minimal dependencies on existing code
- ✓ Well-documented
- ✓ Ready for testing
- ✓ No breaking changes

**Recommendation:** Run all 7 processes first to validate d-selection methods before moving to Pythia.
