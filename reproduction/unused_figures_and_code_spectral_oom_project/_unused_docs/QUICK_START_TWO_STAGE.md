# Quick Start: Two-Stage Estimator

## Run Everything (Recommended)

```bash
cd reproduction
python run_two_stage_estimator.py
```

**Output:**
- Comprehensive report to stdout
- 3 pickle files: `stage0_gate_output.pkl`, `stage1_cca_output.pkl`, `stage2_cluster_separation_output.pkl`
- Plot: `figures/stage2_cluster_separation_results.png`
- Summary: `two_stage_summary.pkl`

**Expected runtime:** ~5-10 minutes (GPU), ~15-20 minutes (CPU)

---

## Run Individual Stages (Debugging)

### Stage 0 (Gate)
```bash
python fig_stage0_gate_rrxor.py
```
**Tests:** Do 36 belief states separate in supervised 5D subspace?  
**Check output for:** `Gate status: PASS` ✓

### Stage 1 (CCA/RRR + Ledoit-Wolf)
```bash
python fig14_cca_rrr_estimator_twostage.py
```
Requires: `stage0_gate_output.pkl` (auto-run Stage 0 if missing)  
**Tests:** Isolate belief-bearing region (d=8-10)  
**Check output for:** CCA spectrum, d_stage1=10

### Stage 2 (Cluster Separation)
```bash
python fig_stage2_cluster_separation.py
```
Requires: `stage0_gate_output.pkl`, `stage1_cca_output.pkl`  
**Tests:** Phase A (oracle) and Phase B (unsupervised) cluster-based re-ranking  
**Check output for:** `Selected d = 5` and plateau plot

---

## Inspect Results in Python

```python
import pickle

# Stage 0: Gate metrics
with open("stage0_gate_output.pkl", "rb") as f:
    s0 = pickle.load(f)
print(f"Gate pass: {s0['gate_pass']}")
print(f"Separation ratio: {s0['separation_ratio']:.4f}")
print(f"Mushy states: {s0['mushy_states']}")

# Stage 2: Final results
with open("stage2_cluster_separation_output.pkl", "rb") as f:
    s2 = pickle.load(f)
print(f"Selected d: {s2['plateau_d']}")
print(f"Acceptance criteria PASS: {s2['overall_pass']}")

# Phase B scores
phase_b = s2['phase_b_scores']
for d in sorted(phase_b.keys()):
    r2 = phase_b[d]['state_r2']
    ratio = phase_b[d]['separation_ratio']
    print(f"  d={d}: state_r2={r2:.4f}, sep_ratio={ratio:.4f}")
```

---

## Expected Output Example

### If successful (d=5 recovery):
```
Stage 0: Gate PASS, separation_ratio=0.52
Stage 1: CCA spectrum smooth, d_stage1=10 selected
Stage 2 Phase A: state_r2(d=5)=0.85
Stage 2 Phase B: plateau at d=5, state_r2(d=5)=0.83

Comparison table:
  spectral-OOM    | d=5 | 0.816
  supervised      | d=5 | 1.000
  two-stage(B)    | d=5 | 0.830 ← beats baseline!

✓ ACCEPTANCE CRITERIA: PASS
```

### If plateau wrong (d≠5):
```
Stage 2 Phase B: plateau at d=7
Selected d = 7

✗ CRITERION 1: FAIL (plateau at d=7, not d=5)
```

### If separation weak (low R²):
```
Stage 2 Phase B: plateau at d=5, state_r2(d=5)=0.80

✗ CRITERION 2: FAIL (0.80 ≤ 0.816 baseline)
```

---

## Files Created

```
reproduction/
├── fig_stage0_gate_rrxor.py                       [13 KB] Stage 0: gate
├── fig14_cca_rrr_estimator_twostage.py            [14 KB] Stage 1: CCA/RRR
├── fig_stage2_cluster_separation.py               [16 KB] Stage 2: cluster sep
├── run_two_stage_estimator.py                     [ 9 KB] Orchestrator (main)
├── QUICK_START_TWO_STAGE.md                       [This file]
├── TWO_STAGE_ESTIMATOR_README.md                  [Full documentation]
├── IMPLEMENTATION_SUMMARY.md                      [Implementation details]
└── stage0_gate_output.pkl                         [Auto-generated]
    stage1_cca_output.pkl                          [Auto-generated]
    stage2_cluster_separation_output.pkl           [Auto-generated]
    two_stage_summary.pkl                          [Auto-generated]
    figures/stage2_cluster_separation_results.png  [Auto-generated]
```

---

## Troubleshooting

### "ModuleNotFoundError: simplexity"
Ensure `simplexity` is installed or available in PYTHONPATH.

### "No model file found"
Check that `models/rrxor_transformer.pt` exists in reproduction directory.

### Stage completes but no pickle saved
Check file permissions in reproduction directory.

### "Stage 0 gate FAIL"
This means the 36 belief states don't separate cleanly even with oracle labels. Indicates intrinsic observability limitation. Proceed anyway for diagnostics, but results may show lower R².

### Plateau detected at d≠5
The method picked a different d. Check:
1. Did Phase A work? (If Phase A failed, direction-selection idea doesn't work)
2. Is K-means converged? (Check cluster sizes in console output)
3. Is the separation ratio curve smooth? (Look at the plot)

---

## Comparison to Baselines

Run all three methods side-by-side:

```bash
# Original spectral-OOM
python fig13_rrxor_unsupervised_oom.py

# Original CCA/RRR (for comparison)
python fig14_cca_rrr_estimator.py

# New two-stage
python run_two_stage_estimator.py

# Compare state_r2(d=5) from all three
```

Expected:
- Spectral-OOM: 0.816 (paper baseline)
- Two-stage: 0.82+ (goal is to beat 0.816)
- Supervised ceiling: 1.000

---

## What Each Stage Does

| Stage | Goal | Input | Output | Success Criterion |
|-------|------|-------|--------|-------------------|
| 0 | Verify signal exists | Activations + beliefs | Separation ratio | Gate PASS: ratio > 0.3 |
| 1 | Isolate belief region | Activations + softmax | d_stage1=10 subspace | Smooth CCA spectrum |
| 2A | Test idea (oracle) | Stage1 + labels | LDA basis | state_r2(d=5) > 0.8 |
| 2B | Select d (unsupervised) | Stage1 + K-means | Plateau d | d_selected = 5 |

---

## Next: Understand the Code

1. Read `IMPLEMENTATION_SUMMARY.md` for design decisions
2. Read `TWO_STAGE_ESTIMATOR_README.md` for full methodology
3. Read the spec: `two_stage_estimator_spec.md` (acceptance criteria)
4. Read stage source code: each is ~300-500 lines, well-commented

---

## Report Structure (Stage 2 Output)

```
STAGE 0 GATE
  - Separation ratio (between/within scatter)
  - Gate status (PASS/FAIL)
  - Mushy states (which transient states fail to separate)

STAGE 1: CCA/RRR
  - CCA spectrum (first 12 singular values)
  - d_stage1 dimension

STAGE 2 PHASE A (Controlled)
  - Direction-selection works? (YES/NO)
  - state_r2 at d=5

STAGE 2 PHASE B (Unsupervised)
  - Selected d (via plateau)
  - state_r2 at d_selected and d=5

ACCEPTANCE CRITERIA
  ✓ Criterion 1: d=5 via plateau?
  ✓ Criterion 2: state_r2(d=5) > 0.816?
  ✓ Overall: PASS/FAIL

COMPARISON TABLE
  Method          | d   | R²
  spectral-OOM    | 5   | 0.816
  supervised      | 5   | 1.000
  two-stage       | d*  | ???
```

---

## Minimal Test (If You're In a Hurry)

```bash
# Just gate + Phase A (fast, ~30 sec)
python fig_stage0_gate_rrxor.py
python -c "
import fig14_cca_rrr_estimator_twostage as s1
import pickle
with open('stage0_gate_output.pkl','rb') as f: s0=pickle.load(f)
s1.run_stage1('RRXOR','rrxor_transformer.pt',s0['T'],s0['STATIONARY'],max_horizon=2)
" 2>&1 | grep "Stage-1 subspace"

python -c "
import fig_stage2_cluster_separation as s2
s2.phase_a_controlled(*[pickle.load(open(f,'rb')) for f in ['stage1_cca_output.pkl','stage0_gate_output.pkl']])
"
```

---

**Ready? Run:** `python run_two_stage_estimator.py` 🚀
