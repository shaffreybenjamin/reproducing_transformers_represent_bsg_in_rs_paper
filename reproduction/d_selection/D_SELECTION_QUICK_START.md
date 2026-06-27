# D-Selection Quick Start

## One-Line Summary
Two new d-selection methods (effective rank, MI saturation) added to spectral-OOM and CCA/RRR, ready to validate on 7 processes.

## Run All 7 Processes (Single Command)
```bash
cd reproduction
python run_d_selection_all_7_processes.py
```

**Output:**
- Console: progress for each process
- `d_selection_PROCESS_results.txt` (7 files, one per process)
- `D_SELECTION_COMPREHENSIVE_REPORT.txt` (master report)

**Time:** ~10-15 minutes on GPU (depends on model loading)

## Run Single Process
```bash
cd reproduction
python fig1_mess3_d_selection.py       # d=3 (easy baseline)
python fig1_arch_d_selection.py        # d=4 (tetrahedron)
python fig1_rrxor_d_selection.py       # d=5 (harder)
```

## What Gets Output

For each process, a file like `d_selection_mess3_results.txt`:
```
D-Selection Results: mess3
======================================================================
True d: 3

SPECTRAL-OOM:
  Effective Rank (95%): 3
  Effective Rank (99%): 3
  MI Saturation:        3
  Agreement:            True

CCA/RRR:
  Effective Rank (95%): 3
  Effective Rank (99%): 3
  MI Saturation:        3
  Agreement:            True
```

## Success Looks Like

✓ Effective Rank (95%) picks d within ±1 of true d  
✓ Effective Rank (99%) picks d within ±1 of true d  
✓ MI Saturation picks d within ±1 of true d  
✓ Both estimators (spectral-OOM, CCA/RRR) agree  

Example success:
- True d = 3
- ER95 selects 3, ER99 selects 3, MI selects 3 ✓

Example partial success:
- True d = 5 (RRXOR, harder)
- ER95 selects 4, ER99 selects 5, MI selects 5 ✓

## File Locations

| File | Purpose |
|------|---------|
| `d_selection.py` | Core methods (effective rank, MI saturation) |
| `d_selection_pipeline.py` | Unified orchestrator |
| `fig1_*_d_selection.py` (×7) | Individual process scripts |
| `run_d_selection_all_7_processes.py` | Run all 7 + generate report |
| `D_SELECTION_README.md` | Full documentation |
| `IMPLEMENTATION_D_SELECTION.md` | Implementation details |
| `D_SELECTION_QUICK_START.md` | This file |

## Troubleshooting

**Q: Script crashes on import**  
A: Make sure you're in the `reproduction/` directory when running.

**Q: "Model file not found"**  
A: Models should be in `reproduction/models/`. Check `ls models/`.

**Q: Process hangs on "Running Spectral-OOM"**  
A: If you have limited GPU memory, reduce `max_d_test` parameter or run on CPU.

**Q: Results don't match expected d**  
A: RRXOR is expected to be harder (next-token degeneracy). Check report for method comparison.

## Next Steps After Validation

1. Review `D_SELECTION_COMPREHENSIVE_REPORT.txt`
2. If methods work (recover d on all 7):
   - Proceed to Pythia 70M experiment
   - Use same d-selection pipeline for real model
3. If methods struggle (e.g., on RRXOR):
   - Diagnose which method fails
   - Consider improved tuning or alternative approaches

---

**Documentation:** See `D_SELECTION_README.md` for full details.  
**Theory:** See `d_selection.py` docstrings and `PYTHIA_EXPERIMENT_CONTEXT.md` for methodology.
