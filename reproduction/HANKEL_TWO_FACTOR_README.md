# Hankel Two-Factor Spectral-OOM Estimator

## Overview

This implementation extends the spectral-OOM estimator to compute **both the controllability factor P and observability factor S**, forming the full Hankel matrix **H = P·S** in activation space.

The two-sided factorization improves conditioning by leveraging mutual constraints between past statistics (P) and future statistics (S), enabling more reliable dimensionality selection via elbow detection on singular values.

## Files Created

### Core Implementation
- **`spectral_oom_hankel.py`** — Main estimator
  - `run_hankel_spectral_oom()`: Full pipeline
  - `build_controllability_factor()`: Evolve initial state through A_x operators
  - `build_observability_factor()`: Stack multi-step readouts (C, G_x C, ...)
  - `compute_hankel_matrix()`: H = P·S
  - `detect_elbow()`: Select d from singular value spectrum

### Process-Specific Wrappers
- `fig1_arch_hankel.py` — Arch (4-state, 3D visualization)
- `fig1_mess3_hankel.py` — Mess3 (3-state, 2D triangle)
- `fig1_fern_hankel.py` — Fern (3-state, 2D triangle)
- `fig1_strata_hankel.py` — Strata (3-state, 2D triangle)
- `fig1_wing_hankel.py` — Wing (3-state, 2D triangle)
- `fig1_zero_one_random_hankel.py` — Zero-one-random (3-state, 2D triangle)
- `fig1_rrxor_hankel.py` — RRXOR (5-state, 2D PCA projection)

Each script:
- Loads the trained model and HMM for the process
- Calls `run_hankel_spectral_oom()` with full verbosity
- Prints elbow-detected d vs true d (with match indicator)
- Generates a 2-panel figure:
  - **Left**: Singular value spectrum (log scale) with true/detected d marked
  - **Right**: Recovered belief geometry vs ground truth
- Reports belief-decode R² for the ground truth dimensionality

### Orchestrator
- **`run_hankel_all_processes.py`** — Run all 7 processes
  - Executes each script sequentially
  - Parses output to extract d_detected
  - Generates summary: `HANKEL_ELBOW_RESULTS.txt`

## Usage

### Run a single process:
```bash
cd reproduction
python fig1_arch_hankel.py
```

**Output:**
- Console: progress + full singular value spectrum + d-selection results
- Figure: `figures/fig1_arch_hankel.png`

### Run all 7 processes:
```bash
cd reproduction
python run_hankel_all_processes.py
```

**Output:**
- Individual figures: `figures/fig1_*_hankel.png` (one per process)
- Summary report: `HANKEL_ELBOW_RESULTS.txt`
  - Table: true d vs detected d vs match status
  - Accuracy: # correct / 7
  - Detailed output per process

## Key Design Choices

### 1. Controllability Factor P
- **Construction**: Evolve initial state forward through fitted A_x operators to all reachable prefixes up to depth L
- **Column space meaning**: Represents belief directions as they accumulate from the initial state
- **Depth**: Same as observability matrix (full context_len)

### 2. Observability Factor S
- **Construction**: Stack multi-step readouts [C | G_x C | G_x G_y C | ...] to same depth L
- **Reuse**: Adapts existing observable_subspace logic from fig14_observable_oom.py
- **Readout C**: Ridge regression from activations to softmax (next-token observable)
- **Operators G_x**: Rescaled transition maps (a(w) G_x ≈ P(x|w) a(wx)) in observation space

### 3. Hankel Matrix H = P·S
- **Size**: (num_reachable_states, num_columns) where num_columns = D + D·vocab + D·vocab² + ...
- **Semantics**: Each column is a multi-step observable; rows are reachable state representations
- **SVD**: Left singular vectors U give belief subspace in activation coordinates
- **Spectrum**: Singular values ranked by their contribution to the Hankel factorization

### 4. Elbow Detection
- **Method 1**: Find first d where sv[d] / sv[d-1] < 5% (default threshold)
- **Method 2**: Fallback to cumulative variance = 90% if no clear gap
- **Robustness**: Designed for smooth spectra without sharp drops

## Expected Results

| Process | True d | Vocab | Notes |
|---------|--------|-------|-------|
| arch | 4 | 3 | 4-state tetrahedron |
| mess3 | 3 | 3 | Sierpinski fractal |
| fern | 3 | 2 | Deterministic |
| strata | 3 | 2 | Strong structure |
| wing | 3 | 2 | Geometric pattern |
| zero_one_random | 3 | 2 | Stochastic |
| rrxor | 5 | 2 | Next-token degenerate, challenging |

**Success criteria:** Elbow method should recover true d for all 7 processes (or ±1 tolerance if the spectrum is ambiguous).

## Theory Behind Two-Factor Factorization

In the spectral learning literature (HKZ, Balle et al., Thon-Jaeger), the central theorem states:

```
Hankel matrix H = P·S
  where P maps past to belief states (reachability)
        S maps belief states to futures (observability)
  
  rank(H) = rank(P) = rank(S) = number of hidden states
```

**One-sided estimation (S alone):**
- Directly builds observables from activations
- Captures what the network *outputs* about future
- Can be biased if operator estimates are noisy

**Two-sided estimation (H = P·S):**
- Builds both reachability (P) and observability (S) from operators
- Past and future statistics mutually constrain each other
- Better-conditioned singular vectors on finite data
- More robust elbow detection when S alone has smooth spectrum

This is especially valuable for weak observability (e.g., next-token-degenerate RRXOR directions) where conditioning matters.

## Relationship to Existing Code

- **Does NOT modify**: unsupervised_belief_oom.py, fig14_observable_oom.py, stage1_general.py
- **Parallel to**: fig14_observable_oom.py (one-sided observable subspace)
- **Reuses infrastructure**: activation collection, operator fitting, prefix tree enumeration

## File Structure

```
reproduction/
├── spectral_oom_hankel.py              # Core two-factor estimator (NEW)
│
├── fig1_arch_hankel.py                 # Arch process (NEW)
├── fig1_mess3_hankel.py                # Mess3 process (NEW)
├── fig1_fern_hankel.py                 # Fern process (NEW)
├── fig1_strata_hankel.py               # Strata process (NEW)
├── fig1_wing_hankel.py                 # Wing process (NEW)
├── fig1_zero_one_random_hankel.py      # Zero-one-random process (NEW)
├── fig1_rrxor_hankel.py                # RRXOR process (NEW)
│
├── run_hankel_all_processes.py         # Orchestrator (NEW)
├── HANKEL_TWO_FACTOR_README.md         # This file (NEW)
│
├── figures/
│   ├── fig1_arch_hankel.png            # Output figures
│   ├── fig1_mess3_hankel.png
│   └── ... (one per process)
│
└── HANKEL_ELBOW_RESULTS.txt            # Summary report (generated)
```

## Next Steps

1. **Run the orchestrator:** `python run_hankel_all_processes.py`
2. **Review summary:** Check `HANKEL_ELBOW_RESULTS.txt` for accuracy
3. **Inspect individual spectra:** Check figures for elbow clarity
4. **Compare to existing method:** How does H = P·S elbow compare to S-alone?
5. **If successful:** Use Hankel two-factor for subsequent analysis

## Implementation Notes

- **Depth**: Currently uses full context_len for both P and S. Can be tuned if memory is a constraint.
- **Ridge regularization**: Fixed at 1e-2 (RIDGE_ALPHA). Matches existing spectral-OOM code.
- **Reachability**: Model-determined via softmax > EPS threshold; no DGP knowledge required.
- **Visualization**: Ground truth d is hardcoded for plotting; elbow-detected d is reported separately.

---

**Implementation date:** 2026-06-26  
**Status:** Ready for testing on all 7 processes
