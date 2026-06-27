# Two-Stage Belief-Subspace Estimator (RRXOR)

This directory contains the implementation of the two-stage estimator for recovering RRXOR's belief-state geometry, as specified in `two_stage_estimator_spec.md`.

## Problem

The spectral-OOM unsupervised estimator recovers RRXOR's belief geometry poorly (state-level R²≈0.82 at d=5) because it anchors to next-token predictions (the softmax observable). But RRXOR exhibits **next-token degeneracy**: geometrically distinct belief states share similar next-token distributions (R²≈0.31), so output-anchored methods systematically bury true belief directions.

CCA/RRR improves to R²≈0.89, but only at d=10 (overshooting the true d=5). The issue is **ranking/concentration**: CCA ranks by predictive relevance, which ignores belief identity.

## Solution

The two-stage approach uses a different re-ranking signal: **cluster separation** (between/within scatter ratio, LDA-style), which is orthogonal to predictive relevance.

- **Stage 1** (CCA/RRR): Isolates belief-bearing region of activation space (d_stage1=8–10)
- **Stage 2** (Cluster separation): Within that region, re-ranks by belief-identity geometry, selecting the correct d=5

## Files

### Core Implementation

1. **`fig_stage0_gate_rrxor.py`**
   - Pre-gate: Verifies that 36 belief states *actually separate* in a supervised 5D subspace
   - Computes between-cluster / within-cluster scatter ratio (LDA-style)
   - Flags "mushy" transient states that fail to separate (intrinsic gap indicator)
   - Output: `stage0_gate_output.pkl`

2. **`fig14_cca_rrr_estimator_twostage.py`**
   - Stage 1: Regularized CCA/RRR with **Ledoit-Wolf shrinkage** on future covariance
   - Uses h=2 horizon (avoids finite-sample noise from longer horizons)
   - Recovers generous d_stage1=8–10 candidate subspace
   - Logs CCA spectrum
   - Output: `stage1_cca_output.pkl`

3. **`fig_stage2_cluster_separation.py`**
   - Stage 2: Within-subspace cluster-based direction re-ranking
   - **Phase A (Controlled)**: Uses known RRXOR state assignments
     - Tests whether direction-selection principle works with oracle labels
     - If Phase A fails, the idea itself is flawed (stop here)
   - **Phase B (Unsupervised)**: K-means clustering in Stage-1 subspace
     - Cluster label-free (K=36 or auto-discovered)
     - LDA on cluster assignments finds separable directions
     - Sweep d to find plateau in separation ratio
     - Plateau location = selected d (label-free dimension criterion)
   - Output: `stage2_cluster_separation_output.pkl`

### Orchestration

4. **`run_two_stage_estimator.py`** (main entry point)
   - Runs all three stages in sequence
   - Checks inputs/outputs at each stage
   - Generates comprehensive report following spec's format
   - Summary output: `two_stage_summary.pkl`

## Usage

### Quick start (run full pipeline)

```bash
cd reproduction
python run_two_stage_estimator.py
```

This will:
1. Run Stage 0 (gate) → saves `stage0_gate_output.pkl`
2. Run Stage 1 (CCA/RRR) → saves `stage1_cca_output.pkl`
3. Run Stage 2 (cluster separation) → saves `stage2_cluster_separation_output.pkl`
4. Generate a comprehensive report to stdout

### Run individual stages (for debugging)

```bash
# Stage 0 only
python fig_stage0_gate_rrxor.py

# Stage 1 only (requires Stage 0 outputs)
python fig14_cca_rrr_estimator_twostage.py

# Stage 2 only (requires Stage 0 and 1 outputs)
python fig_stage2_cluster_separation.py
```

## Report Format

The final report follows this structure (per spec):

### Stage 0 Gate Result
- Do the 36 / 31-transient states separate under supervised labels?
- Which (if any) fail to separate?
- Between/within scatter ratio

### Stage 1: CCA Spectrum
- CCA singular-value spectrum
- D_stage1 = 8–10 selected

### Stage 2 Phase A (Controlled)
- Did within-subspace separation pick the right 5 directions?
- State-level R² at d=5 using oracle labels

### Stage 2 Phase B (Unsupervised)
- Selected d (plateau location)
- State-level & per-position R² at that d
- Plateau separation ratio vs d plot

### Comparison Table
```
Metric              | Method          | d   | R²
-----------
state-level decode  | spectral-OOM    | 5   | 0.816
state-level decode  | supervised      | 5   | 1.000
state-level decode  | two-stage(B)    | 5   | ???
state-level decode  | two-stage(B)    | d*  | ???
```

(Report clearly labels any d>5 numbers as diagnostic only, never as the result.)

## Acceptance Criteria (from spec)

1. **d-selection returns d=5 on RRXOR** via the separability plateau (Stage 2, Phase B)
   - This is the primary success condition
   - A great fit at the wrong d = FAIL

2. **At selected d=5**, **state-level R² beats 0.816** (spectral-OOM paper baseline)
   - Report per-position R² alongside

3. **Mess3 control**: Not tested here (Mess3 is RRXOR-specific cluster-separation)
   - Note: Stage 1 regression should not degrade on Mess3

## Key Design Decisions

### Why Ledoit-Wolf on future covariance?
CCA is aggressive at inverting small covariances in low-sample regime. Ledoit-Wolf shrinkage reduces this amplification of noisy directions without re-ranking (which would break the output-anchored step).

### Why h=2 (not h=3 or longer)?
With ~613 reachable prefixes, longer horizons are undersampled. h=3+ inject finite-sample noise. h=2 is the sweet spot.

### Why generous d_stage1=8–10?
To ensure true belief directions are contained in the candidate subspace before Stage 2 operates. Stage 2 does the actual d-selection.

### Why cluster separation re-ranks orthogonally to predictive relevance?
- CCA ranks by: "which directions predict the future?"
- LDA on belief states ranks by: "which directions separate belief clusters?"
- RRXOR's next-token degeneracy makes these antithetical.
- Cluster geometry is belief-intrinsic; it exists whether or not states are next-token-distinguishable.

### Why K-means in Phase B?
Unsupervised clustering without labels. K-means is simple and fast. The spec allows cluster-count selection via elbow or silhouette; here we use k=36 (known from ground truth for now—a future refinement).

### Why plateau to select d?
The spec's label-free d criterion: "sweep d, plateau location = selected d". As you add more separable directions, between/within ratio improves, but eventually plateaus (diminishing returns). The plateau is a data-driven signal that you've captured the essential structure.

## Outputs & Artifacts

- **Pickle files** (`stage{0,1,2}_*.pkl`, `two_stage_summary.pkl`): Python dict with all intermediate results
- **Plots** (`stage2_cluster_separation_results.png`): Separation ratio vs d for Phase A and B
- **Console report**: Full text output with all metrics and acceptance criteria

## Dependencies

- `torch`, `transformer_lens`: Model loading and inference
- `numpy`, `scipy`: Linear algebra
- `sklearn`: LinearRegression, LDA, KMeans, PCA
- `simplexity`: Hidden Markov model definitions (rrxor, mess3)
- `matplotlib`: Plotting

## Notes

- All three stages are independent executables but orchestrated by `run_two_stage_estimator.py`
- Existing estimators (`unsupervised_belief_oom.py`, original `fig14_cca_rrr_estimator.py`) are untouched
- All new files named with prefix `fig_stage{N}_` or `fig14_*_twostage` to avoid collisions
- Code uses `# %%` VSCode cell style (not Jupyter), can be run cell-by-cell in IDE

## Known Limitations / Future Refinements

1. **K-means cluster count in Phase B**: Currently uses k=36 (known state count). A fully blind version would use elbow method or silhouette to auto-discover k.

2. **Phase B plateau detection**: Currently picks first d where improvement < 10% of max diff. More sophisticated criteria (e.g., KL divergence of successive separation ratios) could improve robustness.

3. **Mess3 control**: Spec notes Mess3 cluster-separation is not expected to work (continuous/fractal data). This version only tests RRXOR; Mess3 can be verified separately via Stage 1 alone.

4. **Generalization**: Currently hardcoded for RRXOR. To apply to other processes, parameterize process name, model path, transition matrix, and state count.

## Citation

Based on: `two_stage_estimator_spec.md`

Background: writeup_v1.tex (spectral-OOM results showing the problem)
