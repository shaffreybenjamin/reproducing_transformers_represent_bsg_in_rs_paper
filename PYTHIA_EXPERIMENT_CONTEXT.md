# Pythia 70M Belief Geometry Extraction Experiment

## Project Overview

**Goal:** Apply unsupervised belief-state geometry estimation to a real, pre-trained 70M-parameter transformer (Pythia 70M) trained on the Pile dataset to reverse-engineer learned algorithms without ground-truth beliefs.

**Key insight:** On toy HMM processes (3-5 hidden states, small vocab), spectral-OOM and CCA/RRR successfully extract belief geometry unsupervised (R² = 0.80–0.999). At real scale (50k+ vocab, billions of tokens), we expect:
- Abundant prefix data (removes sparsity bottleneck that plagued RRXOR)
- Unknown latent structure (may be classical, quantum, or post-quantum)
- Opportunity to validate via mechanistic interpretability (reverse-engineer what the model learned)

---

## Core Methods

### Spectral-OOM (Observability-OOM)

**Pipeline:**
1. Enumerate all reachable prefixes w of length 1 to L_max
2. Collect residual-stream activations: resid[w] ∈ ℝ^d_model for each prefix
3. Estimate single-step readout C (activation → softmax next-token)
4. Estimate single-step transition operators A_x (parent activation → child activation, weighted by likelihood)
5. Build observability matrix O = [C | A_x C | A_x A_y C | …] by composing operators up to horizon h
6. SVD(O) → keep top-d left singular vectors as belief subspace basis B

**Bottleneck:** Belief directions orthogonal to next-token prediction are invisible (funnel-through-C problem). For RRXOR, this limited performance to R²=0.651 at correct d=5.

### CCA/RRR (Canonical Correlation Analysis / Reduced Rank Regression)

**Pipeline:**
1. Enumerate prefixes (same as spectral-OOM)
2. For each horizon k=1..h, directly measure empirical P(x_{t+1:t+k}|w) from data
3. Fit readouts C_k: activations → P(x_{t+1:t+k}|w) via ridge regression
4. Stack future-readouts: Φ = [C_1 | C_2 | … | C_h]
5. Whiten both sides (activation covariance, future covariance)
6. SVD of whitened correlation → keep top-d canonical directions as basis B

**Advantage:** Routes around next-token bias by using multi-step futures directly (no operator composition). For RRXOR at d=10, reached supervised ceiling (R²=0.890).

**Regularization:** Ledoit-Wolf shrinkage on future covariance (shrinkage_intensity=0.05) to control noise amplification during whitening.

---

## D-Selection Without Discrete Assumptions

### Problem
Spectral-OOM and CCA/RRR both produce a spectrum of singular values. Traditional **elbow detection** assumes discrete hidden states (sharp drop at true d). For continuous/quantum/post-quantum processes, the spectrum decays smoothly—no elbow exists.

### Solution: Three complementary methods

#### 1. **Effective Rank**
- Count singular values needed to explain 95–99% of variance
- Formula: d_eff = Σ σ_i / σ_1, where σ_i normalized by max singular value
- **Advantage:** Geometry-agnostic, works for any structure
- **Returns:** pragmatic d based on variance concentration

#### 2. **Mutual Information Saturation**
- Project activations onto d dimensions: A_d = X @ B[:, :d]
- Train linear decoder: A_d → next-token logits
- Compute I(A_d, next_token) for d=1, 2, …, d_max
- Find plateau: first d where MI improvement < 5–10% of max improvement
- **Advantage:** Picks d that matters for the task
- **Returns:** functional d (prediction-relevant dimensionality)

#### 3. **Topological Data Analysis (TDA)**
- Apply persistent homology to activation manifold (optional; lower priority)
- Identifies intrinsic structure (curve vs surface vs higher-d manifold)
- **Advantage:** No assumptions about process type
- **Returns:** intrinsic geometric dimensionality

### Validation Strategy
Run all three, check agreement:
- If all three agree on d, high confidence
- If MI saturation picks much larger d than effective rank, indicates noise/overfitting
- If TDA detects smooth low-d manifold but spectrum is smooth (no elbow), confirms continuous structure

---

## Implementation Modules (Existing Code)

### Key Functions to Reuse

**unsupervised_belief_oom.py:**
- `collect_prefix_features_enumerated(model, hmm, context_len, device)` — enumerate prefixes, collect activations + beliefs + softmax
- `fit_operators(A, P, Bc, Wt)` — fit rescaled transition operators
- `recover_eval_functional(A)` — recover evaluation functional from operator matrix
- `rollout_states(proj, ops, e, max_len, vocab)` — rollout beliefs from root using learned operators
- `belief_decode_r2(features, Y, w)` — compute state-level R² given beliefs and weights
- `simplex_to_xy(beliefs)` — project 3-simplex beliefs to 2D for visualization

**fig14_observable_oom.py:**
- `observable_subspace(resid, soft, reach, vocab, wmap=None)` — build observability matrix, return SVD basis

**stage1_cca_general.py:**
- `run_stage1(resid, soft, reach, vocab, d_keep=None)` — CCA/RRR with Ledoit-Wolf shrinkage

---

## Validation on 7 Known Processes (Phase 1)

### Goal
Verify that effective rank + MI saturation correctly identify d on processes where ground truth is known:

| Process | True d | Vocab | Current Best R² | Notes |
|---------|--------|-------|-----------------|-------|
| zero_one_random | 3 | 2 | 0.803 (spectral-OOM) | Weaker recovery |
| fern | 3 | 2 | 0.910 (spectral-OOM) | Strong |
| strata | 3 | 2 | 0.999 (spectral-OOM) | Excellent |
| wing | 3 | 2 | 0.875 (spectral-OOM) | Solid |
| arch | 4 | 3 | 0.954 (spectral-OOM) | Very strong |
| mess3 | 3 | 3 | ~0.97 (spectral-OOM) | Paper baseline |
| rrxor | 5 | 2 | 0.816 (spectral-OOM) | Next-token degenerate, sparse data |

### Experimental Plan

For each process:
1. **Run spectral-OOM**: collect activations, build O, SVD
2. **Run CCA/RRR**: collect activations, build Φ (direct futures), whitened SVD
3. **Apply d-selection methods:**
   - Effective rank (95% and 99% variance)
   - MI saturation (5% plateau criterion)
   - Compare to true d
4. **Report:**
   - Do effective rank / MI saturation recover true d?
   - Which method (spectral-OOM or CCA/RRR) produces cleaner spectrum for d-selection?
   - State-level R² at selected d

### Success Criteria
- Effective rank / MI saturation pick d within ±1 of true d on all 7 processes
- Agreement across methods on most processes
- If both methods succeed, both are viable for Pythia; if one fails, that's diagnostic

---

## Pythia 70M Experiment (Phase 2)

### Model Details
- **Model:** Pythia 70M (EleutherAI)
- **Training data:** The Pile (825GB, diverse, includes code, math, prose, etc.)
- **Vocab size:** 50,257 (standard BPE)
- **Residual stream dimension:** ~512
- **Context length:** 2048
- **Pre-trained checkpoints:** Available at huggingface.co/EleutherAI/pythia-70m

### Data Availability
- No ground-truth beliefs (unlike toy HMMs)
- No ground-truth hidden states
- **Validation strategy:** use extracted beliefs to reverse-engineer learned algorithms (via mechanistic interpretability)

### Experimental Plan

1. **Load Pythia 70M + Pile**
   - Download model checkpoint (or use huggingface transformer_lens integration)
   - Sample prefixes from Pile (billions of tokens available, no sparsity issue)
   - Enumerate up to length L=10 or L=12 (manageable on single GPU with batching)

2. **Apply spectral-OOM and CCA/RRR**
   - Collect residual-stream activations for all enumerated prefixes
   - Build observability matrices (spectral-OOM) and future readouts (CCA/RRR)
   - Compute SVD spectra

3. **D-selection via effective rank + MI saturation**
   - Apply both methods
   - Report selected d and spectrum structure
   - Check for elbow (discrete structure) vs smooth decay (continuous structure)

4. **Validation via mechanistic interpretability**
   - Extract beliefs at selected d using each method
   - Test: can extracted beliefs predict next-token logits? (surrogate for "are beliefs real?")
   - Test: do extracted beliefs correlate with attention patterns? (indicate geometric structure)
   - Test: do probing classifiers trained on extracted beliefs generalize to auxiliary tasks?

5. **Comparison: spectral-OOM vs CCA/RRR**
   - Which method produces cleaner spectrum for d-selection?
   - Which method's extracted beliefs are more predictive / coherent?
   - Which scales better (memory, computation)?

### Success Criteria
- Both methods extract low-d geometry from Pythia activations (d << 512)
- Extracted beliefs are predictive of next-token distribution (>80% of full model)
- Smooth spectrum (no elbow) → confirms continuous/unknown structure
- Effective rank + MI saturation agree on d (within 5–10)
- Extracted beliefs reveal interpretable structure (e.g., correlate with linguistic/mathematical features)

---

## Fast Validation Tests (2-3 days on GPU)

Once d is selected, rapid tests to show extracted geometry is real:

1. **Next-token prediction:**
   - Linear readout: extracted d-D beliefs → next-token logits
   - Compare R² to: full model (~0.95), random basis (~0.05)
   - Goal: extracted beliefs achieve >85% of full model's predictive power

2. **Attention-belief alignment:**
   - For each layer, do attention patterns correlate with belief transitions?
   - Use attention weights × value vectors as surrogate for "what attention computes"
   - Correlation > 0.5 would suggest geometry is real

3. **Information content:**
   - Compute I(extracted beliefs, next-token distribution)
   - Compare to I(full residual stream, next-token)
   - Goal: extracted beliefs capture >90% of mutual information in 1/10 the dimensions

---

## Potential Outcomes & Fallbacks

| Outcome | Interpretation | Next Step |
|---------|---|---|
| Clean elbow in spectrum at d=20–100 | Discrete latent structure | Suggests classical process; validate via reverse-engineering |
| Smooth decay, effective rank d=200+, MI saturation d=50 | Continuous or high-d structure | Pythia may be learning complex features; TDA might help |
| Extracted beliefs predictive (>85% R²) + interpretable | Method works at scale | Strong result; proceed to full mechanistic analysis |
| Extracted beliefs unpredictive (<60% R²) | Method fails on real data | Fall back to clean RRXOR experiment to isolate whether issue is spectral-OOM, CCA/RRR, or real-data complexity |
| Attention-belief alignment weak | Geometry may not match model's computation | Use TDA or alternative validation; extracted geometry may be statistical artifact |

---

## File Structure

```
reproduction/
├── unsupervised_belief_oom.py          # Core spectral-OOM functions
├── fig14_observable_oom.py              # Observable subspace extraction
├── stage1_cca_general.py                # CCA/RRR with Ledoit-Wolf
├── fig1_*_principled.py                 # Reference implementations (7 processes)
├── pythia_experiment_spectral_oom.py    # [NEW] Spectral-OOM on Pythia
├── pythia_experiment_cca_rrr.py         # [NEW] CCA/RRR on Pythia
├── pythia_d_selection.py                # [NEW] Effective rank + MI saturation
├── pythia_validation.py                 # [NEW] Next-token pred + attention align
└── PYTHIA_EXPERIMENT_CONTEXT.md         # [This file]
```

---

## Key References & Context

**Prior work on these methods:**
- Spectral-OOM: Hsu–Kakade–Zhang (2012), arXiv:0811.4413
- CCA/RRR: Hotelling (1935), Izenman (1975), Donnat & Tuzhilina (2024, arXiv:2405.19539)
- Belief geometry in transformers: Constrained Belief Simplex paper (references folder)
- Quantum/post-quantum representations: neural_networks_leverage_nominally_quantum_and_post_quantum_representations.pdf

**Paper findings that motivate this work:**
- Transformers trained on next-token prediction learn belief geometries that match optimal Bayesian filtering (sometimes quantum/post-quantum)
- These geometries are linearly present in residual streams (validated via linear regression to ground-truth beliefs)
- Unsupervised spectral methods can extract these geometries without ground-truth labels

---

## Notes for Implementation

1. **Batching:** For Pythia on Pile, you'll need to batch prefix enumeration (process ~10k–100k prefixes at a time) to avoid OOM. Use prefix-length stratification to ensure good coverage.

2. **Horizon selection:** Start with h=2–3 (longer horizons require more dense data). For Pile with billions of tokens, h=4–5 may be tractable.

3. **Ledoit-Wolf shrinkage:** Keep at 0.05 for CCA/RRR; this was tuned on RRXOR and seems robust.

4. **D-selection implementation:**
   - Effective rank: numpy; compute from singular values directly
   - MI saturation: train lightweight linear models at each d, measure I(activations, logits) via empirical entropy
   - TDA: use ripser or gudhi if needed (lower priority)

5. **Validation on 7 processes first:** Run all three d-selection methods on known processes to build confidence before Pythia.

---

## Expected Timeline

- **Phase 1 (D-selection validation):** 1–2 days
  - Run spectral-OOM + CCA/RRR on 7 processes
  - Apply d-selection methods
  - Report: do methods recover true d?

- **Phase 2 (Pythia extraction):** 1–2 days
  - Load Pythia, enumerate prefixes, build bases
  - Apply d-selection
  - Generate spectra plots

- **Phase 3 (Fast validation):** 1 day
  - Next-token prediction
  - Attention alignment
  - Information content

- **Total:** 3–5 days start-to-finish, GPU required

---

## Contact / Questions

For clarifications on methodology, refer to:
- Spectral-OOM: unsupervised_belief_oom.py, fig14_observable_oom.py
- CCA/RRR: stage1_cca_general.py (Ledoit-Wolf variant)
- D-selection: effective rank / MI saturation (to be implemented)
