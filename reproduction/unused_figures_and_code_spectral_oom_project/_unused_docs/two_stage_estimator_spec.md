# Implementation Spec — Two-Stage Belief-Subspace Estimator (RRXOR)

## Context for the implementing agent
This continues work on an unsupervised estimator that recovers belief-state
geometry from a transformer's residual-stream activations + its own softmax,
with no ground-truth belief labels. The existing method ("spectral-OOM") builds
an observability matrix by composing estimated transition operators and SVDs it.
A CCA/RRR variant (direct multi-step readouts + whitening) was already
implemented and tested. **Read the constraints before coding.**

### What the prior experiments established (do not re-litigate)
- CCA/RRR at horizon 2 beat the per-position baseline but **underperformed on
  state-level R² at d=5** (0.743 vs spectral-OOM's 0.816).
- Critically: at **d=8 → state-level 0.882; d=10 → 0.890 = supervised ceiling.**
- Interpretation (this is the whole basis for this spec): the belief information
  **is** linearly present and label-free-recoverable. CCA's failure at d=5 is a
  **ranking/concentration** failure — it spreads the 5 belief directions across
  ~10 singular directions, so truncating at d=5 grabs a mix of belief directions
  and loud belief-irrelevant directions.

### The hard constraint that defines success
RRXOR's true hidden-state dimensionality is **5**. The claim under test is that
the transformer represents a **5-dimensional** belief geometry linearly.
**A subspace with d>5 is the WRONG object**, regardless of fit quality. A
solution that only works by overshooting d (e.g. d=10 matching supervised) is
**not an acceptable solution.** The deliverable must recover **d=5** and score
well at d=5. Report any d>5 numbers only as diagnostics, never as the result.

### Why regularization alone won't fix this (don't expect it to)
Shrinkage on the future covariance fixes **variance** (CCA amplifying noisy
future directions). It does **not** fix **concentration/ranking**: every
output-anchored method ranks directions by predictive relevance to the future,
and RRXOR's defining pathology (next-token degeneracy, R²≈0.31 in the original
paper) is that some belief directions have *low* predictive relevance. So any
output-anchored ranking buries those belief directions, and truncating at d=5
drops them. Regularization sharpens the spectrum; it does not re-rank. We need a
direction-scoring signal that is **orthogonal to predictive relevance**.

---

## The idea: two-stage estimator
Stage 1 (output-anchored) isolates the *belief-bearing region* of activation
space and filters pure junk — what it's good at. Stage 2 (cluster-separation)
re-ranks *within that region* by **belief-identity** rather than predictive
relevance, selecting the correct 5 directions and the correct d — the one signal
the output cannot provide. RRXOR's 36 belief states are discrete points in
activation space; that discreteness exists whether or not the states are
next-token-distinguishable, so cluster geometry carries belief-identity
information the output ranking structurally cannot.

---

## STAGE 0 — Pre-gate (run FIRST; everything downstream depends on it)
Before building anything, verify the signal Stage 2 will rely on actually exists.

- Take the **supervised** 5-D belief subspace (you have ground-truth beliefs for
  RRXOR).
- Project activations into it. Check whether the **36 belief states separate** —
  i.e. compute a between-cluster / within-cluster separation ratio
  (LDA-style scatter ratio) using the known state assignments.
- **GATE:**
  - If the 36 states (especially the 31 transient states, which sit close
    together) separate cleanly → Stage 2 has signal; proceed.
  - If the transient states are mushy *even with labels* → this is evidence the
    gap is **partly intrinsic**, and the honest result is about *that*. Stop and
    report; do not force the rest.
- Report the per-state separation, and flag specifically which of the 31
  transient states (if any) fail to separate.

---

## STAGE 1 — Regularized CCA/RRR (isolate the belief-bearing region)
Reuse the existing CCA/RRR code; add regularization and a generous d.

- **Horizon = 2** (h=3,4 already shown to inject finite-sample noise; ~613
  reachable prefixes make long-horizon future distributions undersampled).
- **Ledoit–Wolf shrinkage** (or a ridge floor) on the future covariance before
  the whitening inverse — CCA is notoriously aggressive at inverting small
  covariances in the low-sample regime. This is for **variance control only**;
  it is not expected to fix d-selection.
- Recover a **generous** candidate subspace, **d_stage1 = 8–10**. We are NOT
  keeping this as the answer — it is the region inside which the 5 true belief
  directions are known to live (from the d=10=supervised result).
- Log the CCA spectrum.

## STAGE 2 — Within-subspace cluster separation (select the correct 5)
Operate **inside** the Stage-1 subspace (NOT raw 64-D activations — that's where
unsupervised clustering is fragile; restricting to the 8–10-D belief-bearing
region dissolves that fragility and the cluster-count circularity).

- Within the Stage-1 subspace, find the linear sub-subspace that **maximizes
  activation-cluster separation** (between/within scatter ratio, LDA-style).
- **Two-phase to de-risk:**
  - **Phase A (controlled):** use the known RRXOR state structure to fix cluster
    assignments. This tests whether the *direction-selection* idea works, before
    adding clustering noise. If Phase A can't pick the right 5, the idea fails
    and nothing downstream matters — report and stop.
  - **Phase B (fully unsupervised):** cluster the activations in the Stage-1
    subspace label-free (assignments from the data, not the known structure),
    then run the same separation. This is the real unsupervised deliverable.
- **Label-free d-selection:** sweep the number of cluster-separation directions;
  the correct d is where the between/within separation ratio **plateaus** (adding
  another direction stops improving separability). This is a d-criterion grounded
  in the representation's own geometry — not the known state count, not a
  fit-quality elbow (RRXOR has none).

---

## Acceptance criteria (the only numbers that count)
1. **d-selection returns d=5 on RRXOR** via the separability plateau (Stage 2,
   Phase B). This is the primary success condition. A great fit at the wrong d
   is a failure.
2. At the selected **d=5**, **state-level R² beats 0.816** (the spectral-OOM
   number in the paper). Report per-position R² alongside.
3. **Mess3 control:** the discretized analogue returns **d=3** and stays at
   R²≈0.994+. (Note: cluster-separation is RRXOR-specific because Mess3 is
   continuous/fractal — for Mess3, only verify Stage 1 doesn't regress; the
   cluster step is not expected to apply. Say so explicitly in the report.)

## Reporting back (structure the writeup this way)
- Stage 0 gate result: do the 36 / 31-transient states separate under supervised
  labels? Which fail?
- Stage 1: CCA spectrum, what d_stage1 region was kept.
- Stage 2 Phase A (controlled): did within-subspace separation pick the right 5?
- Stage 2 Phase B (unsupervised): selected d (plateau location), state-level &
  per-position R² at that d.
- Comparison table vs. spectral-OOM (0.816 / 0.651) and supervised ceiling
  (1.00 / 0.91), all **at d=5**.
- Any d>5 numbers clearly labelled as diagnostic only.

---

## Constraints (carry over from prior handoff)
- **Do not edit originals.** Duplicate any file you change
  (`<name>.py` → `<name>_twostage.py`) and work in the copy; originals are
  read-only. List your intended duplications and confirm before starting.
- Cell-based Python, `# %%` VSCode style — **not** Jupyter notebooks.
- Keep the existing spectral-OOM and CCA/RRR estimators intact and runnable for
  same-input comparison.
- If a step requires editing an original, **stop and ask.**

## Out of scope
- Do **not** try to "fix" the RRXOR operator rollout collapse — it's faithful to
  the true nilpotent/defective operator algebra (T_1^5 = 0), not a bug.
- Do **not** report any d>5 result as the headline. The correct dimensionality is
  the deliverable.
