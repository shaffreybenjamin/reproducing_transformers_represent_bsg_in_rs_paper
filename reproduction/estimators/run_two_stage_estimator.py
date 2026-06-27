"""Master orchestration script: Run two-stage estimator (Stages 0, 1, 2) end-to-end.

This script runs the full pipeline:
  Stage 0: Gate — verify belief-state cluster separation exists
  Stage 1: Regularized CCA/RRR (Ledoit-Wolf + h=2) — isolate belief-bearing region
  Stage 2: Within-subspace cluster separation — select correct d via separability plateau

After all stages complete, generates a comprehensive writeup following the spec's
reporting format.

Usage:
  python run_two_stage_estimator.py
"""

from pathlib import Path
import pickle
import sys

OUT_DIR = Path(__file__).parent

def run_all_stages():
    """Run Stages 0, 1, 2 in sequence."""
    print("\n" + "=" * 100)
    print(" " * 25 + "TWO-STAGE BELIEF-SUBSPACE ESTIMATOR (RRXOR)")
    print("=" * 100)

    # Stage 0: Gate
    print("\n>>> RUNNING STAGE 0: Gate...")
    try:
        import fig_stage0_gate_rrxor as stage0
        stage0_results = stage0.main()
    except Exception as e:
        print(f"ERROR in Stage 0: {e}")
        import traceback
        traceback.print_exc()
        return None

    gate_pass = stage0_results.get("gate_pass", False)
    if not gate_pass:
        print("\n" + "!" * 100)
        print("Stage 0 gate FAILED. The 36 belief states do not separate cleanly.")
        print("This indicates the gap is partly intrinsic. Proceeding anyway for diagnostic purposes.")
        print("!" * 100)

    # Stage 1: Regularized CCA/RRR
    print("\n>>> RUNNING STAGE 1: Regularized CCA/RRR...")
    try:
        import fig14_cca_rrr_estimator_twostage as stage1
        T_stage1 = stage0_results.get("T")
        STATIONARY_stage1 = stage0_results.get("STATIONARY")
        stage1_results = stage1.run_stage1(
            "RRXOR",
            "rrxor_transformer.pt",
            T_stage1,
            STATIONARY_stage1,
            max_horizon=2,
        )
    except Exception as e:
        print(f"ERROR in Stage 1: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Stage 2: Cluster Separation
    print("\n>>> RUNNING STAGE 2: Cluster Separation...")
    try:
        import fig_stage2_cluster_separation as stage2
        stage2_results = stage2.main()
    except Exception as e:
        print(f"ERROR in Stage 2: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        "stage0": stage0_results,
        "stage1": stage1_results,
        "stage2": stage2_results,
    }


def generate_report(all_results):
    """Generate comprehensive report following spec's reporting format."""
    if all_results is None:
        print("\nERROR: Pipeline failed. No report generated.")
        return

    stage0 = all_results["stage0"]
    stage1 = all_results["stage1"]
    stage2 = all_results["stage2"]

    print("\n" + "=" * 100)
    print(" " * 30 + "TWO-STAGE ESTIMATOR: FINAL REPORT")
    print("=" * 100)

    # Stage 0 results
    print("\n" + "-" * 100)
    print("STAGE 0: GATE — Do 36 belief states separate in supervised 5D subspace?")
    print("-" * 100)
    gate_pass = stage0.get("gate_pass", False)
    sep_ratio = stage0.get("separation_ratio", 0.0)
    print(f"\nSeparation ratio (between/within scatter):  {sep_ratio:.4f}")
    print(f"Gate status:                               {'PASS ✓' if gate_pass else 'FAIL ✗'}")

    mushy_states = stage0.get("mushy_states", [])
    print(f"Mushy transient states (>threshold):       {len(mushy_states)}")
    if len(mushy_states) > 0 and len(mushy_states) <= 5:
        print(f"  States: {mushy_states}")

    per_state = stage0.get("per_state_compactness", {})
    if per_state:
        transient_compact = [per_state[s] for s in sorted(per_state.keys()) if s != 0]
        print(f"Transient states (1-35):")
        print(f"  Mean compactness:    {np.mean(transient_compact):.4f}")
        print(f"  Std:                 {np.std(transient_compact):.4f}")
        print(f"  Range:               [{np.min(transient_compact):.4f}, {np.max(transient_compact):.4f}]")

    # Stage 1 results
    print("\n" + "-" * 100)
    print("STAGE 1: REGULARIZED CCA/RRR — Isolate belief-bearing region")
    print("-" * 100)
    sv = stage1.get("sv", [])
    print(f"\nCCA/RRR spectrum (first 12 singular values, normalized):")
    if len(sv) > 0:
        sv_norm = sv / sv[0]
        print(f"  {np.round(sv_norm[:12], 3)}")
    D_stage1 = stage1.get("D_stage1", 10)
    print(f"\nStage-1 subspace dimension:  d_stage1 = {D_stage1}")
    print(f"Note: This is a *generous* candidate subspace. Stage 2 will select true d from within it.")

    # Stage 2 results
    print("\n" + "-" * 100)
    print("STAGE 2: CLUSTER SEPARATION — Within-subspace d-selection")
    print("-" * 100)

    phase_a_success = stage2.get("phase_a_success", False)
    print(f"\nPhase A (Controlled with known labels):")
    print(f"  Direction-selection works?  {'YES ✓' if phase_a_success else 'NO ✗'}")

    phase_a_scores = stage2.get("phase_a_scores", {})
    if 5 in phase_a_scores:
        print(f"  State-level R² at d=5:      {phase_a_scores[5]['state_r2']:.4f}")

    plateau_d = stage2.get("plateau_d", None)
    phase_b_scores = stage2.get("phase_b_scores", {})
    print(f"\nPhase B (Fully unsupervised):")
    print(f"  Selected d (via plateau):   d = {plateau_d}")
    if plateau_d in phase_b_scores:
        r2_plateau = phase_b_scores[plateau_d]["state_r2"]
        print(f"  State-level R² at d_{plateau_d}:      {r2_plateau:.4f}")

    # Acceptance criteria
    print("\n" + "-" * 100)
    print("ACCEPTANCE CRITERIA")
    print("-" * 100)

    criterion_1 = stage2.get("criterion_1", False)
    criterion_2 = stage2.get("criterion_2", False)
    overall = stage2.get("overall_pass", False)

    print(f"\n1. Phase B d-selection returns d=5 via plateau:")
    print(f"   {'✓ PASS' if criterion_1 else '✗ FAIL'} (selected d = {plateau_d})")

    print(f"\n2. At d=5, state-level R² > 0.816 (spectral-OOM baseline):")
    if 5 in phase_b_scores:
        r2_d5 = phase_b_scores[5]["state_r2"]
        print(f"   {'✓ PASS' if criterion_2 else '✗ FAIL'} (R² = {r2_d5:.4f})")
    else:
        print(f"   ✗ FAIL (d=5 not in sweep range)")

    print(f"\nOverall: {'✓ PASS' if overall else '✗ FAIL'}")

    # Comparison table
    print("\n" + "-" * 100)
    print("COMPARISON TABLE: State-level R² at d=5")
    print("-" * 100)
    print(f"\n{'Method':<25} {'d':<5} {'State R²':<12} {'Notes'}")
    print("-" * 60)

    print(f"{'spectral-OOM':<25} {'5':<5} {'0.816':<12} {'Paper baseline'}")
    print(f"{'supervised ceiling':<25} {'5':<5} {'1.000':<12} {'Upper bound'}")

    if 5 in phase_b_scores:
        r2_d5 = phase_b_scores[5]["state_r2"]
        print(f"{'two-stage (Phase B)':<25} {'5':<5} {r2_d5:<12.3f} {'At true d=5'}")

    if plateau_d != 5 and plateau_d in phase_b_scores:
        r2_plateau = phase_b_scores[plateau_d]["state_r2"]
        print(f"{'two-stage (Phase B)':<25} {str(plateau_d):<5} {r2_plateau:<12.3f} {'At selected d'}")

    # Summary
    print("\n" + "=" * 100)
    print(" " * 35 + "SUMMARY")
    print("=" * 100)
    if overall:
        print("\n✓ SUCCESS: Two-stage estimator recovered correct d=5 and beats spectral-OOM baseline.")
        print("  The cluster-separation approach successfully re-ranked directions orthogonal to")
        print("  predictive relevance, isolating the true belief geometry from next-token-degenerate")
        print("  subspaces.")
    else:
        print("\n✗ The two-stage approach did not meet acceptance criteria.")
        if criterion_1:
            print("  Phase B recovered d=5, but state-level R² remains below baseline.")
        else:
            print(f"  Phase B selected d={plateau_d} ≠ 5. The plateau criterion did not identify")
            print("  the true dimensionality.")
        print("\n  Diagnostic notes:")
        print(f"  - Stage 0 gate: {'PASS' if gate_pass else 'FAIL (intrinsic separation gap)'}")
        print(f"  - Phase A (controlled): {'PASS' if phase_a_success else 'FAIL (within-subspace direction selection doesn\'t work)'}")
        print(f"  - Phase B unsupervised: check K-means convergence and LDA basis quality")

    print("\n" + "=" * 100 + "\n")


def save_results_summary(all_results):
    """Save a summary pickle for downstream analysis."""
    summary = {
        "stage0_gate_pass": all_results["stage0"].get("gate_pass"),
        "stage1_spectrum": all_results["stage1"].get("sv"),
        "stage2_plateau_d": all_results["stage2"].get("plateau_d"),
        "stage2_phase_b_r2_at_5": all_results["stage2"]["phase_b_scores"].get(5, {}).get("state_r2"),
        "overall_pass": all_results["stage2"].get("overall_pass"),
    }
    out = OUT_DIR / "two_stage_summary.pkl"
    with open(out, "wb") as f:
        pickle.dump(summary, f)
    print(f"Summary saved to {out}")


if __name__ == "__main__":
    import numpy as np

    # Run all stages
    all_results = run_all_stages()

    # Generate report
    if all_results is not None:
        generate_report(all_results)
        save_results_summary(all_results)
    else:
        print("\n" + "!" * 100)
        print("Pipeline failed. Check errors above.")
        print("!" * 100)
        sys.exit(1)
