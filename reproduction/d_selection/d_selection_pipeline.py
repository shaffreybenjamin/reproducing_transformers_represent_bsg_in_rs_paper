"""Unified pipeline for d-selection testing across all processes.

Runs both spectral-OOM and CCA/RRR, applies d-selection methods (effective rank + MI saturation),
and reports results for any HMM process.

Usage (from within a process-specific script):
    results = run_d_selection_pipeline(model, hmm, context_len, device, process_name, true_d)
"""

import numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path

import reproduction.estimators.unsupervised_belief_oom as U
import stage1_cca_general as S1
import d_selection as D_sel
import fig14_observable_oom as FIG14


def run_d_selection_pipeline(model, hmm, context_len, device, process_name, true_d,
                             max_d_test=50, verbose=True):
    """Run spectral-OOM + CCA/RRR d-selection pipeline on a process.

    Args:
        model: HookedTransformer model (loaded and eval mode)
        hmm: HMM process (from simplexity)
        context_len: context length
        device: torch device
        process_name: str name of process (for output)
        true_d: int ground truth dimensionality
        max_d_test: max d to test for MI saturation
        verbose: print progress

    Returns:
        dict with all results:
            - process_name
            - true_d
            - spectral_oom: {effective_rank_95, effective_rank_99, mi_saturation, sv, best_d}
            - cca_rrr: {effective_rank_95, effective_rank_99, mi_saturation, sv, best_d}
            - activations (for further analysis)
            - beliefs (ground truth, for validation)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"D-SELECTION PIPELINE: {process_name} (true d={true_d})")
        print(f"{'='*70}")

    vocab = hmm.vocab_size

    # Step 1: Collect prefix features
    if verbose:
        print(f"\n[1/3] Collecting prefix features...")
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(
        model, hmm, context_len, device
    )

    good = {
        w for w in resid
        if w in belief and np.all(np.isfinite(belief[w])) and prefix_prob.get(w, 0.0) > 1e-12
    }
    reach = {w: (w in good) for w in resid}

    if verbose:
        print(f"  Enumerated prefixes: {len(resid)}")
        print(f"  Reachable (prob > 0): {len(good)}")
        print(f"  Max prefix length: {max_len}")

    # Prepare inputs for d-selection
    seqs = [s for s in resid if s in good]
    true_b = np.array([belief[s] for s in seqs])
    A = np.array([resid[s] for s in seqs])
    P = np.array([soft[s] for s in seqs])
    wcol = np.array([prefix_prob[s] for s in seqs])

    # Step 2: Run Spectral-OOM
    if verbose:
        print(f"\n[2/3] Running Spectral-OOM...")
    rows_oom, A_oom, P_oom, Gs, U_oom, sv_oom = FIG14.observable_subspace(
        resid, soft, reach, vocab, depth=3, wmap=None
    )
    if verbose:
        print(f"  Observable matrix rank: {len(sv_oom)}")
        print(f"  Top-5 singular values: {sv_oom[:5]}")

    # Step 3: Run CCA/RRR
    if verbose:
        print(f"\n[3/3] Running CCA/RRR...")
    (rows_cca, A_cca, P_cca, C_ks, O_cca, U_cca_plain, sv_cca_plain,
     future_collections, sw) = S1.direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=S1.MAX_HORIZON, wmap=None
    )
    U_cca, sv_cca = S1.cca_rrr_subspace_ledoit(
        A_cca, future_collections, ridge_cca=S1.RIDGE_CCA, sw=sw,
        ledoit_shrinkage=S1.LEDOIT_WOLF_SHRINKAGE
    )
    if verbose:
        print(f"  CCA/RRR basis shape: {U_cca.shape}")
        print(f"  Top-5 singular values: {sv_cca[:5]}")

    # Apply d-selection methods to spectral-OOM
    if verbose:
        print(f"\n--- SPECTRAL-OOM D-SELECTION ---")
    er_oom = D_sel.effective_rank(sv_oom, variance_thresholds=(0.95, 0.99))
    elbow_oom = D_sel.elbow_method(sv_oom, verbose=verbose)
    mi_d_oom, mi_curve_oom, mi_explanation_oom = D_sel.mutual_information_saturation(
        A_oom, P_oom, U_oom, max_d=max_d_test, plateau_threshold=0.05, verbose=verbose
    )

    results_oom = D_sel.compare_d_selection(er_oom, elbow_oom, mi_d_oom, true_d=true_d, verbose=verbose)
    results_oom["sv"] = sv_oom
    results_oom["basis_U"] = U_oom
    results_oom["activations"] = A_oom
    results_oom["softmax"] = P_oom
    results_oom["mi_curve"] = mi_curve_oom

    # Apply d-selection methods to CCA/RRR
    if verbose:
        print(f"\n--- CCA/RRR D-SELECTION ---")
    er_cca = D_sel.effective_rank(sv_cca, variance_thresholds=(0.95, 0.99))
    elbow_cca = D_sel.elbow_method(sv_cca, verbose=verbose)
    mi_d_cca, mi_curve_cca, mi_explanation_cca = D_sel.mutual_information_saturation(
        A_cca, P_cca, U_cca, max_d=max_d_test, plateau_threshold=0.05, verbose=verbose
    )

    results_cca = D_sel.compare_d_selection(er_cca, elbow_cca, mi_d_cca, true_d=true_d, verbose=verbose)
    results_cca["sv"] = sv_cca
    results_cca["basis_U"] = U_cca
    results_cca["activations"] = A_cca
    results_cca["softmax"] = P_cca
    results_cca["mi_curve"] = mi_curve_cca

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Process:  {process_name}")
        print(f"True d:   {true_d}")
        print(f"\nSpectral-OOM:")
        print(f"  Effective Rank (95%):  {results_oom['effective_rank_95']}")
        print(f"  Effective Rank (99%):  {results_oom['effective_rank_99']}")
        print(f"  Elbow Method:          {results_oom['elbow']}")
        print(f"  MI Saturation:         {results_oom['mi_saturation']}")
        print(f"\nCCA/RRR:")
        print(f"  Effective Rank (95%):  {results_cca['effective_rank_95']}")
        print(f"  Effective Rank (99%):  {results_cca['effective_rank_99']}")
        print(f"  Elbow Method:          {results_cca['elbow']}")
        print(f"  MI Saturation:         {results_cca['mi_saturation']}")

    return {
        "process_name": process_name,
        "true_d": true_d,
        "spectral_oom": results_oom,
        "cca_rrr": results_cca,
        "ground_truth_beliefs": true_b,
        "ground_truth_beliefs_seqs": seqs,
        "prefix_probs": wcol,
    }
