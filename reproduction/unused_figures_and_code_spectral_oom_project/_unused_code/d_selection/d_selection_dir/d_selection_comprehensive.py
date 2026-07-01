"""Comprehensive d-selection analysis: spectral-OOM vs CCA/RRR across all processes.

Tests:
  - Spectral-OOM with L ∈ {2, 3, 4, 5}
  - CCA/RRR with h ∈ {1, 2, 3, 4, 5, 6}

For each combination, evaluates:
  - Elbow detection (biggest singular-value drop)
  - d_selected (from elbow)
  - R² at d_selected
  - R² at d_true (ground truth)
  - Supervision ceiling
"""

from pathlib import Path
import json
import itertools
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.builder import build_hidden_markov_model

import fig14_observable_oom as F14_OOM
import fig14_cca_rrr_estimator as F14_CCA

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "d_selection_results"

# Process definitions
PROCESSES = {
    "mess3": {"params": {"x": 0.05, "a": 0.85}, "checkpoint": "mess3_transformer.pt", "d_true": 3},
    "rrxor": {"params": {"p1": 0.5, "p2": 0.5}, "checkpoint": "rrxor_transformer.pt", "d_true": 5},
    "arch": {"params": {"a": 0.85}, "checkpoint": "arch_transformer.pt", "d_true": 4},
    "wing": {"params": {"x": 0.5, "y": 0.5}, "checkpoint": "wing_transformer.pt", "d_true": 4},
    "strata": {"params": {"a": 0.85, "t0": 0.5, "t1": 0.5}, "checkpoint": "strata_transformer.pt", "d_true": 5},
    "fern": {"params": {"x": 0.5}, "checkpoint": "fern_transformer.pt", "d_true": 4},
    "zero_one_random": {"params": {"p": 0.5}, "checkpoint": "zero_one_random_transformer.pt", "d_true": 2},
}

SPECTRAL_OOM_DEPTHS = [2, 3, 4, 5]
CCA_RRR_HORIZONS = [1, 2, 3, 4, 5, 6]


def elbow_point(singular_values, method="max_drop"):
    """Detect elbow in singular spectrum.

    Args:
        singular_values: array of singular values (descending)
        method: "max_drop" = biggest relative drop, "knee" = curvature-based

    Returns:
        d_elbow: index where elbow is detected (1-indexed dimension)
    """
    sv_norm = singular_values / singular_values[0]

    if method == "max_drop":
        # Find biggest relative drop between consecutive singular values
        drops = sv_norm[:-1] - sv_norm[1:]
        d_elbow = np.argmax(drops) + 1
    elif method == "knee":
        # Simple knee detection: find max second derivative
        # (curvature of log singular values)
        log_sv = np.log(np.maximum(sv_norm, 1e-10))
        second_diff = np.diff(log_sv, n=2)
        d_elbow = np.argmax(np.abs(second_diff)) + 2
    else:
        raise ValueError(f"Unknown method: {method}")

    return max(1, min(d_elbow, len(singular_values)))


def run_spectral_oom(name, proc_name, depth):
    """Run spectral-OOM with given depth L."""
    hmm = build_hidden_markov_model(proc_name, PROCESSES[proc_name]["params"])
    T = hmm.transition_matrices
    stationary = np.ones(hmm.num_states) / hmm.num_states
    ckpt = PROCESSES[proc_name]["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = F14_OOM._load(ckpt, device)
    resid, soft, belief, pp = F14_OOM._collect(m, T, stationary, device)

    reach = {}
    for w in resid:
        ok = True
        pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14_OOM.EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    vocab = T.shape[0]
    rows, A, P, Gs, U, sv = F14_OOM.observable_subspace(resid, soft, reach, vocab, depth=depth)

    # Detect elbow
    d_elbow = elbow_point(sv, method="max_drop")

    # Evaluate at various dimensions
    Yb = np.stack([belief[w] for w in rows])
    fin = np.isfinite(Yb).all(1)

    decs = {}
    for d in [2, 3, 4, 5, 6, 8, 10]:
        if d <= len(sv):
            s_all, ops, e = F14_OOM.fit_at_dim(rows, A, P, resid, vocab, U[:, :d])
            dec = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
            decs[d] = dec

    # Ceiling
    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])

    # R² at d_elbow and d_true
    d_true = PROCESSES[proc_name]["d_true"]
    if d_elbow <= len(sv):
        s_all, _, _ = F14_OOM.fit_at_dim(rows, A, P, resid, vocab, U[:, :d_elbow])
        r2_elbow = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
    else:
        r2_elbow = np.nan

    if d_true <= len(sv):
        s_all, _, _ = F14_OOM.fit_at_dim(rows, A, P, resid, vocab, U[:, :d_true])
        r2_true = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
    else:
        r2_true = np.nan

    return {
        "method": "spectral-oom",
        "process": proc_name,
        "param": depth,
        "param_name": "L",
        "d_true": d_true,
        "d_elbow": d_elbow,
        "sv_spectrum": sv[:15].tolist(),
        "r2_elbow": float(r2_elbow),
        "r2_true": float(r2_true),
        "ceiling": float(ceiling),
        "decs": {int(d): float(r) for d, r in decs.items()},
    }


def run_cca_rrr(name, proc_name, max_horizon):
    """Run CCA/RRR with given horizon h."""
    hmm = build_hidden_markov_model(proc_name, PROCESSES[proc_name]["params"])
    T = hmm.transition_matrices
    stationary = np.ones(hmm.num_states) / hmm.num_states
    ckpt = PROCESSES[proc_name]["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = F14_CCA._load(ckpt, device)
    resid, soft, belief, pp = F14_CCA._collect(m, T, stationary, device)

    reach = {}
    for w in resid:
        ok = True
        pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14_CCA.EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    vocab = T.shape[0]
    prefix_probs = F14_CCA.analytic_prefix_probs(resid, T, stationary)

    rows, A, P, C_ks, O, U_plain, sv_plain, future_collections, sw = F14_CCA.direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=max_horizon, wmap=prefix_probs
    )

    # Use CCA whitening
    U_cca, sv_cca = F14_CCA.cca_rrr_subspace(A, future_collections, ridge_cca=0.01, sw=sw)
    U, sv = U_cca, sv_cca

    # Detect elbow
    d_elbow = elbow_point(sv, method="max_drop")

    # Evaluate at various dimensions
    Yb = np.stack([belief[w] for w in rows])
    fin = np.isfinite(Yb).all(1)

    decs = {}
    for d in [2, 3, 4, 5, 6, 8, 10]:
        if d <= len(sv):
            s_all, ops, e = F14_CCA.fit_at_dim(rows, A, P, resid, vocab, U[:, :d])
            dec = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
            decs[d] = dec

    # Ceiling
    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])

    # R² at d_elbow and d_true
    d_true = PROCESSES[proc_name]["d_true"]
    if d_elbow <= len(sv):
        s_all, _, _ = F14_CCA.fit_at_dim(rows, A, P, resid, vocab, U[:, :d_elbow])
        r2_elbow = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
    else:
        r2_elbow = np.nan

    if d_true <= len(sv):
        s_all, _, _ = F14_CCA.fit_at_dim(rows, A, P, resid, vocab, U[:, :d_true])
        r2_true = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
    else:
        r2_true = np.nan

    return {
        "method": "cca-rrr",
        "process": proc_name,
        "param": max_horizon,
        "param_name": "h",
        "d_true": d_true,
        "d_elbow": d_elbow,
        "sv_spectrum": sv[:15].tolist(),
        "r2_elbow": float(r2_elbow),
        "r2_true": float(r2_true),
        "ceiling": float(ceiling),
        "decs": {int(d): float(r) for d, r in decs.items()},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Test all processes
    for proc_name in sorted(PROCESSES.keys()):
        print(f"\n{'='*70}")
        print(f"Process: {proc_name} (d_true={PROCESSES[proc_name]['d_true']})")
        print(f"{'='*70}")

        # Spectral-OOM with varying depths
        print(f"\nSpectral-OOM:")
        for L in SPECTRAL_OOM_DEPTHS:
            try:
                print(f"  L={L}...", end=" ", flush=True)
                result = run_spectral_oom(f"{proc_name}_oom_L{L}", proc_name, L)
                results.append(result)
                print(f"d_elbow={result['d_elbow']}, r2_elbow={result['r2_elbow']:.3f}, r2_true={result['r2_true']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")

        # CCA/RRR with varying horizons
        print(f"\nCCA/RRR:")
        for h in CCA_RRR_HORIZONS:
            try:
                print(f"  h={h}...", end=" ", flush=True)
                result = run_cca_rrr(f"{proc_name}_cca_h{h}", proc_name, h)
                results.append(result)
                print(f"d_elbow={result['d_elbow']}, r2_elbow={result['r2_elbow']:.3f}, r2_true={result['r2_true']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")

    # Save results
    out_path = RESULTS_DIR / "d_selection_full_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved results to {out_path}")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Process':<20} {'Method':<12} {'Param':<5} {'d_true':<6} {'d_elbow':<8} {'Elbow R²':<10} {'True d R²':<10} {'Ceiling':<10}")
    print("-"*100)

    for result in sorted(results, key=lambda r: (r["process"], r["method"], r["param"])):
        proc = result["process"]
        method = result["method"]
        param = result["param"]
        d_true = result["d_true"]
        d_elbow = result["d_elbow"]
        r2_elbow = result["r2_elbow"]
        r2_true = result["r2_true"]
        ceiling = result["ceiling"]

        print(f"{proc:<20} {method:<12} {param:<5} {d_true:<6} {d_elbow:<8} {r2_elbow:<10.3f} {r2_true:<10.3f} {ceiling:<10.3f}")


if __name__ == "__main__":
    main()
