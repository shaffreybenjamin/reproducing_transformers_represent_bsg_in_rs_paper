"""Fast d-selection analysis: collect once per process, test all parameter combinations.

Tests:
  - Spectral-OOM with L ∈ {2, 3, 4, 5}
  - CCA/RRR with h ∈ {1, 2, 3, 4, 5, 6}

Optimization: collect activations once per process, reuse for all parameter settings.
"""

from pathlib import Path
import json
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
    """Detect elbow in singular spectrum using multiple criteria."""
    sv_norm = singular_values / singular_values[0]

    if method == "max_drop":
        drops = sv_norm[:-1] - sv_norm[1:]
        d_elbow = np.argmax(drops) + 1
    elif method == "knee":
        # Knee detection: max second derivative of log spectrum
        log_sv = np.log(np.maximum(sv_norm, 1e-10))
        second_diff = np.diff(log_sv, n=2)
        d_elbow = np.argmax(np.abs(second_diff)) + 2
    else:
        raise ValueError(f"Unknown method: {method}")

    return max(1, min(d_elbow, len(singular_values)))


def eval_at_dimension(U, sv, A, P, rows, resid, vocab, Yb, fin, d):
    """Evaluate R² at a specific dimension."""
    if d > len(sv):
        return np.nan
    s_all, ops, e = F14_OOM.fit_at_dim(rows, A, P, resid, vocab, U[:, :d])
    return LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])


def collect_once(proc_name):
    """Collect activations and metadata once per process."""
    print(f"  Collecting activations for {proc_name}...")

    hmm = build_hidden_markov_model(proc_name, PROCESSES[proc_name]["params"])
    T = hmm.transition_matrices
    stationary = np.ones(hmm.num_states) / hmm.num_states
    ckpt = PROCESSES[proc_name]["checkpoint"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = F14_OOM._load(str(MODEL_DIR / ckpt), device)
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

    # Filter to reachable prefixes with valid children
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > F14_OOM.EPS)]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])

    Yb = np.stack([belief[w] for w in rows])
    fin = np.isfinite(Yb).all(1)

    # Compute prefix probabilities for weighting
    prefix_probs = F14_CCA.analytic_prefix_probs(resid, T, stationary)

    return {
        "resid": resid,
        "soft": soft,
        "belief": belief,
        "reach": reach,
        "vocab": vocab,
        "rows": rows,
        "A": A,
        "P": P,
        "Yb": Yb,
        "fin": fin,
        "T": T,
        "stationary": stationary,
        "prefix_probs": prefix_probs,
    }


def test_spectral_oom(data, depth):
    """Test spectral-OOM with given depth."""
    rows, A, P = data["rows"], data["A"], data["P"]
    resid, soft, reach, vocab = data["resid"], data["soft"], data["reach"], data["vocab"]
    Yb, fin = data["Yb"], data["fin"]

    rows_obj, A_obj, P_obj, Gs, U, sv = F14_OOM.observable_subspace(
        resid, soft, reach, vocab, depth=depth
    )

    d_elbow = elbow_point(sv, method="max_drop")

    # Evaluate at d_elbow and d_true
    d_true = PROCESSES[list(PROCESSES.keys())[0]]["d_true"]  # will be set properly in caller
    r2_elbow = eval_at_dimension(U, sv, A, P, rows, resid, vocab, Yb, fin, d_elbow)
    r2_true = eval_at_dimension(U, sv, A, P, rows, resid, vocab, Yb, fin, d_true)

    # Ceiling
    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])

    return {
        "d_elbow": int(d_elbow),
        "sv_spectrum": sv[:15].tolist(),
        "r2_elbow": float(r2_elbow),
        "r2_true": float(r2_true),
        "ceiling": float(ceiling),
    }


def test_cca_rrr(data, max_horizon):
    """Test CCA/RRR with given horizon."""
    rows = data["rows"]
    A, P = data["A"], data["P"]
    resid, soft, reach, vocab = data["resid"], data["soft"], data["reach"], data["vocab"]
    Yb, fin = data["Yb"], data["fin"]
    prefix_probs = data["prefix_probs"]

    # Fit multi-step readouts
    rows_obj, A_obj, P_obj, C_ks, O, U_plain, sv_plain, future_collections, sw = F14_CCA.direct_multistep_readouts(
        resid, soft, reach, vocab, max_horizon=max_horizon, wmap=prefix_probs
    )

    # Apply CCA whitening
    U_cca, sv_cca = F14_CCA.cca_rrr_subspace(A_obj, future_collections, ridge_cca=0.01, sw=sw)
    U, sv = U_cca, sv_cca

    d_elbow = elbow_point(sv, method="max_drop")

    # Evaluate at d_elbow and d_true
    d_true = PROCESSES[list(PROCESSES.keys())[0]]["d_true"]  # will be set properly in caller
    r2_elbow = eval_at_dimension(U, sv, A_obj, P_obj, rows_obj, resid, vocab, Yb, fin, d_elbow)
    r2_true = eval_at_dimension(U, sv, A_obj, P_obj, rows_obj, resid, vocab, Yb, fin, d_true)

    # Ceiling
    ceiling = LinearRegression().fit(A_obj[fin], Yb[fin]).score(A_obj[fin], Yb[fin])

    return {
        "d_elbow": int(d_elbow),
        "sv_spectrum": sv[:15].tolist(),
        "r2_elbow": float(r2_elbow),
        "r2_true": float(r2_true),
        "ceiling": float(ceiling),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for proc_name in sorted(PROCESSES.keys()):
        print(f"\n{'='*70}")
        print(f"Process: {proc_name} (d_true={PROCESSES[proc_name]['d_true']})")
        print(f"{'='*70}")

        # Collect once
        try:
            data = collect_once(proc_name)
        except Exception as e:
            print(f"ERROR collecting data: {e}")
            continue

        d_true = PROCESSES[proc_name]["d_true"]

        # Test spectral-OOM
        print(f"Spectral-OOM:")
        for L in SPECTRAL_OOM_DEPTHS:
            try:
                print(f"  L={L}...", end=" ", flush=True)
                test_result = test_spectral_oom(data, L)
                result = {
                    "method": "spectral-oom",
                    "process": proc_name,
                    "param": L,
                    "param_name": "L",
                    "d_true": d_true,
                    **test_result,
                }
                results.append(result)
                print(f"d_elbow={result['d_elbow']}, r2_elbow={result['r2_elbow']:.3f}, r2_true={result['r2_true']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")

        # Test CCA/RRR
        print(f"CCA/RRR:")
        for h in CCA_RRR_HORIZONS:
            try:
                print(f"  h={h}...", end=" ", flush=True)
                test_result = test_cca_rrr(data, h)
                result = {
                    "method": "cca-rrr",
                    "process": proc_name,
                    "param": h,
                    "param_name": "h",
                    "d_true": d_true,
                    **test_result,
                }
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
    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    print(f"{'Process':<20} {'Method':<12} {'Param':<5} {'d_true':<6} {'d_elbow':<8} {'Correct?':<10} {'Elbow R²':<10} {'True d R²':<10} {'Ceiling':<10}")
    print("-"*110)

    for result in sorted(results, key=lambda r: (r["process"], r["method"], r["param"])):
        proc = result["process"]
        method = result["method"]
        param = result["param"]
        d_true = result["d_true"]
        d_elbow = result["d_elbow"]
        correct = "✓" if d_elbow == d_true else f"✗ ({d_elbow})"
        r2_elbow = result["r2_elbow"]
        r2_true = result["r2_true"]
        ceiling = result["ceiling"]

        print(f"{proc:<20} {method:<12} {param:<5} {d_true:<6} {d_elbow:<8} {correct:<10} {r2_elbow:<10.3f} {r2_true:<10.3f} {ceiling:<10.3f}")


if __name__ == "__main__":
    main()
