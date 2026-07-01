"""D-selection methods for unsupervised belief geometry estimation.

Provides two complementary methods to select the latent dimensionality d:
1. Effective rank: counts singular values needed to explain variance threshold
2. MI saturation: finds plateau in mutual information as d increases
"""

import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import entropy


def elbow_method(singular_values, verbose=False):
    """Detect elbow in singular value spectrum using geometric method.

    The elbow is the point where the spectrum transitions from steep to flat.
    Detected as the point with maximum perpendicular distance from the line
    connecting first and last points (classic knee detection).

    Args:
        singular_values: array of singular values (descending order)
        verbose: print progress

    Returns:
        d_elbow: selected dimensionality (int)
    """
    if len(singular_values) < 3:
        return 1

    # Normalize singular values
    sv = np.asarray(singular_values, dtype=float)
    sv = sv / sv[0]  # Normalize by max

    # Geometric elbow detection: find point with max perpendicular distance
    # from line connecting start to end of spectrum
    x = np.arange(len(sv))
    y = sv

    # Line from (0, sv[0]) to (len-1, sv[-1])
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # Perpendicular distance of each point to this line
    # Formula: |ax + by + c| / sqrt(a^2 + b^2)
    # where line is: (y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0 = 0
    numerator = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominator = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    distances = numerator / denominator

    # Elbow is at max distance (point farthest from line)
    d_elbow = np.argmax(distances) + 1  # +1 for 1-indexing
    d_elbow = int(np.clip(d_elbow, 1, len(sv)))

    if verbose:
        print(f"  Elbow method selected d={d_elbow}")

    return d_elbow


def effective_rank(singular_values, variance_thresholds=(0.95, 0.99)):
    """Compute effective rank for one or more variance thresholds.

    Args:
        singular_values: array of singular values (descending order)
        variance_thresholds: float or tuple of floats in (0, 1)

    Returns:
        dict: {threshold: d} mapping each threshold to selected dimension
    """
    if isinstance(variance_thresholds, (int, float)):
        variance_thresholds = (variance_thresholds,)

    # Normalize to variance explained (squared, cumulative)
    sv_squared = singular_values ** 2
    total_var = np.sum(sv_squared)
    cumvar = np.cumsum(sv_squared) / total_var

    results = {}
    for thresh in variance_thresholds:
        # Find first d where cumulative variance exceeds threshold
        d = np.searchsorted(cumvar, thresh) + 1  # +1 because searchsorted is 0-indexed
        results[thresh] = int(np.clip(d, 1, len(singular_values)))

    return results


def mutual_information_saturation(A, softmax_targets, basis_U, max_d=None,
                                  plateau_threshold=0.01, ridge_alpha=1e-2,
                                  verbose=False):
    """Find dimensionality where mutual information saturates.

    Strategy: for each d in 1..max_d:
      1. Project activations to d dimensions
      2. Train linear decoder: projected_A -> next-token logits
      3. Compute R^2 of decoder (proxy for information captured)
      4. Find d where improvement is small and sustained (plateau detected)

    Uses adaptive plateau detection that identifies where diminishing returns begin:
    - Finds the point where successive improvements drop below a threshold
    - Looks for sustained plateau (not just a single small improvement)

    Args:
        A: (n_samples, d_model) activation matrix
        softmax_targets: (n_samples, vocab_size) next-token distribution
        basis_U: (d_model, d_candidate_max) orthonormal basis (left singular vectors)
        max_d: max dimensionality to test; if None, use min(rank(basis_U), 50)
        plateau_threshold: absolute improvement below this indicates plateau (default 0.01 = 1%)
        ridge_alpha: regularization for ridge regression
        verbose: print progress

    Returns:
        d_selected: selected dimensionality (int)
        mi_curve: {d: r2_score} mapping for all tested dimensions
        explanation: string describing the selected d
    """
    if max_d is None:
        max_d = min(basis_U.shape[1], 50)

    mi_curve = {}
    r2_scores = []

    # Test each d from 1 to max_d
    for d in range(1, max_d + 1):
        # Project to d dimensions
        B_d = basis_U[:, :d]
        A_d = A @ B_d

        # Train linear decoder from A_d to softmax
        try:
            decoder = Ridge(alpha=ridge_alpha, fit_intercept=True)
            decoder.fit(A_d, softmax_targets)

            # Compute R^2 on training set (proxy for MI)
            y_pred = decoder.predict(A_d)
            ss_res = np.sum((softmax_targets - y_pred) ** 2)
            ss_tot = np.sum((softmax_targets - softmax_targets.mean(0)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            r2_scores.append(r2)
            mi_curve[d] = float(r2)

            if verbose:
                print(f"  d={d:3d}: R^2={r2:.4f}")
        except Exception as e:
            if verbose:
                print(f"  d={d:3d}: failed ({e})")
            r2_scores.append(0.0)
            mi_curve[d] = 0.0

    # Find plateau: use adaptive detection
    if len(r2_scores) < 2:
        d_selected = 1
        explanation = "insufficient dimensions tested"
    else:
        max_r2 = np.max(r2_scores)
        improvements = np.diff([0] + r2_scores)  # improvement at each step

        # Adaptive plateau detection:
        # Look for where improvements consistently stay below threshold
        # (at least 2 consecutive small improvements)
        d_selected = 1
        consecutive_small = 0
        for d in range(1, len(improvements)):
            if improvements[d] < plateau_threshold:
                consecutive_small += 1
                if consecutive_small >= 2:
                    d_selected = d - 1  # Select one step before plateau starts
                    break
            else:
                consecutive_small = 0

        # Fallback: if no clear plateau, use "elbow" at 90% of max R²
        if d_selected == 1 and improvements[-1] < plateau_threshold:
            target = 0.90 * max_r2
            for d, r2 in enumerate(r2_scores, 1):
                if r2 >= target:
                    d_selected = d
                    break

        explanation = (
            f"plateau around d={d_selected} (improvement drops below {plateau_threshold:.4f}), "
            f"max R^2={max_r2:.4f}"
        )

    if verbose:
        print(f"\nMI saturation selected d={d_selected}: {explanation}")

    return d_selected, mi_curve, explanation


def compare_d_selection(effective_rank_results, elbow_d, mi_d, true_d=None, verbose=False):
    """Compare d-selection methods and report agreement.

    Args:
        effective_rank_results: dict from effective_rank() {threshold: d}
        elbow_d: selected d from elbow_method
        mi_d: selected d from mutual_information_saturation
        true_d: ground truth d (for reference)
        verbose: print comparison

    Returns:
        dict with results and agreement metrics
    """
    er_95 = effective_rank_results.get(0.95, None)
    er_99 = effective_rank_results.get(0.99, None)

    results = {
        "effective_rank_95": er_95,
        "effective_rank_99": er_99,
        "elbow": elbow_d,
        "mi_saturation": mi_d,
        "true_d": true_d,
    }

    # Check agreement
    candidates = [x for x in [er_95, er_99, elbow_d, mi_d] if x is not None]
    if len(candidates) > 0:
        agreement = max(candidates) - min(candidates) <= 1
        results["agreement"] = agreement
    else:
        results["agreement"] = None

    if verbose:
        print("\n" + "="*60)
        print("D-SELECTION RESULTS")
        print("="*60)
        print(f"  Effective Rank (95% var):  d={er_95}")
        print(f"  Effective Rank (99% var):  d={er_99}")
        print(f"  Elbow Method:              d={elbow_d}")
        print(f"  MI Saturation:             d={mi_d}")
        if true_d is not None:
            print(f"  Ground Truth:              d={true_d}")
            if candidates:
                closest = min(candidates, key=lambda x: abs(x - true_d))
                print(f"  Best Method:               d={closest} (±{abs(closest - true_d)} from true)")
                print(f"  All within ±1:             {all(abs(x - true_d) <= 1 for x in candidates)}")
        else:
            print(f"  Agreement (±1):            {agreement if 'agreement' in results else 'N/A'}")
        print("="*60)

    return results
