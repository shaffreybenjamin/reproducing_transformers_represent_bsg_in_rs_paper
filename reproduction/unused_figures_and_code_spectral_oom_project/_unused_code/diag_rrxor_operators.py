"""Verify the claim that RRXOR belief-update operators are non-diagonalizable (nilpotent/defective).

Tests whether the transition matrix T for RRXOR and its belief-update operators
(conditional transition matrices T_x for each observation x) have:
  - Defective (non-diagonalizable) structure
  - Nilpotent or near-nilpotent properties
"""

import numpy as np
from scipy.linalg import svd
from simplexity.generative_processes.transition_matrices import rrxor, mess3


def analyze_matrix_spectrum(M, name="Matrix"):
    """Compute and report spectral properties of a matrix."""
    evals, evecs = np.linalg.eig(M)
    print(f"\n{name}:")
    print(f"  Shape: {M.shape}")
    print(f"  Eigenvalues: {np.sort(np.abs(evals))[::-1]}")

    # Check for repeated eigenvalues (indication of defect)
    unique_evals, counts = np.unique(np.round(evals, 10), return_counts=True)
    repeated = unique_evals[counts > 1]
    if len(repeated) > 0:
        print(f"  Repeated eigenvalues (algebraic multiplicity > 1): {repeated}")

    # Check geometric multiplicity via rank of (M - lambda*I)
    print(f"  Eigenvalue multiplicities:")
    for lam in unique_evals:
        # Algebraic multiplicity
        alg_mult = counts[np.where(unique_evals == lam)[0][0]]
        # Geometric multiplicity via rank
        defect_matrix = M - lam * np.eye(M.shape[0])
        rank = np.linalg.matrix_rank(defect_matrix)
        geom_mult = M.shape[0] - rank
        print(f"    λ={lam:.4f}: algebraic={alg_mult}, geometric={geom_mult}, defect={alg_mult - geom_mult}")

    # Check trace and det (sum and product of eigenvalues)
    print(f"  Trace: {np.trace(M):.6f} (sum of eigenvalues: {np.sum(evals):.6f})")
    print(f"  Det: {np.linalg.det(M):.6f} (product of eigenvalues: {np.prod(evals):.6f})")

    # Check for nilpotency: is M^k = 0 for some k?
    print(f"  Nilpotency check (M^k = 0?):")
    M_power = M.copy()
    for k in range(2, M.shape[0] + 2):
        M_power = M_power @ M
        max_entry = np.max(np.abs(M_power))
        print(f"    M^{k} max|entry| = {max_entry:.2e}")
        if max_entry < 1e-10:
            print(f"    -> Nilpotent! (M^{k} ≈ 0)")
            return True

    # Check Jordan form via SVD of eigenvector matrix
    if np.all(np.isfinite(evecs)):
        cond_number = np.linalg.cond(evecs)
        print(f"  Eigenvector matrix condition number: {cond_number:.2e}")
        if cond_number > 1e10:
            print(f"    -> Eigenvectors are ill-conditioned (sign of defect)")

    is_diagonalizable = np.all([counts[i] - (M.shape[0] - np.linalg.matrix_rank(M - unique_evals[i] * np.eye(M.shape[0]))) == 0 for i in range(len(unique_evals))])
    print(f"  Diagonalizable: {is_diagonalizable}")

    return False


def main():
    # Compute transition matrices
    T_rrxor_raw = rrxor(0.5, 0.5)
    T_mess3_raw = mess3(x=0.05, a=0.85)

    print(f"RRXOR raw shape: {np.array(T_rrxor_raw).shape}")
    print(f"Mess3 raw shape: {np.array(T_mess3_raw).shape}")

    # RRXOR is (observation, next_state, current_state)
    # Mess3 is (observation, next_state, current_state)
    # Extract the transition matrices
    T_rrxor = np.array(T_rrxor_raw)  # shape: (n_obs, n_states, n_states)
    T_mess3 = np.array(T_mess3_raw)  # shape: (n_obs, n_states, n_states)

    print("=" * 70)
    print("RRXOR TRANSITION MATRIX ANALYSIS")
    print("=" * 70)
    print(f"RRXOR structure: {T_rrxor.shape[0]} observations, {T_rrxor.shape[1]} states")

    # Analyze each observation channel
    for obs_idx in range(T_rrxor.shape[0]):
        analyze_matrix_spectrum(T_rrxor[obs_idx], f"RRXOR Observation {obs_idx} Transition Matrix")

    # For comparison, analyze Mess3
    print("\n" + "=" * 70)
    print("MESS3 TRANSITION MATRIX (for comparison)")
    print("=" * 70)
    print(f"Mess3 structure: {T_mess3.shape[0]} observations, {T_mess3.shape[1]} states")
    for obs_idx in range(T_mess3.shape[0]):
        analyze_matrix_spectrum(T_mess3[obs_idx], f"Mess3 Observation {obs_idx} Transition Matrix")

    # Check if T^T has different properties (since belief update uses b @ T)
    print("\n" + "=" * 70)
    print("RRXOR T^T Analysis (belief-update direction)")
    print("=" * 70)
    for obs_idx in range(T_rrxor.shape[0]):
        analyze_matrix_spectrum(T_rrxor[obs_idx].T, f"RRXOR Observation {obs_idx} T^T")


if __name__ == "__main__":
    main()
