"""Verify that Mess3's belief-update operators are diagonalizable strict contractions.

Mess3 has 3 inputs (A, B, C), each inducing a 3x3 transition matrix.
"""

import numpy as np


def analyze_operator(M, name="Operator"):
    """Analyze spectral and contractivity properties of a transition matrix."""
    print(f"\n{name}:")
    print("Matrix:")
    print(M)

    # Spectral properties
    evals, evecs = np.linalg.eig(M)
    evals_sorted = np.sort(np.abs(evals))[::-1]
    print(f"\nEigenvalues (by magnitude): {evals_sorted}")
    print(f"Spectral radius (max|λ|): {np.max(np.abs(evals)):.6f}")

    # Check diagonalizability
    unique_evals, counts = np.unique(np.round(evals, 10), return_counts=True)
    print(f"\nDiagonalizability check:")
    print(f"  Unique eigenvalues: {unique_evals}")
    print(f"  Number of distinct eigenvalues: {len(unique_evals)}")

    # Key fact: if all eigenvalues are distinct, matrix is diagonalizable
    is_diagonalizable = (len(unique_evals) == M.shape[0])

    for lam in unique_evals:
        alg_mult = counts[np.where(unique_evals == lam)[0][0]]
        if alg_mult == 1:
            geom_mult = 1
            defect = 0
        else:
            defect_matrix = M - lam * np.eye(M.shape[0])
            rank = np.linalg.matrix_rank(defect_matrix)
            geom_mult = M.shape[0] - rank
            defect = alg_mult - geom_mult
        print(f"    λ={lam:.6f}: algebraic={alg_mult}, geometric={geom_mult}, defect={defect}")
        if defect > 0:
            is_diagonalizable = False

    # Eigenvector condition number
    if np.all(np.isfinite(evecs)):
        cond_number = np.linalg.cond(evecs)
        print(f"  Eigenvector matrix condition number: {cond_number:.2e}")
        print(f"  Well-conditioned: {cond_number < 1e10}")

    print(f"  ✓ Diagonalizable: {is_diagonalizable}")

    # Check for strict contraction
    spec_radius = np.max(np.abs(evals))
    is_strict_contraction = spec_radius < 1.0
    print(f"\nStrict contraction check:")
    print(f"  Spectral radius: {spec_radius:.6f}")
    print(f"  ✓ Strict contraction (ρ < 1): {is_strict_contraction}")

    # Iterative behavior
    print(f"\nIterative decay:")
    M_power = M.copy()
    for k in range(1, 6):
        max_entry = np.max(np.abs(M_power))
        print(f"  ||M^{k}||_∞ = {max_entry:.6f}")
        M_power = M_power @ M

    # Jordan normal form (for diagonalizable matrices)
    if is_diagonalizable:
        print(f"\nJordan normal form:")
        print(f"  M = P D P⁻¹ where D is diagonal")
        D = np.diag(evals)
        print(f"  D (diagonal eigenvalues):")
        print(f"    {np.diag(D)}")

    return is_diagonalizable, is_strict_contraction


def main():
    # Mess3 transition matrices from writeup_v1.tex
    T_A = np.array([
        [0.765, 0.00375, 0.00375],
        [0.0425, 0.0675, 0.00375],
        [0.0425, 0.00375, 0.0675]
    ])

    T_B = np.array([
        [0.0675, 0.0425, 0.00375],
        [0.00375, 0.765, 0.00375],
        [0.00375, 0.0425, 0.0675]
    ])

    T_C = np.array([
        [0.0675, 0.00375, 0.0425],
        [0.00375, 0.0675, 0.0425],
        [0.00375, 0.00375, 0.765]
    ])

    print("=" * 70)
    print("MESS3 BELIEF-UPDATE OPERATORS ANALYSIS")
    print("=" * 70)
    print("Mess3 has 3 inputs (A, B, C), each with a 3x3 transition matrix.")
    print("These are the DIAGONALIZABLE STRICT CONTRACTIONS claimed in the writeup.")
    print()

    # Analyze each operator
    diag_A, contract_A = analyze_operator(T_A, "T^(A) - Input A operator")
    diag_B, contract_B = analyze_operator(T_B, "T^(B) - Input B operator")
    diag_C, contract_C = analyze_operator(T_C, "T^(C) - Input C operator")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_diagonalizable = diag_A and diag_B and diag_C
    all_contractions = contract_A and contract_B and contract_C

    print(f"Diagonalizable: {all_diagonalizable}")
    print(f"  T^(A): {diag_A}")
    print(f"  T^(B): {diag_B}")
    print(f"  T^(C): {diag_C}")
    print()
    print(f"Strict contractions (ρ < 1): {all_contractions}")
    print(f"  T^(A): {contract_A}")
    print(f"  T^(B): {contract_B}")
    print(f"  T^(C): {contract_C}")
    print()

    if all_diagonalizable and all_contractions:
        print("✓ VERIFIED: Mess3 operators are diagonalizable strict contractions")
        print()
        print("Implication: Belief states evolve via diagonalizable operators with")
        print("spectral radius < 1, meaning beliefs converge to a fixed point or")
        print("attractor set (unlike RRXOR's defective transients).")
    else:
        print("✗ CLAIM FAILED")
        if not all_diagonalizable:
            print("  - Some operators are NOT diagonalizable")
        if not all_contractions:
            print("  - Some operators are NOT strict contractions")


if __name__ == "__main__":
    main()
