"""Test whether multi-step ALS can work in principle using ground-truth operators.

Two tests:
1. Test 1 — belief space, true operators: verify rescaling is correct
2. Test 2 — activation space, true-derived operators: check if multi-step solution exists
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from simplexity.generative_processes.transition_matrices import mess3, rrxor

import sys
sys.path.insert(0, 'reproduction/plotting')
sys.path.insert(0, 'reproduction/estimators')

import fig14_observable_oom as F14
import unsupervised_belief_oom as U

EPS = 1e-3
RIDGE = 1e-2


def test_belief_space_true_operators(name, T, pi, max_order=2):
    """Test 1: Verify multi-step rescaling in belief space with true T^x."""
    print(f"\n{'='*60}")
    print(f"TEST 1: {name} — Belief space, true operators")
    print(f"{'='*60}")

    NS = T.shape[1]
    vocab = T.shape[0]

    # Generate all prefixes and compute true beliefs
    import itertools
    max_len = 10
    all_prefixes = []
    beliefs = {}

    for L in range(max_len + 1):
        for w in itertools.product(range(vocab), repeat=L):
            b = np.array(pi, dtype=float)
            p = 1.0
            for x in w:
                d = np.array([(b @ T[i]).sum() for i in range(vocab)])
                if d[x] < 1e-12:
                    p = 0.0
                    break
                p *= float(d[x])
                b = b @ T[x] / d[x]
            if p > 0:
                all_prefixes.append(w)
                beliefs[w] = (b, p)

    print(f"Generated {len(all_prefixes)} reachable prefixes")

    # Compute multi-step targets in belief space
    # For order-k: b(w) T^{x_1} T^{x_2} ... T^{x_k} ≈ P(x_1...x_k|w) b(w x_1...x_k)

    def compute_belief_space_loss(order=2):
        """Compute multi-step loss in belief space with true T^x."""
        total_loss = 0.0
        count = 0

        for w in all_prefixes:
            if w not in beliefs:
                continue
            b_w, p_w = beliefs[w]

            # Generate all continuations of order k
            for continuation in itertools.product(range(vocab), repeat=order):
                w_cont = w + continuation
                if w_cont not in beliefs:
                    continue

                b_wc, p_wc = beliefs[w_cont]

                # Compute product rescale: P(continuation|w)
                rescale = 1.0
                prefix = w
                for x in continuation:
                    if prefix not in beliefs:
                        rescale = 0.0
                        break
                    b_pre, _ = beliefs[prefix]
                    cond_prob = (b_pre @ T[x]).sum()
                    rescale *= cond_prob
                    prefix = prefix + (x,)

                if rescale < 1e-12:
                    continue

                # True belief multi-step prediction
                b_pred = b_w.copy()
                for x in continuation:
                    b_pred = b_pred @ T[x]  # Un-rescaled product

                b_target = b_wc  # Target is the true belief at wc

                # Loss: ||b_pred - rescale * b_target||^2
                # Actually: b_pred ≈ rescale * b_target, so (b_pred - rescale * b_target)
                residual = b_pred - rescale * b_target
                loss = np.sum(residual ** 2)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1), count

    # Test order-1
    loss_1, count_1 = compute_belief_space_loss(order=1)
    print(f"Order-1 loss: {loss_1:.2e} ({count_1} terms)")

    # Test order-2
    if max_order >= 2:
        loss_2, count_2 = compute_belief_space_loss(order=2)
        print(f"Order-2 loss: {loss_2:.2e} ({count_2} terms)")

    if loss_1 < 1e-10:
        print("✓ Belief-space rescaling is CORRECT")
        return True
    else:
        print("✗ Belief-space rescaling has BUGS")
        return False


def test_activation_space_true_operators(name, T, pi, model_ckpt, device="cpu"):
    """Test 2: Check if multi-step solution exists in activation space with true A_x."""
    print(f"\n{'='*60}")
    print(f"TEST 2: {name} — Activation space, true-derived operators")
    print(f"{'='*60}")

    vocab = T.shape[0]  # T is (vocab, NS)

    # Load model and collect activations
    import torch
    model = F14._load(model_ckpt, device)
    resid, soft, belief_labels, _ = F14._collect(model, T, pi, device)

    # Compute reachability
    reach = {}
    for w in resid:
        ok = True
        pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    # Filter out rows with NaN beliefs
    rows = [w for w in rows if np.isfinite(belief_labels[w]).all()]
    print(f"Reachable rows for regression: {len(rows)}")

    A = np.stack([resid[w] for w in rows])
    B = np.stack([belief_labels[w] for w in rows])

    # Regress to get Ψ: a(w) = b(w) Ψ + c (affine encoding)
    # Work in centered coordinates to avoid affine offset issues in composition
    B_centered = B - B.mean(axis=0, keepdims=True)
    A_centered = A - A.mean(axis=0, keepdims=True)

    psi = LinearRegression(fit_intercept=False).fit(B_centered, A_centered).coef_.T  # (NS, D)
    psi_pinv = np.linalg.pinv(psi)  # (D, NS)

    print(f"Encoding: B {B.shape} → A {A.shape} (centered)")
    print(f"Ψ shape: {psi.shape}, Ψ+ shape: {psi_pinv.shape}")

    # Compute true activation-space operators: A_x = Ψ^+ T^x Ψ
    print("\nComputing true activation-space operators A_x = Ψ⁺ T^x Ψ...")
    Gs_true = []
    for x in range(vocab):
        A_x = psi_pinv @ T[x] @ psi
        Gs_true.append(A_x)

    # Test spectrum of true operators
    print("\nSpectrum test:")
    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, np.stack([soft[w] for w in rows])).coef_.T
    cols, frontier = [C], [C]
    for _ in range(4):  # depth 5
        nxt = [G @ f for G in Gs_true for f in frontier]
        cols += nxt
        frontier = nxt
    O = np.hstack(cols)
    U_obs, sv, _ = np.linalg.svd(O, full_matrices=False)
    print(f"Singular values: {np.round(sv[:10] / sv[0], 3)}")

    correct_d = T.shape[1]  # Number of hidden states
    drops = np.diff(sv[:10] / sv[0])
    print(f"Largest drop: {-drops.min():.3f} at position {np.argmin(drops)}")
    if np.argmin(drops) + 1 == correct_d:
        print(f"✓ Clean elbow at d={correct_d}")
    else:
        print(f"✗ Elbow at d={np.argmin(drops)+1}, expected d={correct_d}")

    # Test 2b: Multi-step ALS objective at true operators (in centered coordinates)
    print("\nMulti-step loss test (true A_x, in centered coordinates):")

    # Build row-to-index lookup for fast indexing
    row_to_idx = {w: i for i, w in enumerate(rows)}

    def compute_multistep_loss_true_ops_centered(Gs, order=2):
        """Compute multi-step loss with true operators in centered coordinates."""
        total_loss = 0.0
        count = 0

        for i, w in enumerate(rows):
            a_w_centered = A_centered[i]

            if order == 1:
                for x in range(vocab):
                    if soft[w][x] > EPS and w + (x,) in row_to_idx:
                        j = row_to_idx[w + (x,)]
                        a_wx_centered = A_centered[j]
                        pred = a_w_centered @ Gs[x]
                        target = soft[w][x] * a_wx_centered
                        loss = np.sum((pred - target) ** 2)
                        total_loss += loss
                        count += 1

            elif order == 2:
                for x in range(vocab):
                    for y in range(vocab):
                        if (soft[w][x] > EPS and w + (x,) in soft and
                            soft[w + (x,)][y] > EPS and w + (x, y) in row_to_idx):

                            j_x = row_to_idx[w + (x,)]
                            j_xy = row_to_idx[w + (x, y)]

                            a_wx_centered = A_centered[j_x]
                            a_wxy_centered = A_centered[j_xy]

                            rescale = soft[w][x] * soft[w + (x,)][y]
                            pred = a_w_centered @ Gs[x] @ Gs[y]
                            target = rescale * a_wxy_centered
                            loss = np.sum((pred - target) ** 2)
                            total_loss += loss
                            count += 1

        return total_loss / max(count, 1) if count > 0 else 0.0, count

    loss_1, count_1 = compute_multistep_loss_true_ops_centered(Gs_true, order=1)
    loss_2, count_2 = compute_multistep_loss_true_ops_centered(Gs_true, order=2)

    print(f"Order-1 loss: {loss_1:.2e} ({count_1} terms)")
    print(f"Order-2 loss: {loss_2:.2e} ({count_2} terms)")

    total_loss = loss_1 + loss_2
    print(f"Total loss: {total_loss:.2e}")

    if total_loss < 1e-3:
        print("✓ Multi-step solution EXISTS — ALS is worth pursuing")
        return True
    else:
        print("✗ Multi-step solution DOESN'T EXIST (or poorly conditioned)")
        return False


def test_belief_recovery_with_true_operators(name, T, pi, model_ckpt, device="cpu"):
    """TEST 3: Can we recover belief geometry using true operators in the observability matrix?

    This shows the theoretical ceiling: if operator estimation were perfect, could we
    recover the belief subspace and geometry?
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: {name} — Belief recovery with true operators")
    print(f"{'='*60}")

    vocab = T.shape[0]

    # Load model and collect activations
    import torch
    model = F14._load(model_ckpt, device)
    resid, soft, belief_labels, _ = F14._collect(model, T, pi, device)

    # Compute reachability and get regression data
    reach = {}
    for w in resid:
        ok = True
        pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:
                ok = False
                break
            pre = pre + (t,)
        reach[w] = ok

    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    rows = [w for w in rows if np.isfinite(belief_labels[w]).all()]

    A = np.stack([resid[w] for w in rows])
    B = np.stack([belief_labels[w] for w in rows])

    # Get encoding Ψ
    B_centered = B - B.mean(axis=0, keepdims=True)
    A_centered = A - A.mean(axis=0, keepdims=True)
    psi = LinearRegression(fit_intercept=False).fit(B_centered, A_centered).coef_.T

    # Build observability matrix using TRUE operators
    print(f"\nBuilding observability matrix with true T^x operators...")
    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A_centered, B_centered).coef_.T  # (D, NS)

    cols = [C]
    frontier = [C]
    for depth_iter in range(4):  # depth 5 total
        nxt = []
        for x in range(vocab):
            A_x = psi.T @ T[x] @ psi  # (256, 256)
            for f in frontier:
                nxt.append(A_x @ f)
        cols.extend(nxt)
        frontier = nxt

    O = np.hstack(cols)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)

    correct_d = T.shape[1]
    print(f"Singular values (first 10): {np.round(sv[:10] / sv[0], 3)}")

    # Find elbow
    drops = -np.diff(sv[:10] / sv[0])
    elbow_d = np.argmax(drops) + 1
    print(f"Largest drop: {drops.max():.3f} at d={elbow_d}, expected d={correct_d}")

    # Project activations onto true observability subspace at the correct dimension d
    S = A_centered @ U[:, :correct_d]  # (N, correct_d)

    # Measure how well beliefs can be decoded from the projected activations
    decode_r2 = LinearRegression().fit(S, B_centered).score(S, B_centered)

    print(f"\nBelief recovery with true operators at d={correct_d}:")
    print(f"  R² (beliefs from projection): {decode_r2:.4f}")

    return decode_r2


def main():
    # Test RRXOR
    T_rrxor = np.array(rrxor(0.5, 0.5))
    pi_rrxor = np.array([2, 1, 1, 1, 1]) / 6.0

    test1_rrxor = test_belief_space_true_operators("RRXOR", T_rrxor, pi_rrxor)
    test2_rrxor = test_activation_space_true_operators("RRXOR", T_rrxor, pi_rrxor,
                                                       "rrxor_transformer.pt")
    test3_rrxor = test_belief_recovery_with_true_operators("RRXOR", T_rrxor, pi_rrxor,
                                                            "rrxor_transformer.pt")

    # Test Mess3
    T_mess3 = np.array(mess3(x=0.05, a=0.85))
    pi_mess3 = np.array([1, 1, 1]) / 3.0

    test1_mess3 = test_belief_space_true_operators("Mess3", T_mess3, pi_mess3)
    test2_mess3 = test_activation_space_true_operators("Mess3", T_mess3, pi_mess3,
                                                       "mess3_transformer.pt")
    test3_mess3 = test_belief_recovery_with_true_operators("Mess3", T_mess3, pi_mess3,
                                                            "mess3_transformer.pt")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"RRXOR: Test1={test1_rrxor}, Test2={test2_rrxor}, Test3_R²={test3_rrxor:.4f}")
    print(f"Mess3: Test1={test1_mess3}, Test2={test2_mess3}, Test3_R²={test3_mess3:.4f}")
    print(f"\nInterpretation:")
    print(f"Test3 shows the THEORETICAL CEILING for belief recovery if operators were perfect.")
    print(f"- RRXOR Test3_R²: if this is low (<0.9), weak observability is fundamental")
    print(f"- Mess3 Test3_R²: should be near 1.0 since the encoding is strong")


if __name__ == "__main__":
    main()
