"""Diagnostic: test whether P(w)-weighting is responsible for poor RRXOR recovery.

Hypothesis: P(w)-weighting concentrates the fit on high-probability (shallow) prefixes,
exactly the prefixes that haven't synchronized yet. For RRXOR (a synchronizing process),
the discriminating information is in the deeper, lower-probability prefixes that reach
the vertices. So P(w)-weighting may be actively down-weighting the transitions that
carry belief-separating signal. Try unweighted and synchronization-aware weighting.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.transition_matrices import rrxor
from fig14_observable_oom import _load, _collect

MAX_LEN = 10
EPS = 1e-3
RIDGE = 1e-2
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def observable_subspace_at_depth(resid, soft, reach, vocab, depth=3, wmap=None):
    """Build observability matrix O = [C, G_x C, G_x G_y C, ...] at specified depth.
    wmap can be None (unweighted), a dict of P(w) values, or "uniform" for explicit unweighted."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])

    # Handle weighting
    if wmap is None or wmap == "uniform":
        sw = None  # Uniform weighting
    else:
        sw = np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P, sample_weight=sw).coef_.T
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        swm = None if sw is None else sw[m]
        Gs.append(Ridge(alpha=RIDGE, fit_intercept=False).fit(A[m], tgt, sample_weight=swm).coef_.T)

    cols, frontier = [C], [C]
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols += nxt
        frontier = nxt
    O = np.hstack(cols)
    U, sv, _ = np.linalg.svd(O, full_matrices=False)
    return rows, A, P, Gs, U, sv


def fit_at_dim(rows, A, P, resid, vocab, B):
    """Project to subspace B (D x d), fit operators A_x and eval functional e."""
    s_all = A @ B
    ops = {}
    for x in range(vocab):
        m = np.array([P[i, x] > EPS for i in range(len(rows))])
        sw = s_all[m]
        sc = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]]) @ B
        ops[x] = np.linalg.lstsq(sw, P[m, x][:, None] * sc, rcond=None)[0]
    e = np.linalg.lstsq(s_all, np.ones(len(s_all)), rcond=None)[0]
    return s_all, ops, e


def test_weighting_variants(resid, soft, reach, belief, vocab, depth=3):
    """Test decode R^2 for unweighted vs P(w)-weighted observability matrix at depth L."""
    rows_pw, A_pw, P_pw, Gs_pw, U_pw, sv_pw = observable_subspace_at_depth(
        resid, soft, reach, vocab, depth=depth, wmap=None)

    # For P(w)-weighted, we need to recompute with actual weights
    # (current code doesn't track them; use analytic weights from DGP)
    from fig14_observable_oom import analytic_prefix_probs
    T = np.array(rrxor(0.5, 0.5))
    pi = np.array([2, 1, 1, 1, 1]) / 6.0
    wmap = analytic_prefix_probs(resid, T, pi)

    rows_uw = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                      for x in range(vocab) if soft[w][x] > EPS)]
    A_uw = np.stack([resid[w] for w in rows_uw])
    P_uw = np.stack([soft[w] for w in rows_uw])

    # Recompute with P(w) weights
    sw = np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows_uw])
    C_pw = Ridge(alpha=RIDGE, fit_intercept=False).fit(A_uw, P_uw, sample_weight=sw).coef_.T
    Gs_pw = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows_uw])
        child = np.stack([resid[rows_uw[i] + (x,)] for i in np.where(m)[0]])
        tgt = P_uw[m, x][:, None] * child
        swm = sw[m]
        Gs_pw.append(Ridge(alpha=RIDGE, fit_intercept=False).fit(A_uw[m], tgt, sample_weight=swm).coef_.T)

    # Build observability matrix for weighted
    cols_pw, frontier = [C_pw], [C_pw]
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs_pw for f in frontier]
        cols_pw += nxt
        frontier = nxt
    O_pw = np.hstack(cols_pw)
    U_pw, sv_pw, _ = np.linalg.svd(O_pw, full_matrices=False)

    # Get ground-truth beliefs
    Yb_uw = np.stack([belief[w] for w in rows_uw])
    fin_uw = np.isfinite(Yb_uw).all(1)

    # Test decode R^2 for various dimensions
    results = {"unweighted": {}, "weighted": {}}

    # Compute unweighted observability matrix once
    C_uw = Ridge(alpha=RIDGE, fit_intercept=False).fit(A_uw, P_uw).coef_.T
    Gs_uw = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows_uw])
        child = np.stack([resid[rows_uw[i] + (x,)] for i in np.where(m)[0]])
        tgt = P_uw[m, x][:, None] * child
        Gs_uw.append(Ridge(alpha=RIDGE, fit_intercept=False).fit(A_uw[m], tgt).coef_.T)

    cols_uw, frontier = [C_uw], [C_uw]
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs_uw for f in frontier]
        cols_uw += nxt
        frontier = nxt
    O_uw = np.hstack(cols_uw)
    U_uw, sv_uw, _ = np.linalg.svd(O_uw, full_matrices=False)

    # Now test decode for various dimensions
    for d in [2, 3, 4, 5, 6, 8, 10]:
        if d > len(sv_pw):
            continue

        s_uw = A_uw @ U_uw[:, :d]
        dec_uw = LinearRegression().fit(s_uw[fin_uw], Yb_uw[fin_uw]).score(s_uw[fin_uw], Yb_uw[fin_uw])

        s_pw = A_uw @ U_pw[:, :d]
        dec_pw = LinearRegression().fit(s_pw[fin_uw], Yb_uw[fin_uw]).score(s_pw[fin_uw], Yb_uw[fin_uw])

        results["unweighted"][d] = dec_uw
        results["weighted"][d] = dec_pw

    return results, sv_pw, sv_uw, A_uw, U_pw, U_uw, Yb_uw, fin_uw


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and collect activations
    m = _load("rrxor_transformer.pt", device)
    T = np.array(rrxor(0.5, 0.5))
    pi = np.array([2, 1, 1, 1, 1]) / 6.0
    resid, soft, belief, pp = _collect(m, T, pi, device)

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

    vocab = T.shape[0]
    print(f"Vocab size: {vocab}")
    print(f"Reachable nodes: {sum(reach.values())}/{len(reach)}")

    # Test weighting at depth 3 (original)
    print("\n=== TESTING WEIGHTING: P(w) vs UNWEIGHTED at L=3 ===")
    results_d3, sv_pw_d3, sv_uw_d3, A, U_pw, U_uw, Yb, fin = test_weighting_variants(
        resid, soft, reach, belief, vocab, depth=3)

    print("\nDecode R^2 vs dimension (L=3):")
    print("d\tUnweighted\tP(w)-weighted\tDiff")
    for d in sorted(results_d3["unweighted"].keys()):
        uw = results_d3["unweighted"][d]
        pw = results_d3["weighted"][d]
        diff = uw - pw
        print(f"{d}\t{uw:.3f}\t\t{pw:.3f}\t\t{diff:+.3f}")

    # Plot comparison
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Decode vs dimension
    ds = sorted(results_d3["unweighted"].keys())
    uw_vals = [results_d3["unweighted"][d] for d in ds]
    pw_vals = [results_d3["weighted"][d] for d in ds]

    ax0.plot(ds, uw_vals, "o-", color="blue", linewidth=2.5, label="Unweighted", markersize=8)
    ax0.plot(ds, pw_vals, "s-", color="red", linewidth=2.5, label="P(w)-weighted", markersize=8)
    ax0.set_xlabel("Subspace dim d")
    ax0.set_ylabel("Belief-decode R$^2$")
    ax0.set_title("RRXOR L=3: Does P(w)-weighting hurt?")
    ax0.set_ylim(0, 0.8)
    ax0.legend(fontsize=10)
    ax0.grid(alpha=0.3)

    # Singular spectra
    ax1.semilogy(range(min(20, len(sv_pw_d3))), sv_pw_d3[:20], "s-", color="red", label="P(w)-weighted", linewidth=2, markersize=6)
    ax1.semilogy(range(min(20, len(sv_uw_d3))), sv_uw_d3[:20], "o-", color="blue", label="Unweighted", linewidth=2, markersize=6)
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value (log)")
    ax1.set_title("Observability spectrum comparison")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_weighting_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {FIG_DIR / 'diag_weighting_comparison.png'}")


if __name__ == "__main__":
    main()
