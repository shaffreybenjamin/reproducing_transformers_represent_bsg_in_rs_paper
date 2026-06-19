"""Diagnostic: test whether increasing observability horizon L improves RRXOR recovery.

Hypothesis: RRXOR's weak observability at L=3 can be overcome by longer horizons.
The 36 discrete belief states are only separable in the predicted futures at deeper
horizons. This script builds the observability matrix at L=3,4,5,6,7,8 and tests:
  1. Singular spectrum: do high-singular-value directions appear at deeper horizons?
  2. Decode R^2 as function of retained rank, for each L
  3. Unweighted vs P(w)-weighted variants

Expected outcome: if horizon is the bottleneck, singular values should lift and
decode should improve as L increases to some optimal point.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
import itertools

from simplexity.generative_processes.transition_matrices import rrxor
from fig14_observable_oom import _load, _collect, analytic_prefix_probs

MAX_LEN = 10
EPS = 1e-3
RIDGE = 1e-2
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def observable_subspace_at_depth(resid, soft, reach, vocab, depth=3, wmap=None):
    """Build observability matrix O = [C, G_x C, G_x G_y C, ...] at specified depth.
    Returns: rows, A, P, Gs, U (left singular vectors), sv (singular values)
    """
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    sw = None if wmap is None else np.array([max(float(wmap.get(w, 0.0)), 1e-12) for w in rows])

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


def test_horizons(resid, soft, reach, belief, vocab, ckpt_name, horizons=(3, 4, 5, 6, 7, 8), wmap=None):
    """Test state-level R^2 (COM-geom) and spectrum for multiple horizons with P(w)-weighting."""
    from collections import defaultdict
    from fig13_rrxor_unsupervised_oom import msp_index

    results = {}
    for L in horizons:
        print(f"\n  Testing horizon L={L}...")
        rows, A, P, Gs, U, sv = observable_subspace_at_depth(resid, soft, reach, vocab, depth=L, wmap=wmap)

        # Get ground-truth beliefs for rows in this subspace
        Yb_rows = np.stack([belief[w] for w in rows])

        # Map beliefs to states for COM-geom scoring
        _, index = msp_index()
        idx = np.array([index.get(tuple(np.round(belief[w], 5)), -1) for w in rows])
        groups = defaultdict(list)
        for i, s in enumerate(idx):
            if s >= 0:
                groups[s].append(i)

        def com_geom(B):
            """State-level R^2: COM (centre of mass) of each belief state with per-state-equal weighting.
            Each state counts equally (weight=1), regardless of how many points it contains."""
            s = A @ B
            com = np.array([s[g].mean(0) for g in groups.values() if len(g) > 0])
            bel = np.array([Yb_rows[g[0]] for g in groups.values() if len(g) > 0])
            if len(com) > 0:
                # Per-state-equal weighting: each state contributes equally
                sw = np.ones(len(com))
                return LinearRegression().fit(com, bel, sample_weight=sw).score(com, bel, sample_weight=sw)
            return 0.0

        # Test COM-geom R^2 for various dimensions
        decs = []
        for d in [2, 3, 4, 5, 6, 8, 10]:
            if d > len(sv):
                continue
            cg = com_geom(U[:, :d])
            decs.append((d, cg))

        # Also get ceiling (per-position on full concat)
        fin_rows = np.isfinite(Yb_rows).all(1)
        ceiling = LinearRegression().fit(A[fin_rows], Yb_rows[fin_rows]).score(A[fin_rows], Yb_rows[fin_rows])

        results[L] = {
            "sv": sv,
            "decs": decs,
            "ceiling": ceiling,
            "rows": rows,
            "A": A,
            "U": U,
            "Yb": Yb_rows,
            "groups": groups
        }

        print(f"    Spectrum (top 15): {np.round(sv[:15] / sv[0], 4)}")
        print(f"    Decode R^2 vs d: {[(d, round(r, 3)) for d, r in decs]}")
        print(f"    Supervised ceiling: {ceiling:.3f}")

    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and collect activations
    m = _load("rrxor_transformer.pt", device)
    resid, soft, belief, pp = _collect(m, np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0, device)

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

    vocab = np.array(rrxor(0.5, 0.5)).shape[0]  # RRXOR vocabulary size (2)
    print(f"Vocab size: {vocab}")
    print(f"Reachable nodes: {sum(reach.values())}/{len(reach)}")

    # Get P(w) weights (consistent with fig26)
    from fig14_observable_oom import analytic_prefix_probs
    T = np.array(rrxor(0.5, 0.5))
    pi = np.array([2, 1, 1, 1, 1]) / 6.0
    wmap = analytic_prefix_probs(resid, T, pi)

    # Test multiple horizons with P(w)-weighting (extended to include L=10,12,15)
    print("\n=== TESTING HORIZONS L=3,4,5,6,7,8,10,12,15 (P(w)-weighted, COM-geom R²) ===")
    results = test_horizons(resid, soft, reach, belief, vocab, "rrxor_transformer.pt",
                           horizons=(3, 4, 5, 6, 7, 8, 10, 12, 15), wmap=wmap)

    # Plot 1: Singular spectrum for each horizon
    horizons_to_plot = sorted(results.keys())
    n_horizons = len(horizons_to_plot)
    n_cols = 3
    n_rows = (n_horizons + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_horizons > 1 else [axes]

    for idx, L in enumerate(horizons_to_plot):
        ax = axes[idx]
        if L not in results:
            ax.text(0.5, 0.5, f"L={L} not computed", ha="center", va="center")
            continue
        sv = results[L]["sv"]
        ax.semilogy(range(len(sv[:30])), sv[:30], "ko-", markersize=4)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value (log scale)")
        ax.set_title(f"Observability spectrum at L={L}")
        ax.grid(alpha=0.3)
        ax.set_ylim(sv[-1] / 10, sv[0] * 2)

    # Hide unused subplots
    for idx in range(len(horizons_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_horizon_spectra.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {FIG_DIR / 'diag_horizon_spectra.png'}")

    # Plot 2: Decode R^2 vs dimension for each horizon
    horizons_to_plot = sorted(results.keys())
    n_horizons = len(horizons_to_plot)
    n_cols = 3
    n_rows = (n_horizons + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_horizons > 1 else [axes]

    for idx, L in enumerate(horizons_to_plot):
        ax = axes[idx]
        if L not in results:
            ax.text(0.5, 0.5, f"L={L} not computed", ha="center", va="center")
            continue
        decs = results[L]["decs"]
        ds = [d for d, _ in decs]
        rs = [r for _, r in decs]
        ax.plot(ds, rs, "o-", color="black", linewidth=2, markersize=6)
        ax.axhline(results[L]["ceiling"], color="red", ls="--", label=f"ceiling={results[L]['ceiling']:.3f}")
        ax.set_xlabel("Subspace dim d")
        ax.set_ylabel("Belief-decode R$^2$")
        ax.set_title(f"L={L}: decode R$^2$ vs d")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(horizons_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_horizon_decode_vs_d.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {FIG_DIR / 'diag_horizon_decode_vs_d.png'}")

    # Plot 3: Best decode R^2 (at optimal d) vs horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    ls = sorted(results.keys())
    best_decs = []
    for L in ls:
        decs = results[L]["decs"]
        best = max([r for _, r in decs])
        best_decs.append(best)

    ax.plot(ls, best_decs, "o-", color="black", linewidth=2.5, markersize=8, label="Best R$^2$ at each L")
    ax.fill_between(ls, best_decs, alpha=0.2)
    ax.set_xlabel("Horizon L")
    ax.set_ylabel("Best belief-decode R$^2$")
    ax.set_title("RRXOR: Does longer horizon improve recovery?")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diag_horizon_best_r2.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {FIG_DIR / 'diag_horizon_best_r2.png'}")

    # Summary table
    print("\n=== SUMMARY ===")
    print("L\tCeiling\tBest R^2\tBest d")
    for L in sorted(results.keys()):
        decs = results[L]["decs"]
        best_d, best_r2 = max(decs, key=lambda x: x[1])
        ceiling = results[L]["ceiling"]
        print(f"{L}\t{ceiling:.3f}\t{best_r2:.3f}\t{best_d}")


if __name__ == "__main__":
    main()
