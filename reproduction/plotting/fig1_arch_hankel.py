"""Arch process: Hankel matrix two-factor estimator with elbow d-selection.

Tests whether the two-sided Hankel factorization H = P·S improves the spectrum
and elbow-based d-selection for the arch 4-state process.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import spectral_oom_observability_only as hankel

ARCH_PARAMS = {"a": 0.85}
FIG_DIR = Path(__file__).parent.parent / "figures"
MODEL_DIR = Path(__file__).parent.parent / "models"
TRUE_D = 4  # 4 hidden states

# Regular 3-simplex (tetrahedron): equilateral-triangle base (states 0,1,2) + apex (state 3).
TETRA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3) / 2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
])

CORNER_COLORS = np.array([
    [1.0, 0.0, 0.0],   # state 0 red
    [0.0, 0.8, 0.0],   # state 1 green
    [0.0, 0.0, 1.0],   # state 2 blue
    [1.0, 0.6, 0.0],   # state 3 orange
])


def simplex_to_xyz(beliefs):
    """Map 4-state beliefs to 3D tetrahedron coordinates."""
    return np.asarray(beliefs) @ TETRA


def belief_colors(beliefs):
    return np.clip(np.asarray(beliefs) @ CORNER_COLORS, 0, 1)


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "arch_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("arch", ARCH_PARAMS)
    model, ctx = load_model(device)
    print(f"=== Arch (TRUE_D={TRUE_D}) ===")
    print(f"loaded model (ctx={ctx}, device={device})")

    # Run Hankel spectral-OOM
    result = hankel.run_hankel_spectral_oom(model, hmm, ctx, device, verbose=True)

    d_detected = result["d_elbow"]
    U = result["U"]
    sv = result["sv"]
    belief = result["belief"]
    resid = result["resid"]

    print(f"\n>>> Elbow-detected d: {d_detected}   True d: {TRUE_D}   Match: {d_detected == TRUE_D}")

    # Collect valid sequences with beliefs
    seqs = [s for s in resid if s in belief and np.all(np.isfinite(belief[s]))]
    true_b = np.array([belief[s] for s in seqs])
    true_xyz = simplex_to_xyz(true_b)
    color = belief_colors(true_b)

    # Supervised: Use FULL activations (oracle - what's theoretically possible)
    activations = np.stack([resid[s] for s in seqs])
    reg_true = LinearRegression().fit(activations, true_b)
    supervised_b = np.clip(reg_true.predict(activations), 0, None)
    supervised_b = supervised_b / supervised_b.sum(axis=1, keepdims=True)
    supervised_xyz = simplex_to_xyz(supervised_b)
    r2_true = reg_true.score(activations, true_b)

    # Unsupervised: Use Hankel basis with detected d dimension
    basis_detected = U[:, :max(1, d_detected)]
    proj_detected = activations @ basis_detected
    reg_detected = LinearRegression().fit(proj_detected, true_b)
    unsupervised_b = np.clip(reg_detected.predict(proj_detected), 0, None)
    unsupervised_b = unsupervised_b / unsupervised_b.sum(axis=1, keepdims=True)
    unsupervised_xyz = simplex_to_xyz(unsupervised_b)
    r2_detected = reg_detected.score(proj_detected, true_b)

    print(f"Belief-decode R^2 (supervised, full activations): {r2_true:.4f}")
    print(f"Belief-decode R^2 (unsupervised, d={d_detected}): {r2_detected:.4f}")

    # === GEOMETRY PLOT: 3 panels (3D) ===
    fig = plt.figure(figsize=(18, 5))

    # Panel A: Ground truth
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], c=color, s=8, alpha=0.8, depthshade=False, linewidths=0)
    ax.set_title("A. Ground Truth", fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-60)

    # Panel B: Supervised (true d) - no ground truth overlay
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(supervised_xyz[:, 0], supervised_xyz[:, 1], supervised_xyz[:, 2], c=color, s=8, alpha=0.8, depthshade=False, linewidths=0, marker="o")
    ax.set_title(f"B. Supervised (d={TRUE_D}, R²={r2_true:.3f})", fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-60)

    # Panel C: Unsupervised (detected d) - no ground truth overlay
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(unsupervised_xyz[:, 0], unsupervised_xyz[:, 1], unsupervised_xyz[:, 2], c=color, s=8, alpha=0.8, depthshade=False, linewidths=0, marker="s")
    ax.set_title(f"C. Unsupervised (d={d_detected}, R²={r2_detected:.3f})", fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-60)

    out_geom = FIG_DIR / "fig1_arch_hankel_geometry.png"
    fig.savefig(out_geom, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved geometry figure -> {out_geom}")

    # === SPECTRUM PLOT: Separate figure ===
    fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
    sv_norm = sv / sv[0]
    ax_spec.semilogy(range(len(sv_norm[:20])), sv_norm[:20], "o-", color="black", linewidth=2, markersize=6)
    ax_spec.axvline(TRUE_D - 0.5, color="red", linestyle="--", linewidth=2.5, label=f"True d={TRUE_D}")
    ax_spec.axvline(d_detected - 0.5, color="green", linestyle="--", linewidth=2.5, label=f"Detected d={d_detected}")
    ax_spec.set_xlabel("Singular value index", fontsize=12)
    ax_spec.set_ylabel("Normalized singular value (log scale)", fontsize=12)
    ax_spec.set_title("Hankel Matrix Singular Value Spectrum", fontsize=13, fontweight="bold")
    ax_spec.legend(fontsize=11)
    ax_spec.grid(alpha=0.3)

    out_spec = FIG_DIR / "fig1_arch_hankel_spectrum.png"
    fig_spec.savefig(out_spec, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved spectrum figure -> {out_spec}")


if __name__ == "__main__":
    main()
