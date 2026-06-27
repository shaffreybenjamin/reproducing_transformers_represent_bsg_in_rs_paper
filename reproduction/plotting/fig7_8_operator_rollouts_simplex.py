"""Operator rollout plots for 3-4 state processes (zero_one_random, fern, strata, wing, arch).

Generates Figure 7-8 style operator rollout visualizations for each process:
  * Panel 1: Ground-truth belief simplex (analytic)
  * Panel 2: Unsupervised decode (spectral-OOM raw activations)
  * Panel 3: Operator rollout (learned dynamics from root)

Uses the observable-OOM subspace (D-D, D = #hidden states) and fits operators via
rescaled-children trick (ALS-refined), then rolls out from the root belief.
For 3-state processes: 2D triangle simplex visualization.
For 4-state arch: 3D tetrahedron visualization.

Outputs:
  figures/fig7_8_<process>_operator_rollout.png   GT | raw | rollout
"""
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import unsupervised_belief_oom as U
import fig14_observable_oom as F14

FIG_DIR = Path(__file__).parent.parent / "figures"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Regular 3-simplex (tetrahedron): equilateral-triangle base (states 0,1,2) + apex (state 3).
TETRA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3) / 2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
])
CORNER_COLORS_TETRA = np.array([
    [1.0, 0.0, 0.0],   # state 0 red
    [0.0, 0.8, 0.0],   # state 1 green
    [0.0, 0.0, 1.0],   # state 2 blue
    [1.0, 0.6, 0.0],   # state 3 orange
])
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

PROCESSES = {
    "zero_one_random": {"params": {"p": 0.5}, "n_states": 3, "vocab": 2},
    "fern": {"params": {"x": 0.5}, "n_states": 3, "vocab": 2},
    "strata": {"params": {"a": 0.85, "t0": 0.5, "t1": 0.5}, "n_states": 3, "vocab": 2},
    "wing": {"params": {"x": 0.5, "y": 0.5}, "n_states": 3, "vocab": 2},
    "arch": {"params": {"a": 0.85}, "n_states": 4, "vocab": 3},
}


def simplex_to_xyz(beliefs):
    """Map 4-state beliefs (rows sum to 1) to 3D tetrahedron coordinates."""
    return np.asarray(beliefs) @ TETRA


def belief_colors_tetra(beliefs):
    return np.clip(np.asarray(beliefs) @ CORNER_COLORS_TETRA, 0, 1)


def load_model(process_name, device):
    """Load transformer checkpoint for a process."""
    ckpt = torch.load(
        MODEL_DIR / f"{process_name}_transformer.pt",
        map_location=device,
        weights_only=False,
    )
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def transitions_for_operators(resid, soft, reach, vocab, max_len):
    """Collect transition data for operator fitting."""
    rows = [
        w
        for w in resid
        if reach[w] and len(w) < max_len and all((w + (x,)) in resid for x in range(vocab))
    ]
    X = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    Yc = np.stack([[resid[w + (x,)] for x in range(vocab)] for w in rows])
    return rows, X, P, Yc


def _panel_3d(fig, idx, xyz, color, title):
    """Create a 3D tetrahedron panel for 4-state visualization."""
    ax = fig.add_subplot(1, 3, idx, projection="3d")
    for a, b in TETRA_EDGES:
        ax.plot(*zip(TETRA[a], TETRA[b]), color="lightgray", lw=0.6, zorder=0)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=3, alpha=0.45,
               depthshade=False, linewidths=0)
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    return ax


def operator_rollout_figure(process_name):
    """Generate operator rollout figure for a process."""
    config = PROCESSES[process_name]
    n_states = config["n_states"]
    vocab = config["vocab"]
    params = config["params"]
    is_4d = n_states == 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and HMM
    hmm = build_hidden_markov_model(process_name, params)
    model, ctx, step = load_model(process_name, device)
    print(f"\nloaded {process_name} checkpoint (step={step}, ctx={ctx}, device={device})")

    # Collect prefix features
    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(
        model, hmm, ctx, device
    )

    # Filter reachable prefixes
    good = {
        w
        for w in resid
        if w in belief
        and np.all(np.isfinite(belief[w]))
        and prefix_prob.get(w, 0.0) > 1e-12
    }
    print(f"enumerated MSP prefixes: {len(resid)}  reachable (prob>0): {len(good)}")
    reach = {w: (w in good) for w in resid}

    # Extract observable subspace (unsupervised)
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)
    B_unsup = Uobs[:, :n_states]

    # Collect ground-truth beliefs for all reachable prefixes
    seqs = [s for s in resid if s in good]
    true_b = np.array([belief[s] for s in seqs])

    if is_4d:
        true_coords = simplex_to_xyz(true_b)
        color = belief_colors_tetra(true_b)
    else:
        true_coords = U.simplex_to_xy(true_b)
        color = np.clip(true_b, 0, 1)

    # Prepare data for operator fitting
    Xfull = np.stack([resid[s] for s in seqs])
    wcol = np.array([prefix_prob[s] for s in seqs])

    # Get transition data for all reachable prefixes (weighted)
    rows, X, P, Yc = transitions_for_operators(resid, soft, reach, vocab, max_len)
    Wt = np.array([prefix_prob[w] for w in rows])

    def to_simplex(p):
        p = np.clip(p, 0, None)
        return p / p.sum(1, keepdims=True)

    # --- RAW DECODE (unsupervised OOM subspace, no rollout) ---
    A_raw = Xfull @ B_unsup
    raw_r2 = U.belief_decode_r2(A_raw, true_b, wcol)
    decode_raw = LinearRegression().fit(A_raw, true_b)
    pred_raw = to_simplex(decode_raw.predict(A_raw))

    if is_4d:
        raw_coords = simplex_to_xyz(pred_raw)
    else:
        raw_coords = U.simplex_to_xy(pred_raw)

    # --- OPERATOR ROLLOUT (fit operators in unsupervised subspace, rollout from root) ---
    # ALS refinement for operator consistency
    B_als = U.als_refine_basis(X, P, Yc, Wt, B_unsup, n_states)

    # Fit operators
    ops, _ = U.fit_operators(X @ B_als, P, Yc @ B_als, Wt)

    # Recover evaluation functional
    e, _ = U.recover_eval_functional(X @ B_als)

    # Project all activations to basis for rollout
    proj = {w: resid[w] @ B_als for w in resid if reach[w]}

    # Rollout from root
    roll = U.rollout_states(proj, ops, e, max_len, vocab)

    # Collect rollout results for reachable prefixes with valid rollouts
    rw = [w for w in seqs if w in roll and np.isfinite(roll[w]).all()]
    roll_states = np.stack([roll[w] for w in rw])
    roll_b = np.stack([belief[w] for w in rw])
    rwcol = np.array([prefix_prob[w] for w in rw])

    roll_r2 = U.belief_decode_r2(roll_states, roll_b, rwcol)
    decode_roll = LinearRegression().fit(roll_states, roll_b)
    pred_roll = to_simplex(decode_roll.predict(roll_states))

    if is_4d:
        roll_coords = simplex_to_xyz(pred_roll)
        rcol = belief_colors_tetra(roll_b)
    else:
        roll_coords = U.simplex_to_xy(pred_roll)
        rcol = np.clip(roll_b, 0, 1)

    print(
        f"[{process_name}] OOM-ALS raw R^2={raw_r2:.3f}   operator-rollout R^2={roll_r2:.3f}"
    )

    # --- RENDER 3-PANEL FIGURE ---
    if is_4d:
        fig = plt.figure(figsize=(18, 6))
        _panel_3d(fig, 1, true_coords, color, f"{process_name.replace('_', ' ').title()} ground truth")
        _panel_3d(fig, 2, raw_coords, color, f"Spectral-OOM - raw activations\n(belief-decode R$^2$={raw_r2:.3f})")
        _panel_3d(fig, 3, roll_coords, rcol, f"Spectral-OOM - operator rollout\n(belief-decode R$^2$={roll_r2:.3f})")
    else:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(U.rasterize_simplex(true_coords, color, px=2), origin="lower")
        ax[0].set_title(f"{process_name.replace('_', ' ').title()} ground truth")
        ax[1].imshow(U.rasterize_simplex(raw_coords, color, px=2), origin="lower")
        ax[1].set_title(f"Spectral-OOM - raw activations\n(belief-decode R$^2$={raw_r2:.3f})")
        ax[2].imshow(U.rasterize_simplex(roll_coords, rcol, px=2), origin="lower")
        ax[2].set_title(f"Spectral-OOM - operator rollout\n(belief-decode R$^2$={roll_r2:.3f})")
        for a in ax:
            a.axis("off")

    out = FIG_DIR / f"fig7_8_{process_name}_operator_rollout.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for process in PROCESSES.keys():
        operator_rollout_figure(process)
