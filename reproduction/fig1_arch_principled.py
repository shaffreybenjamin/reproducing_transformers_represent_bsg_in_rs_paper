"""Arch GT | supervised | unsupervised comparison (4-state -> 3D tetrahedron).

The arch analogue of fig1_mess3_principled.py (Figure 1 in writeup_v1.tex): a 3-panel
belief-geometry comparison.
  * Panel 1: analytic ground-truth belief simplex.
  * Panel 2: supervised decode of belief from the FULL single-layer residual stream (64-D).
  * Panel 3: unsupervised decode from the observability-OOM subspace (D-D, D = #hidden states).

arch has 4 hidden states (vocab 3), so belief is 4-D and lives on a 3-simplex (a tetrahedron).
We therefore visualise in true 3D (one 4-state-belief -> one point in the tetrahedron) rather
than the 2-D triangle used for the 3-state processes, and rather than the PCA-to-2D fallback
used for the 5-state rrxor. Same enumerated MSP prefixes, same belief-decode metric, same
observability-OOM subspace machinery (just D=4); only the projection/rendering is 3D.

Uses the arch checkpoint (the overnight run was interrupted by a Windows Update reboot at
~195k steps; the model is already at its optimal windowed loss).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import unsupervised_belief_oom as U
import fig14_observable_oom as F14

ARCH_PARAMS = {"a": 0.85}
FIG_DIR = Path(__file__).parent / "figures"
MODEL_DIR = Path(__file__).parent / "models"
D = 4   # = number of hidden states (belief is 4-D -> tetrahedron)

# Regular 3-simplex (tetrahedron): equilateral-triangle base (states 0,1,2) + apex (state 3).
TETRA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3) / 2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
])
# One distinct colour per hidden state; a belief is shown as its belief-weighted blend.
CORNER_COLORS = np.array([
    [1.0, 0.0, 0.0],   # state 0 red
    [0.0, 0.8, 0.0],   # state 1 green
    [0.0, 0.0, 1.0],   # state 2 blue
    [1.0, 0.6, 0.0],   # state 3 orange
])
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def simplex_to_xyz(beliefs):
    """Map 4-state beliefs (rows sum to 1) to 3D tetrahedron coordinates."""
    return np.asarray(beliefs) @ TETRA


def belief_colors(beliefs):
    return np.clip(np.asarray(beliefs) @ CORNER_COLORS, 0, 1)


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "arch_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device  # checkpoint trained on cuda; honor the local device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def _panel(fig, idx, xyz, color, title):
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


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("arch", ARCH_PARAMS)
    vocab = hmm.vocab_size
    model, ctx, step = load_model(device)
    print(f"loaded arch checkpoint (step={step}, ctx={ctx}, device={device})")

    resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, device)
    # keep only reachable prefixes (impossible token prefixes get prob 0 / NaN belief in the MSP tree)
    good = {
        w for w in resid
        if w in belief and np.all(np.isfinite(belief[w])) and prefix_prob.get(w, 0.0) > 1e-12
    }
    print(f"enumerated MSP prefixes: {len(resid)}  reachable (prob>0): {len(good)}  (lengths 1..{max_len})")
    reach = {w: (w in good) for w in resid}

    # unsupervised observability-OOM subspace (single layer), P(w)-weighted
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab, wmap=prefix_prob)
    B = Uobs[:, :D]

    seqs = [s for s in resid if s in good]
    true_b = np.array([belief[s] for s in seqs])
    true_xyz = simplex_to_xyz(true_b)
    color = belief_colors(true_b)
    wcol = np.array([prefix_prob[s] for s in seqs])
    Xfull = np.stack([resid[s] for s in seqs])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    def decode_panel(feat):
        r2 = U.belief_decode_r2(feat, true_b, wcol)
        dec = LinearRegression().fit(feat, true_b)
        xyz = simplex_to_xyz(to_simplex(dec.predict(feat)))
        return xyz, r2

    xyz_sup, r2_sup = decode_panel(Xfull)
    xyz_uns, r2_uns = decode_panel(Xfull @ B)
    print(f"arch belief-decode R^2:  supervised(64-D)={r2_sup:.3f}   unsupervised({D}-D)={r2_uns:.3f}")

    fig = plt.figure(figsize=(18, 6))
    _panel(fig, 1, true_xyz, color, "Arch ground truth")
    _panel(fig, 2, xyz_sup, color, f"Supervised: full residual stream (64-D)\nbelief-decode R$^2$={r2_sup:.3f}")
    _panel(fig, 3, xyz_uns, color, f"Unsupervised: spectral-OOM subspace ({D}-D)\nbelief-decode R$^2$={r2_uns:.3f}")
    out = FIG_DIR / "fig1_arch_principled.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
