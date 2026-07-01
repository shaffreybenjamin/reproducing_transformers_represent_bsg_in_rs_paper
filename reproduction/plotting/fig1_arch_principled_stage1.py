"""Arch GT | supervised | Stage 1 CCA/RRR (minimal wrapper, 3D tetrahedron).

Identical to fig1_arch_principled.py but uses Stage 1 CCA/RRR subspace.
To revert: delete this file.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U
import stage1_cca_general as S1

ARCH_PARAMS = {"a": 0.85}
FIG_DIR = Path(__file__).parent / "figures"
MODEL_DIR = Path(__file__).parent / "models"
D = 4

# Regular 3-simplex (tetrahedron)
TETRA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3) / 2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
])
CORNER_COLORS_TETRA = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.6, 0.0],
])
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def simplex_to_xyz(beliefs):
    """Map 4-state beliefs to 3D tetrahedron coordinates."""
    return np.asarray(beliefs) @ TETRA


def belief_colors_tetra(beliefs):
    return np.clip(np.asarray(beliefs) @ CORNER_COLORS_TETRA, 0, 1)


def load_model(device):
    ckpt = torch.load(MODEL_DIR / "arch_transformer.pt", map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def _panel_3d(fig, idx, xyz, color, title):
    """Create a 3D tetrahedron panel."""
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
    good = {
        w for w in resid
        if w in belief and np.all(np.isfinite(belief[w])) and prefix_prob.get(w, 0.0) > 1e-12
    }
    print(f"enumerated MSP prefixes: {len(resid)}  reachable (prob>0): {len(good)}")
    reach = {w: (w in good) for w in resid}

    # Stage 1 subspace
    B_stage1 = S1.run_stage1(resid, soft, reach, vocab, d_keep=D)
    print(f"Stage 1 subspace shape: {B_stage1.shape}")

    seqs = [s for s in resid if s in good]
    true_b = np.array([belief[s] for s in seqs])
    true_xyz = simplex_to_xyz(true_b)
    color = belief_colors_tetra(true_b)
    wcol = np.array([prefix_prob[s] for s in seqs])
    Xfull = np.stack([resid[s] for s in seqs])

    def to_simplex(p):
        p = np.clip(p, 0, None)
        return p / p.sum(1, keepdims=True)

    def decode_panel(feat):
        r2 = U.belief_decode_r2(feat, true_b, wcol)
        dec = LinearRegression().fit(feat, true_b)
        xyz = simplex_to_xyz(to_simplex(dec.predict(feat)))
        return xyz, r2

    xyz_sup, r2_sup = decode_panel(Xfull)
    xyz_s1, r2_s1 = decode_panel(Xfull @ B_stage1)
    print(f"arch belief-decode R^2:  supervised(64-D)={r2_sup:.3f}   stage1({D}-D)={r2_s1:.3f}")

    fig = plt.figure(figsize=(18, 6))
    _panel_3d(fig, 1, true_xyz, color, "Arch ground truth")
    _panel_3d(fig, 2, xyz_sup, color, f"Supervised: full residual stream (64-D)\nbelief-decode R$^2$={r2_sup:.3f}")
    _panel_3d(fig, 3, xyz_s1, color, f"Stage 1 CCA/RRR ({D}-D)\nbelief-decode R$^2$={r2_s1:.3f}")

    out = FIG_DIR / "fig1_arch_principled_stage1.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
