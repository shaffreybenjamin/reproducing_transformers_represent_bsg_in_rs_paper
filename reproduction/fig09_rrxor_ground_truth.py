"""Figure 7B reproduction: ground-truth belief geometry of Random-Random-XOR.

RRXOR (p1=p2=0.5) is a 5-state unifilar HMM whose MSP has 36 distinct belief
states (a 4-simplex). Faithful to the authors' rrxor_simplex.ipynb: collect the
unique MSP beliefs, PCA-project the 5-D belief vectors to 2-D, and scatter each
belief state in a distinct colour.

simplexity's MixedStateTreeGenerator enumerates *all* token strings, including
the zero-probability ones forbidden by RRXOR's XOR constraint (their belief is
0/0 = NaN); we drop those. The 36 = 35 positive-probability prefix beliefs + the
root (stationary) belief.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator

RRXOR_PARAMS = {"p1": 0.5, "p2": 0.5}
DEPTH = 14
OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"


def unique_msp_beliefs(hmm, depth=DEPTH):
    """Unique positive-probability MSP belief states (incl. the root), and a
    {rounded-belief-tuple -> index} map for colouring later panels consistently."""
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=depth).generate()
    beliefs = {}
    root = tree.nodes.get(())
    if root is not None:
        beliefs[tuple(np.round(np.asarray(root.belief_state), 5))] = np.asarray(root.belief_state)
    for seq, v in tree.nodes.items():
        if len(seq) == 0:
            continue
        b = np.asarray(v.belief_state)
        if not np.isfinite(b).all():
            continue
        beliefs.setdefault(tuple(np.round(b, 5)), b)
    keys = list(beliefs.keys())
    B = np.array([beliefs[k] for k in keys])
    index = {k: i for i, k in enumerate(keys)}
    return B, index


def distinct_colors(n):
    base = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return [base[i % len(base)] for i in range(n)]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hmm = build_hidden_markov_model("rrxor", RRXOR_PARAMS)
    B, index = unique_msp_beliefs(hmm)
    print(f"RRXOR MSP unique belief states: {B.shape[0]}  (belief dim {B.shape[1]})")

    pca = PCA(n_components=2).fit(B)
    xy = pca.transform(B)
    colors = distinct_colors(len(B))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=90, edgecolors="white", linewidths=0.5)
    ax.set_title("Ground Truth Belief Geometry  (RRXOR, 36 MSP states)")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig09_rrxor_ground_truth.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
