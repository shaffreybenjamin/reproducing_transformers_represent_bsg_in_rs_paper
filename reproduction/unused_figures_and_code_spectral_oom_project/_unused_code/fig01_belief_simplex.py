"""Figure 1 - Ground-truth belief-state geometry of the mess3 process.

Concept
-------
`mess3` is a 3-state Hidden Markov Model. An observer never sees the hidden
state; it only sees emitted tokens. The optimal thing to track is a *belief*:
the posterior distribution over the 3 hidden states given the history so far.

Every distinct history (sequence prefix) maps to one belief = one point on the
2-simplex (a triangle, because the 3 probabilities sum to 1). The set of all
reachable beliefs is the Mixed-State Presentation (MSP). For mess3 it is a
self-similar *fractal*. This script draws that ground-truth geometry - the thing
the paper later claims to find linearly encoded in a transformer's residual
stream.

We color each point by its belief (R,G,B) = P(state0,state1,state2), so position
and color carry the same information.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator

# mess3 parameters from the paper (Appendix A.3 transition matrices ->
# simplexity mess3(x, a) gives x=0.05, a=0.85).
MESS3_PARAMS = {"x": 0.05, "a": 0.85}
MAX_SEQUENCE_LENGTH = 10  # tree depth; deeper -> finer fractal but slower
OUT_DIR = Path(__file__).parent / "figures"


def simplex_to_xy(beliefs: np.ndarray) -> np.ndarray:
    """Map points on the 2-simplex (rows sum to 1) to 2D triangle coordinates.

    We use two basis vectors at 60 degrees, then project the first two barycentric
    coordinates. Because the three coordinates sum to 1, two of them fully
    determine the point, so this is a faithful (affine) embedding of the simplex.
    """
    theta = np.pi / 3.0
    basis = np.array([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
    return beliefs[:, :2] @ basis


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the process and enumerate its mixed-state presentation.
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    generator = MixedStateTreeGenerator(hmm, max_sequence_length=MAX_SEQUENCE_LENGTH)
    tree = generator.generate()

    # 2. Pull belief states (distribution over hidden states) for every node.
    beliefs = np.array([node.belief_state for node in tree.nodes.values()])  # (N, 3)
    probs = np.array([node.probability for node in tree.nodes.values()])  # (N,)
    print(f"MSP nodes: {len(beliefs)} (depth {MAX_SEQUENCE_LENGTH})")
    print(f"belief simplex check: rows sum to ~1 -> {np.allclose(beliefs.sum(1), 1.0)}")

    # 3. Project to 2D and plot, coloring each belief by its (R,G,B) coordinates.
    xy = simplex_to_xy(beliefs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xy[:, 0], xy[:, 1], c=beliefs, s=2, edgecolors="none")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"mess3 ground-truth belief-state geometry (MSP)\n{len(beliefs)} reachable beliefs, depth {MAX_SEQUENCE_LENGTH}")

    out = OUT_DIR / "fig01_belief_simplex_mess3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
