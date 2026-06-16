"""Mess3 belief geometry recovered by the observability OOM (fig14) -- the positive
control. The same DGP-free estimator that only partially recovers RRXOR (COM 0.685,
weakly observable) recovers Mess3's full fractal cleanly (decode R^2 ~ 0.997), because
Mess3's belief is strongly observable and richly sampled.

Left: ground-truth Mess3 belief simplex. Right: belief decoded from the unsupervised
observability-OOM subspace (subspace found from activations + model softmax + tree
only; the linear readout to the simplex is the validation map). Rendered identically
(datashader-style raster, coloured by belief).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import mess3
import fig14_observable_oom as F14
import fig15_oom_denoised as F15
import unsupervised_belief_oom as U

D = 3
FIG_DIR = Path(__file__).parent / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    T = np.array(mess3(x=0.05, a=0.85)); pi = np.array([1, 1, 1]) / 3.0
    resid, soft, belief, reach = F15.prepare("mess3_transformer.pt", T, pi)
    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, T.shape[0])
    B = Uobs[:, :D]

    # decode (validation map) fit on the OOM rows
    Yb = np.stack([belief[w] for w in rows])
    decode = LinearRegression().fit(A @ B, Yb)
    print(f"observability-OOM Mess3 belief-decode R^2 = {decode.score(A @ B, Yb):.4f}  (d={D})")

    # dense display set: every reachable prefix (Mess3 is full-support)
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Aa = np.stack([resid[w] for w in allw])
    pred = decode.predict(Aa @ B)
    pred = np.clip(pred, 0, None); pred = pred / pred.sum(1, keepdims=True)   # back onto the simplex
    true = np.stack([belief[w] for w in allw])
    print(f"display points: {len(allw)}")

    img_t = U.rasterize_simplex(U.simplex_to_xy(true), np.clip(true, 0, 1))
    img_p = U.rasterize_simplex(U.simplex_to_xy(pred), np.clip(pred, 0, 1))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.6))
    ax0.imshow(img_t, origin="lower"); ax0.set_title("Mess3 ground-truth belief geometry")
    ax1.imshow(img_p, origin="lower")
    ax1.set_title(f"Recovered by observability OOM (unsupervised)\nbelief-decode R$^2$ = {decode.score(A @ B, Yb):.3f}")
    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "fig16_oom_mess3_geometry.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
