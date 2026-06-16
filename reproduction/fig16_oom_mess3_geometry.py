"""Mess3 belief geometry recovered by the observability OOM (fig14) -- the positive
control, on a SINGLE residual stream (blocks.{last}.hook_resid_post), matching the
paper / the supervised fig03 / the older OOM fig04 (apples-to-apples).

Panels: (1) ground-truth belief simplex; (2) belief decoded from the unsupervised
observability-OOM subspace (raw per-prefix projection); (3) operator-rollout: each
prefix regenerated purely from the recovered operators (one-shot denoising, as in
fig04). Subspace + operators are found from activations + model softmax + tree only;
the linear readout to the simplex is the validation map. Rendered identically.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import mess3
import fig14_observable_oom as F14
import unsupervised_belief_oom as U

D = 3
FIG_DIR = Path(__file__).parent / "figures"


def collect_single(model, T, pi, layer, device="cpu"):
    """Single-layer resid_post activation, softmax, analytic belief; one row per prefix."""
    hook = f"blocks.{layer}.hook_resid_post"
    NS = T.shape[1]

    def ndist(b): return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])
    def upd(b, x): nb = b @ T[x]; return nb / nb.sum()
    def bp(w):
        b = pi
        for x in w:
            d = ndist(b)
            if d[x] < 1e-12: return None
            b = upd(b, x)
        return b

    resid, soft, belief = {}, {}, {}
    for L in range(1, F14.MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n == hook)
            act = c[hook][:, -1, :].cpu().numpy()
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = act[j]; soft[w] = sm[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan)
    return resid, soft, belief


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    T = np.array(mess3(x=0.05, a=0.85)); pi = np.array([1, 1, 1]) / 3.0
    model = F14._load("mess3_transformer.pt", "cpu")
    layer = model.cfg.n_layers - 1
    resid, soft, belief = collect_single(model, T, pi, layer)
    reach = {w: True for w in resid}                       # Mess3 is full-support
    print(f"single layer: blocks.{layer}.hook_resid_post  (dim {len(next(iter(resid.values())))})")

    rows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, T.shape[0])
    B = Uobs[:, :D]
    s_all, ops, e = F14.fit_at_dim(rows, A, P, resid, T.shape[0], B)
    Yb = np.stack([belief[w] for w in rows])
    decode = LinearRegression().fit(s_all, Yb)
    print(f"observability-OOM Mess3 belief-decode R^2 = {decode.score(s_all, Yb):.4f}  (d={D})")

    allw = [w for w in resid if np.isfinite(belief[w]).all()]
    Aa = np.stack([resid[w] for w in allw])
    true = np.stack([belief[w] for w in allw])

    def to_simplex(p):
        p = np.clip(p, 0, None); return p / p.sum(1, keepdims=True)

    raw = to_simplex(decode.predict(Aa @ B))

    # operator rollout (one-shot denoise): regenerate each prefix from the operators
    proj = {w: resid[w] @ B for w in allw}
    sh = U.rollout_states(proj, ops, e, F14.MAX_LEN, T.shape[0])
    rw = [w for w in allw if w in sh and np.isfinite(sh[w]).all()]
    roll = to_simplex(decode.predict(np.stack([sh[w] for w in rw])))
    print(f"display points: raw={len(allw)}  rollout={len(rw)}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    panels = [("Mess3 ground truth", U.simplex_to_xy(true), np.clip(true, 0, 1)),
              (f"Observability OOM (raw)  R$^2$={decode.score(s_all, Yb):.3f}",
               U.simplex_to_xy(raw), np.clip(raw, 0, 1)),
              ("Observability OOM (operator rollout, denoised)",
               U.simplex_to_xy(roll), np.clip(roll, 0, 1))]
    for ax, (title, xy, col) in zip(axes, panels):
        ax.imshow(U.rasterize_simplex(xy, col), origin="lower")
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "fig16_oom_mess3_geometry.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
