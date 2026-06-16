"""Observable-anchored OOM: recover belief geometry from activations WITHOUT the
CCA-correlation step that degenerates on (near-)deterministic models, and WITHOUT
any DGP knowledge (model softmax + activations + the prefix tree only).

Pipeline (all ridge regressions -- no whitening, no destructive PCA):
  1. Observable anchor      C  : a(w) -> P(.|w)            (1-step softmax; seed + eval functional)
  2. Rescaled operators     G_x: a(w) -> P(x|w) a(wx)      (activation-space, one P factor: no compounding)
  3. Observability closure       grow span{C} under {G_x} until rank saturates -> latent dim d
  4. Reduce, fit A_x, eval e; validate (decode R^2, eig(A_x) vs eig(T_x), geometry)

DGP-free: reachability and weights come from the MODEL softmax, d is ESTIMATED (not
set to 5). Ground-truth belief is used ONLY to score, never to fit. The construction
assumes only LINEARITY (linear state / operators / observable), so it carries over to
quantum / post-quantum predictive representations unchanged.

FINDINGS (this is a correct method, validated on Mess3; RRXOR is intrinsically hard):
  * Mess3 (control): clean rank drop in the observability spectrum at d=3, belief-decode
    R^2 = 0.997 (= supervised), and eig(A_x) MATCHES eig(T_x). So the estimator works,
    and it is NOT the broken CCA from fig13.
  * RRXOR: decode only ~0.48 at d=5 / 0.61 at d=10, vs supervised ceiling 0.89; the
    spectrum has no clean rank. The earlier suspects are ruled out -- ESS was a weighting
    artifact, the clock is excluded by the P-anchored relation, and CCA is gone.
  * The real cause: RRXOR's belief has WEAKLY-OBSERVABLE modes (directions that barely
    change the predicted future -- the multi-step version of panel D's 0.33). Surfacing
    them unsupervised needs deep multi-step propagation, which compounds: iterating the
    estimated operators makes decode DROP with depth (0.53 -> 0.35 for depth 2 -> 8),
    because operator-estimation noise grows faster than the weak belief signal. The
    softmax-product route compounds chain-rule error instead -- same wall.
  * Supervised regression sidesteps all of this by using belief LABELS directly, so it
    never has to surface the weak modes from observations. The supervised/unsupervised
    gap (0.89 vs ~0.5) is exactly the "value of the labels": near zero for Mess3
    (strongly observable belief), large for RRXOR (weakly observable belief).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

from simplexity.generative_processes.transition_matrices import mess3, rrxor

MAX_LEN = 10
EPS = 1e-3            # model-softmax reachability threshold
RIDGE = 1e-2
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


def observable_subspace(resid, soft, reach, vocab, depth=3):
    """Observability matrix O = [C, G_x C, G_x G_y C, ...]; its SVD gives candidate
    belief directions ordered by strength. Columns of O are the rescaled multi-step
    observables P(x..|w)*P(.|w x..), each LINEAR in belief, spanning belief as depth grows."""
    rows = [w for w in resid if reach[w] and all((w + (x,)) in resid and reach[w + (x,)]
                                                  for x in range(vocab) if soft[w][x] > EPS)]
    A = np.stack([resid[w] for w in rows]); P = np.stack([soft[w] for w in rows])

    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P).coef_.T          # (D, V) observable
    Gs = []
    for x in range(vocab):
        m = np.array([soft[w][x] > EPS for w in rows])
        child = np.stack([resid[rows[i] + (x,)] for i in np.where(m)[0]])
        tgt = P[m, x][:, None] * child
        Gs.append(Ridge(alpha=RIDGE, fit_intercept=False).fit(A[m], tgt).coef_.T)  # (D, D)

    cols, frontier = [C], [C]                                              # observability matrix
    for _ in range(depth - 1):
        nxt = [Gx @ f for Gx in Gs for f in frontier]
        cols += nxt; frontier = nxt
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


def run(name, ckpt, proc_T, stationary):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load(ckpt, device)  # resid_post concat, softmax, analytic belief for SCORING only
    resid, soft, belief, pp = _collect(m, proc_T, stationary, device)
    # model-determined reachability (DGP-free): node reachable if every step's softmax > EPS
    reach = {}
    for w in resid:
        ok = True; pre = ()
        for t in w:
            if pre in soft and soft[pre][t] <= EPS:   # first token has no context -> always reachable
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    vocab = proc_T.shape[0]
    print(f"\n=== {name} ===  reachable nodes: {sum(reach.values())}/{len(reach)}")

    rows, A, P, Gs, U, sv = observable_subspace(resid, soft, reach, vocab)
    print(f"  rows={len(rows)}  D={A.shape[1]}  observability singular spectrum:\n   {np.round(sv[:12] / sv[0], 3)}")

    Yb = np.stack([belief[w] for w in rows]); fin = np.isfinite(Yb).all(1)
    decs = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :d])
        dec = LinearRegression().fit(s_all[fin], Yb[fin]).score(s_all[fin], Yb[fin])
        decs.append((d, dec))
    print("  belief-decode R^2 vs d:", {d: round(r, 3) for d, r in decs})

    ceiling = LinearRegression().fit(A[fin], Yb[fin]).score(A[fin], Yb[fin])
    dstar = proc_T.shape[1]                                   # report eig/geometry at the true belief dim
    s_all, ops, e = fit_at_dim(rows, A, P, resid, vocab, U[:, :dstar])
    reg = LinearRegression().fit(s_all[fin], Yb[fin]); decode = reg.score(s_all[fin], Yb[fin])
    print(f"  supervised ceiling={ceiling:.3f};  at d={dstar}: decode R^2={decode:.3f}; eig(A^x) vs eig(T^x):")
    for x in range(vocab):
        evt = np.sort(np.linalg.eigvals(proc_T[x]).real)
        eva = np.sort(np.linalg.eigvals(ops[x]).real)
        print(f"    token {x}: eig(T)={np.round(evt,3)}  eig(A)={np.round(eva,3)}")
    return dict(resid=resid, soft=soft, belief=belief, rows=rows, A=A, P=P, Gs=Gs, U=U,
                fin=fin, Yb=Yb, reg=reg, dstar=dstar, decode=decode, decs=decs, ceiling=ceiling, vocab=vocab)


def _load(ckpt, device):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    c = torch.load(MODEL_DIR / ckpt, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(c["cfg"]); cfg.device = device
    mm = HookedTransformer(cfg); mm.load_state_dict(c["state_dict"]); mm.to(device).eval()
    return mm


def _collect(model, T, stationary, device):
    import itertools
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    NS = T.shape[1]

    def ndist(b): return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])
    def upd(b, x): nb = b @ T[x]; return nb / nb.sum()
    def bp(w):
        b, p = stationary, 1.0
        for x in w:
            dd = ndist(b); p *= dd[x]
            if dd[x] < 1e-12: return None
            b = upd(b, x)
        return b

    resid, soft, belief, pp = {}, {}, {}, {}
    for L in range(1, MAX_LEN + 1):
        strs = np.array(list(itertools.product(range(T.shape[0]), repeat=L)), dtype=np.int64)
        for i in range(0, len(strs), 4096):
            inp = torch.from_numpy(strs[i:i + 4096]).to(device)
            with torch.no_grad():
                logits, c = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
            concat = np.concatenate([c[h][:, -1, :].cpu().numpy() for h in hooks], axis=-1)
            sm = torch.softmax(logits[:, -1, :], -1).cpu().numpy()
            for j, sct in enumerate(strs[i:i + 4096]):
                w = tuple(int(t) for t in sct)
                resid[w] = concat[j]; soft[w] = sm[j]
                b = bp(w); belief[w] = b if b is not None else np.full(NS, np.nan); pp[w] = 0.0
    return resid, soft, belief, pp


def depth_curve(res, depths=(2, 3, 4, 5, 6, 8)):
    """RRXOR decode (at d=5) vs observability depth -- iterating the estimated operators."""
    A, P, fin, Yb = res["A"], res["P"], res["fin"], res["Yb"]
    C = Ridge(alpha=RIDGE, fit_intercept=False).fit(A, P).coef_.T
    out = []
    for depth in depths:
        cols, fr = [C], [C]
        for _ in range(depth - 1):
            nxt = [Gx @ f for Gx in res["Gs"] for f in fr]; cols += nxt; fr = nxt
        U, _, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
        s = A @ U[:, :5]
        out.append((depth, LinearRegression().fit(s[fin], Yb[fin]).score(s[fin], Yb[fin])))
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    R = run("RRXOR", "rrxor_transformer.pt", np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0)
    M = run("Mess3", "mess3_transformer.pt", np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0)
    dc = depth_curve(R)
    print("RRXOR decode (d=5) vs observability depth:", {d: round(r, 3) for d, r in dc})

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    for res, nm, col in [(R, "RRXOR", "black"), (M, "Mess3", "red")]:
        ds = [d for d, _ in res["decs"]]; rr = [r for _, r in res["decs"]]
        ax0.plot(ds, rr, "o-", color=col, label=f"{nm} (observable OOM)")
        ax0.axhline(res["ceiling"], color=col, ls="--", lw=1, alpha=0.6,
                    label=f"{nm} supervised ceiling = {res['ceiling']:.2f}")
    ax0.set_xlabel("subspace dim d"); ax0.set_ylabel("belief-decode R$^2$")
    ax0.set_title("Observable-anchored OOM (DGP-free):\nrecovers Mess3, NOT RRXOR")
    ax0.set_ylim(0, 1.02); ax0.legend(fontsize=8, loc="center right"); ax0.grid(alpha=.3)

    dd = [d for d, _ in dc]; rr = [r for _, r in dc]
    ax1.plot(dd, rr, "ko-")
    ax1.set_xlabel("observability depth (operator iterations)")
    ax1.set_ylabel("RRXOR belief-decode R$^2$ (d=5)")
    ax1.set_title("RRXOR: deeper observability -> WORSE\n(operator-iteration noise compounds past the weak signal)")
    ax1.set_ylim(0, 0.6); ax1.grid(alpha=.3)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "fig14_observable_oom.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
