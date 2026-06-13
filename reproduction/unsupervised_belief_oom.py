"""Unsupervised activation-side OOM / DMD estimator for the Mess3 belief geometry.

Goal
----
Recover the belief-state geometry from a trained transformer's residual stream
*without ever regressing onto analytic belief targets*. The only ingredients are
(i) residual-stream activations and (ii) the network's own softmax next-token
probabilities. This implements the estimator proposed in the compass_artifact
note (the "rescaling trick").

The idea
--------
The optimal belief update is projective:
    b(wx) = b(w) T^(x) / ( b(w) T^(x) 1 ),    where  b(w) T^(x) 1 = P(x|w).
Naive linear regression z(wx) ~ z(w) is therefore mis-specified. But the
normalizer P(x|w) is exactly what the transformer outputs as its softmax. So
multiply it through:
    P(x|w) * z(wx)  ≈  z(w) * A^(x)                          (now LINEAR)
and fit one operator A^(x) per token by ordinary least squares, using only
activations + softmax. Because activations are an (uncentered) linear image of
the belief, a(w) = b(w) Ψ, one has A^(x) = Ψ^{-1} T^(x) Ψ — similar to the true
per-token operator. Hence eig(A^(x)) ≈ eig(T^(x)): a gauge-invariant validation
that needs no ground truth.

Why uncentered, dimension d=3
-----------------------------
Mess3 has 3 hidden states; the belief lives on the 2-simplex (2 free dims). The
affine offset of the activation->belief map is absorbed into a linear term via
sum(b)=1, so an *uncentered* rank-3 projection makes a(w) = b(w) Ψ exactly
linear with a 3x3 invertible Ψ. The operators are then 3x3, directly comparable
to the 3x3 T^(x).

What is and isn't unsupervised
------------------------------
- Operator fit, eigenvalue check, eval-functional recovery, one-step rollout R^2:
  fully target-free (activations + softmax only).
- Final overlay on the canonical fractal and RGB coloring use the analytic
  beliefs *for display/interpretation only* (positions/structure come from the
  estimator). This mirrors the paper, which colors activations by ground truth.
"""

from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.generative_processes.transition_matrices import mess3

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
MSP_DEPTH = 10          # enumerate prefixes up to this length
LATENT_D = 3            # operator dimension (3 hidden states)
SEED = 0

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"


# --------------------------------------------------------------------------- #
# Model + data
# --------------------------------------------------------------------------- #
def load_model(device):
    ckpt = torch.load(MODEL_DIR / "mess3_transformer.pt", map_location=device, weights_only=False)
    model = HookedTransformer(HookedTransformerConfig.from_dict(ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"]


def collect_prefix_features(model, hmm, depth, context_len, device):
    """For every MSP prefix (up to `depth`, capped at context_len), record:
        resid[seq]  = final-position residual stream (pre-final-LayerNorm), R^d_model
        soft[seq]   = final-position softmax over next token, R^vocab
        belief[seq] = analytic belief (for evaluation/coloring ONLY)
    """
    layer = model.cfg.n_layers - 1
    max_len = min(depth, context_len)
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=max_len).generate()

    by_len = defaultdict(lambda: ([], [], []))
    for seq, v in tree.nodes.items():
        if len(seq) == 0:
            continue
        seqs, bels, probs = by_len[len(seq)]
        seqs.append(seq)
        bels.append(v.belief_state)
        probs.append(v.probability)

    resid, soft, belief, prefix_prob = {}, {}, {}, {}
    for L in sorted(by_len):
        seqs, bels, probs = by_len[L]
        inp = torch.tensor(np.array(seqs, dtype=np.int64), device=device)
        with torch.no_grad():
            logits, cache = model.run_with_cache(inp, names_filter=f"blocks.{layer}.hook_resid_post")
        z = cache["resid_post", layer][:, -1, :].cpu().numpy()           # (n, d_model)
        p = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy()        # (n, vocab)
        for i, seq in enumerate(seqs):
            resid[seq] = z[i]
            soft[seq] = p[i]
            belief[seq] = np.array(bels[i])
            prefix_prob[seq] = probs[i]
    return resid, soft, belief, prefix_prob, max_len


# --------------------------------------------------------------------------- #
# Subspace + operator fit (the unsupervised core)
# --------------------------------------------------------------------------- #
def fit_subspace(resid, d):
    """Uncentered rank-d projection of the activation snapshot matrix."""
    seqs = list(resid.keys())
    Z = np.stack([resid[s] for s in seqs])              # (N, d_model)
    # uncentered SVD: keep the mean/offset direction so a(w) = b(w) Psi is linear
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    basis = Vt[:d].T                                    # (d_model, d)
    proj = {s: resid[s] @ basis for s in seqs}          # a(w) in R^d
    return proj, basis, S


def build_triples(proj, soft, prefix_prob, max_len, vocab):
    """(a(w), a(wx), P(x|w), P(w)) per prefix/next-token, grouped by token x."""
    triples = {x: ([], [], [], []) for x in range(vocab)}
    for w, aw in proj.items():
        if len(w) >= max_len:           # child wx would exceed enumerated depth
            continue
        for x in range(vocab):
            child = w + (x,)
            if child not in proj:
                continue
            A_rows, B_rows, P, Wt = triples[x]
            A_rows.append(aw)
            B_rows.append(proj[child])
            P.append(soft[w][x])
            Wt.append(prefix_prob[w])
    return {x: (np.array(a), np.array(b), np.array(p), np.array(wt)) for x, (a, b, p, wt) in triples.items()}


def fit_operators(triples):
    """Rescaled, probability-weighted least squares:  a(w) A_x ≈ P(x|w) a(wx).

    Weighting rows by sqrt(P(w)) makes the operators most accurate on the
    realized (high-probability) part of the distribution that rollout follows.
    """
    ops, fit_r2 = {}, {}
    for x, (A, B, P, Wt) in triples.items():
        target = P[:, None] * B                         # rescale -> linear
        sw = np.sqrt(np.clip(Wt, 1e-12, None))[:, None]
        Op, *_ = np.linalg.lstsq(A * sw, target * sw, rcond=None)
        pred = A @ Op
        wss = Wt[:, None]
        ss_res = (wss * (target - pred) ** 2).sum()
        ss_tot = (wss * (target - np.average(target, axis=0, weights=Wt)) ** 2).sum()
        ops[x] = Op
        fit_r2[x] = 1.0 - ss_res / ss_tot
    return ops, fit_r2


def recover_eval_functional(proj):
    """Find e with a(w) . e = 1 for all w (the gauge image of the all-ones sum).

    Fully unsupervised: encodes that normalized beliefs sum to 1.
    """
    A = np.stack(list(proj.values()))
    ones = np.ones(A.shape[0])
    e, *_ = np.linalg.lstsq(A, ones, rcond=None)
    resid = A @ e - 1.0
    return e, float(np.sqrt((resid ** 2).mean()))


# --------------------------------------------------------------------------- #
# Validation (target-free)
# --------------------------------------------------------------------------- #
def eigenvalue_comparison(ops, params):
    tm = np.array(mess3(**params))                      # (vocab, 3, 3)
    rows = []
    for x in sorted(ops):
        ev_op = np.sort_complex(np.linalg.eigvals(ops[x]))
        ev_true = np.sort_complex(np.linalg.eigvals(tm[x]))
        rows.append((x, ev_true, ev_op))
    return rows


def one_step_rollout_r2(model, hmm, basis, ops, context_len, device, n_seq=400):
    """One-step operator prediction on freshly sampled sequences (target-free).

    Predict a_hat(wx) = a(w) A_x / P(x|w) and compare to the actual projected
    activation a(wx). Uses model softmax for P(x|w); no analytic beliefs.
    """
    layer = model.cfg.n_layers - 1
    key = jax.random.PRNGKey(SEED)
    init = jnp.array(hmm.initial_state)
    gs = jnp.repeat(init[None, :], n_seq, axis=0)
    _, inputs, _ = generate_data_batch(gs, hmm, n_seq, context_len + 1, key, device=device)
    inputs = inputs.long().to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names_filter=f"blocks.{layer}.hook_resid_post")
    resid = cache["resid_post", layer].cpu().numpy()                 # (n, T, d_model)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()              # (n, T, vocab)
    toks = inputs.cpu().numpy()
    A = resid @ basis                                                # (n, T, d)

    preds, actuals = [], []
    T = A.shape[1]
    for t in range(T - 1):
        x_next = toks[:, t + 1]                                      # token emitted at t+1
        p = probs[np.arange(len(toks)), t, x_next]                  # P(x_{t+1} | prefix_t)
        for x in ops:
            m = x_next == x
            if not m.any():
                continue
            ahat = (A[m, t] @ ops[x]) / np.clip(p[m, None], 1e-8, None)
            preds.append(ahat)
            actuals.append(A[m, t + 1])
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    ss_res = ((actuals - preds) ** 2).sum()
    ss_tot = ((actuals - actuals.mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Geometry recovery by operator rollout + visualization
# --------------------------------------------------------------------------- #
def rollout_states(proj, ops, e, max_len, vocab):
    """Generate states by the *projective* rollout of the recovered operators.

    At each step:  s(wx) = s(w) A_x ,  then renormalize so s(wx) . e = 1.
    Renormalizing every step (rather than once at the end) keeps states bounded
    and reproduces the belief metadynamics; the eigenvalues<1 contraction is
    exactly cancelled by the P(x|w)=s.A_x.e normalizer. The root state is the
    in-gauge mean of length-1 projected activations.
    Returns {seq: normalized_state_in_R^d}.
    """
    init_seqs = [s for s in proj if len(s) == 1]
    s0 = np.mean([proj[s] for s in init_seqs], axis=0)
    s0 = s0 / (s0 @ e)
    states = {(): s0}
    frontier = [()]
    for _ in range(max_len):
        new_frontier = []
        for w in frontier:
            sw = states[w]
            for x in range(vocab):
                child = w + (x,)
                if child not in proj:
                    continue
                sc = sw @ ops[x]
                denom = sc @ e
                states[child] = sc / denom if abs(denom) > 1e-12 else sc
                new_frontier.append(child)
        frontier = new_frontier
    states.pop((), None)
    return states


def plane_coords(points):
    """2D coordinates for points lying ~on an affine plane in R^d (PCA to 2D)."""
    P = np.asarray(points)
    mu = P.mean(0)
    Pc = P - mu
    _, _, vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ vt[:2].T


def affine_align(src_xy, dst_xy):
    """Least-squares affine map src->dst (for display overlay only)."""
    X = np.concatenate([src_xy, np.ones((len(src_xy), 1))], axis=1)
    M, *_ = np.linalg.lstsq(X, dst_xy, rcond=None)
    return X @ M


def simplex_to_xy(beliefs):
    theta = np.pi / 3.0
    basis = np.array([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])
    return beliefs[:, :2] @ basis


# --------------------------------------------------------------------------- #
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    vocab = hmm.vocab_size
    model, context_len = load_model(device)

    resid, soft, belief, prefix_prob, max_len = collect_prefix_features(
        model, hmm, MSP_DEPTH, context_len, device
    )
    print(f"prefixes collected: {len(resid)}  (lengths 1..{max_len})")

    proj, basis, S = fit_subspace(resid, LATENT_D)
    print("top-8 singular values:", np.round(S[:8], 2))
    print(f"singular-value gaps after rank: 3 -> {S[2] / S[3]:.2f}x")

    triples = build_triples(proj, soft, prefix_prob, max_len, vocab)
    ops, fit_r2 = fit_operators(triples)
    print("per-token operator fit R^2:", {x: round(r, 4) for x, r in fit_r2.items()})

    e, e_rmse = recover_eval_functional(proj)
    print(f"recovered eval functional rmse(a.e - 1) = {e_rmse:.2e}")

    print("\n=== TARGET-FREE VALIDATION ===")
    for x, ev_true, ev_op in eigenvalue_comparison(ops, MESS3_PARAMS):
        print(f"token {x}: eig(T^x)  = {np.round(ev_true, 3)}")
        print(f"         eig(A^x)  = {np.round(ev_op, 3)}")
    r2_roll = one_step_rollout_r2(model, hmm, basis, ops, context_len, device)
    print(f"one-step operator-rollout R^2 (held-out, target-free): {r2_roll:.4f}")

    # ---- geometry recovery (positions from activations only) ----
    seqs = [s for s in proj if s in belief]
    true_b = np.array([belief[s] for s in seqs])
    true_xy = simplex_to_xy(true_b)
    color = np.clip(true_b, 0, 1)

    # Activation-cloud panel: every prefix (activations are bounded).
    act_xy = plane_coords([proj[s] for s in seqs])
    act_aligned = affine_align(act_xy, true_xy)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].scatter(true_xy[:, 0], true_xy[:, 1], c=color, s=2, edgecolors="none")
    axes[0].set_title("Ground truth (analytic belief)")
    axes[1].scatter(act_aligned[:, 0], act_aligned[:, 1], c=color, s=2, edgecolors="none")
    axes[1].set_title(
        "Unsupervised recovery from activations\n"
        f"(positions: activations only, no targets; eig & rollout certified; color=belief)"
    )
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    out = FIG_DIR / "fig04_unsupervised_oom_mess3.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
