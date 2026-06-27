"""Stage 0 (Pre-gate): Verify belief-state cluster separation exists.

Gate: before building Stage 2's cluster-separation estimator, verify that the
36 belief states of RRXOR *actually separate* in a supervised 5D belief subspace.

If the 36 states (especially the 31 transient states) are mushy even with
ground-truth labels, the gap is partly intrinsic. This gate catches that early.

Method:
  1. Fit supervised regression: concatenated residual-stream activations → belief
  2. Recover the top-5 belief subspace via supervised PCA
  3. For each of the 36 unique belief states, compute center-of-mass in that subspace
  4. Compute between-cluster / within-cluster scatter ratio (LDA-style)
  5. Report: per-state separation and flag which transient states (if any) fail to separate

Output: pass/fail gate, separation metrics, and the supervised 5D basis for Stage 1's reference.
"""

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.transition_matrices import rrxor

P1 = P2 = 0.5
MAX_LEN = 10
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"

T = np.array(rrxor(P1, P2))  # (2, 5, 5)
NSTATES = T.shape[1]
STATIONARY = np.array([2, 1, 1, 1, 1]) / 6.0


def next_dist(b):
    """Next-token distribution P(x|b) for state b."""
    return np.array([(b @ T[x]).sum() for x in range(T.shape[0])])


def update(b, x):
    """Bayesian belief update: b' = update(b, x)."""
    nb = b @ T[x]
    return nb / nb.sum()


def msp_states():
    """36 unique positive-probability MSP beliefs (incl. root), + {rounded->index}."""
    beliefs = {tuple(np.round(STATIONARY, 5)): STATIONARY}
    frontier = [STATIONARY]
    for _ in range(20):
        nxt = []
        for b in frontier:
            d = next_dist(b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                b2 = update(b, x)
                k = tuple(np.round(b2, 5))
                if k not in beliefs:
                    beliefs[k] = b2
                    nxt.append(b2)
        frontier = nxt
    keys = list(beliefs.keys())
    B = np.array([beliefs[k] for k in keys])
    index = {k: i for i, k in enumerate(keys)}
    return B, index


def enumerate_inputs(n_ctx, index):
    """All positive-prob length-n_ctx sequences, with per-position belief and 36-state index."""
    frontier = [((), [])]
    for _ in range(n_ctx):
        nxt = []
        for seq, bels in frontier:
            b = STATIONARY if not bels else bels[-1]
            d = next_dist(b)
            for x in range(len(d)):
                if d[x] < 1e-12:
                    continue
                nxt.append((seq + (x,), bels + [update(b, x)]))
        frontier = nxt
    seqs = np.array([s for s, _ in frontier], dtype=np.int64)
    beliefs = np.array([b for _, b in frontier], dtype=np.float32)  # (N, n_ctx, 5)
    idx = np.array([[index[tuple(np.round(b, 5))] for b in bels] for _, bels in frontier])
    return seqs, beliefs, idx


def load_model(path, device):
    if not Path(path).exists():
        path = MODEL_DIR / path
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ck["cfg"])
    cfg.device = device
    m = HookedTransformer(cfg)
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m


def collect_activations(model, seqs, device, hooks):
    """Return {hook_name: (N, n_ctx, d)} for the requested hooks, batched."""
    out = {h: [] for h in hooks}
    for i in range(0, len(seqs), 4096):
        inp = torch.from_numpy(seqs[i : i + 4096]).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, names_filter=lambda n: n in hooks)
        for h in hooks:
            out[h].append(cache[h].cpu().numpy())
    for h in hooks:
        out[h] = np.concatenate(out[h], axis=0)
    return out


def flatten_concat_activations(activations_dict, seqs):
    """Concatenate residual-stream activations across layers.

    Returns (N, n_ctx, D_total) for the last token of each sequence.
    """
    nL = max(1 + int(k.split(".")[1]) for k in activations_dict.keys())
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    concat_list = []
    for h in hooks:
        concat_list.append(activations_dict[h])
    concat_act = np.concatenate(concat_list, axis=-1)
    return concat_act[:, -1, :]  # last token only


def between_within_scatter(projections, state_indices):
    """Compute LDA-style between/within cluster scatter ratio.

    Args:
        projections: (N, d) activation projections
        state_indices: (N,) integer state index for each sample

    Returns:
        between_scatter, within_scatter, ratio
    """
    unique_states = np.unique(state_indices[state_indices >= 0])
    global_mean = projections.mean(axis=0)

    # Between-cluster scatter: sum of squared distances of class means from global mean
    between = np.zeros((projections.shape[1], projections.shape[1]))
    for state in unique_states:
        mask = state_indices == state
        class_mean = projections[mask].mean(axis=0)
        diff = class_mean - global_mean
        between += mask.sum() * np.outer(diff, diff)

    # Within-cluster scatter: sum of squared distances of points from their class mean
    within = np.zeros((projections.shape[1], projections.shape[1]))
    for state in unique_states:
        mask = state_indices == state
        class_mean = projections[mask].mean(axis=0)
        centered = projections[mask] - class_mean
        within += centered.T @ centered

    # Trace ratio (generalized eigenvalue): tr(S_b) / tr(S_w)
    trace_between = np.trace(between)
    trace_within = np.trace(within)
    ratio = trace_between / (trace_within + 1e-10)

    return between, within, ratio


def per_state_separation(projections, state_indices, unique_states):
    """Compute per-state within-cluster compactness (for reporting)."""
    separations = {}
    for state in unique_states:
        mask = state_indices == state
        if mask.sum() > 1:
            state_points = projections[mask]
            state_mean = state_points.mean(axis=0)
            # Average distance from mean (compactness)
            distances = np.linalg.norm(state_points - state_mean, axis=1)
            separations[state] = float(distances.mean())
        else:
            separations[state] = 0.0
    return separations


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 80)
    print("STAGE 0: Gate - Verify belief-state separation in supervised 5D subspace")
    print("=" * 80)

    # Load model and generate all prefixes
    print(f"\nLoading model from {MODEL_DIR / 'rrxor_transformer.pt'}...")
    model = load_model("rrxor_transformer.pt", device)

    print("Enumerating prefixes and beliefs...")
    B_states, state_index = msp_states()
    seqs, beliefs, state_indices = enumerate_inputs(MAX_LEN, state_index)
    print(f"  Total sequences: {len(seqs)}")
    print(f"  Unique belief states: {len(B_states)}")

    # Collect activations
    print("Collecting residual-stream activations...")
    nL = model.cfg.n_layers
    hooks = [f"blocks.{i}.hook_resid_post" for i in range(nL)]
    act_dict = collect_activations(model, seqs, device, hooks)

    # Flatten to (N, D_total)
    print("Flattening and concatenating activations across layers...")
    A = flatten_concat_activations(act_dict, seqs)
    print(f"  Activation shape: {A.shape}")

    # Reshape beliefs from (N, n_ctx, 5) to (N*n_ctx, 5) and correspondingly A
    N, n_ctx = seqs.shape[0], seqs.shape[1] if seqs.ndim > 1 else 1
    if beliefs.ndim == 3:
        # Expand activations per position: (N, D) -> (N*n_ctx, D)
        beliefs_flat = beliefs.reshape(-1, 5)
        if A.shape[0] == N:
            # Need to expand A per position
            A_expanded = []
            for layer_i in range(nL):
                act_layer = act_dict[f"blocks.{layer_i}.hook_resid_post"]
                A_expanded.append(act_layer.reshape(-1, act_layer.shape[-1]))
            A = np.concatenate(A_expanded, axis=-1)
        state_indices_flat = state_indices.reshape(-1)
    else:
        beliefs_flat = beliefs
        state_indices_flat = state_indices

    print(f"  Flattened: beliefs {beliefs_flat.shape}, states {state_indices_flat.shape}")

    # Supervised regression: A -> beliefs
    print("\nFitting supervised regression: activations -> beliefs...")
    fin = np.isfinite(beliefs_flat).all(axis=1)
    reg = LinearRegression().fit(A[fin], beliefs_flat[fin])
    r2_full = reg.score(A[fin], beliefs_flat[fin])
    print(f"  Supervised decode (full 64D): R² = {r2_full:.4f}")

    # Get 5D supervised subspace via PCA of predictions
    print("Extracting 5D supervised subspace via PCA of predicted beliefs...")
    beliefs_pred = reg.predict(A[fin])
    pca = PCA(n_components=5)
    beliefs_proj = pca.fit_transform(beliefs_pred)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    # Supervised subspace (basis)
    B_supervised = pca.components_.T  # (64, 5)
    beliefs_proj_full = (beliefs_pred @ pca.components_.T)  # (N_fin, 5)

    # Evaluate: R² of supervised 5D projection
    reg_5d = LinearRegression().fit(beliefs_proj, beliefs_pred)
    r2_5d = reg_5d.score(beliefs_proj, beliefs_pred)
    print(f"  Supervised 5D projection R²: {r2_5d:.4f}")

    # Gate criterion: between/within scatter
    print("\n" + "-" * 80)
    print("GATE CRITERION: Between/within cluster separation in supervised 5D")
    print("-" * 80)

    between, within, ratio = between_within_scatter(beliefs_proj_full, state_indices_flat[fin])
    print(f"\nBetween-cluster scatter trace: {np.trace(between):.4f}")
    print(f"Within-cluster scatter trace:  {np.trace(within):.4f}")
    print(f"Separation ratio (between/within): {ratio:.4f}")

    # Per-state compactness
    unique_states = np.unique(state_indices_flat[fin])
    per_state = per_state_separation(beliefs_proj_full, state_indices_flat[fin], unique_states)

    # Identify transient states (all except root state 0)
    root_state = 0
    transient_states = [s for s in unique_states if s != root_state]
    transient_compact = [per_state[s] for s in transient_states if s in per_state]

    if root_state in per_state:
        print(f"\nRoot state (0) compactness: {per_state[root_state]:.4f}")
    else:
        print(f"\nRoot state (0) not in valid beliefs")
    if transient_compact:
        print(f"Transient states (1-35) mean compactness: {np.mean(transient_compact):.4f}")
        print(f"Transient states std: {np.std(transient_compact):.4f}")
        print(f"Transient states range: [{np.min(transient_compact):.4f}, {np.max(transient_compact):.4f}]")
    else:
        print(f"No transient states in valid beliefs")

    # Identify "mushy" states (high within-state dispersion)
    if transient_compact:
        mushy_threshold = np.mean(transient_compact) + 1.5 * np.std(transient_compact)
        mushy_states = [s for s in transient_states if s in per_state and per_state[s] > mushy_threshold]
    else:
        mushy_threshold = np.inf
        mushy_states = []

    print(f"\nMushy threshold (mean + 1.5*std): {mushy_threshold:.4f}")
    print(f"Mushy transient states (>threshold): {len(mushy_states)}")
    if len(mushy_states) <= 5:
        print(f"  States: {mushy_states}")

    # Gate decision
    print("\n" + "-" * 80)
    if ratio > 0.5 and len(mushy_states) == 0:
        gate_pass = True
        print("✓ GATE PASS: States separate cleanly. Proceed to Stage 1.")
    elif ratio > 0.3:
        gate_pass = True
        print(f"✓ GATE PASS (weak): Ratio={ratio:.3f} > 0.3, {len(mushy_states)} mushy states.")
        print("  Proceeding to Stage 1, but note partial intrinsic gap.")
    else:
        gate_pass = False
        print(f"✗ GATE FAIL: Ratio={ratio:.3f} < 0.3. Gap is partly intrinsic.")
        print("  The 36 states do not separate cleanly even with supervision.")

    # Save outputs for Stage 1/2
    outputs = {
        "gate_pass": gate_pass,
        "separation_ratio": float(ratio),
        "B_supervised": B_supervised,  # (64, 5) basis
        "pca_model": pca,
        "mushy_states": mushy_states,
        "per_state_compactness": per_state,
        "A": A,  # Full activations for Stage 1
        "beliefs_flat": beliefs_flat,
        "state_indices_flat": state_indices_flat,
        "fin": fin,
        "r2_supervised": r2_full,
        "B_states": B_states,
        "state_index": state_index,
        "T": T,
        "STATIONARY": STATIONARY,
    }

    # Save to pickle for Stage 1 to load
    import pickle
    stage0_out = OUT_DIR / "stage0_gate_output.pkl"
    with open(stage0_out, "wb") as f:
        pickle.dump(outputs, f)
    print(f"\nSaved Stage 0 outputs to {stage0_out}")

    print("\n" + "=" * 80)
    print("STAGE 0 COMPLETE")
    print("=" * 80)

    return outputs


if __name__ == "__main__":
    main()
