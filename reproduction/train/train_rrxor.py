"""Train a transformer on RRXOR with the QUANTUM-PAPER setup, but CONTEXT WINDOW 14.

Same configuration as the other quantum-paper runs (Adam beta1 0.9 / beta2 0.999 / eps 1e-8,
lr 1e-4, no weight decay, batch 128, 200 batches/epoch x 20,000 epochs, ReduceLROnPlateau,
transformer with d_model 64, 4 heads of dim 16, 4 layers, d_mlp 256, ReLU, LayerNorm, seed 42)
EXCEPT n_ctx = 14.

ctx 14 keeps the EXACT enumerate+multinomial pipeline (true P(w)), but enumerates in PURE NUMPY
(no jax). A vectorised breadth-first walk over RRXOR's reachable prefixes builds every reachable
length-(n_ctx+1) window with its exact probability P(w), plus the exact myopic (windowed) entropy
-- only reachable nodes are built (RRXOR forbids most transitions), so this is fast and small
(~5k windows at ctx 14). This deliberately avoids the simplexity/jax mixed-state tree, which
hangs on the many-core training pod under JAX_PLATFORMS=cpu. Training then uses the enumerated
table: torch.multinomial batches, FULL-enumeration probability-weighted validation, exact
optimal-loss line.

ALL checkpoints are kept (not overwritten): epoch 0 (random init), a dense early schedule to
capture the emergence, then every 100 epochs, each to reproduction/models/rrxor_checkpoints/
rrxor_epoch_<NNNNN>.pt; the final model is also written to reproduction/models/rrxor_transformer.pt.
The checkpoint dir is cleared at startup so a reused volume can't mix in stale checkpoints.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

PROCESS = "rrxor"
RRXOR_PARAMS = {"p1": 0.5, "p2": 0.5}
CONTEXT_LEN = 14                       # longer context (was 8), still enumerable
BATCH_SIZE = 128
BATCHES_PER_EPOCH = 200
N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000  # paper value = 20,000
LEARNING_RATE = 1e-4
SEED = 42
LOG_EVERY_EPOCHS = 25

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR.parent / "models"          # fresh models/ (ctx-8 run archived as models_paper1_adam_ctx8)
FIG_DIR = OUT_DIR.parent / "figures"
CKPT = MODEL_DIR / "rrxor_transformer.pt"       # final model (canonical name)
CKPT_DIR = MODEL_DIR / "rrxor_checkpoints"      # ALL kept checkpoints, tagged by epoch


def rrxor_tensor(p1=0.5, p2=0.5):
    """RRXOR joint transition tensor T[token, from_state, to_state] (numpy). States S,0,1,T,F=0..4;
    stationary = [2,1,1,1,1]/6. (Reproduces simplexity.rrxor.)"""
    T = np.zeros((2, 5, 5), dtype=np.float64)
    T[0, 0, 1] = p1;     T[1, 0, 2] = 1 - p1
    T[0, 1, 4] = p2;     T[1, 1, 3] = 1 - p2
    T[0, 2, 3] = p2;     T[1, 2, 4] = 1 - p2
    T[1, 3, 0] = 1.0;    T[0, 4, 0] = 1.0
    return T


def enumerate_windows(T_np, stationary, n_ctx):
    """Vectorised BFS over reachable prefixes -> (windows (N, n_ctx+1) int64, probs (N,) sum 1,
    optimal_loss). Builds only reachable (P(w)>0) nodes; computes exact myopic entropy en route."""
    window_len = n_ctx + 1
    vocab = T_np.shape[0]
    seqs = np.zeros((1, 0), dtype=np.int64)
    beliefs = stationary.reshape(1, -1).astype(np.float64)
    probs = np.array([1.0])
    ent_per_ctx = []
    for L in range(window_len + 1):
        if 1 <= L <= n_ctx:                                    # myopic entropy at context length L
            nd = np.einsum("mi,xij->mx", beliefs, T_np)
            ndn = nd / np.clip(nd.sum(1, keepdims=True), 1e-12, None)
            H = -(ndn * np.log(np.clip(ndn, 1e-12, None))).sum(1)
            ent_per_ctx.append(float((probs * H).sum()))
        if L == window_len:
            break
        child = np.einsum("mi,xij->mxj", beliefs, T_np)        # (M, vocab, nstates) unnorm child beliefs
        pcond = child.sum(2)                                   # (M, vocab) P(token | belief)
        mi, xi = np.where(pcond > 1e-12)
        probs = probs[mi] * pcond[mi, xi]
        beliefs = child[mi, xi] / pcond[mi, xi][:, None]
        seqs = np.concatenate([seqs[mi], xi[:, None]], axis=1)
    return seqs, probs / probs.sum(), float(np.mean(ent_per_ctx))


def checkpoint_epochs(n_epochs):
    early = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75]
    return {e for e in early if e <= n_epochs} | set(range(100, n_epochs + 1, 100)) | {n_epochs}


def build_model(vocab_size: int, device: str) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        d_model=64, d_head=16, n_heads=4, n_layers=4, d_mlp=256, n_ctx=CONTEXT_LEN,
        d_vocab=vocab_size, act_fn="relu", normalization_type="LN", seed=SEED, device=device,
    )
    return HookedTransformer(cfg)


def save_ckpt(model, losses, epochs_done, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model.cfg.to_dict(), "losses": losses,
         "rrxor_params": RRXOR_PARAMS, "context_len": CONTEXT_LEN,
         "epochs_done": epochs_done, "step": epochs_done * BATCHES_PER_EPOCH},
        path,
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for old in CKPT_DIR.glob("rrxor_epoch_*.pt"):   # clear stale checkpoints (reused volume)
        old.unlink()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_np = rrxor_tensor(**RRXOR_PARAMS)
    stationary = np.array([2.0, 1.0, 1.0, 1.0, 1.0]) / 6.0
    windows_np, probs_np, optimal_loss = enumerate_windows(T_np, stationary, CONTEXT_LEN)
    vocab = T_np.shape[0]
    sequences = torch.from_numpy(windows_np).to(device)
    probs = torch.from_numpy(probs_np).float().to(device)
    x_val, y_val = sequences[:, :-1], sequences[:, 1:]   # full enumerated validation table

    total_steps = N_EPOCHS * BATCHES_PER_EPOCH
    print(f"device: {device}  ctx: {CONTEXT_LEN}  epochs: {N_EPOCHS}  steps: {total_steps}  "
          f"windows: {sequences.shape[0]}  optimal loss: {optimal_loss:.4f}", flush=True)

    model = build_model(vocab, device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}  vocab: {vocab}", flush=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1000, cooldown=200, threshold=1e-6
    )
    gen = torch.Generator(device=device).manual_seed(SEED)

    targets = checkpoint_epochs(N_EPOCHS)
    save_ckpt(model, [], 0, CKPT_DIR / "rrxor_epoch_00000.pt")   # random init

    epoch_losses: list[float] = []
    for epoch in range(N_EPOCHS):
        model.train()
        run = torch.zeros((), device=device)
        for _ in range(BATCHES_PER_EPOCH):
            idx = torch.multinomial(probs, BATCH_SIZE, replacement=True, generator=gen)
            w = sequences[idx]
            x, y = w[:, :-1], w[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run += loss.detach()
        mean_train = (run / BATCHES_PER_EPOCH).item()
        epoch_losses.append(mean_train)

        # Full-enumeration probability-weighted validation (their validate_epoch_all).
        model.eval()
        with torch.no_grad():
            ce = F.cross_entropy(
                model(x_val).reshape(-1, vocab), y_val.reshape(-1), reduction="none"
            ).reshape(x_val.shape[0], -1)
            weighted = ce * probs.unsqueeze(1)
            scheduler.step(weighted.mean())
            val_loss = weighted.sum(dim=0).mean().item()

        done = epoch + 1
        if done in targets:
            save_ckpt(model, epoch_losses, done, CKPT_DIR / f"rrxor_epoch_{done:05d}.pt")
        if epoch % LOG_EVERY_EPOCHS == 0 or done == N_EPOCHS:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:6d}  step {done * BATCHES_PER_EPOCH:8d}  train {mean_train:.4f}  "
                  f"val {val_loss:.4f}  lr {lr:.2e}  (opt {optimal_loss:.4f})", flush=True)

    save_ckpt(model, epoch_losses, N_EPOCHS, CKPT)
    print(f"saved final model -> {CKPT}", flush=True)
    print(f"kept {len(targets)} checkpoints in {CKPT_DIR}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_losses, alpha=0.4, lw=0.7, label="train (epoch mean)")
    w = 50
    if len(epoch_losses) > w:
        smooth = np.convolve(epoch_losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(epoch_losses)), smooth, "r-", lw=1.5, label="smoothed")
    ax.axhline(optimal_loss, color="k", ls="--", lw=1, label=f"optimal {optimal_loss:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"RRXOR ctx={CONTEXT_LEN} (quantum-paper config) training (final {epoch_losses[-1]:.4f})")
    ax.legend()
    fig.savefig(FIG_DIR / "fig_rrxor_ctx14_training_loss.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
