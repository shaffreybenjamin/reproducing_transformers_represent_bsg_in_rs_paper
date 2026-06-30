"""Train a transformer on mess3 with the QUANTUM-PAPER training setup.

Reproduces the training configuration of Riechers, Elliott & Shai, "Neural networks
leverage nominally quantum and post-quantum representations" (epsilon-transformers
`quantum-public`), Tables I-II + the mess3/Bloch experiment config:

  Model (their exact transformer):  n_ctx 8, d_model 64, 4 heads of dim 16
    (4*16 = 64), 4 layers, d_mlp 256 (= 4*d_model), ReLU, LayerNorm, seed 42.
  Optimizer:  Adam (beta1 0.9, beta2 0.999, eps 1e-8), lr 1e-4, no weight decay.
  Schedule:   200 batches/epoch x 20,000 epochs (= 4,000,000 steps), batch size 128.
  LR scheduler: ReduceLROnPlateau(factor 0.5, patience 1000, cooldown 200, thr 1e-6),
    stepped once per epoch on a freshly sampled validation batch.
  Checkpoint every 100 epochs; analysis is done on the final-layer residual stream.

Data is the exact-distribution sampler (exact_sampler.py): each batch is `BATCH_SIZE`
i.i.d. draws from the enumerated length-(n_ctx+1) mess3 window distribution. The
optimal-loss line is the exact myopic-entropy bound for this context length.

Long run: we checkpoint periodically so the analysis figures can be regenerated from
the latest checkpoint at any time, and so the run is resumable.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

from exact_sampler import build_process_data

PROCESS = "mess3"
MESS3_PARAMS = {"x": 0.05, "a": 0.85}  # paper values (Appendix A.3)
CONTEXT_LEN = 8                        # quantum-paper n_ctx
BATCH_SIZE = 128
BATCHES_PER_EPOCH = 200
N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000  # paper value = 20,000
LEARNING_RATE = 1e-4                   # Adam
SEED = 42
LOG_EVERY_EPOCHS = 25
CKPT_EVERY_EPOCHS = 100                # paper checkpoints every 100 epochs

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR.parent / "models"    # reproduction/models -- where the analysis scripts read
FIG_DIR = OUT_DIR.parent / "figures"
CKPT = MODEL_DIR / "mess3_transformer.pt"


def build_model(vocab_size: int, device: str) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_mlp=256,
        n_ctx=CONTEXT_LEN,
        d_vocab=vocab_size,
        act_fn="relu",
        normalization_type="LN",
        seed=SEED,
        device=device,
    )
    return HookedTransformer(cfg)


def save_ckpt(model, losses, step):
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model.cfg.to_dict(), "losses": losses,
         "mess3_params": MESS3_PARAMS, "context_len": CONTEXT_LEN, "step": step},
        CKPT,
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = build_process_data(PROCESS, MESS3_PARAMS, CONTEXT_LEN, device)
    total_steps = N_EPOCHS * BATCHES_PER_EPOCH
    print(f"device: {device}  epochs: {N_EPOCHS}  steps: {total_steps}  "
          f"windows: {data.sequences.shape[0]}  optimal loss: {data.optimal_loss:.4f}")

    model = build_model(data.vocab_size, device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}  vocab: {data.vocab_size}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1000, cooldown=200, threshold=1e-6
    )

    train_gen = torch.Generator(device=device).manual_seed(SEED)
    x_val, y_val, val_probs = data.validation_data()  # full enumerated table (their validate_epoch_all)

    epoch_losses: list[float] = []  # per-epoch mean train loss (light; for the loss curve)
    step = 0
    for epoch in range(N_EPOCHS):
        model.train()
        run = torch.zeros((), device=device)
        for _ in range(BATCHES_PER_EPOCH):
            x, y = data.sample_batch(BATCH_SIZE, generator=train_gen)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run += loss.detach()   # accumulate on-GPU; avoids a CPU<->GPU sync every step
            step += 1
        mean_train = (run / BATCHES_PER_EPOCH).item()   # one sync per epoch, not per step
        epoch_losses.append(mean_train)

        # Validation over the FULL enumerated distribution (their validate_epoch_all): probability-
        # weighted cross-entropy over every window. Step the LR scheduler on it, exactly as they do.
        model.eval()
        with torch.no_grad():
            ce = F.cross_entropy(
                model(x_val).reshape(-1, data.vocab_size), y_val.reshape(-1), reduction="none"
            ).reshape(x_val.shape[0], -1)
            weighted = ce * val_probs.unsqueeze(1)
            scheduler.step(weighted.mean())               # scheduler signal (their loss.mean())
            val_loss = weighted.sum(dim=0).mean().item()  # expected CE per position (for logging)

        if epoch % LOG_EVERY_EPOCHS == 0 or epoch == N_EPOCHS - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:6d}  step {step:8d}  train {mean_train:.4f}  val {val_loss:.4f}  "
                  f"lr {lr:.2e}  (opt {data.optimal_loss:.4f})", flush=True)
        if epoch > 0 and epoch % CKPT_EVERY_EPOCHS == 0:
            save_ckpt(model, epoch_losses, step)

    save_ckpt(model, epoch_losses, total_steps)
    print(f"saved model -> {CKPT}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_losses, alpha=0.4, lw=0.7, label="train (epoch mean)")
    w = 50
    if len(epoch_losses) > w:
        smooth = np.convolve(epoch_losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(epoch_losses)), smooth, "r-", lw=1.5, label="smoothed")
    ax.axhline(data.optimal_loss, color="k", ls="--", lw=1, label=f"optimal {data.optimal_loss:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"mess3 (quantum-paper config) training (final {epoch_losses[-1]:.4f})")
    ax.legend()
    fig.savefig(FIG_DIR / "fig02_training_loss_mess3.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
