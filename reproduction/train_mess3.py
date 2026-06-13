"""Step 2 - Train a transformer on mess3 with the PAPER'S EXACT hyperparameters.

Appendix A.6 of Shai et al. (arXiv:2405.15943):
  context window 10, ReLU, head dimension 8, model dimension 64, 1 attention head
  in each of 4 layers, MLPs of dimension 256, causal masking, LayerNorm.
  Optimizer SGD, batch size 64, 1,000,000 steps, learning rate 0.01, no weight
  decay. Each batch = 64 sequences from mess3, initial hidden state from the
  stationary distribution (mess3's initial_state is already stationary = 1/3 each).
  Analysis is done on the final-layer residual stream, before LayerNorm/unembed.

Long run: we checkpoint periodically so the analysis figures can be regenerated
from the latest checkpoint at any time, and so the run is resumable.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch

MESS3_PARAMS = {"x": 0.05, "a": 0.85}  # paper values (Appendix A.3)
CONTEXT_LEN = 10
NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000  # paper value = 1e6
BATCH_SIZE = 64
LEARNING_RATE = 0.01           # SGD
SEED = 42
LOG_EVERY = 500
CKPT_EVERY = 5_000
OPTIMAL_LOSS = 0.7935          # myopic-entropy optimum for these params

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
CKPT = MODEL_DIR / "mess3_transformer.pt"


def build_model(vocab_size: int, device: str) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        d_model=64,
        d_head=8,
        n_heads=1,
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
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  steps: {NUM_STEPS}  optimal loss: {OPTIMAL_LOSS}")

    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model = build_model(hmm.vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    init_state = jnp.array(hmm.initial_state)  # stationary (1/3,1/3,1/3)
    key = jax.random.PRNGKey(SEED)

    losses: list[float] = []
    model.train()
    for step in range(NUM_STEPS):
        key, batch_key = jax.random.split(key)
        gen_states = jnp.repeat(init_state[None, :], BATCH_SIZE, axis=0)
        _, inputs, labels = generate_data_batch(
            gen_states, hmm, BATCH_SIZE, CONTEXT_LEN + 1, batch_key, device=device
        )
        inputs = inputs.long()
        labels = labels.long()

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            recent = np.mean(losses[-LOG_EVERY:])
            print(f"step {step:7d}  loss {loss.item():.4f}  recent_mean {recent:.4f}  (opt {OPTIMAL_LOSS})", flush=True)
        if step > 0 and step % CKPT_EVERY == 0:
            save_ckpt(model, losses, step)

    save_ckpt(model, losses, NUM_STEPS)
    print(f"saved model -> {CKPT}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, lw=0.5)
    w = 500
    if len(losses) > w:
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(losses)), smooth, "r-", lw=1.5, label="smoothed")
    ax.axhline(OPTIMAL_LOSS, color="k", ls="--", lw=1, label=f"optimal {OPTIMAL_LOSS}")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"mess3 (paper config) training (final {losses[-1]:.4f})")
    ax.legend()
    fig.savefig(FIG_DIR / "fig02_training_loss_mess3.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
