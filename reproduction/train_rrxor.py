"""Train a transformer on RRXOR with the paper's exact hyperparameters (App. A.6).

Same architecture/optimiser as the Mess3 run -- context window 10, d_model 64,
head dim 8, 1 head, 4 layers, MLP 256, ReLU, LayerNorm; SGD, batch 64, lr 0.01,
no weight decay, 1,000,000 steps -- only the process differs (RRXOR p1=p2=0.5,
vocab 2).

SPEED / COST: the model is tiny (143K params), so a per-step jax data-generation
call + host<->device transfer dominates wall-clock (it was ~120 ms/step, i.e. the
GPU sat idle). We instead generate data in large ON-DEVICE BLOCKS and slice them
into minibatches, amortising the jax dispatch ~BLOCK_STEPS-fold so the loop is
compute-bound. This is the single biggest lever on RunPod cost (see runpod/README).
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

RRXOR_PARAMS = {"p1": 0.5, "p2": 0.5}
CONTEXT_LEN = 10
NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000  # paper value = 1e6
BATCH_SIZE = 64
LEARNING_RATE = 0.01           # SGD
SEED = 42
LOG_EVERY = 500
CKPT_EVERY = 5_000
BLOCK_STEPS = 2_000            # generate this many steps of data per jax call (on device)
OPTIMAL_LOSS = float(2 * np.log(2) / 3)  # 0.4621 nats: 2 random + 1 deterministic token per RRXOR triplet

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
CKPT = MODEL_DIR / "rrxor_transformer.pt"


def build_model(vocab_size: int, device: str) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        d_model=64, d_head=8, n_heads=1, n_layers=4, d_mlp=256, n_ctx=CONTEXT_LEN,
        d_vocab=vocab_size, act_fn="relu", normalization_type="LN", seed=SEED, device=device,
    )
    return HookedTransformer(cfg)


def save_ckpt(model, losses, step):
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model.cfg.to_dict(), "losses": losses,
         "rrxor_params": RRXOR_PARAMS, "context_len": CONTEXT_LEN, "step": step},
        CKPT,
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  steps: {NUM_STEPS}  block_steps: {BLOCK_STEPS}  optimal loss: {OPTIMAL_LOSS:.4f}")

    hmm = build_hidden_markov_model("rrxor", RRXOR_PARAMS)
    model = build_model(hmm.vocab_size, device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}  vocab: {hmm.vocab_size}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    init_state = jnp.array(hmm.initial_state)
    key = jax.random.PRNGKey(SEED)

    losses: list[float] = []
    inp = lab = None
    model.train()
    for step in range(NUM_STEPS):
        b = step % BLOCK_STEPS
        if b == 0:  # generate a fresh on-device block of BLOCK_STEPS*BATCH sequences
            key, batch_key = jax.random.split(key)
            n = BLOCK_STEPS * BATCH_SIZE
            gen_states = jnp.repeat(init_state[None, :], n, axis=0)
            _, inp, lab = generate_data_batch(gen_states, hmm, n, CONTEXT_LEN + 1, batch_key, device=device)
            inp = inp.long().to(device)
            lab = lab.long().to(device)
        x = inp[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        y = lab[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            recent = np.mean(losses[-LOG_EVERY:])
            print(f"step {step:7d}  loss {loss.item():.4f}  recent_mean {recent:.4f}  (opt {OPTIMAL_LOSS:.4f})", flush=True)
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
    ax.axhline(OPTIMAL_LOSS, color="k", ls="--", lw=1, label=f"optimal {OPTIMAL_LOSS:.4f}")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"RRXOR (paper config) training (final {losses[-1]:.4f})")
    ax.legend()
    fig.savefig(FIG_DIR / "fig_rrxor_training_loss.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
