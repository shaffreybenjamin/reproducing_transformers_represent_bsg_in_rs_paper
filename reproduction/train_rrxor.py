"""Train a transformer on RRXOR — EXACT continuous-stream pipeline (paper / epsilon-transformers).

Matches epsilon-transformers' ProcessDataset (IterableDataset) + DataLoader:
  * the process emits ONE continuous stream of tokens,
  * chopped into consecutive, non-overlapping length-(n_ctx+1)=11 windows,
    each window -> (input = w[:-1], target = w[1:]),
  * batched 64 CONSECUTIVE windows per gradient step (no shuffle).
So the hidden state is continuous across windows (each window's start state is the
continuation of the previous one), exactly as in their pipeline.

Hyperparameters are the paper's App. A.6: context 10, d_model 64, head dim 8,
1 head, 4 layers, MLP 256, ReLU, LayerNorm; SGD, batch 64, lr 0.01, no weight
decay, 1,000,000 steps. RRXOR p1=p2=0.5 (vocab 2).

The stream is generated in memory BLOCKS via jax.lax.scan, carrying the hidden
state AND the rng key across blocks -- so it is a single unbroken stream (identical
data to generating it all at once) but produced in chunks so the tiny model's GPU
isn't stalled by per-step generation (the RunPod cost lever; see runpod/COST_NOTES.md).
"""

import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.transition_matrices import rrxor

RRXOR_PARAMS = {"p1": 0.5, "p2": 0.5}
CONTEXT_LEN = 10
NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000  # paper value = 1e6
BATCH_SIZE = 64
LEARNING_RATE = 0.01           # SGD
SEED = 42
LOG_EVERY = 500
CKPT_EVERY = 5_000
BLOCK_STEPS = 2_000            # generate this many steps' worth of the stream per jax.lax.scan
N_TOKENS = BLOCK_STEPS * BATCH_SIZE * (CONTEXT_LEN + 1)   # tokens per generated block
OPTIMAL_LOSS = float(2 * np.log(2) / 3)  # 0.4621 nats: 2 random + 1 deterministic token per RRXOR triplet

OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
CKPT = MODEL_DIR / "rrxor_transformer.pt"

# RRXOR transition tensor T[token, from, to]; from a state s, the joint distribution over
# (token, next_state) is T[:, s, :] (shape 2x5, sums to 1). States S,0,1,T,F.
_T = jnp.asarray(np.array(rrxor(**RRXOR_PARAMS)))
_NSTATES = _T.shape[1]
STATIONARY = jnp.array([2.0, 1.0, 1.0, 1.0, 1.0]) / 6.0


def _emit(carry, _):
    s, key = carry
    key, sub = jax.random.split(key)
    logp = jnp.log(_T[:, s, :].reshape(-1))     # log P(token*NS + next | s); forbidden -> -inf
    idx = jax.random.categorical(sub, logp)
    token = idx // _NSTATES
    nxt = idx % _NSTATES
    return (nxt, key), token


@partial(jax.jit, static_argnums=2)
def gen_stream_block(s0, key0, n):
    """Continue the single emission stream for n tokens; return (final_state, final_key, tokens)."""
    (s_final, key_final), tokens = jax.lax.scan(_emit, (s0, key0), None, length=n)
    return s_final, key_final, tokens.astype(jnp.int32)


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

    vocab_size = int(_T.shape[0])
    model = build_model(vocab_size, device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}  vocab: {vocab_size}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    key = jax.random.PRNGKey(SEED)
    key, sk = jax.random.split(key)
    state = jax.random.categorical(sk, jnp.log(STATIONARY))   # initial hidden state ~ stationary

    losses: list[float] = []
    inp_all = lab_all = None
    model.train()
    for step in range(NUM_STEPS):
        b = step % BLOCK_STEPS
        if b == 0:  # continue the single stream for one block, chop into consecutive windows
            state, key, toks = gen_stream_block(state, key, N_TOKENS)
            windows = np.asarray(toks).reshape(BLOCK_STEPS * BATCH_SIZE, CONTEXT_LEN + 1)
            w = torch.from_numpy(windows).long().to(device)
            inp_all, lab_all = w[:, :-1], w[:, 1:]
        x = inp_all[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        y = lab_all[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

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
    ax.set_title(f"RRXOR (paper config, continuous stream) training (final {losses[-1]:.4f})")
    ax.legend()
    fig.savefig(FIG_DIR / "fig_rrxor_training_loss.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
