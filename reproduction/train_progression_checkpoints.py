"""Train mess3 and save SEPARATE checkpoints at chosen steps (for Figure 6 top row).

Identical config / seed / data order to train_mess3.py, so the trajectory matches
the canonical 1,000,000-step RunPod model (verified: losses[0]=1.2215 either way).
We therefore only need the early checkpoints here (the converged 1M model is reused
as the final point). Checkpoints go to models/progression/step_{N}.pt and this
script NEVER touches models/mess3_transformer.pt.

Usage: python train_progression_checkpoints.py [comma-sep steps]
  default steps: 100,1000,5000  (paper points are ~100,1000,4980,983000; the last
  is the existing 1M model).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch

MESS3_PARAMS = {"x": 0.05, "a": 0.85}
CONTEXT_LEN = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SEED = 42

OUT_DIR = Path(__file__).parent
CKPT_DIR = OUT_DIR / "models" / "progression"

CKPT_STEPS = sorted(int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["100", "1000", "5000"]))


def build_model(vocab_size, device):
    cfg = HookedTransformerConfig(
        d_model=64, d_head=8, n_heads=1, n_layers=4, d_mlp=256, n_ctx=CONTEXT_LEN,
        d_vocab=vocab_size, act_fn="relu", normalization_type="LN", seed=SEED, device=device,
    )
    return HookedTransformer(cfg)


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model("mess3", MESS3_PARAMS)
    model = build_model(hmm.vocab_size, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    init_state = jnp.array(hmm.initial_state)
    key = jax.random.PRNGKey(SEED)
    max_step = max(CKPT_STEPS)
    print(f"device {device}  saving checkpoints at steps {CKPT_STEPS}  (train to {max_step})")

    def save(step):
        path = CKPT_DIR / f"step_{step}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "cfg": model.cfg.to_dict(),
             "mess3_params": MESS3_PARAMS, "context_len": CONTEXT_LEN, "step": step},
            path,
        )
        print(f"  saved {path}", flush=True)

    model.train()
    losses = []
    for step in range(max_step + 1):
        if step in CKPT_STEPS:
            save(step)
        if step == max_step:
            break
        key, bk = jax.random.split(key)
        gs = jnp.repeat(init_state[None, :], BATCH_SIZE, axis=0)
        _, inputs, labels = generate_data_batch(gs, hmm, BATCH_SIZE, CONTEXT_LEN + 1, bk, device=device)
        logits = model(inputs.long())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.long().reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
