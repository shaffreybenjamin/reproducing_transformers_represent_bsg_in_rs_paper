# Why the Mess3 run cost ~$19, and how to make RRXOR cheap

## What happened
The model is tiny (143K params), but training ran for 1,000,000 steps where **each
step made its own `jax` data-generation call + host↔device transfer**. That
per-step pipeline — not the GPU — was the bottleneck: locally it was ~120 ms/step
(~8 steps/s). At that rate 1e6 steps ≈ **28–35 h**, and you were paying for an
(otherwise idle) GPU the whole time: ~30 h × ~$0.5–0.7/h ≈ **$19**.

The expensive part was **wall-clock hours of a mostly-idle GPU**, because the
workload was CPU-bound on data generation, not GPU-bound on the model.

## The fix that matters most: generate data in blocks (done in `train_rrxor.py`)
Generate `BLOCK_STEPS` worth of sequences in one on-device `jax` call and slice
minibatches out of it, instead of one call per step. This amortises the jax
dispatch and keeps the loop compute-bound.
- Measured locally (CPU): **8 → 20 steps/s** (2.6×).
- On a GPU the model forward/backward is effectively free, so the remaining cost
  is just the (now-amortised, vectorised) generation — the multi-hour run should
  drop to **well under an hour**.

`train_mess3.py` is deliberately left on the per-step path (its committed model +
progression checkpoints depend on that exact data order); use `train_rrxor.py` as
the template for new, cheap runs.

## Other levers (in priority order)
1. **Cheapest GPU is correct.** `RUNPOD_GPU_TYPE` is already cheapest-first
   (A4500/RTX2000Ada/L4/4090). After the block fix the model saturates any of
   them; do **not** pay for an A100/H100.
2. **Community vs Secure cloud (~2× price).** `lib.sh` uses `"cloudType": "SECURE"`.
   Community Cloud is ~half the price — but network volumes are typically
   Secure-Cloud/datacenter-bound, so switching may break the volume reuse this
   workflow relies on. Verify volume support before changing it.
3. **Fewer steps if exact 1e6 isn't required.** The belief geometry emerges well
   before 1e6 steps; checkpoint and stop on a loss plateau (CKPT_EVERY already
   writes periodically).
4. **Confirm the pod actually terminates.** `train.sh` calls `terminate_pod`
   after pulling results; check the RunPod console for stray running pods (a
   failed run or an interrupted `setup.sh` can leave one billing).

## Running RRXOR
```
./runpod/train.sh 1000000 train_rrxor.py
```
(pulls back `reproduction/models/rrxor_transformer.pt`).
