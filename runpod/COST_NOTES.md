# RunPod cost notes

## History: why the old Mess3 run cost ~$19
The original `train_mess3.py` ran 1,000,000 steps where **each step made its own
`jax` data-generation call + host↔device transfer** (~120 ms/step, ~8 steps/s). That
per-step pipeline — not the GPU — was the bottleneck: 1e6 steps ≈ 28–35 h of an
otherwise-idle GPU ≈ ~$19. The cost was wall-clock hours of a mostly-idle GPU,
because the workload was CPU-bound on data generation, not GPU-bound on the model.

## Current pipeline (quantum-paper config) removes that bottleneck
All 7 training scripts now share `reproduction/train/exact_sampler.py`:

- **One-time** at startup: enumerate every length-(n_ctx+1) window with its exact
  probability via the simplexity mixed-state presentation (this is the only `jax` in
  the run) and compute the exact loss bound.
- **Per step** (4,000,000 times for a full run): a pure-torch `torch.multinomial`
  gather over the precomputed table + the tiny model's forward/backward. **No jax,
  no host↔device transfer per step.**

So the per-step CPU bottleneck is gone for *every* process. The run is now compute-
bound on a ~200K-param model, i.e. GPU-bound and fast. The full run is 4M steps (4×
the old 1M) but each step is cheap, so expect far less than the old per-model cost.

**Measure before committing to 7 full runs:** do a short timed run first, e.g.
`./runpod/train.sh 100 train/train_mess3.py` (100 epochs = 20,000 steps), read the
steps/s from the log, and extrapolate to 4,000,000 steps.

## Cost levers (in priority order)
1. **Cheapest GPU is correct.** `RUNPOD_GPU_TYPE` is cheapest-first
   (A4500/RTX2000Ada/L4/4090). The tiny model saturates any of them; do **not** pay
   for an A100/H100.
2. **Community vs Secure cloud (~2× price).** `lib.sh` uses `"cloudType": "SECURE"`.
   Community is ~half price, but network volumes are typically Secure-Cloud/
   datacenter-bound, so switching may break the volume reuse this workflow relies on.
3. **Fewer epochs if exact 20,000 isn't required.** Belief geometry emerges well
   before the end; the LR scheduler (ReduceLROnPlateau) decays once the loss
   plateaus, and checkpoints are written every 100 epochs, so you can stop early.
4. **Confirm the pod actually terminates.** `train.sh` calls `terminate_pod` after
   pulling results; check the RunPod console for stray running pods.

## Arg semantics (important)
`train.sh` takes **EPOCHS**, not steps: `./runpod/train.sh 20000 train/train_rrxor.py`
trains 20,000 epochs (= 4,000,000 steps) and pulls back
`reproduction/models/rrxor_transformer.pt`. Passing a step count like `1000000` would
request a million epochs — don't.
