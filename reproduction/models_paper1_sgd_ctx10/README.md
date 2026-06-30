# Archived checkpoints — paper 1 (BSG) training config

These are the original training runs reproducing **Shai et al., "Transformers
Represent Belief State Geometry in their Residual Stream"** (App. A.6 config):

- Optimizer **SGD**, lr 0.01, batch 64, 1,000,000 steps, no weight decay.
- Architecture: context window **10**, d_model 64, **1 head** of dim 8, 4 layers,
  d_mlp 256, ReLU, LayerNorm, seed 42.
- Data: continuous-stream / per-step HMM simulation.

They were moved here (from `reproduction/models/`) when the training scripts were
switched to the **quantum-paper** setup (Riechers, Elliott & Shai — Adam lr 1e-4,
batch 128, 200 batches/epoch × 20,000 epochs, ReduceLROnPlateau, context window 8,
4 heads of dim 16, exact-distribution sampling). New runs land in
`reproduction/models/`.

To analyse one of these old checkpoints, point the analysis script's `MODEL_DIR`
at this folder (or copy the `.pt` back into `reproduction/models/`).
