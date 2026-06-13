# RunPod training automation

Train on a rented GPU with one command. Each run spins up a **fresh ephemeral pod**,
reuses dependencies from a **persistent network volume**, pulls results back to your
laptop, commits them to git, and **terminates the pod** so billing stops.

```
./runpod/train.sh 1000000      # train 1e6 steps on a GPU, then auto-terminate
```

## How it works

```
laptop  ──create pod (volume attached)──►  RunPod GPU pod (ephemeral)
        ──rsync reproduction/ up──────►     /workspace/repo
        ◄─tail log over SSH────────────     tmux: .venv/bin/python train_mess3.py
        ◄─rsync models+figures down───      /workspace/repo/reproduction/{models,figures}
        ──DELETE /pods/{id}────────────►     (terminated)
   git commit + push results

  network volume (persists)  ──►  /workspace/.venv, /workspace/simplexity, uv python+cache
```

The **network volume** is the trick behind "keep deps but terminate the pod":
terminating a pod wipes the container disk, but the volume survives and re-attaches
to the next pod at `/workspace`. The venv, uv-managed Python 3.12, and the uv cache all
live on it, so `train.sh` never reinstalls — it just runs.

## One-time setup

1. **Account + credit.** Sign up at https://console.runpod.io and add credit.

2. **SSH key on your account.** You already have `~/.ssh/id_ed25519.pub`. Add its
   contents under console → Settings → *SSH Public Keys*. (Generate one with
   `ssh-keygen -t ed25519` if missing.)

3. **API key.** Console → Settings → *API Keys* → create one (read/write).

4. **Config.**
   ```
   cp runpod/config.env.example runpod/config.env
   # edit runpod/config.env: paste RUNPOD_API_KEY; confirm RUNPOD_GPU_TYPE / RUNPOD_IMAGE
   ```
   `config.env` is gitignored (holds your secret key). Leave `RUNPOD_VOLUME_ID` blank.

5. **Bootstrap the volume** (creates the network volume + installs all deps, ~10–15 min once):
   ```
   ./runpod/setup.sh
   ```
   This writes the new `RUNPOD_VOLUME_ID` back into `config.env`. Re-run it only when
   simplexity's dependencies change.

## Daily use

```
./runpod/train.sh            # 1,000,000 steps (paper default)
./runpod/train.sh 50000      # a quick smoke run
```

When it finishes, `reproduction/models/mess3_transformer.pt` and
`reproduction/figures/*.png` are updated locally and (if `AUTO_PUSH=true`) committed
and pushed. Then regenerate any analysis figures locally, e.g.:

```
python reproduction/fig03_residual_belief_geometry.py
```

## Notes / gotchas

- **Always terminates.** The scripts trap exit/Ctrl-C and `DELETE` the pod, so you
  won't leak billing. The only standing cost between runs is the volume
  (~`VOLUME_SIZE_GB` × $0.07/GB-month ≈ $3.50/mo for 50 GB).
- **SSH drops are tolerated.** Training runs in `tmux` on the pod, so a transient network
  blip during the tail loop won't kill it — the loop just retries. **Ctrl-C, however,
  terminates the pod** (via the EXIT trap, by design). If you want to leave a long run
  unattended, start it and let it run to completion rather than backgrounding the script.
- **GPU availability.** A network volume pins the datacenter (`RUNPOD_DATACENTER`). If
  your chosen GPU isn't available there, pick a different `RUNPOD_GPU_TYPE` or recreate
  the volume in another DC.
- **Image tag.** `RUNPOD_IMAGE` only needs a recent CUDA-12 driver (uv installs torch/jax
  itself). If create fails on the image, copy a current tag from the console's RunPod
  PyTorch template.
- **Verify the venv interactively** (optional): the volume also works with VSCode
  Remote-SSH — connect to a pod, open `/workspace`, select `/workspace/.venv/bin/python`.
```
