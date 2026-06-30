#!/usr/bin/env bash
# Train on a fresh RunPod GPU, pull results back, commit them, terminate the pod.
#
#   ./runpod/train.sh [EPOCHS] [SCRIPT...]   # default: 20000  train/train_mess3.py
#   ./runpod/train.sh 20000 train/train_rrxor.py
#   ./runpod/train.sh 20000 all              # all 7 processes CONCURRENTLY on one GPU
#   ./runpod/train.sh 20000 train/train_wing.py train/train_fern.py   # a chosen subset, concurrently
#
# Multiple scripts run as separate processes on the SAME single-GPU pod (the models are
# tiny, so they share the GPU). Each gets its own log + completion sentinel; we wait for
# all of them, then pull every checkpoint back at once.
#
# NOTE: the training scripts take an EPOCH count (200 batches/epoch); the quantum-paper
# full run is 20,000 epochs (= 4,000,000 steps). Do NOT pass a step count like 1000000
# here -- that would request a million epochs.
#
# Reuses the venv installed by setup.sh on the network volume (no reinstall).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EPOCHS="${1:-20000}"
SCRIPTS=("${@:2}")
[[ ${#SCRIPTS[@]} -eq 0 ]] && SCRIPTS=("train/train_mess3.py")
# "all" -> the full set of 7 processes
if [[ ${#SCRIPTS[@]} -eq 1 && "${SCRIPTS[0]}" == "all" ]]; then
  SCRIPTS=(train/train_mess3.py train/train_wing.py train/train_fern.py train/train_arch.py \
           train/train_strata.py train/train_zero_one_random.py train/train_rrxor.py)
fi
[[ -n "${RUNPOD_VOLUME_ID:-}" ]] || { echo "No RUNPOD_VOLUME_ID. Run ./runpod/setup.sh first." >&2; exit 1; }

create_pod
wait_running

# Guard: make sure setup.sh has run on this volume.
if ! pod_ssh "test -x /workspace/.venv/bin/python"; then
  echo "ERROR: no venv on the volume. Run ./runpod/setup.sh first." >&2
  exit 1
fi

# Ensure the tools we drive the pod with exist (the image omits them; container disk is
# ephemeral so this runs each time, but it's a small, fast apt install).
echo ">> ensuring rsync/tmux on pod"
pod_ssh "command -v rsync >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux)"

# 1. Push the latest training code to the pod (only reproduction/, deps come from the venv).
echo ">> syncing code to pod"
pod_ssh "mkdir -p /workspace/repo/reproduction"
pod_rsync_up \
  --exclude '__pycache__' --exclude 'models' --exclude 'figures' --exclude '*.log' \
  "$REPO_ROOT/reproduction/" "root@$POD_IP:/workspace/repo/reproduction/"

# 1b. Sync THIS working tree's simplexity package source over the pod's (editable) install.
# The pod's simplexity is a fresh clone of upstream, which lacks the locally-added processes
# (arch/fern/strata/wing). simplexity is installed with `pip install -e`, so overwriting the
# package source makes our processes importable without a reinstall.
echo ">> syncing local simplexity package to pod"
pod_rsync_up --exclude '__pycache__' \
  "$REPO_ROOT/simplexity/simplexity/" "root@$POD_IP:/workspace/simplexity/simplexity/"

# 2. Launch each training script in its own detached tmux session, writing a per-script
#    log and a per-script completion sentinel (/workspace/DONE_<name>).
# JAX_PLATFORMS=cpu pins jax (used only for the one-time MSP enumeration) to the CPU, so the
# GPU is left entirely to torch. Without it, many concurrent jax processes fail to init
# cuSOLVER on the shared GPU ("gpusolverDnCreate failed"); it also speeds startup.
echo ">> launching ${#SCRIPTS[@]} run(s) concurrently ($EPOCHS epochs each): ${SCRIPTS[*]}"
pod_ssh "rm -f /workspace/DONE_*"
for s in "${SCRIPTS[@]}"; do
  name="$(basename "$s" .py)"
  pod_ssh "cd /workspace/repo && tmux new-session -d -s '$name' \
    \"JAX_PLATFORMS=cpu /workspace/.venv/bin/python -u reproduction/$s $EPOCHS > reproduction/${name}.log 2>&1; echo \\\$? > /workspace/DONE_${name}\""
done

# 3. Poll until ALL scripts have written their sentinel, streaming each log's latest line.
# Training runs in tmux on the pod, so transient SSH drops during this poll loop don't kill
# it. (Ctrl-C here DOES terminate the pod via the EXIT trap — that's intentional.)
echo ">> ${#SCRIPTS[@]} run(s) in progress; tailing latest line of each:"
while true; do
  done_count="$(pod_ssh 'ls -1 /workspace/DONE_* 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
  pod_ssh "cd /workspace/repo/reproduction && for f in *.log; do printf '%-22s ' \"\$f\"; tail -n 1 \"\$f\" 2>/dev/null; done" || true
  [[ "$done_count" -ge "${#SCRIPTS[@]}" ]] && break
  sleep 30
done

# 4. Collect per-script exit codes from the sentinels.
EXIT_CODE=0
for s in "${SCRIPTS[@]}"; do
  name="$(basename "$s" .py)"
  rc="$(pod_ssh "cat /workspace/DONE_${name}" 2>/dev/null || echo 1)"
  echo ">> ${name} finished (exit code ${rc})"
  [[ "$rc" != "0" ]] && EXIT_CODE=1
done

# 5. Pull results back to the laptop (all checkpoints + figures + each log).
echo ">> pulling results back"
mkdir -p "$REPO_ROOT/reproduction/models" "$REPO_ROOT/reproduction/figures"
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/models/"  "$REPO_ROOT/reproduction/models/"  || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/figures/" "$REPO_ROOT/reproduction/figures/" || true
for s in "${SCRIPTS[@]}"; do
  name="$(basename "$s" .py)"
  pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/${name}.log" "$REPO_ROOT/reproduction/${name}.log" || true
done

# 6. Terminate now (don't wait for trap) so billing stops asap.
terminate_pod

if [[ "$EXIT_CODE" != "0" ]]; then
  echo "At least one run exited non-zero; skipping git commit. See reproduction/*.log." >&2
  exit "$EXIT_CODE"
fi

# 7. Commit results to git so others can use them without retraining.
if [[ "${AUTO_PUSH:-false}" == "true" ]]; then
  cd "$REPO_ROOT"
  git add reproduction/models reproduction/figures
  if git diff --cached --quiet; then
    echo ">> no result changes to commit"
  else
    git commit -m "RunPod training run: ${#SCRIPTS[@]} process(es), ${EPOCHS} epochs

Scripts: ${SCRIPTS[*]}
Trained on RunPod ($RUNPOD_GPU_TYPE), results pulled back automatically.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    git push && echo ">> results committed and pushed"
  fi
fi
echo ">> done."
