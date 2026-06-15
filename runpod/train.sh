#!/usr/bin/env bash
# Train on a fresh RunPod GPU, pull results back, commit them, terminate the pod.
#
#   ./runpod/train.sh [STEPS] [SCRIPT]   # default: 1000000  train_mess3.py
#   ./runpod/train.sh 1000000 train_rrxor.py
#
# Reuses the venv installed by setup.sh on the network volume (no reinstall).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

STEPS="${1:-1000000}"
SCRIPT="${2:-train_mess3.py}"   # which reproduction/ training script to run
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

# 2. Launch training in a detached tmux session; write a sentinel on completion.
echo ">> starting training: $SCRIPT  ($STEPS steps)"
pod_ssh "bash -se" <<EOF
set -e
$UV_ENV
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)
rm -f /workspace/DONE
cd /workspace/repo
tmux new-session -d -s train \
  "cd /workspace/repo && /workspace/.venv/bin/python -u reproduction/$SCRIPT $STEPS > reproduction/train_run.log 2>&1; echo \\\$? > /workspace/DONE"
EOF

# 3. Poll until done, streaming recent log lines.
# Training runs in tmux on the pod, so transient SSH drops during this poll loop don't
# kill it. (Ctrl-C here DOES terminate the pod via the EXIT trap — that's intentional.)
echo ">> training running; tailing progress:"
while ! pod_ssh "test -f /workspace/DONE" 2>/dev/null; do
  pod_ssh "tail -n 3 /workspace/repo/reproduction/train_run.log 2>/dev/null" || true
  sleep 30
done
EXIT_CODE="$(pod_ssh 'cat /workspace/DONE' 2>/dev/null || echo 1)"
echo ">> training finished (exit code $EXIT_CODE)"
pod_ssh "tail -n 5 /workspace/repo/reproduction/train_run.log" || true

# 4. Pull results back to the laptop.
echo ">> pulling results back"
mkdir -p "$REPO_ROOT/reproduction/models" "$REPO_ROOT/reproduction/figures"
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/models/"  "$REPO_ROOT/reproduction/models/"  || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/figures/" "$REPO_ROOT/reproduction/figures/" || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/train_run.log" "$REPO_ROOT/reproduction/train_run.log" || true

# 5. Terminate now (don't wait for trap) so billing stops asap.
terminate_pod

if [[ "$EXIT_CODE" != "0" ]]; then
  echo "Training exited non-zero ($EXIT_CODE); skipping git commit. See reproduction/train_run.log." >&2
  exit "$EXIT_CODE"
fi

# 6. Commit results to git so others can use them without retraining.
if [[ "${AUTO_PUSH:-false}" == "true" ]]; then
  cd "$REPO_ROOT"
  git add reproduction/models reproduction/figures
  if git diff --cached --quiet; then
    echo ">> no result changes to commit"
  else
    git commit -m "RunPod training run: ${STEPS} steps

Trained on RunPod ($RUNPOD_GPU_TYPE), results pulled back automatically.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    git push && echo ">> results committed and pushed"
  fi
fi
echo ">> done."
