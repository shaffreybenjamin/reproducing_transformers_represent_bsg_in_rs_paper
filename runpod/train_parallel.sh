#!/usr/bin/env bash
# Multi-GPU training on ONE multi-GPU RunPod pod, mirroring epsilon-transformers'
# launcher_cuda_parallel.py: one training process per GPU, GPUs recycled as jobs finish.
#
#   ./runpod/train_parallel.sh [EPOCHS] [PROC...]   # default: 20000  all 7 processes
#   ./runpod/train_parallel.sh 20000 mess3 arch wing
#
# Set RUNPOD_GPU_COUNT in config.env to how many GPUs to rent on the single pod (the pod
# attaches your one existing network volume regardless of GPU count). The scheduler
# (reproduction/train/launch_parallel.py) handles any number of jobs on any number of GPUs.
#
# EPOCHS, not steps (200 batches/epoch). Reuses the venv from setup.sh on the volume.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EPOCHS="${1:-20000}"
shift 2>/dev/null || true
PROCS=("$@")   # process short-names (mess3 wing ...); empty -> launch_parallel.py runs all 7
[[ -n "${RUNPOD_VOLUME_ID:-}" ]] || { echo "No RUNPOD_VOLUME_ID. Run ./runpod/setup.sh first." >&2; exit 1; }

create_pod
wait_running

if ! pod_ssh "test -x /workspace/.venv/bin/python"; then
  echo "ERROR: no venv on the volume. Run ./runpod/setup.sh first." >&2; exit 1
fi

echo ">> ensuring rsync/tmux on pod"
pod_ssh "command -v rsync >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux)"

# 1. Sync training code + the local simplexity package (which carries the locally-added
#    processes arch/fern/strata/wing that upstream simplexity lacks).
echo ">> syncing code to pod"
pod_ssh "mkdir -p /workspace/repo/reproduction"
pod_rsync_up --exclude '__pycache__' --exclude 'models' --exclude 'figures' --exclude '*.log' \
  "$REPO_ROOT/reproduction/" "root@$POD_IP:/workspace/repo/reproduction/"
echo ">> syncing local simplexity package to pod"
pod_rsync_up --exclude '__pycache__' \
  "$REPO_ROOT/simplexity/simplexity/" "root@$POD_IP:/workspace/simplexity/simplexity/"

echo ">> GPUs visible on pod:"
pod_ssh "/workspace/.venv/bin/python -c 'import torch; print(torch.cuda.device_count())'"

# 2. Run the GPU-pool scheduler in a detached tmux session; sentinel on completion.
echo ">> launching scheduler (epochs=$EPOCHS  procs=${PROCS[*]:-ALL})"
pod_ssh "bash -se" <<EOF
set -e
rm -f /workspace/DONE
cd /workspace/repo
tmux new-session -d -s sched \
  "cd /workspace/repo && SLOTS_PER_GPU=${RUNPOD_SLOTS_PER_GPU:-1} /workspace/.venv/bin/python -u reproduction/train/launch_parallel.py $EPOCHS ${PROCS[*]} > reproduction/launch_parallel.log 2>&1; echo \\\$? > /workspace/DONE"
EOF

# 3. Poll until the scheduler finishes, tailing its log + the latest line of each job log.
echo ">> scheduler running; tailing progress:"
while ! pod_ssh "test -f /workspace/DONE" 2>/dev/null; do
  pod_ssh "tail -n 3 /workspace/repo/reproduction/launch_parallel.log 2>/dev/null; \
           cd /workspace/repo/reproduction/train_logs 2>/dev/null && for f in *.log; do printf '  %-12s ' \"\$f\"; tail -n 1 \"\$f\" 2>/dev/null; done" || true
  sleep 30
done
EXIT_CODE="$(pod_ssh 'cat /workspace/DONE' 2>/dev/null || echo 1)"
echo ">> scheduler finished (exit code $EXIT_CODE)"
pod_ssh "tail -n 12 /workspace/repo/reproduction/launch_parallel.log" || true

# 4. Pull results (all checkpoints + figures + per-job logs).
echo ">> pulling results back"
mkdir -p "$REPO_ROOT/reproduction/models" "$REPO_ROOT/reproduction/figures" "$REPO_ROOT/reproduction/train_logs"
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/models/"     "$REPO_ROOT/reproduction/models/"     || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/figures/"    "$REPO_ROOT/reproduction/figures/"    || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/train_logs/" "$REPO_ROOT/reproduction/train_logs/" || true

# 5. Terminate now so billing stops asap.
terminate_pod

if [[ "$EXIT_CODE" != "0" ]]; then
  echo "Scheduler exited non-zero ($EXIT_CODE); skipping git commit. See reproduction/train_logs/." >&2
  exit "$EXIT_CODE"
fi

# 6. Commit results.
if [[ "${AUTO_PUSH:-false}" == "true" ]]; then
  cd "$REPO_ROOT"
  git add reproduction/models reproduction/figures
  if git diff --cached --quiet; then
    echo ">> no result changes to commit"
  else
    git commit -m "RunPod multi-GPU training: ${EPOCHS} epochs (${RUNPOD_GPU_COUNT:-1} GPUs)

Processes: ${PROCS[*]:-all 7}
Ran one process per GPU via reproduction/train/launch_parallel.py (port of
epsilon-transformers' launcher_cuda_parallel.py).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    git push && echo ">> results committed and pushed"
  fi
fi
echo ">> done."
