#!/usr/bin/env bash
# Run d-selection CV analysis on RunPod GPU
# Usage: ./runpod/analysis.sh [PROCESS]
#   ./runpod/analysis.sh              # all 7 processes (~45 min)
#   ./runpod/analysis.sh mess3        # just mess3 (~5-10 min)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

PROCESS="${1:-all}"   # 'all' or specific process name (mess3, rrxor, arch, wing, strata, fern, zero_one_random)

[[ -n "${RUNPOD_VOLUME_ID:-}" ]] || { echo "No RUNPOD_VOLUME_ID. Run ./runpod/setup.sh first." >&2; exit 1; }

create_pod
wait_running

# Ensure rsync/tmux (small, fast apt install)
echo ">> ensuring rsync/tmux on pod"
pod_ssh "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux" 2>/dev/null || true

# Push only reproduction/ code (not models/figures — reuse from volume)
echo ">> syncing code to pod"
pod_ssh "mkdir -p /workspace/repo/reproduction"
pod_rsync_up \
  --exclude '__pycache__' --exclude 'models' --exclude 'figures' --exclude '*.log' \
  "$REPO_ROOT/reproduction/" "root@$POD_IP:/workspace/repo/reproduction/"

# Run analysis
if [[ "$PROCESS" == "all" ]]; then
  echo ">> starting CV analysis (all 7 processes, ~45 min)"
  ANALYSIS_CMD="/workspace/.venv/bin/python -u reproduction/d_selection_cv.py"
else
  echo ">> starting CV analysis (process=$PROCESS, ~5-10 min)"
  ANALYSIS_CMD="/workspace/.venv/bin/python -u reproduction/d_selection_cv.py --process $PROCESS"
fi

pod_ssh "bash -se" <<EOF
set -e
$UV_ENV
rm -f /workspace/DONE
cd /workspace/repo
tmux new-session -d -s analysis \
  "cd /workspace/repo && $ANALYSIS_CMD > reproduction/cv_analysis.log 2>&1; echo \\\$? > /workspace/DONE"
EOF

# Stream progress (tail log every 30s until done)
echo ">> analysis running; tailing progress:"
while ! pod_ssh "test -f /workspace/DONE" 2>/dev/null; do
  pod_ssh "tail -n 5 /workspace/repo/reproduction/cv_analysis.log 2>/dev/null" || true
  sleep 30
done

EXIT_CODE="$(pod_ssh 'cat /workspace/DONE' 2>/dev/null || echo 1)"
echo ">> analysis finished (exit code $EXIT_CODE)"
pod_ssh "tail -n 10 /workspace/repo/reproduction/cv_analysis.log" || true

# Pull results back to laptop
echo ">> pulling results back"
mkdir -p "$REPO_ROOT/reproduction/d_selection_results"
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/d_selection_results/" \
  "$REPO_ROOT/reproduction/d_selection_results/" || true
pod_rsync_down "root@$POD_IP:/workspace/repo/reproduction/cv_analysis.log" \
  "$REPO_ROOT/reproduction/cv_analysis.log" || true

# Terminate pod (billing stops immediately)
terminate_pod

if [[ "$EXIT_CODE" != "0" ]]; then
  echo "Analysis failed (exit code $EXIT_CODE); see cv_analysis.log" >&2
  exit "$EXIT_CODE"
fi

echo ">> done. Results in reproduction/d_selection_results/"
