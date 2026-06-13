#!/usr/bin/env bash
# One-time: create the network volume (if needed) and install all dependencies onto it.
# After this, train.sh spins up ephemeral pods that reuse the volume's venv — no reinstall.
#
#   ./runpod/setup.sh
#
# Safe to re-run: it only reinstalls deps (e.g. after editing simplexity's pyproject).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

# 1. Ensure a network volume exists (persists across pod termination -> keeps deps).
if [[ -z "${RUNPOD_VOLUME_ID:-}" ]]; then
  echo ">> no RUNPOD_VOLUME_ID set; creating a ${VOLUME_SIZE_GB}GB volume in $RUNPOD_DATACENTER"
  RUNPOD_VOLUME_ID="$(create_volume)"
  echo ">> created volume: $RUNPOD_VOLUME_ID"
  # persist it back into config.env so future runs reuse it
  if grep -q '^RUNPOD_VOLUME_ID=' "$HERE/config.env"; then
    sed -i "s|^RUNPOD_VOLUME_ID=.*|RUNPOD_VOLUME_ID=$RUNPOD_VOLUME_ID|" "$HERE/config.env"
  else
    echo "RUNPOD_VOLUME_ID=$RUNPOD_VOLUME_ID" >> "$HERE/config.env"
  fi
fi

# 2. Boot an ephemeral pod with the volume attached.
create_pod
wait_running

# 3. Install uv + python(3.12) + simplexity + cuda deps INTO the volume's venv.
echo ">> installing dependencies onto the volume (this is the slow one-time step)"
pod_ssh "bash -se" <<EOF
set -euo pipefail
$UV_ENV
mkdir -p /workspace/.uv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace
[ -d simplexity/.git ] || git clone https://github.com/Astera-org/simplexity.git
cd simplexity && git pull --ff-only || true
[ -x /workspace/.venv/bin/python ] || uv venv --python 3.12 /workspace/.venv  # reuse if it exists
uv pip install --python /workspace/.venv -e ".[cuda]"
# PyPI's default torch 2.9 wheel targets CUDA 13.0 (needs a rare 13.0-driver host). Pin the
# cu128 build so torch runs on the common CUDA 12.8 hosts (see RUNPOD_ALLOWED_CUDA).
uv pip install --python /workspace/.venv "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
echo "---- sanity ----"
/workspace/.venv/bin/python - <<'PY'
import torch, transformer_lens, jax, simplexity
print("torch", torch.__version__, "built-for-cuda", torch.version.cuda)
x = torch.zeros(1, device="cuda")          # raises loudly if the driver is too old
print("cuda tensor ok on:", x.device)
print("jax", jax.__version__, "devices", jax.devices())
print("simplexity ok")
PY
EOF

echo ">> setup complete. The venv lives on volume $RUNPOD_VOLUME_ID and survives termination."
echo ">> run a training job with:  ./runpod/train.sh 1000000"
# pod terminated by the EXIT trap
