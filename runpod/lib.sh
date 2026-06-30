#!/usr/bin/env bash
# Shared helpers for the RunPod automation scripts. Sourced by setup.sh and train.sh.
# Talks to the RunPod REST API (https://rest.runpod.io/v1) with curl, parses JSON with python3.

set -euo pipefail

API="https://rest.runpod.io/v1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# --- config -----------------------------------------------------------------
load_config() {
  local cfg="$HERE/config.env"
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: $cfg not found. Copy runpod/config.env.example to runpod/config.env and fill it in." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a; source "$cfg"; set +a
  : "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in config.env}"
  SSH_KEY="${SSH_KEY/#\~/$HOME}"
  SSH_PUBKEY="${SSH_PUBKEY/#\~/$HOME}"
}

# --- tiny JSON helpers (no jq dependency) -----------------------------------
# json_get '<python expression on `d`>'  <<< "$json"
json_get() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }

# --- REST calls -------------------------------------------------------------
api() { # api METHOD PATH [json-body]
  local method="$1" path="$2" body="${3:-}"
  local args=(-sS -X "$method" "$API$path"
    -H "Authorization: Bearer $RUNPOD_API_KEY"
    -H "Content-Type: application/json")
  [[ -n "$body" ]] && args+=(-d "$body")
  curl "${args[@]}"
}

# --- network volume ---------------------------------------------------------
create_volume() { # echoes new volume id
  local body
  body="$(python3 -c 'import json,sys; print(json.dumps({"name":sys.argv[1],"size":int(sys.argv[2]),"dataCenterId":sys.argv[3]}))' \
        "${RUNPOD_POD_NAME}-vol" "$VOLUME_SIZE_GB" "$RUNPOD_DATACENTER")"
  local resp; resp="$(api POST /networkvolumes "$body")"
  local id; id="$(json_get 'd.get("id","")' <<<"$resp" 2>/dev/null || true)"
  if [[ -z "$id" ]]; then echo "ERROR creating volume: $resp" >&2; exit 1; fi
  echo "$id"
}

# --- pod lifecycle ----------------------------------------------------------
POD_ID=""

create_pod() { # sets POD_ID
  local pubkey; pubkey="$(cat "$SSH_PUBKEY")"
  local name="${RUNPOD_POD_NAME}-$(date +%H%M%S)"
  local body
  body="$(python3 - "$name" "$RUNPOD_IMAGE" "$RUNPOD_GPU_TYPE" "$RUNPOD_CONTAINER_DISK_GB" \
                  "$RUNPOD_VOLUME_ID" "$pubkey" "${RUNPOD_ALLOWED_CUDA:-}" "${RUNPOD_GPU_COUNT:-1}" <<'PY'
import json, sys
name, image, gpu, disk, vol, pubkey, cuda, gpu_count = sys.argv[1:9]
gpu_ids = [s.strip() for s in gpu.split(",") if s.strip()]   # ordered fallback list
body = {
    "name": name,
    "imageName": image,
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuCount": int(gpu_count),
    "gpuTypeIds": gpu_ids,
    "gpuTypePriority": "availability",
    "containerDiskInGb": int(disk),
    "networkVolumeId": vol,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "env": {"PUBLIC_KEY": pubkey},
    "supportPublicIp": True,
}
cuda_versions = [s.strip() for s in cuda.split(",") if s.strip()]
if cuda_versions:   # only place us on hosts whose driver supports these CUDA versions
    body["allowedCudaVersions"] = cuda_versions
print(json.dumps(body))
PY
)"
  local resp; resp="$(api POST /pods "$body")"
  POD_ID="$(json_get 'd.get("id","")' <<<"$resp" 2>/dev/null || true)"
  if [[ -z "$POD_ID" ]]; then echo "ERROR creating pod: $resp" >&2; exit 1; fi
  echo ">> pod created: $POD_ID  (${RUNPOD_GPU_COUNT:-1}x gpu: $RUNPOD_GPU_TYPE)"
}

terminate_pod() {
  [[ -z "${POD_ID:-}" ]] && return 0
  echo ">> terminating pod $POD_ID"
  api DELETE "/pods/$POD_ID" >/dev/null 2>&1 || true
  POD_ID=""
}

# Always terminate on exit so we never leak billing.
trap terminate_pod EXIT INT TERM

POD_IP=""; POD_PORT=""
wait_running() { # polls until SSH (publicIp + port 22 mapping) is available; sets POD_IP/POD_PORT
  echo -n ">> waiting for pod to expose SSH "
  for _ in $(seq 1 120); do            # up to ~10 min
    local resp; resp="$(api GET "/pods/$POD_ID" || true)"
    POD_IP="$(json_get 'd.get("publicIp") or ""' <<<"$resp" 2>/dev/null || true)"
    POD_PORT="$(json_get 'd.get("portMappings",{}).get("22","") if d.get("portMappings") else ""' <<<"$resp" 2>/dev/null || true)"
    if [[ -n "$POD_IP" && -n "$POD_PORT" ]]; then
      echo " -> $POD_IP:$POD_PORT"
      break
    fi
    echo -n "."; sleep 5
  done
  if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo; echo "ERROR: pod never exposed a public SSH port. Check the datacenter supports public IP." >&2
    exit 1
  fi
  # wait for sshd to actually accept connections
  echo -n ">> waiting for sshd "
  for _ in $(seq 1 60); do
    if pod_ssh true 2>/dev/null; then echo " ok"; return 0; fi
    echo -n "."; sleep 5
  done
  echo; echo "ERROR: could not establish SSH to the pod." >&2; exit 1
}

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null
          -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=10)

pod_ssh() { ssh -i "$SSH_KEY" -p "$POD_PORT" "${SSH_OPTS[@]}" "root@$POD_IP" "$@"; }

# -rltz (recurse, links, times, compress) WITHOUT owner/group/perms: the network volume
# is a networked FS that forbids chown, and preserving ownership across hosts is pointless.
RSYNC_FLAGS=(-rltz --no-owner --no-group --no-perms --info=stats0)
pod_rsync_up()   { rsync "${RSYNC_FLAGS[@]}" -e "ssh -i $SSH_KEY -p $POD_PORT ${SSH_OPTS[*]}" "$@"; }
pod_rsync_down() { rsync "${RSYNC_FLAGS[@]}" -e "ssh -i $SSH_KEY -p $POD_PORT ${SSH_OPTS[*]}" "$@"; }

# env line that pins uv's python + cache onto the network volume so deps persist
UV_ENV='export UV_CACHE_DIR=/workspace/.uv/cache UV_PYTHON_INSTALL_DIR=/workspace/.uv/python PATH=$HOME/.local/bin:$PATH'
