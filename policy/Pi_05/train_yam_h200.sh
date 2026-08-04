#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${POLICY_DIR}/openpi"
RHOS_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
PYTHON="${OPENPI_ROOT}/.venv-h200-jax/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing H200 environment: ${PYTHON}" >&2
  exit 1
fi

# This overlay is intentionally inconsistent with uv.lock. Never invoke uv for
# this environment; it would downgrade JAX 0.6.2 to the locked JAX 0.5.3 stack.
"${PYTHON}" - <<'PY'
import jax
import jaxlib

expected = "0.6.2"
if jax.__version__ != expected or jaxlib.__version__ != expected:
    raise SystemExit(
        f"H200 environment corrupted: expected jax/jaxlib {expected}, "
        f"got {jax.__version__}/{jaxlib.__version__}"
    )
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
gpu_names=()
for gpu_id in "${gpu_ids[@]}"; do
  gpu_names+=("$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_id}" | head -n1)")
done
gpu_name="$(IFS=' + '; echo "${gpu_names[*]}")"
if [[ "${OPENPI_ALLOW_NON_H200:-0}" != "1" ]]; then
  if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
    echo "Expected exactly two visible H200 GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  for name in "${gpu_names[@]}"; do
    if [[ "${name}" != *H200* ]]; then
      echo "Refusing dual-H200 full fine-tuning on GPU set: ${gpu_name}" >&2
      echo "Set OPENPI_ALLOW_NON_H200=1 only for a non-training environment check." >&2
      exit 1
    fi
  done
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset XLA_FLAGS CUDNN_FRONTEND_LOG_INFO CUDNN_FRONTEND_LOG_FILE
export NO_PROXY="*" no_proxy="*"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${RHOS_ROOT}/data}"
export OPENPI_YAM_DATA_REPO_ID="${OPENPI_YAM_DATA_REPO_ID:-yam-entong-fanya-box-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${RHOS_ROOT}/XPolicyLab/.cache/openpi}"
export OPENPI_YAM_ASSETS_BASE_DIR="${OPENPI_YAM_ASSETS_BASE_DIR:-${RHOS_ROOT}/assets/pi05}"
export OPENPI_YAM_CHECKPOINT_BASE_DIR="${OPENPI_YAM_CHECKPOINT_BASE_DIR:-${RHOS_ROOT}/checkpoints/pi05-h200}"
export OPENPI_PI05_BASE_PARAMS="${OPENPI_PI05_BASE_PARAMS:-${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params}"

"${PYTHON}" - <<'PY'
from importlib import metadata

expected = {
    "lerobot": "0.4.4",
    "datasets": "4.8.5",
    "huggingface-hub": "0.35.3",
    "pyarrow": "24.0.0",
    "av": "15.1.0",
}
actual = {package: metadata.version(package) for package in expected}
wrong = {package: (expected[package], version) for package, version in actual.items() if version != expected[package]}
if wrong:
    details = ", ".join(f"{package}: expected {want}, got {got}" for package, (want, got) in wrong.items())
    raise SystemExit(f"H200 offline data stack corrupted: {details}")
PY

dataset_dir="${HF_LEROBOT_HOME}/${OPENPI_YAM_DATA_REPO_ID}"
if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
  echo "Missing offline LeRobot dataset metadata: ${dataset_dir}/meta/info.json" >&2
  echo "HF_HUB_OFFLINE=1 is intentional; copy the complete dataset to this shared path." >&2
  exit 1
fi

if [[ ! -d "${OPENPI_PI05_BASE_PARAMS}" ]]; then
  echo "Missing offline pi0.5 base params: ${OPENPI_PI05_BASE_PARAMS}" >&2
  exit 1
fi

config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_yam_green_block_circle}"
exp_name="${1:-h200_jax062_pi05_full_$(date +%Y%m%d_%H%M%S)}"
shift || true

echo "[Pi_05/H200] gpus=${gpu_name}"
echo "[Pi_05/H200] python=${PYTHON}"
echo "[Pi_05/H200] config=${config_name}"
echo "[Pi_05/H200] exp_name=${exp_name}"
echo "[Pi_05/H200] data=${dataset_dir}"
echo "[Pi_05/H200] base_params=${OPENPI_PI05_BASE_PARAMS}"
echo "[Pi_05/H200] checkpoints=${OPENPI_YAM_CHECKPOINT_BASE_DIR}"

cd "${OPENPI_ROOT}"
exec "${PYTHON}" scripts/train.py "${config_name}" \
  --exp-name="${exp_name}" \
  "$@"
