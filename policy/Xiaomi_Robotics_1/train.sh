#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR1_DIR="${POLICY_DIR}/post_training/xr1"

if [[ ! -f "${XR1_DIR}/scripts/train.sh" ]]; then
    echo "Xiaomi-Robotics-1 post-training submodule is not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 2
fi

cd "${XR1_DIR}"
exec bash scripts/train.sh "$@"
