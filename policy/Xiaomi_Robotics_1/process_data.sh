#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAT_DOC="${POLICY_DIR}/post_training/xr1/docs/data_format.md"

if [[ ! -f "${FORMAT_DOC}" ]]; then
    echo "Xiaomi-Robotics-1 post-training submodule is not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 2
fi

echo "The official trainer consumes one JSON annotation and three videos per episode."
echo "Format: ${FORMAT_DOC}"
echo "Robot-specific conversion belongs in the integrating robot repository."
