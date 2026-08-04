# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${POLICY_DIR}/openpi"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

# Never inherit a container/system proxy. Resolve wheels through a configurable
# mirror and keep uv/Python caches on the shared disk.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*" no_proxy="*"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${UV_DEFAULT_INDEX}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XPOLICYLAB_ROOT}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${XPOLICYLAB_ROOT}/.cache/uv-python}"

echo "[Pi_05] OPENPI_ROOT=${OPENPI_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
uv python install 3.11
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[Pi_05] Installation finished."
echo "[Pi_05] Activate: source ${OPENPI_ROOT}/.venv/bin/activate"
