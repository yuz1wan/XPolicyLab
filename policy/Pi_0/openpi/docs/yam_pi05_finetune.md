# Fine-tuning pi0.5 on the YAM green-block task

The checked-in training config is `pi05_yam_green_block_circle`. It performs full
fine-tuning from `gs://openpi-assets/checkpoints/pi05_base/params` on the LeRobot v3
dataset `rhospolicy/yam-entong-fanya-box-1`.

The config deliberately computes YAM-specific normalization statistics. YAM joint
zero positions are not assumed to use the standard Trossen ALOHA convention, so
`adapt_to_pi` is disabled. The 12 arm joint targets are converted from absolute
positions to deltas by `LeRobotAlohaDataConfig`; the two normalized gripper targets
remain absolute.

## Server setup

Clone RhOSPolicy and initialize only the policy submodule; the YAM control submodule
is not required on a training-only server:

```bash
git clone git@github.com:yuz1wan/RhOSPolicy.git
cd RhOSPolicy
git submodule update --init XPolicyLab
```

Copy the dataset so its directory matches the LeRobot repository ID:

```text
/data/lerobot/
└── rhospolicy/
    └── yam-entong-fanya-box-1/
        ├── data/
        ├── meta/
        └── videos/
```

Install the pinned OpenPI environment. This command bypasses configured proxies and
uses the Aliyun Python mirror for Torch/CUDA and other Python wheels, as required by
the project deployment policy:

```bash
cd XPolicyLab/policy/Pi_0/openpi
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='*' no_proxy='*' \
    UV_DEFAULT_INDEX='https://mirrors.aliyun.com/pypi/simple/' \
    uv --no-progress sync --frozen --group lerobot
```

The released pi0.5 checkpoint is hosted on Google Cloud Storage, so the server still
needs direct access to `gs://openpi-assets` when normalization or training first
loads the base weights/tokenizer.

## Normalize, then train

Use absolute paths for generated assets and checkpoints if they should live outside
the Git checkout:

```bash
export HF_LEROBOT_HOME=/data/lerobot
export OPENPI_YAM_DATA_REPO_ID=rhospolicy/yam-entong-fanya-box-1
export OPENPI_DATA_HOME=/data/openpi-cache
export OPENPI_YAM_ASSETS_BASE_DIR=/data/rhospolicy-pi05/assets
export OPENPI_YAM_CHECKPOINT_BASE_DIR=/data/rhospolicy-pi05/checkpoints

cd /path/to/RhOSPolicy/XPolicyLab/policy/Pi_0/openpi
uv run scripts/compute_norm_stats.py \
  --config-name pi05_yam_green_block_circle

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_yam_green_block_circle \
  --exp-name=green-block-circle \
  --overwrite
```

With the environment above, normalization statistics are written under
`/data/rhospolicy-pi05/assets/pi05_yam_green_block_circle/rhospolicy/yam-entong-fanya-box-1/`
and are automatically loaded by the subsequent training command. If the path
overrides are omitted, local `assets/` and `checkpoints/` directories are used; both
are ignored by this repository.

The default is a global batch size of 64, 20,000 steps, one FSDP device, checkpoints
every 1,000 steps, and retention every 5,000 steps. Full fine-tuning generally needs
more than 70 GB of accelerator memory. On a two-GPU server, enable model sharding and
keep the global batch divisible by the number of devices:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_yam_green_block_circle \
  --exp-name=green-block-circle \
  --fsdp-devices=2 \
  --batch-size=64 \
  --overwrite
```

To continue an existing run, replace `--overwrite` with `--resume`; never pass both.
The language instruction loaded from the LeRobot task table is:

```text
Place the green block inside the circle.
```
