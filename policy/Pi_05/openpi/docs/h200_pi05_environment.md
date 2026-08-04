# Pi0.5 full fine-tuning on H200

This project has a dedicated H200 overlay copied from the JAX 0.6.2 environment
validated by `openpi_multidata/docs/H200_NEW_JAX_EXPERIMENT_GUIDE.md`.

## Environment

The environment and its Python runtime are both on the shared RhOSPolicy disk:

```text
XPolicyLab/policy/Pi_05/openpi/.python-h200-jax/runtime/bin/python3.11
XPolicyLab/policy/Pi_05/openpi/.venv-h200-jax/bin/python
```

Pinned runtime versions:

```text
Python 3.11.14
JAX / jaxlib / CUDA plugin / PJRT 0.6.2
CUDA runtime 12.8
cuDNN 9.10.2.21
cuBLAS 12.8.4.1
NumPy 1.26.4
SciPy 1.15.3
ml-dtypes 0.5.3
Flax 0.10.2
PyTorch 2.8.0+cu128
LeRobot 0.4.4
datasets 4.8.5
huggingface-hub 0.35.3
PyArrow 24.0.0
PyAV 15.1.0
```

Do not run `uv sync`, `uv run`, or an automatic dependency repair against
`.venv-h200-jax`. The repository lock pins JAX 0.5.3 and would silently destroy
the H200-compatible overlay. Invoke its Python executable directly.

The environment's editable `openpi` and `openpi-client` paths have been rewritten
to this Pi_05 checkout. `src/openpi/models/siglip.py` remains on the original
`nn.Conv` patch extraction implementation.

## Offline full fine-tuning

The launcher defaults to `CUDA_VISIBLE_DEVICES=0,1`, requires exactly two H200
GPUs, verifies JAX/JAXLIB 0.6.2 and the offline
LeRobot data stack, checks that local dataset metadata exists, removes proxy and
XLA/cuDNN diagnostic variables, forces Hugging Face offline mode, disables JAX
preallocation, and uses only shared dataset/assets/weight/checkpoint paths:

```bash
cd XPolicyLab/policy/Pi_05
bash train_yam_h200.sh
```

The YAM config performs full fine-tuning on two devices with `fsdp_devices=1`
(two-way data parallelism without parameter sharding), a global batch size of
64, 30,000 steps, checkpoint saves every 5,000 steps, and durable checkpoint
retention every 10,000 steps.

The default experiment name contains a timestamp. To specify a name and training
overrides:

```bash
bash train_yam_h200.sh h200_jax062_pi05_full_bz32_s50000_YYYYmmdd_HHMMSS \
  --batch-size=32 \
  --num-train-steps=50000 \
  --log-interval=10
```

Use a new experiment name for a new run. For an existing checkpoint, reuse its
name and pass `--resume`; do not combine `--resume` and `--overwrite`.

The H200 source environment originally contained LeRobot 0.1.0. That release
queries Hugging Face dataset refs before opening this repository's v3 dataset,
so strict offline mode failed with `OfflineModeIsEnabled` even though the files
were present. The data-only packages above were overlaid from a mirror without
dependencies; the validated JAX 0.6.2/CUDA stack was left unchanged.

## Preflight on the H200 container

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
  --format=csv,noheader

XPolicyLab/policy/Pi_05/openpi/.venv-h200-jax/bin/python -c \
  'import jax,jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())'

grep -n -A8 'Patch extraction' \
  XPolicyLab/policy/Pi_05/openpi/src/openpi/models/siglip.py
```

Expected: an NVIDIA H200 with compute capability 9.0, JAX/JAXLIB 0.6.2, a CUDA
device, and `nn.Conv` in the patch extraction block.
