# Xiaomi_Robotics_1

**Contributor:** Xiaomi Corporation | **Paper:** Not released | **arXiv:** Not released | **Original code:** [`XiaomiRobotics/Xiaomi-Robotics-1`](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1).

`Xiaomi_Robotics_1` contains two deliberately separate integrations. The existing
`xiaomi_robotics_1/` tree remains the XPolicyLab inference adapter used by RoboDojo.
The official Xiaomi repository is tracked as the nested `post_training/` submodule
and supplies the XR-1 post-training implementation under `post_training/xr1/`.
Keeping that boundary avoids copying upstream training code and makes upstream
updates an explicit submodule-pointer change.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash install.sh
conda activate <policy_env>  # e.g. mibot (override the name via MIBOT_CONDA_ENV)
```

The installer creates the `mibot` conda environment with PyTorch 2.8, Flash Attention, and the other core dependencies. Read `INSTALLATION.md` for the manual install equivalent, checkpoint preparation, and smoke checks.

## Data Processing

The official trainer consumes one metadata JSON and three synchronized videos per
episode. Run `process_data.sh` to locate the pinned upstream format document.
Embodiment-specific conversion is intentionally owned by the robot integration
repository; for YAM, use RhOSPolicy's `yam_xiaomi_robotics_1 prepare` workflow.

## Model Assets

Download the inference checkpoint from the [RoboDojo official dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo); only the `ckpt/RoboDojo/Xiaomi_Robotics_1/` folder is needed:

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
mkdir -p checkpoints

hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "ckpt/RoboDojo/Xiaomi_Robotics_1/*" \
  --local-dir checkpoints/Xiaomi_Robotics_1 \
  --local-dir-use-symlinks False
```

At evaluation time the checkpoint is resolved from the `model_dir` field in `deploy.yml` (absolute path) or, when unset, from `checkpoints/<ckpt_name>/`.

## Training

Initialize nested submodules, prepare an official-format dataset and launch the
pinned official trainer through the stable XPolicyLab wrapper:

```bash
git submodule update --init --recursive
RESOURCE_GPU=1 bash train.sh \
  --config-dir /absolute/path/to/generated/configs \
  data=yam \
  model.params.pretrained=/absolute/path/to/model_states.pt \
  trainer.project=xiaomi-robotics-1 \
  trainer.exp_name=yam-posttrain
```

`train.sh` changes into `post_training/xr1/` before invoking the official script,
so it is safe to call from any working directory. Extra arguments are passed to
Hydra unchanged.

## Evaluation

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the downloaded official checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls Xiaomi_Robotics_1 arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `action_type` (`joint` or `ee`; default `ee`), `model_dir`, `action_max_length`, `action_length`, `state_token_length`, `input_length`, `crop_ratio`, `task_id` (selects normalization parameters from the training config), `vlm_processor_path` (HuggingFace repo id or local path; default `Qwen/Qwen3-VL-4B-Instruct`), `default_prompt`, and `eval_env`. The only script environment variable is `MIBOT_CONDA_ENV` (conda env name used by `install.sh`; default `mibot`).

## Notes

- The input state is always joint-space regardless of `action_type`; state and action use a 60-dim vector with 8 slots per arm (see the layout documented in `model.py`).
- The model predicts relative actions with respect to the current state; the adapter restores absolute actions, mapping end-effector rotations between the MiBot and simulator frames for `action_type=ee`.
