"""Shared, dependency-free YAM task contracts for training and fast statistics."""

from dataclasses import dataclass, replace
import os
from pathlib import Path


@dataclass(frozen=True)
class YamTask:
    name: str
    repo_id: str
    prompt: str
    action_space: str = "joints"
    state_space: str = "joints"
    state_rotation: str = "quaternion"
    num_train_steps: int = 50_000
    outcome: str = "success"
    action_horizon: int = 50
    action_hz: int = 30
    batch_size: int = 64
    delta_mask: tuple[bool, ...] = (True,) * 6 + (False,) + (True,) * 6 + (False,)
    state_key: str = "observation.state"
    action_key: str = "action"
    project: str = "rhospolicy-pi05"
    entity: str = "HandZero"

    @property
    def task_id(self) -> int:
        return int(self.name.split("_task_", 1)[1].split("_", 1)[0])

    def resolved_repo_id(self) -> str:
        return os.environ.get(
            "OPENPI_LEROBOT_REPO_ID",
            os.environ.get("OPENPI_YAM_DATA_REPO_ID", os.environ.get("OPENPI_DATA_REPO_ID", self.repo_id)),
        )

    def assets_dir(self) -> Path:
        return (Path(os.environ.get("OPENPI_YAM_ASSETS_BASE_DIR", "./assets")) / self.name).resolve()


TASK_YAM_0004 = YamTask(
    name="pi05_yam_task_0004",
    repo_id="rhospolicy/task-yam-0004",
    prompt="Grab the ham sausage to the foam box",
)
TASK_YAM_0006 = YamTask(
    name="pi05_yam_task_0006",
    repo_id="rhospolicy/task-yam-0006",
    prompt="Pick up the small square and place it in the groove",
)
TASK_YAM_0010 = YamTask(
    name="pi05_yam_task_0010",
    repo_id="rhospolicy/task-yam-0010",
    prompt="One gripper holds the chewing gum bottle, while the other one removes the lid.",
)
TASK_YAM_0008 = YamTask(
    name="pi05_yam_task_0008",
    repo_id="rhospolicy/task-yam-0008",
    prompt="One gripper clamps the bottle, and the other gripper rotates to open the cap.",
)
TASK_YAM_0015 = YamTask(
    name="pi05_yam_task_0015",
    repo_id="rhospolicy/task-yam-0015",
    prompt="Put items in the drawer.",
)
TASKS = {task.name: task for task in (
    TASK_YAM_0004, TASK_YAM_0006, TASK_YAM_0008, TASK_YAM_0010, TASK_YAM_0015,
)}

# EEF variants deliberately have separate dataset/config/asset identities.
TASK_YAM_0004_EEF = replace(
    TASK_YAM_0004, name="pi05_yam_task_0004_eef",
    repo_id="rhospolicy/task-yam-0004-eef", action_space="eef",
)
TASK_YAM_0004_EEF_STATE = replace(
    TASK_YAM_0004_EEF, name="pi05_yam_task_0004_eef_state",
    state_space="eef", state_key="observation.eef_pose",
)
TASK_YAM_0004_EEF_STATE_ROT6D = replace(
    TASK_YAM_0004_EEF_STATE, name="pi05_yam_task_0004_eef_state_rot6d",
    state_rotation="rot6d",
)
TASK_YAM_0006_EEF = replace(
    TASK_YAM_0006, name="pi05_yam_task_0006_eef",
    repo_id="rhospolicy/task-yam-0006-eef", action_space="eef",
)
TASK_YAM_0006_EEF_60HZ = replace(
    TASK_YAM_0006_EEF, name="pi05_yam_task_0006_eef_60hz",
    repo_id="rhospolicy/task-yam-0006-eef-60hz", action_hz=60,
)
TASK_YAM_0006_EEF_STATE_ROT6D = replace(
    TASK_YAM_0006_EEF, name="pi05_yam_task_0006_eef_state_rot6d",
    state_space="eef", state_key="observation.eef_pose", state_rotation="rot6d",
)
TASK_YAM_0010_EEF = replace(
    TASK_YAM_0010, name="pi05_yam_task_0010_eef",
    repo_id="rhospolicy/task-yam-0010-eef", action_space="eef",
)
TASK_YAM_0010_EEF_60HZ = replace(
    TASK_YAM_0010_EEF, name="pi05_yam_task_0010_eef_60hz",
    repo_id="rhospolicy/task-yam-0010-eef-60hz", action_hz=60,
)
TASK_YAM_0010_EEF_STATE_ROT6D = replace(
    TASK_YAM_0010_EEF, name="pi05_yam_task_0010_eef_state_rot6d",
    state_space="eef", state_key="observation.eef_pose", state_rotation="rot6d",
)
TASK_YAM_0008_EEF = replace(
    TASK_YAM_0008, name="pi05_yam_task_0008_eef",
    repo_id="rhospolicy/task-yam-0008-eef", action_space="eef",
)
TASK_YAM_0008_EEF_60HZ = replace(
    TASK_YAM_0008_EEF, name="pi05_yam_task_0008_eef_60hz",
    repo_id="rhospolicy/task-yam-0008-eef-60hz", action_hz=60,
)
TASK_YAM_0015_EEF = replace(
    TASK_YAM_0015, name="pi05_yam_task_0015_eef",
    repo_id="rhospolicy/task-yam-0015-eef", action_space="eef",
)
EEF_TASKS = {task.name: task for task in (
    TASK_YAM_0004_EEF, TASK_YAM_0004_EEF_STATE, TASK_YAM_0004_EEF_STATE_ROT6D,
    TASK_YAM_0006_EEF, TASK_YAM_0006_EEF_60HZ, TASK_YAM_0006_EEF_STATE_ROT6D,
    TASK_YAM_0010_EEF, TASK_YAM_0010_EEF_60HZ, TASK_YAM_0010_EEF_STATE_ROT6D,
    TASK_YAM_0008_EEF, TASK_YAM_0008_EEF_60HZ, TASK_YAM_0015_EEF,
)}
ALL_TASKS = {**TASKS, **EEF_TASKS}


@dataclass(frozen=True)
class YamMultiTask:
    """Joint fine-tuning over many instructions held in one LeRobot dataset.

    The prompt of every frame comes from the dataset's own task row, so this record
    carries no single prompt and deliberately has no numeric `task_id`: the member
    tasks are listed in the conversion manifest and the versioned task table.
    """

    name: str
    repo_id: str
    task_table: str
    task_count: int
    dataset_tag: str
    # Joint runs use a 4x larger global batch than the single-task recipe. openpi's own
    # batch-256 configs (pi05_libero, pi05 DROID) pair that batch with peak_lr 5e-5, so
    # this follows the model authors rather than the batch-64 YAM value. The 10:1
    # peak-to-floor ratio is unchanged.
    peak_lr: float = 5e-5
    decay_lr: float = 5e-6
    project: str = "GM1k"
    entity: str = "rhos-ai"
    num_train_steps: int = 100_000
    batch_size: int = 256
    action_horizon: int = 50
    action_hz: int = 30
    action_space: str = "joints"
    state_space: str = "joints"
    outcome: str = "success"
    delta_mask: tuple[bool, ...] = (True,) * 6 + (False,) + (True,) * 6 + (False,)
    state_key: str = "observation.state"
    action_key: str = "action"

    def resolved_repo_id(self) -> str:
        return os.environ.get(
            "OPENPI_LEROBOT_REPO_ID",
            os.environ.get("OPENPI_YAM_DATA_REPO_ID", os.environ.get("OPENPI_DATA_REPO_ID", self.repo_id)),
        )

    def assets_dir(self) -> Path:
        return (Path(os.environ.get("OPENPI_YAM_ASSETS_BASE_DIR", "./assets")) / self.name).resolve()


# 100 TASK-TEL instructions selected on 2026-11-13; see the RhOSPolicy task table at
# configs/tasks/yam_tel_100/tasks.json for ids, English prompts and selection evidence.
TASK_YAM_TEL100 = YamMultiTask(
    name="pi05_yam_tel100_joint",
    repo_id="rhospolicy/yam-tel100-joint",
    task_table="configs/tasks/yam_tel_100/tasks.json",
    task_count=100,
    dataset_tag="task-tel-100",
)
MULTITASK_TASKS = {task.name: task for task in (TASK_YAM_TEL100,)}
