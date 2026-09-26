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
TASK_YAM_0010_EEF = replace(
    TASK_YAM_0010, name="pi05_yam_task_0010_eef",
    repo_id="rhospolicy/task-yam-0010-eef", action_space="eef",
)
TASK_YAM_0010_EEF_60HZ = replace(
    TASK_YAM_0010_EEF, name="pi05_yam_task_0010_eef_60hz",
    repo_id="rhospolicy/task-yam-0010-eef-60hz", action_hz=60,
)
TASK_YAM_0008_EEF = replace(
    TASK_YAM_0008, name="pi05_yam_task_0008_eef",
    repo_id="rhospolicy/task-yam-0008-eef", action_space="eef",
)
TASK_YAM_0015_EEF = replace(
    TASK_YAM_0015, name="pi05_yam_task_0015_eef",
    repo_id="rhospolicy/task-yam-0015-eef", action_space="eef",
)
EEF_TASKS = {task.name: task for task in (
    TASK_YAM_0004_EEF, TASK_YAM_0004_EEF_STATE, TASK_YAM_0004_EEF_STATE_ROT6D,
    TASK_YAM_0006_EEF, TASK_YAM_0006_EEF_60HZ,
    TASK_YAM_0010_EEF, TASK_YAM_0010_EEF_60HZ,
    TASK_YAM_0008_EEF, TASK_YAM_0015_EEF,
)}
ALL_TASKS = {**TASKS, **EEF_TASKS}
