"""Shared, dependency-free YAM task contracts for training and fast statistics."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class YamTask:
    name: str
    repo_id: str
    prompt: str
    num_train_steps: int = 50_000
    outcome: str = "success"
    action_horizon: int = 50
    batch_size: int = 64
    delta_mask: tuple[bool, ...] = (True,) * 6 + (False,) + (True,) * 6 + (False,)
    state_key: str = "observation.state"
    action_key: str = "action"

    @property
    def task_id(self) -> int:
        return int(self.name.rsplit("_", 1)[1])

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
TASKS = {task.name: task for task in (TASK_YAM_0004, TASK_YAM_0006, TASK_YAM_0010)}
