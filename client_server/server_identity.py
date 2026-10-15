"""Read-only identity of a loaded policy, exposed through the generic CALL RPC."""
from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone


def model_identity(config, *, checkpoint=None, action_space="unknown", action_horizon=None):
    # Deliberate allowlist: deployment configs can also contain credentials.
    return {
        "schema_version": 1,
        "instance_id": uuid.uuid4().hex,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "policy_name": config.get("policy_name"),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "train_config": config.get("train_config_name"),
        "repo_id": config.get("repo_id"),
        "action_space": action_space,
        "action_horizon": action_horizon,
    }


def attach_model_identity(model, identity):
    snapshot = copy.deepcopy(identity)
    model.get_server_info = lambda: copy.deepcopy(snapshot)
    return model
