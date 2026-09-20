"""YAM EEF action adapter: configured proprioception, ABC-130k relative EEF targets."""

from dataclasses import dataclass
import numpy as np

from openpi.policies import aloha_policy
from openpi.shared.yam_eef import encode_bimanual


@dataclass(frozen=True)
class YamEEFInputs:
    def __call__(self, data: dict) -> dict:
        data = dict(data)
        if 'actions' in data:
            data['actions'] = encode_bimanual(data['actions'], data['eef_state'])
        # Only image layout and state passthrough are reused, no Aloha joint/gripper conversion.
        return aloha_policy.AlohaInputs(adapt_to_pi=False)(data)


@dataclass(frozen=True)
class YamEEFOutputs:
    def __call__(self, data: dict) -> dict:
        # Relative EEF outputs require reanchoring + IK before physical execution.
        # Deliberately do not apply Aloha AbsoluteActions (which would add joint angles).
        return {'actions': np.asarray(data['actions'])[..., :14]}
