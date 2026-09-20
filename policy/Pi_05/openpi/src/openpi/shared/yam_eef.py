"""ABC-130k chunk-relative EEF geometry, without the reference data-loader dependencies.

Adapted from HarmonicRhOS data/eef_actions.py at a9360e360971ce5b5be69a2258b4c50ad478a3e7.
Poses are xyz + quaternion xyzw; each arm's action is local xyz, principal rotvec,
and aperture delta, all anchored to the observation at the start of the chunk.
"""

import math
import numpy as np

EEF_ACTION_DIM = 7
EEF_GRIPPER_INDEX = 6

def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert one 3x3 rotation matrix to canonical xyzw quaternion form."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape [3,3]")
    quaternion = np.asarray(
        [
            math.copysign(
                0.5 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])),
                matrix[2, 1] - matrix[1, 2],
            ),
            math.copysign(
                0.5 * math.sqrt(max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])),
                matrix[0, 2] - matrix[2, 0],
            ),
            math.copysign(
                0.5 * math.sqrt(max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])),
                matrix[1, 0] - matrix[0, 1],
            ),
            0.5 * math.sqrt(max(0.0, 1.0 + np.trace(matrix))),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("rotation matrix produced a degenerate quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions with arbitrary leading dimensions."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError("quaternion must end in four xyzw values")
    value = value / np.linalg.norm(value, axis=-1, keepdims=True).clip(1e-12)
    x, y, z, w = np.moveaxis(value, -1, 0)
    result = np.empty((*value.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result.astype(np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quaternion_inverse(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result[..., :3] *= -1.0
    return result


def _quaternion_to_rotvec(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-12)
    quaternion = np.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, quaternion[..., 3:4].clip(0.0))
    scale = np.where(vector_norm > 1e-7, angle / vector_norm.clip(1e-12), 2.0)
    return (vector * scale).astype(np.float32)


def _rotvec_to_quaternion(value: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(value, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    scale = np.where(angle > 1e-7, np.sin(angle / 2.0) / angle.clip(1e-12), 0.5)
    vector = rotvec * scale
    quaternion = np.concatenate((vector, np.cos(angle / 2.0)), axis=-1)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-12)
    return quaternion.astype(np.float32)


def _rotate_vectors(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(quaternion)
    return np.einsum("...ij,...j->...i", rotation, vectors).astype(np.float32)


def encode_chunk_relative_eef_actions(
    action_poses: np.ndarray,
    action_grippers: np.ndarray,
    reference_pose: np.ndarray,
    reference_gripper: np.ndarray | float,
) -> np.ndarray:
    """Encode one arm relative to its chunk-start observed pose and gripper."""

    actions = np.asarray(action_poses, dtype=np.float32)
    grippers = np.asarray(action_grippers, dtype=np.float32)
    reference = np.asarray(reference_pose, dtype=np.float32)
    gripper_reference = np.asarray(reference_gripper, dtype=np.float32)
    if actions.shape[-1] != 7 or grippers.shape != actions.shape[:-1]:
        raise ValueError("single-arm action poses/grippers must end in [7] and scalar")
    if reference.shape[-1] != 7:
        raise ValueError("single-arm reference pose must end in [7]")
    while reference.ndim < actions.ndim:
        reference = np.expand_dims(reference, axis=-2)
    while gripper_reference.ndim < grippers.ndim:
        gripper_reference = np.expand_dims(gripper_reference, axis=-1)
    inverse = _quaternion_inverse(reference[..., 3:7])
    translation = _rotate_vectors(inverse, actions[..., :3] - reference[..., :3])
    rotation = _quaternion_to_rotvec(_quaternion_multiply(inverse, actions[..., 3:7]))
    encoded = np.empty((*actions.shape[:-1], EEF_ACTION_DIM), dtype=np.float32)
    encoded[..., :3] = translation
    encoded[..., 3:6] = rotation
    encoded[..., EEF_GRIPPER_INDEX] = grippers - gripper_reference
    return encoded


def decode_chunk_relative_eef_actions(
    relative_actions: np.ndarray,
    reference_pose: np.ndarray,
    reference_gripper: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover one arm's world-frame pose and absolute gripper targets."""

    actions = np.asarray(relative_actions, dtype=np.float32)
    reference = np.asarray(reference_pose, dtype=np.float32)
    gripper_reference = np.asarray(reference_gripper, dtype=np.float32)
    if actions.shape[-1] != EEF_ACTION_DIM or reference.shape[-1] != 7:
        raise ValueError("relative actions/reference poses have incompatible shapes")
    while reference.ndim < actions.ndim:
        reference = np.expand_dims(reference, axis=-2)
    while gripper_reference.ndim < actions.ndim - 1:
        gripper_reference = np.expand_dims(gripper_reference, axis=-1)
    world_translation = reference[..., :3] + _rotate_vectors(reference[..., 3:7], actions[..., :3])
    world_rotation = _quaternion_multiply(reference[..., 3:7], _rotvec_to_quaternion(actions[..., 3:6]))
    poses = np.concatenate((world_translation, world_rotation), axis=-1).astype(np.float32)
    grippers = actions[..., EEF_GRIPPER_INDEX] + gripper_reference
    return poses, grippers.astype(np.float32)


def encode_bimanual(action_poses, reference_poses):
    """[... H, 16] pose+aperture pairs and [... 16] observed pairs → [... H, 14]."""
    actions = np.asarray(action_poses, dtype=np.float32)
    reference = np.asarray(reference_poses, dtype=np.float32)
    if actions.shape[-1] != 16 or reference.shape[-1] != 16:
        raise ValueError("Expected two xyz/xyzw/aperture poses (16 dimensions)")
    if not np.isfinite(actions).all() or not np.isfinite(reference).all():
        raise ValueError("EEF poses must be finite")
    return np.concatenate([
        encode_chunk_relative_eef_actions(
            actions[..., start:start+7], actions[..., start+7],
            reference[..., start:start+7], reference[..., start+7],
        )
        for start in (0, 8)
    ], axis=-1)


def decode_bimanual(relative_actions, reference_poses):
    """Recover absolute pose+aperture targets, never joint motor commands."""
    actions = np.asarray(relative_actions, dtype=np.float32)
    reference = np.asarray(reference_poses, dtype=np.float32)
    if actions.shape[-1] != 14 or reference.shape[-1] != 16:
        raise ValueError("Expected 14 relative actions and 16 reference pose values")
    outputs = []
    for arm in range(2):
        poses, grippers = decode_chunk_relative_eef_actions(
            actions[..., arm*7:(arm+1)*7],
            reference[..., arm*8:arm*8+7], reference[..., arm*8+7],
        )
        outputs.extend([poses, grippers[..., None]])
    return np.concatenate(outputs, axis=-1)
