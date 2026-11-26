"""JAX PI-R2 schedules using openpi's t=0 clean, t=1 noise convention."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def staircase_t(slide_steps: jax.Array, horizon: int, *, dtype=jnp.float32) -> jax.Array:
    slide_steps = jnp.asarray(slide_steps, dtype=jnp.int32)
    positions = jnp.arange(horizon)[None, :]
    d = slide_steps[:, None]
    interior = (positions - d).astype(dtype) / (horizon - 2 * d).astype(dtype)
    return jnp.where(positions < d, 0.0, jnp.where(positions >= horizon - d, 1.0, interior)).astype(dtype)


def training_sample(
    rng: jax.Array,
    actions: jax.Array,
    noise: jax.Array,
    shared_time: jax.Array,
    *,
    mode: str,
    d_max: int,
    flow_probability: float,
    jitter: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return x_t, per-position time, target velocity, and active loss mask."""
    d_rng, jitter_rng, flow_rng = jax.random.split(rng, 3)
    batch, horizon = actions.shape[:2]
    d = jax.random.randint(d_rng, (batch,), 1, d_max + 1)
    positions = jnp.arange(horizon)[None, :]
    front = positions < d[:, None]

    if mode == "rtc":
        timestep = jnp.broadcast_to(shared_time[:, None], (batch, horizon))
        standard_flow = jnp.zeros((batch,), dtype=bool)
    elif mode == "staircase":
        timestep = staircase_t(d, horizon, dtype=actions.dtype)
        if jitter:
            timestep = jnp.clip(
                timestep + jax.random.uniform(jitter_rng, timestep.shape, minval=-jitter, maxval=jitter),
                0,
                1,
            )
        standard_flow = jax.random.uniform(flow_rng, (batch,)) < flow_probability
        timestep = jnp.where(standard_flow[:, None], shared_time[:, None], timestep)
    else:
        raise ValueError(f"Unsupported PI-R2 mode: {mode}")

    active_front = front & ~standard_flow[:, None]
    x_t = timestep[..., None] * noise + (1 - timestep[..., None]) * actions
    x_t = jnp.where(active_front[..., None], actions, x_t)
    return x_t, timestep, noise - actions, ~active_front
