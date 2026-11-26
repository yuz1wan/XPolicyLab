import jax
import jax.numpy as jnp

from openpi.models import pir2_schedule


def test_staircase_has_clean_front_and_noisy_tail():
    tau = pir2_schedule.staircase_t(jnp.array([2], dtype=jnp.int32), 16)
    assert jnp.all(tau[:, :2] == 0)
    assert jnp.all(tau[:, -2:] == 1)


def test_training_sample_masks_clean_front():
    actions = jnp.ones((2, 16, 3))
    noise = jnp.zeros_like(actions)
    x_t, timestep, _, mask = pir2_schedule.training_sample(
        jax.random.key(0),
        actions,
        noise,
        jnp.full((2,), 0.5),
        mode="staircase",
        d_max=2,
        flow_probability=0.0,
        jitter=0.0,
    )
    assert x_t.shape == actions.shape
    assert timestep.shape == (2, 16)
    assert mask.shape == (2, 16)
    assert jnp.all(x_t[~mask] == 1)
