import jax

from openpi.models import pi0_config


def test_pi05_staircase_dummy_loss():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_horizon=16,
        pir2_mode="staircase",
        pir2_d_max=3,
    )
    model = config.create(jax.random.key(0))
    loss = model.compute_loss(jax.random.key(1), config.fake_obs(1), config.fake_act(1))
    assert loss.shape == (1, config.action_horizon)
