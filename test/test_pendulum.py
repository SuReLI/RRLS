from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

pendulum_env = gym.make("rrls/robust-invertedpendulum-v0")


@pytest.mark.parametrize("env", [pendulum_env])
def test_pendulum_change_params(env):
    desired_polemass = 3.0
    desired_cartmass = 4.0

    env.set_params(
        polemass=desired_polemass,
        cartmass=desired_cartmass,
    )

    assert env.unwrapped.model.body_mass[1] == desired_cartmass
    assert env.unwrapped.model.body_mass[2] == desired_polemass

    expected_values = {
        "polemass": desired_polemass,
        "cartmass": desired_cartmass,
    }
    # Filter env.get_params() to only include keys that are in expected_values and have non-None values
    filtered_env_params = {
        k: v
        for k, v in env.get_params().items()
        if k in expected_values and v is not None
    }
    assert filtered_env_params == expected_values
