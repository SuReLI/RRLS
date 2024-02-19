from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

ant_env = gym.make("rrls/robust-ant-v0")


@pytest.mark.parametrize("env", [ant_env])
def test_ant_change_params(env):
    desired_torsomass = 3.0
    desired_frontleftlegmass = 4.0
    desired_frontrightlegmass = 5.0

    env.set_params(
        torsomass=desired_torsomass,
        frontleftlegmass=desired_frontleftlegmass,
        frontrightlegmass=desired_frontrightlegmass,
    )

    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[2] == desired_frontleftlegmass
    assert env.unwrapped.model.body_mass[5] == desired_frontrightlegmass

    expected_values = {
        "torsomass": desired_torsomass,
        "frontleftlegmass": desired_frontleftlegmass,
        "frontrightlegmass": desired_frontrightlegmass,
    }
    # Filter env.get_params() to only include keys that are in expected_values and have non-None values
    filtered_env_params = {
        k: v
        for k, v in env.get_params().items()
        if k in expected_values and v is not None
    }
    assert filtered_env_params == expected_values
