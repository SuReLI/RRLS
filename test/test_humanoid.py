from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

humanoid_env = gym.make("rrls/robust-humanoidstandup-v0")


@pytest.mark.parametrize("env", [humanoid_env])
def test_humanoid_change_params(env):
    desired_torsomass = 3.0
    desired_leftthighmass = 4.0
    desired_rightfootmass = 5.0

    env.set_params(
        torsomass=desired_torsomass,
        leftthighmass=desired_leftthighmass,
        rightfootmass=desired_rightfootmass,
    )

    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[7] == desired_leftthighmass
    assert env.unwrapped.model.body_mass[6] == desired_rightfootmass

    expected_values = {
        "torsomass": desired_torsomass,
        "leftthighmass": desired_leftthighmass,
        "rightfootmass": desired_rightfootmass,
    }
    # Filter env.get_params() to only include keys that are in expected_values and have non-None values
    filtered_env_params = {
        k: v
        for k, v in env.get_params().items()
        if k in expected_values and v is not None
    }
    assert filtered_env_params == expected_values
