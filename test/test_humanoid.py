from __future__ import annotations

import gymnasium as gym
import pytest

env = gym.make("robust-humanoidstandup")


@pytest.mark.parametrize("humanoid_env", [env])
def test_humanoid_change_params(humanoid_env):
    desired_torsomass = 3.0
    desired_leftthighmass = 4.0
    desired_rightfootmass = 5.0

    env.unwrapped.set_params(
        torsomass=desired_torsomass,
        leftthighmass=desired_leftthighmass,
        rightfootmass=desired_rightfootmass,
    )

    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[7] == desired_leftthighmass
    assert env.unwrapped.model.body_mass[6] == desired_rightfootmass

    assert {k: v for k, v in env.unwrapped.get_params().items() if v is not None} == {
        "torsomass": desired_torsomass,
        "leftthighmass": desired_leftthighmass,
        "rightfootmass": desired_rightfootmass,
    }
