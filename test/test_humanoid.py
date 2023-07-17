from __future__ import annotations

import gymnasium as gym
import pytest

env = gym.make("robust-humanoidstandup")


@pytest.mark.parametrize("humanoid_env", [env])
def test_humanoid_change_params(humanoid_env):
    desired_torsomass = 3.0
    desired_leftthighmass = 4.0
    desired_rightfootmass = 5.0

    env.set_params(
        torsomass=desired_torsomass,
        leftthighmass=desired_leftthighmass,
        rightfootmass=desired_rightfootmass,
    )
    env.change_params()

    assert env.model.body_mass[1] == desired_torsomass
    assert env.model.body_mass[7] == desired_leftthighmass
    assert env.model.body_mass[6] == desired_rightfootmass

    assert env.get_params() == {
        "torsomass": desired_torsomass,
        "leftthighmass": desired_leftthighmass,
        "rightfootmass": desired_rightfootmass,
    }
