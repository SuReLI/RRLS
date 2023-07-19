from __future__ import annotations

import gymnasium as gym
import pytest

env = gym.make("robust-ant")


@pytest.mark.parametrize("ant_env", [env])
def test_ant_change_params(ant_env):
    desired_torsomass = 3.0
    desired_frontleftlegmass = 4.0
    desired_frontrightlegmass = 5.0

    env.unwrapped.set_params(
        torsomass=desired_torsomass,
        frontleftlegmass=desired_frontleftlegmass,
        frontrightlegmass=desired_frontrightlegmass,
    )
    env.unwrapped.change_params()

    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[2] == desired_frontleftlegmass
    assert env.unwrapped.model.body_mass[5] == desired_frontrightlegmass
    # assert env.model.body_mass[4] == desired_frontrightlegmass

    assert {k: v for k, v in env.unwrapped.get_params().items() if v is not None} == {
        "torsomass": desired_torsomass,
        "frontleftlegmass": desired_frontleftlegmass,
        "frontrightlegmass": desired_frontrightlegmass,
    }
