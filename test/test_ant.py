from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

env = gym.make("rrls/robust-ant-v0")


@pytest.mark.parametrize("ant_env", [env])
def test_ant_change_params(ant_env):
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
    # assert env.get_wrapper_attr('model').body_mass[4] == desired_frontrightlegmass

    assert {k: v for k, v in env.get_params().items() if v is not None} == {
        "torsomass": desired_torsomass,
        "frontleftlegmass": desired_frontleftlegmass,
        "frontrightlegmass": desired_frontrightlegmass,
    }
