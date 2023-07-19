from __future__ import annotations

import gymnasium as gym
import pytest

env = gym.make("robust-invertedpendulum")


@pytest.mark.parametrize("pendulum_env", [env])
def test_pendulum_change_params(pendulum_env):
    desired_polemass = 3.0
    desired_cartmass = 4.0

    env.unwrapped.set_params(
        polemass=desired_polemass,
        cartmass=desired_cartmass,
    )
    env.unwrapped.change_params()

    assert env.unwrapped.model.body_mass[1] == desired_cartmass
    assert env.unwrapped.model.body_mass[2] == desired_polemass

    assert env.unwrapped.get_params() == {
        "polemass": desired_polemass,
        "cartmass": desired_cartmass,
    }
