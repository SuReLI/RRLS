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

    assert env.get_params() == {
        "polemass": desired_polemass,
        "cartmass": desired_cartmass,
    }
