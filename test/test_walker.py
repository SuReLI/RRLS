from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

walker_env = gym.make("robust-walker")


@pytest.mark.parametrize("env", [walker_env])
def test_walker_change_params(env):
    desired_worldfriction = 3.0
    desired_torsomass = 4.0
    desired_thighmass = 5.0

    env.unwrapped.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        thighmass=desired_thighmass,
    )

    assert np.array_equal(
        env.get_wrapper_attr("model").geom_friction[:, 0],
        np.array([3.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.9]),
    )
    assert env.get_wrapper_attr("model").body_mass[1] == desired_torsomass
    assert env.get_wrapper_attr("model").body_mass[2] == desired_thighmass

    assert {k: v for k, v in env.unwrapped.get_params().items() if v is not None} == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "thighmass": desired_thighmass,
    }
