from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

env = gym.make("robust-walker")


@pytest.mark.parametrize("walker_env", [env])
def test_walker_change_params(walker_env):
    desired_worldfriction = 3.0
    desired_torsomass = 4.0
    desired_thighmass = 5.0

    env.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        thighmass=desired_thighmass,
    )
    env.change_params()

    assert np.array_equal(
        env.model.geom_friction[:, 0],
        np.array([3.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.9]),
    )
    assert env.model.body_mass[1] == desired_torsomass
    assert env.model.body_mass[2] == desired_thighmass

    assert {k: v for k, v in env.get_params().items() if v is not None} == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "thighmass": desired_thighmass,
    }
