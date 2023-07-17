from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

env = gym.make("robust-halfcheetah")


@pytest.mark.parametrize("halfcheetah_env", [env])
def test_halfcheetah_change_params(halfcheetah_env):
    desired_worldfriction = 3.0
    desired_torsomass = 4.0
    desired_backthighmass = 5.0

    env.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        backthighmass=desired_backthighmass,
    )
    env.change_params()

    assert np.array_equal(
        env.model.geom_friction[:, 0], np.array([desired_worldfriction] * 9)
    )
    assert env.model.body_mass[1] == desired_torsomass
    assert env.model.body_mass[2] == desired_backthighmass

    assert env.get_params() == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "backthighmass": desired_backthighmass,
    }
