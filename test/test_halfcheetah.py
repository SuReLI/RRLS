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

    env.unwrapped.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        backthighmass=desired_backthighmass,
    )

    assert np.array_equal(
        env.unwrapped.model.geom_friction[:, 0], np.array([desired_worldfriction] * 9)
    )
    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[2] == desired_backthighmass

    assert {k: v for k, v in env.unwrapped.get_params().items() if v is not None} == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "backthighmass": desired_backthighmass,
    }
