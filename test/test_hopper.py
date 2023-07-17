from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

env = gym.make("robust-hopper")


@pytest.mark.parametrize("hopper_env", [env])
def test_hopper_change_params(hopper_env):
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
        env.model.geom_friction[0, 0], np.array(desired_worldfriction)
    )
    assert env.model.body_mass[1] == desired_torsomass
    assert env.model.body_mass[2] == desired_thighmass
    print(env.get_params())
    assert env.get_params() == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "thighmass": desired_thighmass,
    }
