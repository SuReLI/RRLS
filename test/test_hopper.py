from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

hopper_env = gym.make("robust-hopper")


@pytest.mark.parametrize("env", [hopper_env])
def test_hopper_change_params(env):
    desired_worldfriction = 3.0
    desired_torsomass = 4.0
    desired_thighmass = 5.0

    env.unwrapped.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        thighmass=desired_thighmass,
    )

    assert np.array_equal(
        env.get_wrapper_attr("model").geom_friction[0, 0],
        np.array(desired_worldfriction),
    )
    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[2] == desired_thighmass
    assert {k: v for k, v in env.unwrapped.get_params().items() if v is not None} == {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "thighmass": desired_thighmass,
    }
