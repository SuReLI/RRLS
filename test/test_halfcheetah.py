from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import rrls  # noqa: F401

halfcheetah_env = gym.make("rrls/robust-halfcheetah-v0")


@pytest.mark.parametrize("env", [halfcheetah_env])
def test_halfcheetah_change_params(env):
    desired_worldfriction = 3.0
    desired_torsomass = 4.0
    desired_backthighmass = 5.0

    env.set_params(
        worldfriction=desired_worldfriction,
        torsomass=desired_torsomass,
        backthighmass=desired_backthighmass,
    )

    assert np.array_equal(
        env.unwrapped.model.geom_friction[:, 0], np.array([desired_worldfriction] * 9)
    )
    assert env.unwrapped.model.body_mass[1] == desired_torsomass
    assert env.unwrapped.model.body_mass[2] == desired_backthighmass

    expected_values = {
        "worldfriction": desired_worldfriction,
        "torsomass": desired_torsomass,
        "backthighmass": desired_backthighmass,
    }
    # Filter env.get_params() to only include keys that are in expected_values and have non-None values
    filtered_env_params = {
        k: v
        for k, v in env.get_params().items()
        if k in expected_values and v is not None
    }
    assert filtered_env_params == expected_values
