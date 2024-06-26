from __future__ import annotations

from copy import deepcopy
from typing import Annotated

import gymnasium as gym
import numpy as np
import pytest

import rrls

envs = [
    gym.make("rrls/robust-ant-v0"),
    gym.make("rrls/robust-halfcheetah-v0"),
    gym.make("rrls/robust-hopper-v0"),
    gym.make("rrls/robust-invertedpendulum-v0"),
    gym.make("rrls/robust-humanoidstandup-v0"),
    gym.make("rrls/robust-walker-v0"),
    gym.make("rrls/force-ant-v0"),
    gym.make("rrls/force-halfcheetah-v0"),
    gym.make("rrls/force-hopper-v0"),
    gym.make("rrls/force-invertedpendulum-v0"),
    gym.make("rrls/force-humanoidstandup-v0"),
    gym.make("rrls/force-walker-v0"),
]

bounds = [
    rrls.envs.AntParamsBound.THREE_DIM.value,
    rrls.envs.HalfCheetahParamsBound.THREE_DIM.value,
    rrls.envs.HopperParamsBound.THREE_DIM.value,
    rrls.envs.InvertedPendulumParamsBound.TWO_DIM.value,
    rrls.envs.HumanoidStandupParamsBound.THREE_DIM.value,
    rrls.envs.Walker2dParamsBound.THREE_DIM.value,
    rrls.envs.AntParamsBound.RARL.value,
    rrls.envs.HalfCheetahParamsBound.RARL.value,
    rrls.envs.HopperParamsBound.RARL.value,
    rrls.envs.InvertedPendulumParamsBound.RARL.value,
    rrls.envs.HumanoidStandupParamsBound.RARL.value,
    rrls.envs.Walker2dParamsBound.RARL.value,
]

envs_and_bounds = zip(envs, bounds)


@pytest.mark.parametrize(
    "env, bounds",
    zip(envs, bounds),
)
def test_change_params_is_effective(env, bounds: dict[str, Annotated[tuple[float], 2]]):
    action_high = env.action_space.high
    for param, interval in bounds.items():
        for value in interval:
            done, truncated = False, False
            obs, _ = env.reset(seed=0)
            env_copied = deepcopy(env)
            obs_copied, _ = env_copied.reset(seed=0)
            while not done and not truncated:
                env_copied = deepcopy(env)
                env_copied.set_params(**{param: value})
                obs_copied, _, _, _, _ = env_copied.step(action_high)
                obs, _, done, truncated, _ = env.step(action_high)
                assert not np.array_equal(obs_copied, obs)
