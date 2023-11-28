from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

import rrls  # noqa: F401

envs = [
    gym.make("rrls/robust-ant-v0"),
    gym.make("rrls/robust-halfcheetah-v0"),
    gym.make("rrls/robust-hopper-v0"),
    gym.make("rrls/robust-invertedpendulum-v0"),
    gym.make("rrls/robust-humanoidstandup-v0"),
    gym.make("rrls/robust-walker-v0"),
]


@pytest.mark.parametrize(
    "env",
    envs,
)
def test_change_params_is_effective(env):
    # The copied env should have the same state as the original env
    done = False
    truncated = False
    state_original, _ = env.reset(seed=0)
    copied_env = deepcopy(env)
    copied_state, _ = copied_env.reset(seed=0)
    assert np.array_equal(copied_state, state_original)
    step_number = 0
    while not done and not truncated:
        action = env.action_space.sample()
        copied_state, _, done_copied, truncated_copied, _ = copied_env.step(action)
        state_original, _, done, truncated, _ = env.step(action)
        assert np.array_equal(
            copied_state, state_original
        ), f" states are different at step_number: {step_number}"
        assert done_copied == done
        assert truncated_copied == truncated
        step_number += 1
