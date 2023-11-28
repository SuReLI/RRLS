from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

dr_envs = []
envs = gym.envs.registry  # pyright: ignore
for env in envs:
    if ("robust" in env) and ("domain" in env):
        dr_envs.append(gym.make(env))


@pytest.mark.parametrize("env", dr_envs)
def test_run_multiple_episode(env):
    done = False
    truncated = False
    _, _ = env.reset()
    while not done and not truncated:
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)
