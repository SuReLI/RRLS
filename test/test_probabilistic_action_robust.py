from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

probabilistic_envs = []
envs = gym.envs.registry  # pyright: ignore
for env in envs:
    if ("robust" in env) and ("probabilistic" in env):
        probabilistic_envs.append(gym.make(env))


@pytest.mark.parametrize("env", probabilistic_envs)
def test_run_an_episode(env):
    done = False
    truncated = False
    _, _ = env.reset()
    while not done and not truncated:
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)
