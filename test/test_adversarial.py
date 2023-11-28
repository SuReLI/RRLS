from __future__ import annotations

import gymnasium as gym
import pytest

import rrls  # noqa: F401

adversarial_envs = []
envs = gym.envs.registry  # pyright: ignore
for env in envs:
    if ("robust" in env) and ("adversarial" in env):
        adversarial_envs.append(gym.make(env))


@pytest.mark.parametrize("env", adversarial_envs)
def test_run_an_episode(env):
    done = False
    truncated = False
    _, _ = env.reset()
    while not done and not truncated:
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)


@pytest.mark.parametrize("env", adversarial_envs)
def test_adversarial_params_change(env):
    params_bound = env.get_wrapper_attr("params_bound")
    done = False
    truncated = False
    _, _ = env.reset()
    while not done and not truncated:
        action, action_nature = env.action_space.sample()
        _, _, done, truncated, _ = env.step((action, action_nature))

        action_formated = {  # type: ignore
            k: v for k, v in zip(params_bound.keys(), action_nature)
        }
        action_formated = _unnormalize_action_nature(action_formated, params_bound)  # type: ignore

        assert {
            k: v for k, v in env.get_params().items() if v is not None
        } == action_formated


def _unnormalize_action_nature(
    action_nature: dict[str, float], params_bound: dict[str, list[float]]
) -> dict[str, float]:
    action_nature_unnormalized = {}
    for k, v in action_nature.items():
        action_nature_unnormalized[k] = (
            params_bound[k][0]
            + ((v - (-1)) * (params_bound[k][1] - params_bound[k][0])) / 2  # type: ignore
        )
    return action_nature_unnormalized
