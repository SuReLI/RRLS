from __future__ import annotations

from typing import Annotated

import gymnasium as gym
import numpy as np

from rrlcs._interface import ModifiedParamsEnv


class DynamicAdversarial(gym.Wrapper):
    """
    The `DynamicAdversarial` class is a Gym Wrapper that enables dynamic
    and adversarial modifications to certain parameters of a `ModifiedParamsEnv`
    environment during an episode.

    It extends the action space of the base environment to include adversarial actions,
    which can modify the environment's parameters within specified bounds.

    The adversarial action is assumed to be a real-valued vector of the same length
    as the number of parameters, each value normalized to the range [-1, 1].

    On each step, the parameters are modified according to the adversarial action,
    and then the base environment's step function is called with the agent's action.
    The agent's reward is passed through unmodified, but the adversarial 'reward'
    (which is simply the negative of the agent's reward) is added to the info dictionary.

    The environment parameters are reset to their default values at the start of each episode.
    Args:
        env (ModifiedParamsEnv): The base environment, which must comply with the `ModifiedParams` protocol.
        params_bound (dict): Dictionary specifying the bounds for each parameter that can be modified.

    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(
            (
                env.action_space,
                gym.spaces.Box(
                    low=-1, high=1, shape=(len(params_bound),)
                ),  # adversarial action
            )
        )
        self.params_bound = params_bound
        self.defaut_params = env.get_params()
        self.env = env

    def step(self, action):
        """
        Steps the environment with the given agent and adversarial actions, modifying the parameters accordingly.

        Args:
            action (tuple): A tuple containing the agent's action and the adversarial action.

        Returns:
            Tuple: A tuple containing the new observation, the agent's reward, whether the episode has ended,
            whether the episode has been truncated, and additional info.
        """
        action_agent: np.ndarray  # type: ignore
        action_nature: np.ndarray  # type: ignore
        action_agent, action_nature = action

        # Apply nature action to the environment
        action_nature: dict[str, float] = {  # type: ignore
            k: v for k, v in zip(self.params_bound.keys(), action_nature)
        }
        unnormalized_action_nature = self._unnormalize_action_nature(action_nature)

        self.env.set_params(**unnormalized_action_nature)
        self.env.change_params()

        # Apply agent action to the environment
        obs, reward, terminated, truncated, info = self.env.step(action_agent)

        info.update(unnormalized_action_nature)
        info.update({"adversarial_reward": -reward})
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Resets the environment to its initial state, including resetting the parameters to their default values.

        Args:
            seed (int, optional): Seed for the environment's random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            Tuple: A tuple containing the initial observation and additional info.
        """
        self.env.set_params(**self.defaut_params)  # type: ignore
        self.env.change_params()  # type: ignore

        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def _unnormalize_action_nature(self, action_nature: dict[str, float]):
        action_nature_unnormalized = {}
        for k, v in action_nature.items():
            action_nature_unnormalized[k] = (
                self.params_bound[k][0]
                + ((v - (-1)) * (self.params_bound[k][1] - self.params_bound[k][0])) / 2  # type: ignore
            )
        return action_nature_unnormalized
