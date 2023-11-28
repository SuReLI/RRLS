from __future__ import annotations

import gymnasium as gym
import numpy as np

from rrls._interface import ModifiedParamsEnv


class ProbabilisticActionRobust(gym.Wrapper):
    """
    An adversarial wrapper that introduces an adversarial action.

    The adversary perturbs the selected action based on a weight, alpha.
    The effect of this adversarial action is bounded by the environment's action space.

    Args:
        env (ModifiedParamsEnv): The base environment. Must adhere to the `ModifiedParams` protocol.
        alpha (float): Weight of the adversarial action. Should be in the range [0, 1].
                    A value of 0 implies no adversarial action, while a value of 1
                    indicates that the agent's action is wholly replaced by the adversarial action.

    References:
        - [1] [Action Robust Reinforcement Learning and Applications in Continuous Control](http://proceedings.mlr.press/v97/tessler19a/tessler19a.pdf)
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        alpha: float = 0.15,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(
            (
                env.action_space,
                env.action_space,  # adversarial action
            )
        )
        self.alpha = alpha
        self.env = env

    def step(self, action):
        """
        Steps the environment with the given agent and adversarial actions,the agent's action.

        Args:
            action (tuple): A tuple containing the agent's action and the adversarial action.

        Returns:
            Tuple: A tuple containing the new observation, the agent's reward, whether the episode has ended,
            whether the episode has been truncated, and additional info.
        """
        action_agent: np.ndarray  # type: ignore
        action_nature: np.ndarray  # type: ignore
        action_agent, action_nature = action

        blend_action = (1 - self.alpha) * action_agent + self.alpha * action_nature
        # Apply agent action to the environment
        obs, reward, terminated, truncated, info = self.env.step(blend_action)
        info.update({"adversarial action": action_nature})
        info.update({"agent action": action_agent})
        info.update({"adversarial_reward": -reward})
        return obs, reward, terminated, truncated, info

    def set_params(self, **params):
        self.env.set_params(**params)

    def get_params(self):
        return self.env.get_params()
