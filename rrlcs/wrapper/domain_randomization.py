from __future__ import annotations

from typing import Annotated, Callable

import gymnasium as gym
import numpy as np

from rrlcs._interface import ModifiedParamsEnv


class DomainRandomizationWrapper(gym.Wrapper):
    """
    The `DomainRandomizationBenchmarkWrapper` is a Gym Wrapper that allows for
    domain randomization by changing the parameters of the environment between episodes.

    This class wraps an environment that follows the `ModifiedParamsEnv` protocol.

    Args:
        env (ModifiedParamsEnv): The environment to be wrapped.
        randomize_fn (Callable): A function that takes the parameter boundaries as input and returns
            a new set of parameters.

    Attributes:
        env (ModifiedParamsEnv): The environment to be wrapped.
        params_bound (dict): Parameter boundaries.
        params (dict): Current parameters.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        randomize_fn: Callable[
            [dict[str, Annotated[tuple[float], 2]]], dict[str, float]
        ]
        | None = None,
    ):
        super().__init__(env)
        self.env = env
        self.params_bound = params_bound

        # If no randomize_fn is provided, use the uniform one
        self.randomize_fn: Callable[
            [dict[str, Annotated[tuple[float], 2]]], dict[str, float]
        ] = (randomize_fn if randomize_fn is not None else self.draw_params_uniform)

        self.draw_params = randomize_fn
        self.params = self.randomize_fn(self.params_bound)

    def reset(self):
        """
        Resets the environment and draws a new set of parameters.

        Returns:
            obj: The initial observation from the environment.
        """
        self.params = self.randomize_fn(self.params_bound)
        self.env.set_params(**self.params)
        self.env.change_physics()
        return self.env.reset()

    def step(self, action):
        """
        Steps the environment using the given action.

        Args:
            action (obj): An action provided by the agent.

        Returns:
            tuple: A tuple containing the new observation, the reward, a boolean indicating whether the episode has ended, and additional info.
        """
        return self.env.step(action)

    def draw_params_uniform(
        self, parameters_space: dict[str, Annotated[tuple[float], 2]]
    ):
        low = np.array([bound[0] for bound in parameters_space.values()])
        high = np.array([bound[1] for bound in parameters_space.values()])  # type: ignore
        params_draw = np.random.uniform(low, high)
        new_params = dict(zip(parameters_space.keys(), params_draw))
        return new_params
