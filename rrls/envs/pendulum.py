from __future__ import annotations

from typing import Any
from enum import Enum

import gymnasium as gym
from gymnasium import Wrapper


class InvertedPendulumParamsBound(Enum):
    ONE_DIM = {
        "polemass": [1.0, 31.0],
    }
    TWO_DIM = {
        "polemass": [1.0, 31.0],
        "cartmass": [1.0, 11.0],
    }


class RobustInvertedPendulum(Wrapper):
    """
    Robust Inverted Pendulum environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - polemass
        - cartmass
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, polemass: float | None = None, cartmass: float | None = None, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("InvertedPendulum-v5", **kwargs))
        self.set_params(polemass=polemass, cartmass=cartmass)

    def set_params(self, polemass: float | None = None, cartmass: float | None = None):
        self.polemass = polemass
        self.cartmass = cartmass
        self._change_params()

    def get_params(self):
        return {
            "polemass": self.polemass,
            "cartmass": self.cartmass,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is not None:
            self.set_params(**options)
        obs, info = self.env.reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def _change_params(self):
        if self.cartmass is not None:
            self.unwrapped.model.body_mass[1] = self.cartmass
        if self.polemass is not None:
            self.unwrapped.model.body_mass[2] = self.polemass
