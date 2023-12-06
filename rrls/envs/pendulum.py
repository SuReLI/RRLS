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
    RARL = {
        "poleforce_x": [-3.0, 3.0],
        "poleforce_y": [-3.0, 3.0],
        "poleforce_z": [-3.0, 3.0],
        "cartforce_x": [-3.0, 3.0],
        "cartforce_y": [-3.0, 3.0],
        "cartforce_z": [-3.0, 3.0],

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

class ForceInvertedPendulum(Wrapper):
    """
    Force InvertedPendulum environment. You can apply forces to the environment using the set_params method.
    The parameters are changed by calling the change_params method. The parameters are:
        - poleforce
        - cartforce
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("InvertedPendulum-v5", **kwargs))
        self.set_params()

    def set_params(
            self,
            poleforce_x: float | None = None,
            poleforce_y: float | None = None,
            poleforce_z: float | None = None, 
            cartforce_x: float | None = None,
            cartforce_y: float | None = None,
            cartforce_z: float | None = None):
        self.poleforce_x = poleforce_x
        self.poleforce_y = poleforce_y
        self.poleforce_z = poleforce_z
        self.cartforce_x = cartforce_x
        self.cartforce_y = cartforce_y
        self.cartforce_z = cartforce_z
        self._change_params()

    def get_params(self):
        return {
            "poleforce_x": self.poleforce_x,
            "poleforce_y": self.poleforce_y,
            "poleforce_z": self.poleforce_z,
            "cartforce_x": self.cartforce_x,
            "cartforce_y": self.cartforce_y,
            "cartforce_z": self.cartforce_z,
        }
    
    def _change_params(self):
        if self.cartforce_x is not None:
            self.unwrapped.data.xfrc_applied[1,0] = self.cartforce_x
        if self.cartforce_y is not None:
            self.unwrapped.data.xfrc_applied[1,1] = self.cartforce_y
        if self.cartforce_z is not None:
            self.unwrapped.data.xfrc_applied[1,2] = self.cartforce_z
        if self.poleforce_x is not None:
            self.unwrapped.data.xfrc_applied[2,0] = self.poleforce_x
        if self.poleforce_y is not None:
            self.unwrapped.data.xfrc_applied[2,1] = self.poleforce_y
        if self.poleforce_z is not None:
            self.unwrapped.data.xfrc_applied[2,2] = self.poleforce_z    
                
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