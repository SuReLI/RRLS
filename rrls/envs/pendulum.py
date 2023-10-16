from __future__ import annotations

from enum import Enum

from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv


class InvertedPendulumParamsBound(Enum):
    ONE_DIM = {
        "polemass": [1.0, 31.0],
    }
    TWO_DIM = {
        "polemass": [1.0, 31.0],
        "cartmass": [1.0, 11.0],
    }


class RobustInvertedPendulum(InvertedPendulumEnv):
    def __init__(self, polemass: float | None = None, cartmass: float | None = None):
        self.polemass = polemass
        self.cartmass = cartmass
        super().__init__()

        self.change_params()

    def set_params(self, polemass: float | None = None, cartmass: float | None = None):
        self.polemass = polemass
        self.cartmass = cartmass

    def get_params(self):
        return {
            "polemass": self.polemass,
            "cartmass": self.cartmass,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is not None:
            self.set_params(**options)
            self.change_params()
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def change_params(self):
        if self.polemass is not None:
            self.model.body_mass[2] = self.polemass
        if self.cartmass is not None:
            self.model.body_mass[1] = self.cartmass
