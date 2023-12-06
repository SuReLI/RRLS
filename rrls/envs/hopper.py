from __future__ import annotations

from typing import Any
from enum import Enum

import gymnasium as gym
from gymnasium import Wrapper


class HopperParamsBound(Enum):
    ONE_DIM = {
        "worldfriction": [0.1, 3.0],
    }
    TWO_DIM = {
        "worldfriction": [0.1, 3.0],
        "torsomass": [0.1, 3.0],
    }
    THREE_DIM = {
        "worldfriction": [0.1, 3.0],
        "torsomass": [0.1, 3.0],
        "thighmass": [0.1, 4.0],
    }


class RobustHopper(Wrapper):
    """
    Robust Hopper environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - worldfriction
        - torsomass
        - thighmass
        - legmass
        - footmass
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        thighmass: float | None = None,
        legmass: float | None = None,
        footmass: float | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(env=gym.make("Hopper-v5", **kwargs))

        self.set_params(
            worldfriction=worldfriction,
            torsomass=torsomass,
            thighmass=thighmass,
            legmass=legmass,
            footmass=footmass,
        )
        self._change_params()

    def set_params(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        thighmass: float | None = None,
        legmass: float | None = None,
        footmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.thighmass = thighmass
        self.legmass = legmass
        self.footmass = footmass
        self._change_params()

    def get_params(self):
        return {
            "worldfriction": self.worldfriction,
            "torsomass": self.torsomass,
            "thighmass": self.thighmass,
            "legmass": self.legmass,
            "footmass": self.footmass,
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
        if self.worldfriction is not None:
            self.unwrapped.model.geom_friction[0, 0] = self.worldfriction

        if self.torsomass is not None:
            self.unwrapped.model.body_mass[1] = self.torsomass

        if self.thighmass is not None:
            self.unwrapped.model.body_mass[2] = self.thighmass

        if self.legmass is not None:
            self.unwrapped.model.body_mass[3] = self.legmass

        if self.footmass is not None:
            self.unwrapped.model.body_mass[4] = self.footmass


# class RobustHopper(HopperEnv):
#     """
#     Robust Hopper environment. You can change the parameters of the environment using options in
#     the reset method or by using the set_params method. The parameters are changed by calling
#     the change_params method. The parameters are:
#         - worldfriction
#         - torsomass
#         - thighmass
#         - legmass
#         - footmass
#     """

#     def __init__(
#         self,
#         worldfriction: float | None = None,
#         torsomass: float | None = None,
#         thighmass: float | None = None,
#         legmass: float | None = None,
#         footmass: float | None = None,
#     ):
#         super().__init__()

#         self.set_params(
#             worldfriction=worldfriction,
#             torsomass=torsomass,
#             thighmass=thighmass,
#             legmass=legmass,
#             footmass=footmass,
#         )
#         self._change_params()

#     def set_params(
#         self,
#         worldfriction: float | None = None,
#         torsomass: float | None = None,
#         thighmass: float | None = None,
#         legmass: float | None = None,
#         footmass: float | None = None,
#     ):
#         self.worldfriction = worldfriction
#         self.torsomass = torsomass
#         self.thighmass = thighmass
#         self.legmass = legmass
#         self.footmass = footmass
#         self._change_params()

#     def get_params(self):
#         return {
#             "worldfriction": self.worldfriction,
#             "torsomass": self.torsomass,
#             "thighmass": self.thighmass,
#             "legmass": self.legmass,
#             "footmass": self.footmass,
#         }

#     def reset(self, *, seed: int | None = None, options: dict | None = None):
#         if options is not None:
#             self.set_params(**options)
#         obs, info = super().reset(seed=seed, options=options)
#         info.update(self.get_params())
#         return obs, info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = super().step(action)
#         info.update(self.get_params())
#         return obs, reward, terminated, truncated, info

#     def _change_params(self):
#         if self.worldfriction is not None:
#             self.model.geom_friction[0, 0] = self.worldfriction

#         if self.torsomass is not None:
#             self.model.body_mass[1] = self.torsomass

#         if self.thighmass is not None:
#             self.model.body_mass[2] = self.thighmass

#         if self.legmass is not None:
#             self.model.body_mass[3] = self.legmass

#         if self.footmass is not None:
#             self.model.body_mass[4] = self.footmass
