from __future__ import annotations

from enum import Enum
from typing import Any

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
    RARL = {
        "footforce_x": [-3.0, 3.0],
        "footforce_y": [-3.0, 3.0],
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


class ForceHopper(Wrapper):
    """
    Force Hopper environment. You can apply forces to the environment using the set_params method.
    The parameters are changed by calling the change_params method. The parameters are:
        - torsoforce_x
        - torsoforce_y
        - torsoforce_z
        - thighforce_x
        - thighforce_y
        - thighforce_z
        - legforce_x
        - legforce_y
        - legforce_z
        - footforce_x
        - footforce_y
        - footforce_z
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("Hopper-v5", **kwargs))
        self.set_params()

    def set_params(
        self,
        torsoforce_x: float | None = None,
        torsoforce_y: float | None = None,
        torsoforce_z: float | None = None,
        thighforce_x: float | None = None,
        thighforce_y: float | None = None,
        thighforce_z: float | None = None,
        legforce_x: float | None = None,
        legforce_y: float | None = None,
        legforce_z: float | None = None,
        footforce_x: float | None = None,
        footforce_y: float | None = None,
        footforce_z: float | None = None,
    ):
        self.torsoforce_x = torsoforce_x
        self.torsoforce_y = torsoforce_y
        self.torsoforce_z = torsoforce_z
        self.thighforce_x = thighforce_x
        self.thighforce_y = thighforce_y
        self.thighforce_z = thighforce_z
        self.legforce_x = legforce_x
        self.legforce_y = legforce_y
        self.legforce_z = legforce_z
        self.footforce_x = footforce_x
        self.footforce_y = footforce_y
        self.footforce_z = footforce_z
        self._change_params()

    def get_params(self):
        return {
            "torsoforce_x": self.torsoforce_x,
            "torsoforce_y": self.torsoforce_y,
            "torsoforce_z": self.torsoforce_z,
            "thighforce_x": self.thighforce_x,
            "thighforce_y": self.thighforce_y,
            "thighforce_z": self.thighforce_z,
            "legforce_x": self.legforce_x,
            "legforce_y": self.legforce_y,
            "legforce_z": self.legforce_z,
            "footforce_x": self.footforce_x,
            "footforce_y": self.footforce_y,
            "footforce_z": self.footforce_z,
        }

    def _change_params(self):
        if self.torsoforce_x is not None:
            self.unwrapped.data.xfrc_applied[1, 0] = self.torsoforce_x

        if self.torsoforce_y is not None:
            self.unwrapped.data.xfrc_applied[1, 1] = self.torsoforce_y

        if self.torsoforce_z is not None:
            self.unwrapped.data.xfrc_applied[1, 2] = self.torsoforce_z

        if self.thighforce_x is not None:
            self.unwrapped.data.xfrc_applied[2, 0] = self.thighforce_x

        if self.thighforce_y is not None:
            self.unwrapped.data.xfrc_applied[2, 1] = self.thighforce_y

        if self.thighforce_z is not None:
            self.unwrapped.data.xfrc_applied[2, 2] = self.thighforce_z

        if self.legforce_x is not None:
            self.unwrapped.data.xfrc_applied[3, 0] = self.legforce_x

        if self.legforce_y is not None:
            self.unwrapped.data.xfrc_applied[3, 1] = self.legforce_y

        if self.legforce_z is not None:
            self.unwrapped.data.xfrc_applied[3, 2] = self.legforce_z

        if self.footforce_x is not None:
            self.unwrapped.data.xfrc_applied[4, 0] = self.footforce_x

        if self.footforce_y is not None:
            self.unwrapped.data.xfrc_applied[4, 1] = self.footforce_y

        if self.footforce_z is not None:
            self.unwrapped.data.xfrc_applied[4, 2] = self.footforce_z

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
