from __future__ import annotations

from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper

DEFAULT_PARAMS = {
    "worldfriction": 0.7,
    "torsomass": 3.6651914291880923,
    "thighmass": 4.057890510886818,
    "legmass": 2.7813566959781637,
    "footmass": 3.1667253948185117,
    "leftthighmass": 4.057890510886818,
    "leftlegmass": 2.7813566959781637,
    "leftfootmass": 3.1667253948185117,
}


class Walker2dParamsBound(Enum):
    ONE_DIM = {
        "worldfriction": [0.1, 4.0],
    }
    TWO_DIM = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 5.0],
    }
    THREE_DIM = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 5.0],
        "thighmass": [0.1, 6.0],
    }
    RARL = {
        "legforce_x": [-3.0, 3.0],
        "legforce_y": [-3.0, 3.0],
        "leftfootforce_x": [-3.0, 3.0],
        "leftfootforce_y": [-3.0, 3.0],
    }


class RobustWalker2d(Wrapper):
    """
    Robust Walker2d environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - worldfriction
        - torsomass
        - thighmass
        - legmass
        - footmass
        - leftthighmass
        - leftlegmass
        - leftfootmass
    """

    metadata = {  # type: ignore
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
        leftthighmass: float | None = None,
        leftlegmass: float | None = None,
        leftfootmass: float | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(env=gym.make("Walker2d-v5", **kwargs))  # type: ignore
        self.set_params(
            worldfriction=worldfriction,
            torsomass=torsomass,
            thighmass=thighmass,
            legmass=legmass,
            footmass=footmass,
            leftthighmass=leftthighmass,
            leftlegmass=leftlegmass,
            leftfootmass=leftfootmass,
        )
        self._change_params()

    def set_params(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        thighmass: float | None = None,
        legmass: float | None = None,
        footmass: float | None = None,
        leftthighmass: float | None = None,
        leftlegmass: float | None = None,
        leftfootmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.thighmass = thighmass
        self.legmass = legmass
        self.footmass = footmass
        self.leftthighmass = leftthighmass
        self.leftlegmass = leftlegmass
        self.leftfootmass = leftfootmass
        self._change_params()

    def get_params(self):
        return {
            "worldfriction": self.worldfriction,
            "torsomass": self.torsomass,
            "thighmass": self.thighmass,
            "legmass": self.legmass,
            "footmass": self.footmass,
            "leftthighmass": self.leftthighmass,
            "leftlegmass": self.leftlegmass,
            "leftfootmass": self.leftfootmass,
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
            self.unwrapped.model.geom_friction[0, 0] = self.worldfriction  # type: ignore

        if self.torsomass is not None:
            self.unwrapped.model.body_mass[1] = self.torsomass  # type: ignore

        if self.thighmass is not None:
            self.unwrapped.model.body_mass[2] = self.thighmass  # type: ignore

        if self.legmass is not None:
            self.unwrapped.model.body_mass[3] = self.legmass  # type: ignore

        if self.footmass is not None:
            self.unwrapped.model.body_mass[4] = self.footmass  # type: ignore

        if self.leftthighmass is not None:
            self.unwrapped.model.body_mass[5] = self.leftthighmass  # type: ignore

        if self.leftlegmass is not None:
            self.unwrapped.model.body_mass[6] = self.leftlegmass  # type: ignore

        if self.leftfootmass is not None:
            self.unwrapped.model.body_mass[7] = self.leftfootmass  # type: ignore


class ForceWalker2d(Wrapper):
    """
    Force Walker2d environment. You can apply forces to the environment using the set_params method.
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
        - leftthighforce_x
        - leftthighforce_y
        - leftthighforce_z
        - leftlegforce_x
        - leftlegforce_y
        - leftlegforce_z
        - leftfootforce_x
        - leftfootforce_y
        - leftfootforce_z
    """

    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("Walker2d-v5", **kwargs))  # type: ignore
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
        leftthighforce_x: float | None = None,
        leftthighforce_y: float | None = None,
        leftthighforce_z: float | None = None,
        leftlegforce_x: float | None = None,
        leftlegforce_y: float | None = None,
        leftlegforce_z: float | None = None,
        leftfootforce_x: float | None = None,
        leftfootforce_y: float | None = None,
        leftfootforce_z: float | None = None,
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
        self.leftthighforce_x = leftthighforce_x
        self.leftthighforce_y = leftthighforce_y
        self.leftthighforce_z = leftthighforce_z
        self.leftlegforce_x = leftlegforce_x
        self.leftlegforce_y = leftlegforce_y
        self.leftlegforce_z = leftlegforce_z
        self.leftfootforce_x = leftfootforce_x
        self.leftfootforce_y = leftfootforce_y
        self.leftfootforce_z = leftfootforce_z
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
            "leftthighforce_x": self.leftthighforce_x,
            "leftthighforce_y": self.leftthighforce_y,
            "leftthighforce_z": self.leftthighforce_z,
            "leftlegforce_x": self.leftlegforce_x,
            "leftlegforce_y": self.leftlegforce_y,
            "leftlegforce_z": self.leftlegforce_z,
            "leftfootforce_x": self.leftfootforce_x,
            "leftfootforce_y": self.leftfootforce_y,
            "leftfootforce_z": self.leftfootforce_z,
        }

    def _change_params(self):
        if self.torsoforce_x is not None:
            self.unwrapped.data.xfrc_applied[1, 0] = self.torsoforce_x  # type: ignore

        if self.torsoforce_y is not None:
            self.unwrapped.data.xfrc_applied[1, 1] = self.torsoforce_y  # type: ignore

        if self.torsoforce_z is not None:
            self.unwrapped.data.xfrc_applied[1, 2] = self.torsoforce_z  # type: ignore

        if self.thighforce_x is not None:
            self.unwrapped.data.xfrc_applied[2, 0] = self.thighforce_x  # type: ignore

        if self.thighforce_y is not None:
            self.unwrapped.data.xfrc_applied[2, 1] = self.thighforce_y  # type: ignore

        if self.thighforce_z is not None:
            self.unwrapped.data.xfrc_applied[2, 2] = self.thighforce_z  # type: ignore

        if self.legforce_x is not None:
            self.unwrapped.data.xfrc_applied[3, 0] = self.legforce_x  # type: ignore

        if self.legforce_y is not None:
            self.unwrapped.data.xfrc_applied[3, 1] = self.legforce_y  # type: ignore

        if self.legforce_z is not None:
            self.unwrapped.data.xfrc_applied[3, 2] = self.legforce_z  # type: ignore

        if self.footforce_x is not None:
            self.unwrapped.data.xfrc_applied[4, 0] = self.footforce_x  # type: ignore # type: ignore

        if self.footforce_y is not None:
            self.unwrapped.data.xfrc_applied[4, 1] = self.footforce_y  # type: ignore

        if self.footforce_z is not None:
            self.unwrapped.data.xfrc_applied[4, 2] = self.footforce_z  # type: ignore

        if self.leftthighforce_x is not None:
            self.unwrapped.data.xfrc_applied[5, 0] = self.leftthighforce_x  # type: ignore

        if self.leftthighforce_y is not None:
            self.unwrapped.data.xfrc_applied[5, 1] = self.leftthighforce_y  # type: ignore

        if self.leftthighforce_z is not None:
            self.unwrapped.data.xfrc_applied[5, 2] = self.leftthighforce_z  # type: ignore

        if self.leftlegforce_x is not None:
            self.unwrapped.data.xfrc_applied[6, 0] = self.leftlegforce_x  # type: ignore

        if self.leftlegforce_y is not None:
            self.unwrapped.data.xfrc_applied[6, 1] = self.leftlegforce_y  # type: ignore

        if self.leftlegforce_z is not None:
            self.unwrapped.data.xfrc_applied[6, 2] = self.leftlegforce_z  # type: ignore

        if self.leftfootforce_x is not None:
            self.unwrapped.data.xfrc_applied[7, 0] = self.leftfootforce_x  # type: ignore

        if self.leftfootforce_y is not None:
            self.unwrapped.data.xfrc_applied[7, 1] = self.leftfootforce_y  # type: ignore

        if self.leftfootforce_z is not None:
            self.unwrapped.data.xfrc_applied[7, 2] = self.leftfootforce_z  # type: ignore

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
