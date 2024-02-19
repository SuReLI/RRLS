from __future__ import annotations

from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper

DEFAULT_PARAMS = {
    "torsomass": 0.32724923474893675,
    "frontleftlegmass": 0.03915775372846671,
    "frontleftlegauxmass": 0.03915775372846671,
    "frontleftleganklemass": 0.06759220453268026,
    "frontrightlegmass": 0.03915775372846671,
    "frontrightlegauxmass": 0.03915775372846671,
    "frontrightleganklemass": 0.06759220453268026,
    "backleftlegmass": 0.03915775372846671,
    "backleftlegauxmass": 0.03915775372846671,
    "backleftleganklemass": 0.06759220453268026,
    "backrightlegmass": 0.03915775372846671,
    "backrightlegauxmass": 0.03915775372846671,
    "backrightleganklemass": 0.06759220453268026,
}


class AntParamsBound(Enum):
    ONE_DIM = {
        "torsomass": [0.1, 3.0],
    }
    TWO_DIM = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
    }
    THREE_DIM = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
        "frontrightlegmass": [0.01, 3.0],
    }
    RARL = {
        "torsoforce_x": [-3.0, 3.0],
        "torsoforce_y": [-3.0, 3.0],
        "frontleftlegforce_x": [-3.0, 3.0],
        "frontleftlegforce_y": [-3.0, 3.0],
        "frontrightlegforce_x": [-3.0, 3.0],
        "frontrightlegforce_y": [-3.0, 3.0],
    }


class RobustAnt(Wrapper):
    """
    Robust Ant environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - torsomass
        - frontleftlegmass
        - frontleftlegauxmass
        - frontleftleganklemass
        - frontrightlegmass
        - frontrightlegauxmass
        - frontrightleganklemass
        - backleftlegmass
        - backleftlegauxmass
        - backleftleganklemass
        - backrightlegmass
        - backrightlegauxmass
        - backrightleganklemass
    """

    # HACK: This is a hack to avoid the following error:
    # gymnasium.error.InvalidMetadata: Expect the environment metadata to be dict, actual type: <class 'module'>
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        torsomass: float | None = None,
        frontleftlegmass: float | None = None,
        frontleftlegauxmass: float | None = None,
        frontleftleganklemass: float | None = None,
        frontrightlegmass: float | None = None,
        frontrightlegauxmass: float | None = None,
        frontrightleganklemass: float | None = None,
        backleftlegmass: float | None = None,
        backleftlegauxmass: float | None = None,
        backleftleganklemass: float | None = None,
        backrightlegmass: float | None = None,
        backrightlegauxmass: float | None = None,
        backrightleganklemass: float | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(env=gym.make("Ant-v5", **kwargs))
        self.set_params(
            torsomass=torsomass,
            frontleftlegmass=frontleftlegmass,
            frontleftlegauxmass=frontleftlegauxmass,
            frontleftleganklemass=frontleftleganklemass,
            frontrightlegmass=frontrightlegmass,
            frontrightlegauxmass=frontrightlegauxmass,
            frontrightleganklemass=frontrightleganklemass,
            backleftlegmass=backleftlegmass,
            backleftlegauxmass=backleftlegauxmass,
            backleftleganklemass=backleftleganklemass,
            backrightlegmass=backrightlegmass,
            backrightlegauxmass=backrightlegauxmass,
            backrightleganklemass=backrightleganklemass,
        )
        self._change_params()

    def set_params(
        self,
        torsomass: float | None = None,
        frontleftlegmass: float | None = None,
        frontleftlegauxmass: float | None = None,
        frontleftleganklemass: float | None = None,
        frontrightlegmass: float | None = None,
        frontrightlegauxmass: float | None = None,
        frontrightleganklemass: float | None = None,
        backleftlegmass: float | None = None,
        backleftlegauxmass: float | None = None,
        backleftleganklemass: float | None = None,
        backrightlegmass: float | None = None,
        backrightlegauxmass: float | None = None,
        backrightleganklemass: float | None = None,
    ):
        self.torsomass = (
            torsomass
            if torsomass is not None
            else getattr(self, "torsomass", DEFAULT_PARAMS["torsomass"])
        )
        self.frontleftlegmass = (
            frontleftlegmass
            if frontleftlegmass is not None
            else getattr(self, "frontleftlegmass", DEFAULT_PARAMS["frontleftlegmass"])
        )
        self.frontleftlegauxmass = (
            frontleftlegauxmass
            if frontleftlegauxmass is not None
            else getattr(
                self, "frontleftlegauxmass", DEFAULT_PARAMS["frontleftlegauxmass"]
            )
        )
        self.frontleftleganklemass = (
            frontleftleganklemass
            if frontleftleganklemass is not None
            else getattr(
                self, "frontleftleganklemass", DEFAULT_PARAMS["frontleftleganklemass"]
            )
        )
        self.frontrightlegmass = (
            frontrightlegmass
            if frontrightlegmass is not None
            else getattr(self, "frontrightlegmass", DEFAULT_PARAMS["frontrightlegmass"])
        )
        self.frontrightlegauxmass = (
            frontrightlegauxmass
            if frontrightlegauxmass is not None
            else getattr(
                self, "frontrightlegauxmass", DEFAULT_PARAMS["frontrightlegauxmass"]
            )
        )
        self.frontrightleganklemass = (
            frontrightleganklemass
            if frontrightleganklemass is not None
            else getattr(
                self, "frontrightleganklemass", DEFAULT_PARAMS["frontrightleganklemass"]
            )
        )
        self.backleftlegmass = (
            backleftlegmass
            if backleftlegmass is not None
            else getattr(self, "backleftlegmass", DEFAULT_PARAMS["backleftlegmass"])
        )
        self.backleftlegauxmass = (
            backleftlegauxmass
            if backleftlegauxmass is not None
            else getattr(
                self, "backleftlegauxmass", DEFAULT_PARAMS["backleftlegauxmass"]
            )
        )
        self.backleftleganklemass = (
            backleftleganklemass
            if backleftleganklemass is not None
            else getattr(
                self, "backleftleganklemass", DEFAULT_PARAMS["backleftleganklemass"]
            )
        )
        self.backrightlegmass = (
            backrightlegmass
            if backrightlegmass is not None
            else getattr(self, "backrightlegmass", DEFAULT_PARAMS["backrightlegmass"])
        )
        self.backrightlegauxmass = (
            backrightlegauxmass
            if backrightlegauxmass is not None
            else getattr(
                self, "backrightlegauxmass", DEFAULT_PARAMS["backrightlegauxmass"]
            )
        )
        self.backrightleganklemass = (
            backrightleganklemass
            if backrightleganklemass is not None
            else getattr(
                self, "backrightleganklemass", DEFAULT_PARAMS["backrightleganklemass"]
            )
        )
        self._change_params()

    def get_params(self):
        return {
            "torsomass": self.torsomass,
            "frontleftlegmass": self.frontleftlegmass,
            "frontleftlegauxmass": self.frontleftlegauxmass,
            "frontleftleganklemass": self.frontleftleganklemass,
            "frontrightlegmass": self.frontrightlegmass,
            "frontrightlegauxmass": self.frontrightlegauxmass,
            "frontrightleganklemass": self.frontrightleganklemass,
            "backleftlegmass": self.backleftlegmass,
            "backleftlegauxmass": self.backleftlegauxmass,
            "backleftleganklemass": self.backleftleganklemass,
            "backrightlegmass": self.backrightlegmass,
            "backrightlegauxmass": self.backrightlegauxmass,
            "backrightleganklemass": self.backrightleganklemass,
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
        if self.torsomass is not None:
            self.unwrapped.model.body_mass[1] = self.torsomass

        if self.frontleftlegmass is not None:
            self.unwrapped.model.body_mass[2] = self.frontleftlegmass

        if self.frontleftlegauxmass is not None:
            self.unwrapped.model.body_mass[3] = self.frontleftlegauxmass

        if self.frontleftleganklemass is not None:
            self.unwrapped.model.body_mass[4] = self.frontleftleganklemass

        if self.frontrightlegmass is not None:
            self.unwrapped.model.body_mass[5] = self.frontrightlegmass

        if self.frontrightlegauxmass is not None:
            self.unwrapped.model.body_mass[6] = self.frontrightlegauxmass

        if self.frontrightleganklemass is not None:
            self.unwrapped.model.body_mass[7] = self.frontrightleganklemass

        if self.backleftlegmass is not None:
            self.unwrapped.model.body_mass[8] = self.backleftlegmass

        if self.backleftlegauxmass is not None:
            self.unwrapped.model.body_mass[9] = self.backleftlegauxmass

        if self.backleftleganklemass is not None:
            self.unwrapped.model.body_mass[10] = self.backleftleganklemass

        if self.backrightlegmass is not None:
            self.unwrapped.model.body_mass[11] = self.backrightlegmass

        if self.backrightlegauxmass is not None:
            self.unwrapped.model.body_mass[12] = self.backrightlegauxmass

        if self.backrightleganklemass is not None:
            self.unwrapped.model.body_mass[13] = self.backrightleganklemass


class ForceAnt(Wrapper):
    """
    Force Ant environment. You can apply forces to the environment using the set_params method.
    The parameters are changed by calling the change_params method. The parameters are:
        - torsoforce_x
        - torsoforce_y
        - torsoforce_z
        - frontleftlegforce_x
        - frontleftlegforce_y
        - frontleftlegforce_z
        - frontleftlegauxforce_x
        - frontleftlegauxforce_y
        - frontleftlegauxforce_z
        - frontleftlegankleforce_x
        - frontleftlegankleforce_y
        - frontleftlegankleforce_z
        - frontrightlegforce_x
        - frontrightlegforce_y
        - frontrightlegforce_z
        - frontrightlegauxforce_x
        - frontrightlegauxforce_y
        - frontrightlegauxforce_z
        - frontrightlegankleforce_x
        - frontrightlegankleforce_y
        - frontrightlegankleforce_z
        - backleftlegforce_x
        - backleftlegforce_y
        - backleftlegforce_z
        - backleftlegauxforce_x
        - backleftlegauxforce_y
        - backleftlegauxforce_z
        - backleftlegankleforce_x
        - backleftlegankleforce_y
        - backleftlegankleforce_z
        - backrightlegforce_x
        - backrightlegforce_y
        - backrightlegforce_z
        - backrightlegauxforce_x
        - backrightlegauxforce_y
        - backrightlegauxforce_z
        - backrightlegankleforce_x
        - backrightlegankleforce_y
        - backrightlegankleforce_z
    """

    # HACK: This is a hack to avoid the following error:
    # gymnasium.error.InvalidMetadata: Expect the environment metadata to be dict, actual type: <class 'module'>
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("Ant-v5", **kwargs))
        self.set_params()

    def set_params(
        self,
        torsoforce_x: float | None = None,
        torsoforce_y: float | None = None,
        torsoforce_z: float | None = None,
        frontleftlegforce_x: float | None = None,
        frontleftlegforce_y: float | None = None,
        frontleftlegforce_z: float | None = None,
        frontleftlegauxforce_x: float | None = None,
        frontleftlegauxforce_y: float | None = None,
        frontleftlegauxforce_z: float | None = None,
        frontleftlegankleforce_x: float | None = None,
        frontleftlegankleforce_y: float | None = None,
        frontleftlegankleforce_z: float | None = None,
        frontrightlegforce_x: float | None = None,
        frontrightlegforce_y: float | None = None,
        frontrightlegforce_z: float | None = None,
        frontrightlegauxforce_x: float | None = None,
        frontrightlegauxforce_y: float | None = None,
        frontrightlegauxforce_z: float | None = None,
        frontrightlegankleforce_x: float | None = None,
        frontrightlegankleforce_y: float | None = None,
        frontrightlegankleforce_z: float | None = None,
        backleftlegforce_x: float | None = None,
        backleftlegforce_y: float | None = None,
        backleftlegforce_z: float | None = None,
        backleftlegauxforce_x: float | None = None,
        backleftlegauxforce_y: float | None = None,
        backleftlegauxforce_z: float | None = None,
        backleftlegankleforce_x: float | None = None,
        backleftlegankleforce_y: float | None = None,
        backleftlegankleforce_z: float | None = None,
        backrightlegforce_x: float | None = None,
        backrightlegforce_y: float | None = None,
        backrightlegforce_z: float | None = None,
        backrightlegauxforce_x: float | None = None,
        backrightlegauxforce_y: float | None = None,
        backrightlegauxforce_z: float | None = None,
        backrightlegankleforce_x: float | None = None,
        backrightlegankleforce_y: float | None = None,
        backrightlegankleforce_z: float | None = None,
    ):
        self.torsoforce_x = torsoforce_x
        self.torsoforce_y = torsoforce_y
        self.torsoforce_z = torsoforce_z
        self.frontleftlegforce_x = frontleftlegforce_x
        self.frontleftlegforce_y = frontleftlegforce_y
        self.frontleftlegforce_z = frontleftlegforce_z
        self.frontleftlegauxforce_x = frontleftlegauxforce_x
        self.frontleftlegauxforce_y = frontleftlegauxforce_y
        self.frontleftlegauxforce_z = frontleftlegauxforce_z
        self.frontleftlegankleforce_x = frontleftlegankleforce_x
        self.frontleftlegankleforce_y = frontleftlegankleforce_y
        self.frontleftlegankleforce_z = frontleftlegankleforce_z
        self.frontrightlegforce_x = frontrightlegforce_x
        self.frontrightlegforce_y = frontrightlegforce_y
        self.frontrightlegforce_z = frontrightlegforce_z
        self.frontrightlegauxforce_x = frontrightlegauxforce_x
        self.frontrightlegauxforce_y = frontrightlegauxforce_y
        self.frontrightlegauxforce_z = frontrightlegauxforce_z
        self.frontrightlegankleforce_x = frontrightlegankleforce_x
        self.frontrightlegankleforce_y = frontrightlegankleforce_y
        self.frontrightlegankleforce_z = frontrightlegankleforce_z
        self.backleftlegforce_x = backleftlegforce_x
        self.backleftlegforce_y = backleftlegforce_y
        self.backleftlegforce_z = backleftlegforce_z
        self.backleftlegauxforce_x = backleftlegauxforce_x
        self.backleftlegauxforce_y = backleftlegauxforce_y
        self.backleftlegauxforce_z = backleftlegauxforce_z
        self.backleftlegankleforce_x = backleftlegankleforce_x
        self.backleftlegankleforce_y = backleftlegankleforce_y
        self.backleftlegankleforce_z = backleftlegankleforce_z
        self.backrightlegforce_x = backrightlegforce_x
        self.backrightlegforce_y = backrightlegforce_y
        self.backrightlegforce_z = backrightlegforce_z
        self.backrightlegauxforce_x = backrightlegauxforce_x
        self.backrightlegauxforce_y = backrightlegauxforce_y
        self.backrightlegauxforce_z = backrightlegauxforce_z
        self.backrightlegankleforce_x = backrightlegankleforce_x
        self.backrightlegankleforce_y = backrightlegankleforce_y
        self.backrightlegankleforce_z = backrightlegankleforce_z
        self._change_params()

    def get_params(self):
        return {
            "torsoforce_x": self.torsoforce_x,
            "torsoforce_y": self.torsoforce_y,
            "torsoforce_z": self.torsoforce_z,
            "frontleftlegforce_x": self.frontleftlegforce_x,
            "frontleftlegforce_y": self.frontleftlegforce_y,
            "frontleftlegforce_z": self.frontleftlegforce_z,
            "frontleftlegauxforce_x": self.frontleftlegauxforce_x,
            "frontleftlegauxforce_y": self.frontleftlegauxforce_y,
            "frontleftlegauxforce_z": self.frontleftlegauxforce_z,
            "frontleftlegankleforce_x": self.frontleftlegankleforce_x,
            "frontleftlegankleforce_y": self.frontleftlegankleforce_y,
            "frontleftlegankleforce_z": self.frontleftlegankleforce_z,
            "frontrightlegforce_x": self.frontrightlegforce_x,
            "frontrightlegforce_y": self.frontrightlegforce_y,
            "frontrightlegforce_z": self.frontrightlegforce_z,
            "frontrightlegauxforce_x": self.frontrightlegauxforce_x,
            "frontrightlegauxforce_y": self.frontrightlegauxforce_y,
            "frontrightlegauxforce_z": self.frontrightlegauxforce_z,
            "frontrightlegankleforce_x": self.frontrightlegankleforce_x,
            "frontrightlegankleforce_y": self.frontrightlegankleforce_y,
            "frontrightlegankleforce_z": self.frontrightlegankleforce_z,
            "backleftlegforce_x": self.backleftlegforce_x,
            "backleftlegforce_y": self.backleftlegforce_y,
            "backleftlegforce_z": self.backleftlegforce_z,
            "backleftlegauxforce_x": self.backleftlegauxforce_x,
            "backleftlegauxforce_y": self.backleftlegauxforce_y,
            "backleftlegauxforce_z": self.backleftlegauxforce_z,
            "backleftlegankleforce_x": self.backleftlegankleforce_x,
            "backleftlegankleforce_y": self.backleftlegankleforce_y,
            "backleftlegankleforce_z": self.backleftlegankleforce_z,
            "backrightlegforce_x": self.backrightlegforce_x,
            "backrightlegforce_y": self.backrightlegforce_y,
            "backrightlegforce_z": self.backrightlegforce_z,
            "backrightlegauxforce_x": self.backrightlegauxforce_x,
            "backrightlegauxforce_y": self.backrightlegauxforce_y,
            "backrightlegauxforce_z": self.backrightlegauxforce_z,
            "backrightlegankleforce_x": self.backrightlegankleforce_x,
            "backrightlegankleforce_y": self.backrightlegankleforce_y,
            "backrightlegankleforce_z": self.backrightlegankleforce_z,
        }

    def _change_params(self):
        if self.torsoforce_x is not None:
            self.unwrapped.data.xfrc_applied[1, 0] = self.torsoforce_x

        if self.torsoforce_y is not None:
            self.unwrapped.data.xfrc_applied[1, 1] = self.torsoforce_y

        if self.torsoforce_z is not None:
            self.unwrapped.data.xfrc_applied[1, 2] = self.torsoforce_z

        if self.frontleftlegforce_x is not None:
            self.unwrapped.data.xfrc_applied[2, 0] = self.frontleftlegforce_x

        if self.frontleftlegforce_y is not None:
            self.unwrapped.data.xfrc_applied[2, 1] = self.frontleftlegforce_y

        if self.frontleftlegforce_z is not None:
            self.unwrapped.data.xfrc_applied[2, 2] = self.frontleftlegforce_z

        if self.frontleftlegauxforce_x is not None:
            self.unwrapped.data.xfrc_applied[3, 0] = self.frontleftlegauxforce_x

        if self.frontleftlegauxforce_y is not None:
            self.unwrapped.data.xfrc_applied[3, 1] = self.frontleftlegauxforce_y

        if self.frontleftlegauxforce_z is not None:
            self.unwrapped.data.xfrc_applied[3, 2] = self.frontleftlegauxforce_z

        if self.frontleftlegankleforce_x is not None:
            self.unwrapped.data.xfrc_applied[4, 0] = self.frontleftlegankleforce_x

        if self.frontleftlegankleforce_y is not None:
            self.unwrapped.data.xfrc_applied[4, 1] = self.frontleftlegankleforce_y

        if self.frontleftlegankleforce_z is not None:
            self.unwrapped.data.xfrc_applied[4, 2] = self.frontleftlegankleforce_z

        if self.frontrightlegforce_x is not None:
            self.unwrapped.data.xfrc_applied[5, 0] = self.frontrightlegforce_x

        if self.frontrightlegforce_y is not None:
            self.unwrapped.data.xfrc_applied[5, 1] = self.frontrightlegforce_y

        if self.frontrightlegforce_z is not None:
            self.unwrapped.data.xfrc_applied[5, 2] = self.frontrightlegforce_z

        if self.frontrightlegauxforce_x is not None:
            self.unwrapped.data.xfrc_applied[6, 0] = self.frontrightlegauxforce_x

        if self.frontrightlegauxforce_y is not None:
            self.unwrapped.data.xfrc_applied[6, 1] = self.frontrightlegauxforce_y

        if self.frontrightlegauxforce_z is not None:
            self.unwrapped.data.xfrc_applied[6, 2] = self.frontrightlegauxforce_z

        if self.frontrightlegankleforce_x is not None:
            self.unwrapped.data.xfrc_applied[7, 0] = self.frontrightlegankleforce_x

        if self.frontrightlegankleforce_y is not None:
            self.unwrapped.data.xfrc_applied[7, 1] = self.frontrightlegankleforce_y

        if self.frontrightlegankleforce_z is not None:
            self.unwrapped.data.xfrc_applied[7, 2] = self.frontrightlegankleforce_z

        if self.backleftlegforce_x is not None:
            self.unwrapped.data.xfrc_applied[8, 0] = self.backleftlegforce_x

        if self.backleftlegforce_y is not None:
            self.unwrapped.data.xfrc_applied[8, 1] = self.backleftlegforce_y

        if self.backleftlegforce_z is not None:
            self.unwrapped.data.xfrc_applied[8, 2] = self.backleftlegforce_z

        if self.backleftlegauxforce_x is not None:
            self.unwrapped.data.xfrc_applied[9, 0] = self.backleftlegauxforce_x

        if self.backleftlegauxforce_y is not None:
            self.unwrapped.data.xfrc_applied[9, 1] = self.backleftlegauxforce_y

        if self.backleftlegauxforce_z is not None:
            self.unwrapped.data.xfrc_applied[9, 2] = self.backleftlegauxforce_z

        if self.backleftlegankleforce_x is not None:
            self.unwrapped.data.xfrc_applied[10, 0] = self.backleftlegankleforce_x

        if self.backleftlegankleforce_y is not None:
            self.unwrapped.data.xfrc_applied[10, 1] = self.backleftlegankleforce_y

        if self.backleftlegankleforce_z is not None:
            self.unwrapped.data.xfrc_applied[10, 2] = self.backleftlegankleforce_z

        if self.backrightlegforce_x is not None:
            self.unwrapped.data.xfrc_applied[11, 0] = self.backrightlegforce_x

        if self.backrightlegforce_y is not None:
            self.unwrapped.data.xfrc_applied[11, 1] = self.backrightlegforce_y

        if self.backrightlegforce_z is not None:
            self.unwrapped.data.xfrc_applied[11, 2] = self.backrightlegforce_z

        if self.backrightlegauxforce_x is not None:
            self.unwrapped.data.xfrc_applied[12, 0] = self.backrightlegauxforce_x

        if self.backrightlegauxforce_y is not None:
            self.unwrapped.data.xfrc_applied[12, 1] = self.backrightlegauxforce_y

        if self.backrightlegauxforce_z is not None:
            self.unwrapped.data.xfrc_applied[12, 2] = self.backrightlegauxforce_z

        if self.backrightlegankleforce_x is not None:
            self.unwrapped.data.xfrc_applied[13, 0] = self.backrightlegankleforce_x

        if self.backrightlegankleforce_y is not None:
            self.unwrapped.data.xfrc_applied[13, 1] = self.backrightlegankleforce_y

        if self.backrightlegankleforce_z is not None:
            self.unwrapped.data.xfrc_applied[13, 2] = self.backrightlegankleforce_z

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
