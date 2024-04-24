from __future__ import annotations

from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper

# from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

DEFAULT_PARAMS = {
    # "worldfriction": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    "worldfriction": 0.4,
    "torsomass": 6.25020920502092,
    "backthighmass": 1.5435146443514645,
    "backshinmass": 1.5874476987447697,
    "backfootmass": 1.0953974895397491,
    "forwardthighmass": 1.4380753138075317,
    "forwardshinmass": 1.200836820083682,
    "forwardfootmass": 0.8845188284518829,
}


class HalfCheetahParamsBound(Enum):
    ONE_DIM = {
        "worldfriction": [0.1, 3.0],
    }
    TWO_DIM = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 7.0],
    }
    THREE_DIM = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 7.0],
        "backthighmass": [0.1, 3.0],
    }
    RARL = {
        "torsoforce_x": [-3.0, 3.0],
        "torsoforce_y": [-3.0, 3.0],
        "backfootforce_x": [-3.0, 3.0],
        "backfootforce_y": [-3.0, 3.0],
        "forwardfootforce_x": [-3.0, 3.0],
        "forwardfootforce_y": [-3.0, 3.0],
    }


class RobustHalfCheetah(Wrapper):
    """
    Robust HalfCheetah environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - worldfriction
        - torsomass
        - backthighmass
        - backshinmass
        - backfootmass
        - forwardthighmass
        - forwardshinmass
        - forwardfootmass
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
        backthighmass: float | None = None,
        backshinmass: float | None = None,
        backfootmass: float | None = None,
        forwardthighmass: float | None = None,
        forwardshinmass: float | None = None,
        forwardfootmass: float | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(env=gym.make("HalfCheetah-v5", **kwargs))  # type: ignore

        self.set_params(
            worldfriction=worldfriction,
            torsomass=torsomass,
            backthighmass=backthighmass,
            backshinmass=backshinmass,
            backfootmass=backfootmass,
            forwardthighmass=forwardthighmass,
            forwardshinmass=forwardshinmass,
            forwardfootmass=forwardfootmass,
        )
        self._change_params()

    def set_params(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        backthighmass: float | None = None,
        backshinmass: float | None = None,
        backfootmass: float | None = None,
        forwardthighmass: float | None = None,
        forwardshinmass: float | None = None,
        forwardfootmass: float | None = None,
    ):
        self.worldfriction = (
            worldfriction
            if worldfriction is not None
            else getattr(self, "worldfriction", DEFAULT_PARAMS["worldfriction"])
        )
        self.torsomass = (
            torsomass
            if torsomass is not None
            else getattr(self, "torsomass", DEFAULT_PARAMS["torsomass"])
        )
        self.backthighmass = (
            backthighmass
            if backthighmass is not None
            else getattr(self, "backthighmass", DEFAULT_PARAMS["backthighmass"])
        )
        self.backshinmass = (
            backshinmass
            if backshinmass is not None
            else getattr(self, "backshinmass", DEFAULT_PARAMS["backshinmass"])
        )
        self.backfootmass = (
            backfootmass
            if backfootmass is not None
            else getattr(self, "backfootmass", DEFAULT_PARAMS["backfootmass"])
        )
        self.forwardthighmass = (
            forwardthighmass
            if forwardthighmass is not None
            else getattr(self, "forwardthighmass", DEFAULT_PARAMS["forwardthighmass"])
        )
        self.forwardshinmass = (
            forwardshinmass
            if forwardshinmass is not None
            else getattr(self, "forwardshinmass", DEFAULT_PARAMS["forwardshinmass"])
        )
        self.forwardfootmass = (
            forwardfootmass
            if forwardfootmass is not None
            else getattr(self, "forwardfootmass", DEFAULT_PARAMS["forwardfootmass"])
        )
        self._change_params()

    def get_params(self):
        return {
            "worldfriction": self.worldfriction,
            "torsomass": self.torsomass,
            "backthighmass": self.backthighmass,
            "backshinmass": self.backshinmass,
            "backfootmass": self.backfootmass,
            "forwardthighmass": self.forwardthighmass,
            "forwardshinmass": self.forwardshinmass,
            "forwardfootmass": self.forwardfootmass,
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
            self.unwrapped.model.geom_friction[:, 0] = self.worldfriction  # type: ignore
        if self.torsomass is not None:
            self.unwrapped.model.body_mass[1] = self.torsomass  # type: ignore
        if self.backthighmass is not None:
            self.unwrapped.model.body_mass[2] = self.backthighmass  # type: ignore
        if self.backshinmass is not None:
            self.unwrapped.model.body_mass[3] = self.backshinmass  # type: ignore
        if self.backfootmass is not None:
            self.unwrapped.model.body_mass[4] = self.backfootmass  # type: ignore
        if self.forwardthighmass is not None:
            self.unwrapped.model.body_mass[5] = self.forwardthighmass  # type: ignore
        if self.forwardshinmass is not None:
            self.unwrapped.model.body_mass[6] = self.forwardshinmass  # type: ignore
        if self.forwardfootmass is not None:
            self.unwrapped.model.body_mass[7] = self.forwardfootmass  # type: ignore


class ForceHalfCheetah(Wrapper):
    """
    Force HalfCheetah environment. You can apply forces to the robot using the env.data.xfrc_applied
    attribute. The parameters are:
        - torsoforce_x
        - torsoforce_y
        - torsoforce_z
        - backthighforce_x
        - backthighforce_y
        - backthighforce_z
        - backshinforce_x
        - backshinforce_y
        - backshinforce_z
        - backfootforce_x
        - backfootforce_y
        - backfootforce_z
        - forwardthighforce_x
        - forwardthighforce_y
        - forwardthighforce_z
        - forwardshinforce_x
        - forwardshinforce_y
        - forwardshinforce_z
        - forwardfootforce_x
        - forwardfootforce_y
        - forwardfootforce_z
    """

    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("HalfCheetah-v5", **kwargs))  # type: ignore
        self.set_params()
        self._change_params()

    def set_params(
        self,
        torsoforce_x: float | None = None,
        torsoforce_y: float | None = None,
        torsoforce_z: float | None = None,
        backthighforce_x: float | None = None,
        backthighforce_y: float | None = None,
        backthighforce_z: float | None = None,
        backshinforce_x: float | None = None,
        backshinforce_y: float | None = None,
        backshinforce_z: float | None = None,
        backfootforce_x: float | None = None,
        backfootforce_y: float | None = None,
        backfootforce_z: float | None = None,
        forwardthighforce_x: float | None = None,
        forwardthighforce_y: float | None = None,
        forwardthighforce_z: float | None = None,
        forwardshinforce_x: float | None = None,
        forwardshinforce_y: float | None = None,
        forwardshinforce_z: float | None = None,
        forwardfootforce_x: float | None = None,
        forwardfootforce_y: float | None = None,
        forwardfootforce_z: float | None = None,
    ):
        self.torsoforce_x = torsoforce_x
        self.torsoforce_y = torsoforce_y
        self.torsoforce_z = torsoforce_z
        self.backthighforce_x = backthighforce_x
        self.backthighforce_y = backthighforce_y
        self.backthighforce_z = backthighforce_z
        self.backshinforce_x = backshinforce_x
        self.backshinforce_y = backshinforce_y
        self.backshinforce_z = backshinforce_z
        self.backfootforce_x = backfootforce_x
        self.backfootforce_y = backfootforce_y
        self.backfootforce_z = backfootforce_z
        self.forwardthighforce_x = forwardthighforce_x
        self.forwardthighforce_y = forwardthighforce_y
        self.forwardthighforce_z = forwardthighforce_z
        self.forwardshinforce_x = forwardshinforce_x
        self.forwardshinforce_y = forwardshinforce_y
        self.forwardshinforce_z = forwardshinforce_z
        self.forwardfootforce_x = forwardfootforce_x
        self.forwardfootforce_y = forwardfootforce_y
        self.forwardfootforce_z = forwardfootforce_z
        self._change_params()

    def get_params(self):
        return {
            "torsoforce_x": self.torsoforce_x,
            "torsoforce_y": self.torsoforce_y,
            "torsoforce_z": self.torsoforce_z,
            "backthighforce_x": self.backthighforce_x,
            "backthighforce_y": self.backthighforce_y,
            "backthighforce_z": self.backthighforce_z,
            "backshinforce_x": self.backshinforce_x,
            "backshinforce_y": self.backshinforce_y,
            "backshinforce_z": self.backshinforce_z,
            "backfootforce_x": self.backfootforce_x,
            "backfootforce_y": self.backfootforce_y,
            "backfootforce_z": self.backfootforce_z,
            "forwardthighforce_x": self.forwardthighforce_x,
            "forwardthighforce_y": self.forwardthighforce_y,
            "forwardthighforce_z": self.forwardthighforce_z,
            "forwardshinforce_x": self.forwardshinforce_x,
            "forwardshinforce_y": self.forwardshinforce_y,
            "forwardshinforce_z": self.forwardshinforce_z,
            "forwardfootforce_x": self.forwardfootforce_x,
            "forwardfootforce_y": self.forwardfootforce_y,
            "forwardfootforce_z": self.forwardfootforce_z,
        }

    def _change_params(self):
        if self.torsoforce_x is not None:
            self.unwrapped.data.xfrc_applied[1, 0] = self.torsoforce_x  # type: ignore
        if self.torsoforce_y is not None:
            self.unwrapped.data.xfrc_applied[1, 1] = self.torsoforce_y  # type: ignore
        if self.torsoforce_z is not None:
            self.unwrapped.data.xfrc_applied[1, 2] = self.torsoforce_z  # type: ignore
        if self.backthighforce_x is not None:
            self.unwrapped.data.xfrc_applied[2, 0] = self.backthighforce_x  # type: ignore
        if self.backthighforce_y is not None:
            self.unwrapped.data.xfrc_applied[2, 1] = self.backthighforce_y  # type: ignore
        if self.backthighforce_z is not None:
            self.unwrapped.data.xfrc_applied[2, 2] = self.backthighforce_z  # type: ignore
        if self.backshinforce_x is not None:
            self.unwrapped.data.xfrc_applied[3, 0] = self.backshinforce_x  # type: ignore
        if self.backshinforce_y is not None:
            self.unwrapped.data.xfrc_applied[3, 1] = self.backshinforce_y  # type: ignore
        if self.backshinforce_z is not None:
            self.unwrapped.data.xfrc_applied[3, 2] = self.backshinforce_z  # type: ignore
        if self.backfootforce_x is not None:
            self.unwrapped.data.xfrc_applied[4, 0] = self.backfootforce_x  # type: ignore
        if self.backfootforce_y is not None:
            self.unwrapped.data.xfrc_applied[4, 1] = self.backfootforce_y  # type: ignore
        if self.backfootforce_z is not None:
            self.unwrapped.data.xfrc_applied[4, 2] = self.backfootforce_z  # type: ignore
        if self.forwardthighforce_x is not None:
            self.unwrapped.data.xfrc_applied[5, 0] = self.forwardthighforce_x  # type: ignore
        if self.forwardthighforce_y is not None:
            self.unwrapped.data.xfrc_applied[5, 1] = self.forwardthighforce_y  # type: ignore
        if self.forwardthighforce_z is not None:
            self.unwrapped.data.xfrc_applied[5, 2] = self.forwardthighforce_z  # type: ignore
        if self.forwardshinforce_x is not None:
            self.unwrapped.data.xfrc_applied[6, 0] = self.forwardshinforce_x  # type: ignore
        if self.forwardshinforce_y is not None:
            self.unwrapped.data.xfrc_applied[6, 1] = self.forwardshinforce_y  # type: ignore
        if self.forwardshinforce_z is not None:
            self.unwrapped.data.xfrc_applied[6, 2] = self.forwardshinforce_z  # type: ignore
        if self.forwardfootforce_x is not None:
            self.unwrapped.data.xfrc_applied[7, 0] = self.forwardfootforce_x  # type: ignore
        if self.forwardfootforce_y is not None:
            self.unwrapped.data.xfrc_applied[7, 1] = self.forwardfootforce_y  # type: ignore
        if self.forwardfootforce_z is not None:
            self.unwrapped.data.xfrc_applied[7, 2] = self.forwardfootforce_z  # type: ignore

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
