from __future__ import annotations

from enum import Enum

from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv


class HumanoidStandupParamsBound(Enum):
    ONE_DIM = {
        "torsomass": [0.1, 16.0],
    }
    TWO_DIM = {
        "torsomass": [0.1, 16.0],
        "rightfootmass": [0.1, 8.0],
    }
    THREE_DIM = {
        "torsomass": [0.1, 16.0],
        "leftthighmass": [0.1, 5.0],
        "rightfootmass": [0.1, 8.0],
    }


class RobustHumanoidStandUp(HumanoidStandupEnv):
    ONE_DIM_PARAMS_BOUND_16 = {
        "torsomass": [0.1, 16.0],
    }
    TWO_DIM_PARAMS_BOUND_16_8 = {
        "torsomass": [0.1, 16.0],
        "rightfootmass": [0.1, 8.0],
    }
    THREE_DIM_PARAMS_BOUND_16_5_8 = {
        "torsomass": [0.1, 16.0],
        "leftthighmass": [0.1, 5.0],
        "rightfootmass": [0.1, 8.0],
    }

    def __init__(
        self,
        torsomass: float | None = None,
        leftthighmass: float | None = None,
        rightfootmass: float | None = None,
    ):
        self.torsomass = torsomass
        self.leftthighmass = leftthighmass
        self.rightfootmass = rightfootmass
        super().__init__()

        self.change_params()

    def set_params(
        self,
        torsomass: float | None = None,
        leftthighmass: float | None = None,
        rightfootmass: float | None = None,
    ):
        self.torsomass = torsomass
        self.leftthighmass = leftthighmass
        self.rightfootmass = rightfootmass

    def get_params(self):
        return {
            "torsomass": self.torsomass,
            "leftthighmass": self.leftthighmass,
            "rightfootmass": self.rightfootmass,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def change_params(self):
        if self.torsomass is not None:
            self.model.body_mass[1] = self.torsomass

        if self.leftthighmass is not None:
            self.model.body_mass[7] = self.leftthighmass

        if self.rightfootmass is not None:
            self.model.body_mass[6] = self.rightfootmass
