from __future__ import annotations

from typing import Any
from enum import Enum

import gymnasium as gym
from gymnasium import Wrapper


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


class RobustHumanoidStandUp(Wrapper):
    """
    Robust Humanoid environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - torsomass
        - lwaistmass
        - pelvismass
        - rightthighmass
        - rightshinmass
        - rightfootmass
        - leftthighmass
        - leftshinmass
        - leftfootmass
        - rightupperarmmass
        - rightlowerarmmass
        - leftupperarmmass
        - leftlowerarmmass
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
        torsomass: float | None = None,
        lwaistmass: float | None = None,
        pelvismass: float | None = None,
        rightthighmass: float | None = None,
        rightshinmass: float | None = None,
        rightfootmass: float | None = None,
        leftthighmass: float | None = None,
        leftshinmass: float | None = None,
        leftfootmass: float | None = None,
        rightupperarmmass: float | None = None,
        rightlowerarmmass: float | None = None,
        leftupperarmmass: float | None = None,
        leftlowerarmmass: float | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(env=gym.make("HumanoidStandup-v5", **kwargs))

        self.set_params(
            torsomass=torsomass,
            lwaistmass=lwaistmass,
            pelvismass=pelvismass,
            rightthighmass=rightthighmass,
            rightshinmass=rightshinmass,
            rightfootmass=rightfootmass,
            leftthighmass=leftthighmass,
            leftshinmass=leftshinmass,
            leftfootmass=leftfootmass,
            rightupperarmmass=rightupperarmmass,
            rightlowerarmmass=rightlowerarmmass,
            leftupperarmmass=leftupperarmmass,
            leftlowerarmmass=leftlowerarmmass,
        )

        self._change_params()

    def set_params(
        self,
        torsomass: float | None = None,
        lwaistmass: float | None = None,
        pelvismass: float | None = None,
        rightthighmass: float | None = None,
        rightshinmass: float | None = None,
        rightfootmass: float | None = None,
        leftthighmass: float | None = None,
        leftshinmass: float | None = None,
        leftfootmass: float | None = None,
        rightupperarmmass: float | None = None,
        rightlowerarmmass: float | None = None,
        leftupperarmmass: float | None = None,
        leftlowerarmmass: float | None = None,
    ):
        self.torsomass = torsomass
        self.lwaistmass = lwaistmass
        self.pelvismass = pelvismass
        self.rightthighmass = rightthighmass
        self.rightshinmass = rightshinmass
        self.rightfootmass = rightfootmass
        self.leftthighmass = leftthighmass
        self.leftshinmass = leftshinmass
        self.leftfootmass = leftfootmass
        self.rightupperarmmass = rightupperarmmass
        self.rightlowerarmmass = rightlowerarmmass
        self.leftupperarmmass = leftupperarmmass
        self.leftlowerarmmass = leftlowerarmmass
        self._change_params()

    def get_params(self):
        return {
            "torsomass": self.torsomass,
            "lwaistmass": self.lwaistmass,
            "pelvismass": self.pelvismass,
            "rightthighmass": self.rightthighmass,
            "rightshinmass": self.rightshinmass,
            "rightfootmass": self.rightfootmass,
            "leftthighmass": self.leftthighmass,
            "leftshinmass": self.leftshinmass,
            "leftfootmass": self.leftfootmass,
            "rightupperarmmass": self.rightupperarmmass,
            "rightlowerarmmass": self.rightlowerarmmass,
            "leftupperarmmass": self.leftupperarmmass,
            "leftlowerarmmass": self.leftlowerarmmass,
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

        if self.lwaistmass is not None:
            self.unwrapped.model.body_mass[2] = self.lwaistmass

        if self.pelvismass is not None:
            self.unwrapped.model.body_mass[3] = self.pelvismass

        if self.rightthighmass is not None:
            self.unwrapped.model.body_mass[4] = self.rightthighmass

        if self.rightshinmass is not None:
            self.unwrapped.model.body_mass[5] = self.rightshinmass

        if self.rightfootmass is not None:
            self.unwrapped.model.body_mass[6] = self.rightfootmass

        if self.leftthighmass is not None:
            self.unwrapped.model.body_mass[7] = self.leftthighmass

        if self.leftshinmass is not None:
            self.unwrapped.model.body_mass[8] = self.leftshinmass

        if self.leftfootmass is not None:
            self.unwrapped.model.body_mass[9] = self.leftfootmass

        if self.rightupperarmmass is not None:
            self.unwrapped.model.body_mass[10] = self.rightupperarmmass

        if self.rightlowerarmmass is not None:
            self.unwrapped.model.body_mass[11] = self.rightlowerarmmass

        if self.leftupperarmmass is not None:
            self.unwrapped.model.body_mass[12] = self.leftupperarmmass

        if self.leftlowerarmmass is not None:
            self.unwrapped.model.body_mass[13] = self.leftlowerarmmass
