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
    RARL = {
        "torsoforce_x": [-3.0, 3.0],
        "torsoforce_y": [-3.0, 3.0],
        "rightthighforce_x": [-3.0, 3.0],
        "rightthighforce_y": [-3.0, 3.0],
        "leftfootforce_x": [-3.0, 3.0],
        "leftfootforce_y": [-3.0, 3.0],
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

class ForceHumanoidStandUp(Wrapper):
    """
    Force HUmanoidStandUp environment. You can apply forces to the environment using the set_params method.
    The parameters are changed by calling the change_params method. The parameters are:
    - torsoforce_x
    - torsoforce_y
    - torsoforce_z
    - lwaisforce_x
    - lwaisforce_y
    - lwaisforce_z
    - pelvisforce_x
    - pelvisforce_y
    - pelvisforce_z
    - rightthighforce_x
    - rightthighforce_y
    - rightthighforce_z
    - rightshinforce_x
    - rightshinforce_y
    - rightshinforce_z
    - rightfootforce_x
    - rightfootforce_y
    - rightfootforce_z
    - leftthighforce_x
    - leftthighforce_y
    - leftthighforce_z
    - leftshinforce_x
    - leftshinforce_y
    - leftshinforce_z
    - leftfootforce_x
    - leftfootforce_y
    - leftfootforce_z
    - rightupperarmforce_x
    - rightupperarmforce_y
    - rightupperarmforce_z
    - rightlowerarmforce_x
    - rightlowerarmforce_y
    - rightlowerarmforce_z
    - leftupperarmforce_x
    - leftupperarmforce_y
    - leftupperarmforce_z
    - leftlowerarmforce_x
    - leftlowerarmforce_y
    - leftlowerarmforce_z
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(env=gym.make("HumanoidStandup-v5", **kwargs))
        self.set_params()

    def set_params(
            self, 
            torsoforce_x: float | None = None,
            torsoforce_y: float | None = None,
            torsoforce_z: float | None = None,
            lwaisforce_x: float | None = None,
            lwaisforce_y: float | None = None,
            lwaisforce_z: float | None = None,
            pelvisforce_x: float | None = None,
            pelvisforce_y: float | None = None,
            pelvisforce_z: float | None = None,
            rightthighforce_x: float | None = None,
            rightthighforce_y: float | None = None,
            rightthighforce_z: float | None = None,
            rightshinforce_x: float | None = None,
            rightshinforce_y: float | None = None,
            rightshinforce_z: float | None = None,
            rightfootforce_x: float | None = None,
            rightfootforce_y: float | None = None,
            rightfootforce_z: float | None = None,
            leftthighforce_x: float | None = None,
            leftthighforce_y: float | None = None,
            leftthighforce_z: float | None = None,
            leftshinforce_x: float | None = None,
            leftshinforce_y: float | None = None,
            leftshinforce_z: float | None = None,
            leftfootforce_x: float | None = None,
            leftfootforce_y: float | None = None,
            leftfootforce_z: float | None = None,
            rightupperarmforce_x: float | None = None,
            rightupperarmforce_y: float | None = None,
            rightupperarmforce_z: float | None = None,
            rightlowerarmforce_x: float | None = None,
            rightlowerarmforce_y: float | None = None,
            rightlowerarmforce_z: float | None = None,
            leftupperarmforce_x: float | None = None,
            leftupperarmforce_y: float | None = None,
            leftupperarmforce_z: float | None = None,
            leftlowerarmforce_x: float | None = None,
            leftlowerarmforce_y: float | None = None,
            leftlowerarmforce_z: float | None = None,
        ):

            self.torsoforce_x = torsoforce_x
            self.torsoforce_y = torsoforce_y
            self.torsoforce_z = torsoforce_z
            self.lwaisforce_x = lwaisforce_x
            self.lwaisforce_y = lwaisforce_y
            self.lwaisforce_z = lwaisforce_z
            self.pelvisforce_x = pelvisforce_x
            self.pelvisforce_y = pelvisforce_y
            self.pelvisforce_z = pelvisforce_z
            self.rightthighforce_x = rightthighforce_x
            self.rightthighforce_y = rightthighforce_y
            self.rightthighforce_z = rightthighforce_z
            self.rightshinforce_x = rightshinforce_x
            self.rightshinforce_y = rightshinforce_y
            self.rightshinforce_z = rightshinforce_z
            self.rightfootforce_x = rightfootforce_x
            self.rightfootforce_y = rightfootforce_y
            self.rightfootforce_z = rightfootforce_z
            self.leftthighforce_x = leftthighforce_x
            self.leftthighforce_y = leftthighforce_y
            self.leftthighforce_z = leftthighforce_z
            self.leftshinforce_x = leftshinforce_x
            self.leftshinforce_y = leftshinforce_y
            self.leftshinforce_z = leftshinforce_z
            self.leftfootforce_x = leftfootforce_x
            self.leftfootforce_y = leftfootforce_y
            self.leftfootforce_z = leftfootforce_z
            self.rightupperarmforce_x = rightupperarmforce_x
            self.rightupperarmforce_y = rightupperarmforce_y
            self.rightupperarmforce_z = rightupperarmforce_z
            self.rightlowerarmforce_x = rightlowerarmforce_x
            self.rightlowerarmforce_y = rightlowerarmforce_y
            self.rightlowerarmforce_z = rightlowerarmforce_z
            self.leftupperarmforce_x = leftupperarmforce_x
            self.leftupperarmforce_y = leftupperarmforce_y
            self.leftupperarmforce_z = leftupperarmforce_z
            self.leftlowerarmforce_x = leftlowerarmforce_x
            self.leftlowerarmforce_y = leftlowerarmforce_y
            self.leftlowerarmforce_z = leftlowerarmforce_z
            self._change_params()        
        
    def get_params(self):
        return {
            "torsoforce_x": self.torsoforce_x,
            "torsoforce_y": self.torsoforce_y,
            "torsoforce_z": self.torsoforce_z,
            "lwaisforce_x": self.lwaisforce_x,
            "lwaisforce_y": self.lwaisforce_y,
            "lwaisforce_z": self.lwaisforce_z,
            "pelvisforce_x": self.pelvisforce_x,
            "pelvisforce_y": self.pelvisforce_y,
            "pelvisforce_z": self.pelvisforce_z,
            "rightthighforce_x": self.rightthighforce_x,
            "rightthighforce_y": self.rightthighforce_y,
            "rightthighforce_z": self.rightthighforce_z,
            "rightshinforce_x": self.rightshinforce_x,
            "rightshinforce_y": self.rightshinforce_y,
            "rightshinforce_z": self.rightshinforce_z,
            "rightfootforce_x": self.rightfootforce_x,
            "rightfootforce_y": self.rightfootforce_y,
            "rightfootforce_z": self.rightfootforce_z,
            "leftthighforce_x": self.leftthighforce_x,
            "leftthighforce_y": self.leftthighforce_y,
            "leftthighforce_z": self.leftthighforce_z,
            "leftshinforce_x": self.leftshinforce_x,
            "leftshinforce_y": self.leftshinforce_y,
            "leftshinforce_z": self.leftshinforce_z,
            "leftfootforce_x": self.leftfootforce_x,
            "leftfootforce_y": self.leftfootforce_y,
            "leftfootforce_z": self.leftfootforce_z,
            "rightupperarmforce_x": self.rightupperarmforce_x,
            "rightupperarmforce_y": self.rightupperarmforce_y,
            "rightupperarmforce_z": self.rightupperarmforce_z,
            "rightlowerarmforce_x": self.rightlowerarmforce_x,
            "rightlowerarmforce_y": self.rightlowerarmforce_y,
            "rightlowerarmforce_z": self.rightlowerarmforce_z,
            "leftupperarmforce_x": self.leftupperarmforce_x,
            "leftupperarmforce_y": self.leftupperarmforce_y,
            "leftupperarmforce_z": self.leftupperarmforce_z,
            "leftlowerarmforce_x": self.leftlowerarmforce_x,
            "leftlowerarmforce_y": self.leftlowerarmforce_y,
            "leftlowerarmforce_z": self.leftlowerarmforce_z,
        }

    def _change_params(self):
        if self.torsoforce_x is not None:
            self.unwrapped.data.xfrc_applied[1][0] = self.torsoforce_x

        if self.torsoforce_y is not None:
            self.unwrapped.data.xfrc_applied[1][1] = self.torsoforce_y

        if self.torsoforce_z is not None:
            self.unwrapped.data.xfrc_applied[1][2] = self.torsoforce_z

        if self.lwaisforce_x is not None:
            self.unwrapped.data.xfrc_applied[2][0] = self.lwaisforce_x

        if self.lwaisforce_y is not None:
            self.unwrapped.data.xfrc_applied[2][1] = self.lwaisforce_y

        if self.lwaisforce_z is not None:
            self.unwrapped.data.xfrc_applied[2][2] = self.lwaisforce_z

        if self.pelvisforce_x is not None:
            self.unwrapped.data.xfrc_applied[3][0] = self.pelvisforce_x

        if self.pelvisforce_y is not None:
            self.unwrapped.data.xfrc_applied[3][1] = self.pelvisforce_y

        if self.pelvisforce_z is not None:
            self.unwrapped.data.xfrc_applied[3][2] = self.pelvisforce_z

        if self.rightthighforce_x is not None:
            self.unwrapped.data.xfrc_applied[4][0] = self.rightthighforce_x

        if self.rightthighforce_y is not None:
            self.unwrapped.data.xfrc_applied[4][1] = self.rightthighforce_y

        if self.rightthighforce_z is not None:
            self.unwrapped.data.xfrc_applied[4][2] = self.rightthighforce_z

        if self.rightshinforce_x is not None:
            self.unwrapped.data.xfrc_applied[5][0] = self.rightshinforce_x

        if self.rightshinforce_y is not None:
            self.unwrapped.data.xfrc_applied[5][1] = self.rightshinforce_y

        if self.rightshinforce_z is not None:
            self.unwrapped.data.xfrc_applied[5][2] = self.rightshinforce_z

        if self.rightfootforce_x is not None:
            self.unwrapped.data.xfrc_applied[6][0] = self.rightfootforce_x

        if self.rightfootforce_y is not None:
            self.unwrapped.data.xfrc_applied[6][1] = self.rightfootforce_y  

        if self.rightfootforce_z is not None:   
            self.unwrapped.data.xfrc_applied[6][2] = self.rightfootforce_z

        if self.leftthighforce_x is not None:
            self.unwrapped.data.xfrc_applied[7][0] = self.leftthighforce_x

        if self.leftthighforce_y is not None:
            self.unwrapped.data.xfrc_applied[7][1] = self.leftthighforce_y

        if self.leftthighforce_z is not None:
            self.unwrapped.data.xfrc_applied[7][2] = self.leftthighforce_z

        if self.leftshinforce_x is not None:
            self.unwrapped.data.xfrc_applied[8][0] = self.leftshinforce_x

        if self.leftshinforce_y is not None:
            self.unwrapped.data.xfrc_applied[8][1] = self.leftshinforce_y

        if self.leftshinforce_z is not None:
            self.unwrapped.data.xfrc_applied[8][2] = self.leftshinforce_z

        if self.leftfootforce_x is not None:
            self.unwrapped.data.xfrc_applied[9][0] = self.leftfootforce_x

        if self.leftfootforce_y is not None:
            self.unwrapped.data.xfrc_applied[9][1] = self.leftfootforce_y

        if self.leftfootforce_z is not None:
            self.unwrapped.data.xfrc_applied[9][2] = self.leftfootforce_z

        if self.rightupperarmforce_x is not None:
            self.unwrapped.data.xfrc_applied[10][0] = self.rightupperarmforce_x

        if self.rightupperarmforce_y is not None:
            self.unwrapped.data.xfrc_applied[10][1] = self.rightupperarmforce_y

        if self.rightupperarmforce_z is not None:
            self.unwrapped.data.xfrc_applied[10][2] = self.rightupperarmforce_z

        if self.rightlowerarmforce_x is not None:
            self.unwrapped.data.xfrc_applied[11][0] = self.rightlowerarmforce_x

        if self.rightlowerarmforce_y is not None:
            self.unwrapped.data.xfrc_applied[11][1] = self.rightlowerarmforce_y

        if self.rightlowerarmforce_z is not None:
            self.unwrapped.data.xfrc_applied[11][2] = self.rightlowerarmforce_z

        if self.leftupperarmforce_x is not None:
            self.unwrapped.data.xfrc_applied[12][0] = self.leftupperarmforce_x

        if self.leftupperarmforce_y is not None:
            self.unwrapped.data.xfrc_applied[12][1] = self.leftupperarmforce_y

        if self.leftupperarmforce_z is not None:
            self.unwrapped.data.xfrc_applied[12][2] = self.leftupperarmforce_z

        if self.leftlowerarmforce_x is not None:
            self.unwrapped.data.xfrc_applied[13][0] = self.leftlowerarmforce_x

        if self.leftlowerarmforce_y is not None:
            self.unwrapped.data.xfrc_applied[13][1] = self.leftlowerarmforce_y

        if self.leftlowerarmforce_z is not None:
            self.unwrapped.data.xfrc_applied[13][2] = self.leftlowerarmforce_z

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