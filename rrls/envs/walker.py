from __future__ import annotations

from enum import Enum

from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv


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


class RobustWalker2d(Walker2dEnv):
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
    ):
        super().__init__()
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
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def _change_params(self):
        if self.worldfriction is not None:
            self.model.geom_friction[0, 0] = self.worldfriction

        if self.torsomass is not None:
            self.model.body_mass[1] = self.torsomass

        if self.thighmass is not None:
            self.model.body_mass[2] = self.thighmass

        if self.legmass is not None:
            self.model.body_mass[3] = self.legmass

        if self.footmass is not None:
            self.model.body_mass[4] = self.footmass

        if self.leftthighmass is not None:
            self.model.body_mass[5] = self.leftthighmass

        if self.leftlegmass is not None:
            self.model.body_mass[6] = self.leftlegmass

        if self.leftfootmass is not None:
            self.model.body_mass[7] = self.leftfootmass
