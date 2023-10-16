from __future__ import annotations

from enum import Enum

from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv


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


class RobustHalfCheetah(HalfCheetahEnv):
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
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.backthighmass = backthighmass
        super().__init__()

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
        self.change_params()

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
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.backthighmass = backthighmass
        self.backshinmass = backshinmass
        self.backfootmass = backfootmass
        self.forwardthighmass = forwardthighmass
        self.forwardshinmass = forwardshinmass
        self.forwardfootmass = forwardfootmass

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
            self.change_params()

        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def change_params(self):
        if self.worldfriction is not None:
            self.model.geom_friction[:, 0] = self.worldfriction
        if self.torsomass is not None:
            self.model.body_mass[1] = self.torsomass
        if self.backthighmass is not None:
            self.model.body_mass[2] = self.backthighmass
        if self.backshinmass is not None:
            self.model.body_mass[3] = self.backshinmass
        if self.backfootmass is not None:
            self.model.body_mass[4] = self.backfootmass
        if self.forwardthighmass is not None:
            self.model.body_mass[5] = self.forwardthighmass
        if self.forwardshinmass is not None:
            self.model.body_mass[6] = self.forwardshinmass
        if self.forwardfootmass is not None:
            self.model.body_mass[7] = self.forwardfootmass
