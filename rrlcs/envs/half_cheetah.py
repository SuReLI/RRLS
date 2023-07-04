from __future__ import annotations

from gymnasium.envs.mujoco.half_cheetah import HalfCheetahEnv


class RobustHalfCheetah(HalfCheetahEnv):
    ONE_DIM_PARAMS_BOUND_3 = {
        "worldfriction": [0.1, 3.0],
    }
    ONE_DIM_PARAMS_BOUND_4 = {
        "worldfriction": [0.1, 4.0],
    }

    TWO_DIM_PARAMS_BOUND_4_7 = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 7.0],
    }
    THREE_DIM_PARAMS_BOUND_3_7_4 = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 7.0],
        "backthighmass": [0.1, 3.0],
    }

    def __init__(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        backthighmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.backthighmass = backthighmass
        super().__init__()

        self.change_params()

    def set_params(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        backthighmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.backthighmass = backthighmass

    def get_params(self):
        return {
            "worldfriction": self.worldfriction,
            "torsomass": self.torsomass,
            "backthighmass": self.backthighmass,
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
        if self.worldfriction is not None:
            self.model.geom_friction[:, 0] = self.worldfriction
        if self.torsomass is not None:
            self.model.body_mass[1] = self.torsomass
        if self.backthighmass is not None:
            self.model.body_mass[2] = self.backthighmass
