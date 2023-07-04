from __future__ import annotations

from gymnasium.envs.mujoco.walker2d import Walker2dEnv


class RobustWalker2d(Walker2dEnv):
    ONE_DIM_PARAMS_BOUND_4 = {
        "worldfriction": [0.1, 4.0],
    }
    TWO_DIM_PARAMS_BOUND_4_5 = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 5.0],
    }
    THREE_DIM_PARAMS_BOUND_4_5_6 = {
        "worldfriction": [0.1, 4.0],
        "torsomass": [0.1, 5.0],
        "thighmass": [0.1, 6.0],
    }

    def __init__(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        thighmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.thighmass = thighmass
        super().__init__()

        self.change_physics()

    def set_params(
        self,
        worldfriction: float | None = None,
        torsomass: float | None = None,
        thighmass: float | None = None,
    ):
        self.worldfriction = worldfriction
        self.torsomass = torsomass
        self.thighmass = thighmass

    def get_params(self):
        return {
            "worldfriction": self.worldfriction,
            "torsomass": self.torsomass,
            "thighmass": self.thighmass,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.get_params())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self.get_params())
        return obs, reward, terminated, truncated, info

    def change_physics(self):
        if self.worldfriction is not None:
            self.model.geom_friction[0, 0] = self.worldfriction

        if self.torsomass is not None:
            self.model.body_mass[1] = self.torsomass

        if self.thighmass is not None:
            self.model.body_mass[2] = self.thighmass
