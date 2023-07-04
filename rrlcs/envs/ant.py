from __future__ import annotations

from gymnasium.envs.mujoco.ant import AntEnv


class RobustAnt(AntEnv):
    ONE_DIM_PARAMS_BOUND_3 = {
        "torsomass": [0.1, 3.0],
    }
    TWO_DIM_PARAMS_BOUND_3_3 = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
    }
    THREE_DIM_PARAMS_BOUND_3_3_3 = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
        "frontrightlegmass": [0.01, 3.0],
    }

    def __init__(
        self,
        torsomass: float | None = None,
        frontleftlegmass: float | None = None,
        frontrightlegmass: float | None = None,
    ):
        self.torsomass = torsomass
        self.frontleftlegmass = frontleftlegmass
        self.frontrightlegmass = frontrightlegmass
        super().__init__()

        self.change_params()

    def set_params(
        self,
        torsomass: float | None = None,
        frontleftlegmass: float | None = None,
        frontrightlegmass: float | None = None,
    ):
        self.torsomass = torsomass
        self.frontleftlegmass = frontleftlegmass
        self.frontrightlegmass = frontrightlegmass

    def get_params(self):
        return {
            "torsomass": self.torsomass,
            "frontleftlegmass": self.frontleftlegmass,
            "frontrightlegmass": self.frontrightlegmass,
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
        if self.frontleftlegmass is not None:
            self.model.body_mass[2] = self.frontleftlegmass
        if self.frontrightlegmass is not None:
            self.model.body_mass[4] = self.frontrightlegmass
