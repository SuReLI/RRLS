from __future__ import annotations

from gymnasium.envs.mujoco.inverted_pendulum import InvertedPendulumEnv


class RobustInvertedPendulum(InvertedPendulumEnv):
    ONE_DIM_PARAMS_BOUND_31 = {
        "polemass": [1, 31],
    }
    ONE_DIM_PARAMS_BOUND_9 = {
        "polemass": [1, 9],  # WATCH OUT in M2TD3 this params is weird
        # https://github.com/akimotolab/M2TD3/blob/main/configs/environment/InvertedPendulumv2-1_9.yaml
    }

    TWO_DIM_PARAMS_BOUND_31_11 = {
        "polemass": [1, 31],
        "cartmass": [1, 11],
    }

    def __init__(self, polemass: float | None = None, cartmass: float | None = None):
        self.polemass = polemass
        self.cartmass = cartmass
        super().__init__()

        self.change_physics()

    def set_params(self, polemass: float | None = None, cartmass: float | None = None):
        self.polemass = polemass
        self.cartmass = cartmass

    def get_params(self):
        return {
            "polemass": self.polemass,
            "cartmass": self.cartmass,
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
        if self.polemass is not None:
            self.model.body_mass[2] = self.polemass
        if self.cartmass is not None:
            self.model.body_mass[1] = self.cartmass
