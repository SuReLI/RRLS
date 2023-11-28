from __future__ import annotations

from enum import Enum

import gymnasium as gym
from gymnasium import Wrapper


class AntParamsBound(Enum):
    ONE_DIM = {
        "torsomass": [0.1, 3.0],
    }
    TWO_DIM = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
    }
    THREE_DIM = {
        "torsomass": [0.1, 3.0],
        "frontleftlegmass": [0.01, 3.0],
        "frontrightlegmass": [0.01, 3.0],
    }


class RobustAnt(Wrapper):
    """
    Robust Ant environment. You can change the parameters of the environment using options in
    the reset method or by using the set_params method. The parameters are changed by calling
    the change_params method. The parameters are:
        - torsomass
        - frontleftlegmass
        - frontleftlegauxmass
        - frontleftleganklemass
        - frontrightlegmass
        - frontrightlegauxmass
        - frontrightleganklemass
        - backleftlegmass
        - backleftlegauxmass
        - backleftleganklemass
        - backrightlegmass
        - backrightlegauxmass
        - backrightleganklemass
    """

    # HACK: This is a hack to avoid the following error:
    # gymnasium.error.InvalidMetadata: Expect the environment metadata to be dict, actual type: <class 'module'>
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
        frontleftlegmass: float | None = None,
        frontleftlegauxmass: float | None = None,
        frontleftleganklemass: float | None = None,
        frontrightlegmass: float | None = None,
        frontrightlegauxmass: float | None = None,
        frontrightleganklemass: float | None = None,
        backleftlegmass: float | None = None,
        backleftlegauxmass: float | None = None,
        backleftleganklemass: float | None = None,
        backrightlegmass: float | None = None,
        backrightlegauxmass: float | None = None,
        backrightleganklemass: float | None = None,
    ):
        super().__init__(env=gym.make("Ant-v5"))
        self.set_params(
            torsomass=torsomass,
            frontleftlegmass=frontleftlegmass,
            frontleftlegauxmass=frontleftlegauxmass,
            frontleftleganklemass=frontleftleganklemass,
            frontrightlegmass=frontrightlegmass,
            frontrightlegauxmass=frontrightlegauxmass,
            frontrightleganklemass=frontrightleganklemass,
            backleftlegmass=backleftlegmass,
            backleftlegauxmass=backleftlegauxmass,
            backleftleganklemass=backleftleganklemass,
            backrightlegmass=backrightlegmass,
            backrightlegauxmass=backrightlegauxmass,
            backrightleganklemass=backrightleganklemass,
        )
        self._change_params()

    def set_params(
        self,
        torsomass: float | None = None,
        frontleftlegmass: float | None = None,
        frontleftlegauxmass: float | None = None,
        frontleftleganklemass: float | None = None,
        frontrightlegmass: float | None = None,
        frontrightlegauxmass: float | None = None,
        frontrightleganklemass: float | None = None,
        backleftlegmass: float | None = None,
        backleftlegauxmass: float | None = None,
        backleftleganklemass: float | None = None,
        backrightlegmass: float | None = None,
        backrightlegauxmass: float | None = None,
        backrightleganklemass: float | None = None,
    ):
        self.torsomass = torsomass
        self.frontleftlegmass = frontleftlegmass
        self.frontleftlegauxmass = frontleftlegauxmass
        self.frontleftleganklemass = frontleftleganklemass
        self.frontrightlegmass = frontrightlegmass
        self.frontrightlegauxmass = frontrightlegauxmass
        self.frontrightleganklemass = frontrightleganklemass
        self.backleftlegmass = backleftlegmass
        self.backleftlegauxmass = backleftlegauxmass
        self.backleftleganklemass = backleftleganklemass
        self.backrightlegmass = backrightlegmass
        self.backrightlegauxmass = backrightlegauxmass
        self.backrightleganklemass = backrightleganklemass
        self._change_params()

    def get_params(self):
        return {
            "torsomass": self.torsomass,
            "frontleftlegmass": self.frontleftlegmass,
            "frontleftlegauxmass": self.frontleftlegauxmass,
            "frontleftleganklemass": self.frontleftleganklemass,
            "frontrightlegmass": self.frontrightlegmass,
            "frontrightlegauxmass": self.frontrightlegauxmass,
            "frontrightleganklemass": self.frontrightleganklemass,
            "backleftlegmass": self.backleftlegmass,
            "backleftlegauxmass": self.backleftlegauxmass,
            "backleftleganklemass": self.backleftleganklemass,
            "backrightlegmass": self.backrightlegmass,
            "backrightlegauxmass": self.backrightlegauxmass,
            "backrightleganklemass": self.backrightleganklemass,
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

        if self.frontleftlegmass is not None:
            self.unwrapped.model.body_mass[2] = self.frontleftlegmass

        if self.frontleftlegauxmass is not None:
            self.unwrapped.model.body_mass[3] = self.frontleftlegauxmass

        if self.frontleftleganklemass is not None:
            self.unwrapped.model.body_mass[4] = self.frontleftleganklemass

        if self.frontrightlegmass is not None:
            self.unwrapped.model.body_mass[5] = self.frontrightlegmass

        if self.frontrightlegauxmass is not None:
            self.unwrapped.model.body_mass[6] = self.frontrightlegauxmass

        if self.frontrightleganklemass is not None:
            self.unwrapped.model.body_mass[7] = self.frontrightleganklemass

        if self.backleftlegmass is not None:
            self.unwrapped.model.body_mass[8] = self.backleftlegmass

        if self.backleftlegauxmass is not None:
            self.unwrapped.model.body_mass[9] = self.backleftlegauxmass

        if self.backleftleganklemass is not None:
            self.unwrapped.model.body_mass[10] = self.backleftleganklemass

        if self.backrightlegmass is not None:
            self.unwrapped.model.body_mass[11] = self.backrightlegmass

        if self.backrightlegauxmass is not None:
            self.unwrapped.model.body_mass[12] = self.backrightlegauxmass

        if self.backrightleganklemass is not None:
            self.unwrapped.model.body_mass[13] = self.backrightleganklemass
