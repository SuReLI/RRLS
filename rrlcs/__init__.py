from __future__ import annotations

from . import envs, wrapper
from ._interface import ModifiedParams, ModifiedParamsEnv


def register_robotics_envs():
    """ """
    pass


__all__ = [
    "ModifiedParams",
    "ModifiedParamsEnv",
    "envs",
    "wrapper",
]
