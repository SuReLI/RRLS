from __future__ import annotations

from gymnasium.envs.registration import register

from .ant import RobustAnt
from .half_cheetah import RobustHalfCheetah
from .hopper import RobustHopper
from .humanoid import RobustHumanoidStandUp
from .pendulum import RobustInvertedPendulum
from .walker import RobustWalker2d

register(
    id="robust-halfcheetah-v4",
    entry_point="rrlcs.envs.half_cheetah:RobustHalfCheehtah",
    max_episode_steps=1000,
)

register(
    id="robust-ant-v4",
    entry_point="rrlcs.envs.ant:RobustAnt",
    max_episode_steps=1000,
)
register(
    id="robust-hopper-v4",
    entry_point="rrlcs.envs.hopper:RobustHopper",
    max_episode_steps=1000,
)
register(
    id="robust-humanoidstandup-v4",
    entry_point="rrlcs.envs.humanoid:RobustHumanoidStandUp",
    max_episode_steps=1000,
)
register(
    id="robust-invertedpendulum-v4",
    entry_point="rrlcs.envs.pendulum:RobustInvertedPendulum",
    max_episode_steps=1000,
)
register(
    id="robust-walker-v4",
    entry_point="rrlcs.envs.walker:RobustWalker2d",
    max_episode_steps=1000,
)

__all__ = [
    "RobustAnt",
    "RobustHalfCheetah",
    "RobustHopper",
    "RobustHumanoidStandUp",
    "RobustInvertedPendulum",
    "RobustWalker2d",
]
