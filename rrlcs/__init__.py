from __future__ import annotations

from gymnasium.envs.registration import register

from . import envs, wrapper
from ._interface import (
    ModifiedParams,
    ModifiedParamsEnv,
    check_protocol_modified_params_env,
)


def register_robotics_envs():
    """ """
    pass


def make_wrapped_env(cls_env, wrapper, params_bound):
    """ """
    env = cls_env()
    wrapped_env = wrapper(env=env, params_bound=params_bound)
    return wrapped_env


# Base and modifiable environments
register(
    id="robust-halfcheetah",
    entry_point="rrlcs.envs.half_cheetah:RobustHalfCheetah",
    max_episode_steps=1000,
)

register(
    id="robust-ant",
    entry_point="rrlcs.envs.ant:RobustAnt",
    max_episode_steps=1000,
)
register(
    id="robust-hopper",
    entry_point="rrlcs.envs.hopper:RobustHopper",
    max_episode_steps=1000,
)
register(
    id="robust-humanoidstandup",
    entry_point="rrlcs.envs.humanoid:RobustHumanoidStandUp",
    max_episode_steps=1000,
)
register(
    id="robust-invertedpendulum",
    entry_point="rrlcs.envs.pendulum:RobustInvertedPendulum",
    max_episode_steps=1000,
)
register(
    id="robust-walker",
    entry_point="rrlcs.envs.walker:RobustWalker2d",
    max_episode_steps=1000,
)


# Advserarial environments
# HalfCheetah
register(
    id="robust-halfcheetah-adversarial-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHalfCheetah,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HalfCheetahParamsBound.THREE_DIM,
    },
)
register(
    id="robust-halfcheetah-adversarial-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHalfCheetah,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HalfCheetahParamsBound.TWO_DIM,
    },
)
register(
    id="robust-halfcheetah-adversarial-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHalfCheetah,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HalfCheetahParamsBound.ONE_DIM,
    },
)
# Ant
register(
    id="robust-ant-adversarial-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustAnt,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.AntParamsBound.THREE_DIM,
    },
)
register(
    id="robust-ant-adversarial-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustAnt,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.AntParamsBound.TWO_DIM,
    },
)
register(
    id="robust-ant-adversarial-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustAnt,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.AntParamsBound.ONE_DIM,
    },
)
# Hopper
register(
    id="robust-hopper-adversarial-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHopper,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HopperParamsBound.THREE_DIM,
    },
)
register(
    id="robust-hopper-adversarial-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHopper,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HopperParamsBound.TWO_DIM,
    },
)
register(
    id="robust-hopper-adversarial-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHopper,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HopperParamsBound.ONE_DIM,
    },
)

# HumanoidStandUp
register(
    id="robust-humanoidstandup-adversarial-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHumanoidStandUp,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM,
    },
)
register(
    id="robust-humanoidstandup-adversarial-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHumanoidStandUp,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM,
    },
)
register(
    id="robust-humanoidstandup-adversarial-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "cls_env": envs.RobustHumanoidStandUp,
        "wrapper": wrapper.DynamicAdversarial,
        "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM,
    },
)

# InvertedPendulum

register(
    id="robust-invertedpendulum-adversarial-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DynamicAdversarial,
        "cls_env": envs.RobustInvertedPendulum,
        "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM,
    },
)

register(
    id="robust-invertedpendulum-adversarial-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DynamicAdversarial,
        "cls_env": envs.RobustInvertedPendulum,
        "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM,
    },
)

# Domainrandomization environments

# HalfCheetah
register(
    id="robust-halfcheetah-domain-randomization-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHalfCheetah,
        "params_bound": envs.HalfCheetahParamsBound.THREE_DIM,
    },
)
register(
    id="robust-halfcheetah-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHalfCheetah,
        "params_bound": envs.HalfCheetahParamsBound.TWO_DIM,
    },
)
register(
    id="robust-halfcheetah-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHalfCheetah,
        "params_bound": envs.HalfCheetahParamsBound.ONE_DIM,
    },
)

# Ant
register(
    id="robust-ant-domain-randomization-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustAnt,
        "params_bound": envs.AntParamsBound.THREE_DIM,
    },
)
register(
    id="robust-ant-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustAnt,
        "params_bound": envs.AntParamsBound.TWO_DIM,
    },
)
register(
    id="robust-ant-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustAnt,
        "params_bound": envs.AntParamsBound.ONE_DIM,
    },
)

# Hopper
register(
    id="robust-hopper-domain-randomization-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHopper,
        "params_bound": envs.HopperParamsBound.THREE_DIM,
    },
)

register(
    id="robust-hopper-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHopper,
        "params_bound": envs.HopperParamsBound.TWO_DIM,
    },
)
register(
    id="robust-hopper-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHopper,
        "params_bound": envs.HopperParamsBound.ONE_DIM,
    },
)
# HumanoidStandUp
register(
    id="robust-humanoidstandup-domain-randomization-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHumanoidStandUp,
        "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM,
    },
)
register(
    id="robust-humanoidstandup-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHumanoidStandUp,
        "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM,
    },
)
register(
    id="robust-humanoidstandup-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustHumanoidStandUp,
        "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM,
    },
)
# InvertedPendulum
register(
    id="robust-invertedpendulum-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustInvertedPendulum,
        "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM,
    },
)
register(
    id="robust-invertedpendulum-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustInvertedPendulum,
        "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM,
    },
)
# Walker2d
register(
    id="robust-walker-domain-randomization-3d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustWalker2d,
        "params_bound": envs.Walker2dParamsBound.THREE_DIM,
    },
)
register(
    id="robust-walker-domain-randomization-2d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustWalker2d,
        "params_bound": envs.Walker2dParamsBound.TWO_DIM,
    },
)
register(
    id="robust-walker-domain-randomization-1d",
    entry_point=make_wrapped_env,  # type: ignore
    max_episode_steps=1000,
    kwargs={
        "wrapper": wrapper.DomainRandomization,
        "cls_env": envs.RobustWalker2d,
        "params_bound": envs.Walker2dParamsBound.ONE_DIM,
    },
)
__all__ = [
    "check_protocol_modified_params_env",
    "ModifiedParams",
    "ModifiedParamsEnv",
    "envs",
    "wrapper",
]
