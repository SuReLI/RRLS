from __future__ import annotations

from gymnasium.envs.registration import register

from . import envs, wrapper
from .evaluate import (
    EVALUATION_ROBUST_ANT_1D,
    EVALUATION_ROBUST_ANT_2D,
    EVALUATION_ROBUST_ANT_3D,
    EVALUATION_ROBUST_HALF_CHEETAH_1D,
    EVALUATION_ROBUST_HALF_CHEETAH_2D,
    EVALUATION_ROBUST_HALF_CHEETAH_3D,
    EVALUATION_ROBUST_HOPPER_1D,
    EVALUATION_ROBUST_HOPPER_2D,
    EVALUATION_ROBUST_HOPPER_3D,
    EVALUATION_ROBUST_HUMANOID_STANDUP_1D,
    EVALUATION_ROBUST_HUMANOID_STANDUP_2D,
    EVALUATION_ROBUST_HUMANOID_STANDUP_3D,
    EVALUATION_ROBUST_INVERTED_PENDULUM_1D,
    EVALUATION_ROBUST_INVERTED_PENDULUM_2D,
    EVALUATION_ROBUST_WALKER_1D,
    EVALUATION_ROBUST_WALKER_2D,
    EVALUATION_ROBUST_WALKER_3D,
    generate_evaluation_set,
)

__all__ = [
    "envs",
    "wrapper",
    "generate_evaluation_set",
    "EVALUATION_ROBUST_ANT_1D",
    "EVALUATION_ROBUST_ANT_2D",
    "EVALUATION_ROBUST_ANT_3D",
    "EVALUATION_ROBUST_HALF_CHEETAH_1D",
    "EVALUATION_ROBUST_HALF_CHEETAH_2D",
    "EVALUATION_ROBUST_HALF_CHEETAH_3D",
    "EVALUATION_ROBUST_HOPPER_1D",
    "EVALUATION_ROBUST_HOPPER_2D",
    "EVALUATION_ROBUST_HOPPER_3D",
    "EVALUATION_ROBUST_HUMANOID_STANDUP_1D",
    "EVALUATION_ROBUST_HUMANOID_STANDUP_2D",
    "EVALUATION_ROBUST_HUMANOID_STANDUP_3D",
    "EVALUATION_ROBUST_INVERTED_PENDULUM_1D",
    "EVALUATION_ROBUST_INVERTED_PENDULUM_2D",
    "EVALUATION_ROBUST_WALKER_1D",
    "EVALUATION_ROBUST_WALKER_2D",
    "EVALUATION_ROBUST_WALKER_3D",
]


def make_wrapped_env(cls_env, wrapper, **kwargs):
    """ """
    env = cls_env()
    wrapped_env = wrapper(env=env, **kwargs)
    return wrapped_env


def register_robotics_envs():
    register(
        id="rrls:robust-halfcheetah",
        entry_point="rrls.envs.half_cheetah:RobustHalfCheetah",
        max_episode_steps=1000,
    )

    register(
        id="rrls:robust-ant",
        entry_point="rrls.envs.ant:RobustAnt",
        max_episode_steps=1000,
    )
    register(
        id="rrls:robust-hopper",
        entry_point="rrls.envs.hopper:RobustHopper",
        max_episode_steps=1000,
    )
    register(
        id="rrls:robust-humanoidstandup",
        entry_point="rrls.envs.humanoid:RobustHumanoidStandUp",
        max_episode_steps=1000,
    )
    register(
        id="rrls:robust-invertedpendulum",
        entry_point="rrls.envs.pendulum:RobustInvertedPendulum",
        max_episode_steps=1000,
    )
    register(
        id="rrls:robust-walker",
        entry_point="rrls.envs.walker:RobustWalker2d",
        max_episode_steps=1000,
    )

    # Advserarial environments
    # HalfCheetah
    register(
        id="rrls:robust-halfcheetah-adversarial-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-halfcheetah-adversarial-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-halfcheetah-adversarial-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.ONE_DIM.value,
        },
    )
    # Ant
    register(
        id="rrls:robust-ant-adversarial-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-ant-adversarial-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-ant-adversarial-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.ONE_DIM.value,
        },
    )
    # Hopper
    register(
        id="rrls:robust-hopper-adversarial-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-hopper-adversarial-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-hopper-adversarial-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.ONE_DIM.value,
        },
    )

    # HumanoidStandUp
    register(
        id="rrls:robust-humanoidstandup-adversarial-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-humanoidstandup-adversarial-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-humanoidstandup-adversarial-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrapper.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM.value,
        },
    )

    # InvertedPendulum

    register(
        id="rrls:robust-invertedpendulum-adversarial-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DynamicAdversarial,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM.value,
        },
    )

    register(
        id="rrls:robust-invertedpendulum-adversarial-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DynamicAdversarial,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM.value,
        },
    )

    # Domainrandomization environments

    # HalfCheetah
    register(
        id="rrls:robust-halfcheetah-domain-randomization-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-halfcheetah-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-halfcheetah-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.ONE_DIM.value,
        },
    )

    # Ant
    register(
        id="rrls:robust-ant-domain-randomization-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-ant-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-ant-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.ONE_DIM.value,
        },
    )

    # Hopper
    register(
        id="rrls:robust-hopper-domain-randomization-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.THREE_DIM.value,
        },
    )

    register(
        id="rrls:robust-hopper-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-hopper-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.ONE_DIM.value,
        },
    )
    # HumanoidStandUp
    register(
        id="rrls:robust-humanoidstandup-domain-randomization-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-humanoidstandup-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-humanoidstandup-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM.value,
        },
    )
    # InvertedPendulum
    register(
        id="rrls:robust-invertedpendulum-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-invertedpendulum-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM.value,
        },
    )
    # Walker2d
    register(
        id="rrls:robust-walker-domain-randomization-3d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls:robust-walker-domain-randomization-2d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls:robust-walker-domain-randomization-1d",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.ONE_DIM.value,
        },
    )
    register(
        id="rrls:probabilistic-action-robust-halfcheetah",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustHalfCheetah,
            "alpha": 0.1,
        },
    )

    register(
        id="rrls:probabilistic-action-robust-ant",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustAnt,
            "alpha": 0.1,
        },
    )

    register(
        id="rrls:probabilistic-action-robust-hopper",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustHopper,
            "alpha": 0.1,
        },
    )

    register(
        id="rrls:probabilistic-action-robust-humanoidstandup",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustHumanoidStandUp,
            "alpha": 0.1,
        },
    )

    register(
        id="rrls:probabilistic-action-robust-invertedpendulum",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustInvertedPendulum,
            "alpha": 0.1,
        },
    )

    register(
        id="rrls:probabilistic-action-robust-walker",
        entry_point=make_wrapped_env,  # type: ignore
        max_episode_steps=1000,
        kwargs={
            "wrapper": wrapper.ProbalistActionRobustWrapper,
            "cls_env": envs.RobustWalker2d,
            "alpha": 0.1,
        },
    )
