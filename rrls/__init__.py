from __future__ import annotations

from gymnasium.envs.registration import register

from . import envs, wrappers
from .evaluate import generate_evaluation_set

__all__ = [
    "envs",
    "wrappers",
    "generate_evaluation_set",
]


def make_wrapped_env(cls_env, wrapper, **kwargs):
    """ """
    env = cls_env()
    wrapped_env = wrapper(env=env, **kwargs)
    return wrapped_env


def register_robotics_envs():
    register(
        id="rrls/robust-halfcheetah-v0",
        entry_point="rrls.envs.half_cheetah:RobustHalfCheetah",
        order_enforce=False,
        disable_env_checker=True,
    )

    register(
        id="rrls/robust-ant-v0",
        entry_point="rrls.envs.ant:RobustAnt",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/robust-hopper-v0",
        entry_point="rrls.envs.hopper:RobustHopper",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/robust-humanoidstandup-v0",
        entry_point="rrls.envs.humanoid:RobustHumanoidStandUp",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/robust-invertedpendulum-v0",
        entry_point="rrls.envs.pendulum:RobustInvertedPendulum",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/robust-walker-v0",
        entry_point="rrls.envs.walker:RobustWalker2d",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-ant-v0",
        entry_point="rrls.envs.ant:ForceAnt",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-halfcheetah-v0",
        entry_point="rrls.envs.half_cheetah:ForceHalfCheetah",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-hopper-v0",
        entry_point="rrls.envs.hopper:ForceHopper",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-humanoidstandup-v0",
        entry_point="rrls.envs.humanoid:ForceHumanoidStandUp",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-invertedpendulum-v0",
        entry_point="rrls.envs.pendulum:ForceInvertedPendulum",
        order_enforce=False,
        disable_env_checker=True,
    )
    register(
        id="rrls/force-walker-v0",
        entry_point="rrls.envs.walker:ForceWalker2d",
        order_enforce=False,
        disable_env_checker=True,
    )
    


    # Advserarial environments
    # HalfCheetah
    register(
        id="rrls/robust-halfcheetah-adversarial-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-halfcheetah-adversarial-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-halfcheetah-adversarial-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHalfCheetah,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.ONE_DIM.value,
        },
    )
    # Ant
    register(
        id="rrls/robust-ant-adversarial-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-ant-adversarial-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-ant-adversarial-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustAnt,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.ONE_DIM.value,
        },
    )
    register(id="rrls/robust-ant-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.ForceAnt,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.AntParamsBound.RARL.value,
        },
    )
    register(id="rrls/robust-halfcheetah-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.ForceHalfCheetah,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HalfCheetahParamsBound.RARL.value,
        },
    )
    register(id="rrls/robust-hopper-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.ForceHopper,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.RARL.value,
        },
    )
    register(id="rrls/robust-humanoidstandup-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.ForceHumanoidStandUp,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.RARL.value,
        },
    )
    register(id="rrls/robust-invertedpendulum-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.ForceInvertedPendulum,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.InvertedPendulumParamsBound.RARL.value,
        },
    )
    register(id="rrls/robust-walker-adversarial-forces-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={ 
            "cls_env": envs.ForceWalker2d,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.Walker2dParamsBound.RARL.value,
        },
    )

    # Hopper
    register(
        id="rrls/robust-hopper-adversarial-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-hopper-adversarial-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-hopper-adversarial-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHopper,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HopperParamsBound.ONE_DIM.value,
        },
    )

    # HumanoidStandUp
    register(
        id="rrls/robust-humanoidstandup-adversarial-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-humanoidstandup-adversarial-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-humanoidstandup-adversarial-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "cls_env": envs.RobustHumanoidStandUp,
            "wrapper": wrappers.DynamicAdversarial,
            "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM.value,
        },
    )

    # InvertedPendulum

    register(
        id="rrls/robust-invertedpendulum-adversarial-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DynamicAdversarial,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM.value,
        },
    )

    register(
        id="rrls/robust-invertedpendulum-adversarial-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DynamicAdversarial,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM.value,
        },
    )

    # Domainrandomization environments

    # HalfCheetah
    register(
        id="rrls/robust-halfcheetah-domain-randomization-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-halfcheetah-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-halfcheetah-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHalfCheetah,
            "params_bound": envs.HalfCheetahParamsBound.ONE_DIM.value,
        },
    )

    # Ant
    register(
        id="rrls/robust-ant-domain-randomization-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-ant-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-ant-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustAnt,
            "params_bound": envs.AntParamsBound.ONE_DIM.value,
        },
    )

    # Hopper
    register(
        id="rrls/robust-hopper-domain-randomization-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.THREE_DIM.value,
        },
    )

    register(
        id="rrls/robust-hopper-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-hopper-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHopper,
            "params_bound": envs.HopperParamsBound.ONE_DIM.value,
        },
    )
    # HumanoidStandUp
    register(
        id="rrls/robust-humanoidstandup-domain-randomization-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-humanoidstandup-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-humanoidstandup-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustHumanoidStandUp,
            "params_bound": envs.HumanoidStandupParamsBound.ONE_DIM.value,
        },
    )
    # InvertedPendulum
    register(
        id="rrls/robust-invertedpendulum-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-invertedpendulum-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustInvertedPendulum,
            "params_bound": envs.InvertedPendulumParamsBound.ONE_DIM.value,
        },
    )
    # Walker2d
    register(
        id="rrls/robust-walker-domain-randomization-3d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.THREE_DIM.value,
        },
    )
    register(
        id="rrls/robust-walker-domain-randomization-2d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.TWO_DIM.value,
        },
    )
    register(
        id="rrls/robust-walker-domain-randomization-1d-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.DomainRandomization,
            "cls_env": envs.RobustWalker2d,
            "params_bound": envs.Walker2dParamsBound.ONE_DIM.value,
        },
    )
    register(
        id="probabilistic-action-robust-halfcheetah-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustHalfCheetah,
            "alpha": 0.1,
        },
    )

    register(
        id="probabilistic-action-robust-ant-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustAnt,
            "alpha": 0.1,
        },
    )

    register(
        id="probabilistic-action-robust-hopper-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustHopper,
            "alpha": 0.1,
        },
    )

    register(
        id="probabilistic-action-robust-humanoidstandup-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustHumanoidStandUp,
            "alpha": 0.1,
        },
    )

    register(
        id="probabilistic-action-robust-invertedpendulum-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustInvertedPendulum,
            "alpha": 0.1,
        },
    )

    register(
        id="probabilistic-action-robust-walker-v0",
        entry_point=make_wrapped_env,  # type: ignore
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "wrapper": wrappers.ProbabilisticActionRobust,
            "cls_env": envs.RobustWalker2d,
            "alpha": 0.1,
        },
    )


register_robotics_envs()
