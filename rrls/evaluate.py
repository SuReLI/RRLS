from __future__ import annotations

import itertools
from typing import Annotated, Callable

import numpy as np

from ._interface import ModifiedParamsEnv
from .envs import (
    AntParamsBound,
    HalfCheetahParamsBound,
    HopperParamsBound,
    HumanoidStandupParamsBound,
    InvertedPendulumParamsBound,
    RobustAnt,
    RobustHalfCheetah,
    RobustHumanoidStandUp,
    RobustInvertedPendulum,
    RobustWalker2d,
    Walker2dParamsBound,
)


def generate_evaluation_set(
    modified_env: Callable[[], ModifiedParamsEnv],
    param_bounds: dict[str, Annotated[list[float], 2]],
    nb_mesh_dim: int = 10,
) -> list[ModifiedParamsEnv]:
    """
    Generate a list of environments to be used for evaluation by meshing the parameter space.

    Args:
        modified_env (Callable[[], ModifiedParamsEnv]): A function that returns a modified environment.
        param_bounds (dict[str, Annotated[list[float], 2]]): Parameter boundaries.
        nb_mesh_dim (int): Number of mesh dimensions.

    Returns:
        list[ModifiedParamsEnv]: A list of environments to be used for evaluation.
    """
    # Generate all combinations of environments given the mesh
    eval_envs = []
    parameters_values = {
        parameter_name: np.arange(
            start=bound_value[0],
            stop=bound_value[1],  # type: ignore
            step=(bound_value[1] - bound_value[0]) / nb_mesh_dim,  # type: ignore
        ).tolist()
        for parameter_name, bound_value in param_bounds.items()
    }
    for values in itertools.product(*parameters_values.values()):
        params = dict(zip(parameters_values.keys(), values))
        env = modified_env(**params)
        eval_envs.append(env)

    return eval_envs


EVALUATION_ROBUST_ANT_1D = generate_evaluation_set(
    modified_env=RobustAnt,  # type: ignore
    param_bounds=AntParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_ANT_2D = generate_evaluation_set(
    modified_env=RobustAnt,  # type: ignore
    param_bounds=AntParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_ANT_3D = generate_evaluation_set(
    modified_env=RobustAnt,  # type: ignore
    param_bounds=AntParamsBound.THREE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HUMANOID_STANDUP_1D = generate_evaluation_set(
    modified_env=RobustHumanoidStandUp,  # type: ignore
    param_bounds=HumanoidStandupParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HUMANOID_STANDUP_2D = generate_evaluation_set(
    modified_env=RobustHumanoidStandUp,  # type: ignore
    param_bounds=HumanoidStandupParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HUMANOID_STANDUP_3D = generate_evaluation_set(
    modified_env=RobustHumanoidStandUp,  # type: ignore
    param_bounds=HumanoidStandupParamsBound.THREE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_WALKER_1D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=Walker2dParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_WALKER_2D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=Walker2dParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_WALKER_3D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=Walker2dParamsBound.THREE_DIM.value,
    nb_mesh_dim=10,
)
EVALUATION_ROBUST_HALF_CHEETAH_1D = generate_evaluation_set(
    modified_env=RobustHalfCheetah,  # type: ignore
    param_bounds=HalfCheetahParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HALF_CHEETAH_2D = generate_evaluation_set(
    modified_env=RobustHalfCheetah,  # type: ignore
    param_bounds=HalfCheetahParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HALF_CHEETAH_3D = generate_evaluation_set(
    modified_env=RobustHalfCheetah,  # type: ignore
    param_bounds=HalfCheetahParamsBound.THREE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_INVERTED_PENDULUM_1D = generate_evaluation_set(
    modified_env=RobustInvertedPendulum,  # type: ignore
    param_bounds=InvertedPendulumParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_INVERTED_PENDULUM_2D = generate_evaluation_set(
    modified_env=RobustInvertedPendulum,  # type: ignore
    param_bounds=InvertedPendulumParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HOPPER_1D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=HopperParamsBound.ONE_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HOPPER_2D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=HopperParamsBound.TWO_DIM.value,
    nb_mesh_dim=10,
)

EVALUATION_ROBUST_HOPPER_3D = generate_evaluation_set(
    modified_env=RobustWalker2d,  # type: ignore
    param_bounds=HopperParamsBound.THREE_DIM.value,
    nb_mesh_dim=10,
)
