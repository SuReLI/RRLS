from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym


class ModifiedParams(Protocol):
    """
    The `ModifiedParams` Protocol defines a set of methods for modifying and accessing parameters.
    """

    def change_params(self, **kwargs):
        """
        Changes parameters according to provided keyword arguments.

        Args:
            kwargs (dict): parameters to be changed along with their new values.
        """
        ...

    def set_params(self, **kwargs):
        """
        Sets parameters according to provided keyword arguments.

        Args:
            kwargs (dict): parameters to be set along with their values.
        """
        ...

    def get_params(self) -> dict[str, float]:
        """
        Fetches the current parameters.

        Returns:
            dict: A dictionary with the current parameters and their values.
        """
        ...


# Python does not support type addition so we need to create a new class
# Unlike Rust where we can just add to the function signature ModifiedPhysics + gym.Env
class ModifiedParamsEnv(ModifiedParams, gym.Env):
    """
    The `ModifiedParamsEnv` class is a Gym Environment that supports parameter modifications.
    """

    def change_params(self, **kwargs):
        """
        Changes parameters according to provided keyword arguments.

        Args:
            kwargs (dict): parameters to be changed along with their new values.
        """
        ...

    def set_params(self, **kwargs):
        """
        Sets parameters according to provided keyword arguments.

        Args:
            kwargs (dict): parameters to be set along with their values.
        """
        ...

    def get_params(self):
        """
        Fetches the current parameters.

        Returns:
            dict: A dictionary with the current parameters and their values.
        """
        ...


def check_protocol_modified_params_env(obj: Any) -> bool:
    """
    Checks if an object complies with the `ModifiedParamsEnv` protocol.
    """
    required_methods = ["change_params", "set_params", "get_params"]
    protocal_compliant = all(
        callable(getattr(obj, method, None)) for method in required_methods
    )
    is_gym_env = isinstance(obj, gym.Env)
    return protocal_compliant and is_gym_env
