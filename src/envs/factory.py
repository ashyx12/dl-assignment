"""MiniGrid environment factory used by experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper


def make_env(
    env_id: str = "MiniGrid-Empty-5x5-v0",
    *,
    visual: bool = True,
    seed: int | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Create a reproducible MiniGrid environment.

    Args:
        env_id: Registered MiniGrid environment ID.
        visual: If True, expose the agent's partial RGB observation.
        seed: Optional seed used for the initial reset.
        **kwargs: Additional arguments passed to ``gym.make``.
    """
    env = gym.make(env_id, **kwargs)

    if visual:
        env = RGBImgPartialObsWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env
