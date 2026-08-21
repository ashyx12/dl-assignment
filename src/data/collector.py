"""Trajectory collection utilities for MiniGrid experiments."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np

from envs import make_env


def collect_trajectory(
    seed: int,
    *,
    env_id: str = "MiniGrid-Empty-5x5-v0",
    max_steps: int = 100,
    actions: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Collect one trajectory from a seeded MiniGrid environment.

    Args:
        seed: Environment seed used for the initial reset.
        env_id: Registered MiniGrid environment ID.
        max_steps: Maximum number of environment steps to collect.
        actions: Optional fixed action sequence. If omitted, actions are
            sampled randomly from the environment's action space.

    Returns:
        A trajectory dictionary containing observations, actions, rewards,
        termination flags, and truncation flags.

    Notes:
        The trajectory always satisfies
        ``len(observations) == len(actions) + 1``.
        Supplying the same seed and the same action sequence makes collection
        reproducible for deterministic MiniGrid environments.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    env = make_env(env_id=env_id, seed=seed)
    env.action_space.seed(seed)

    try:
        observation, _ = env.reset(seed=seed)

        trajectory: dict[str, Any] = {
            "seed": seed,
            "observations": [observation["image"].copy()],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
        }

        action_iterator = iter(actions) if actions is not None else None

        for _ in range(max_steps):
            if action_iterator is None:
                action = int(env.action_space.sample())
            else:
                try:
                    action = int(next(action_iterator))
                except StopIteration:
                    break

            next_observation, reward, terminated, truncated, _ = env.step(action)

            trajectory["actions"].append(action)
            trajectory["rewards"].append(float(reward))
            trajectory["terminated"].append(bool(terminated))
            trajectory["truncated"].append(bool(truncated))
            trajectory["observations"].append(next_observation["image"].copy())

            if terminated or truncated:
                break

        return trajectory
    finally:
        env.close()


def collect_trajectories(
    seeds: Iterable[int],
    *,
    env_id: str = "MiniGrid-Empty-5x5-v0",
    max_steps: int = 100,
) -> list[dict[str, Any]]:
    """Collect one trajectory for each supplied seed."""
    return [
        collect_trajectory(
            seed=seed,
            env_id=env_id,
            max_steps=max_steps,
        )
        for seed in seeds
    ]
