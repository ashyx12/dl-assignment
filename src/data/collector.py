"""Trajectory collection utilities for MiniGrid experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from envs import make_env


@dataclass
class Transition:
    """One action-conditioned environment transition."""

    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    episode_id: int
    timestep: int
    seed: int


@dataclass
class TrajectoryDataset:
    """Collection of transitions represented as NumPy arrays."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    seeds: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: str) -> None:
        """Save the dataset as a compressed NumPy archive."""
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            next_observations=self.next_observations,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_ids=self.episode_ids,
            timesteps=self.timesteps,
            seeds=self.seeds,
        )


def _image(observation: dict[str, Any]) -> np.ndarray:
    """Extract an RGB image and copy it out of the environment buffer."""
    return np.asarray(observation["image"], dtype=np.uint8).copy()


def collect_trajectories(
    seeds: list[int],
    *,
    env_id: str = "MiniGrid-Empty-5x5-v0",
    max_steps: int = 100,
    action_policy: str = "random",
) -> TrajectoryDataset:
    """Collect image transitions from one episode per seed.

    The initial project dataset uses a random policy. Each seed creates a
    separate episode, which keeps episode boundaries explicit and makes the
    generated data reproducible.
    """
    if action_policy != "random":
        raise ValueError("Only action_policy='random' is supported currently")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []
    transition_seeds: list[int] = []

    for episode_id, seed in enumerate(seeds):
        env = make_env(env_id=env_id, visual=True)
        try:
            observation, _ = env.reset(seed=seed)
            env.action_space.seed(seed)

            for timestep in range(max_steps):
                action = int(env.action_space.sample())
                next_observation, reward, done, timeout, _ = env.step(action)

                observations.append(_image(observation))
                actions.append(action)
                next_observations.append(_image(next_observation))
                rewards.append(float(reward))
                terminated.append(bool(done))
                truncated.append(bool(timeout))
                episode_ids.append(episode_id)
                timesteps.append(timestep)
                transition_seeds.append(seed)

                observation = next_observation
                if done or timeout:
                    break
        finally:
            env.close()

    if not observations:
        raise RuntimeError("No transitions were collected")

    return TrajectoryDataset(
        observations=np.stack(observations),
        actions=np.asarray(actions, dtype=np.int64),
        next_observations=np.stack(next_observations),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        seeds=np.asarray(transition_seeds, dtype=np.int64),
    )
