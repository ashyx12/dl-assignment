import numpy as np

from data import collect_trajectories


def test_collect_trajectories_shapes_and_metadata():
    dataset = collect_trajectories([1, 2], max_steps=5)

    assert len(dataset) > 0
    assert dataset.observations.shape[1:] == (56, 56, 3)
    assert dataset.next_observations.shape == dataset.observations.shape
    assert dataset.actions.shape == (len(dataset),)
    assert dataset.rewards.shape == (len(dataset),)
    assert dataset.seeds.shape == (len(dataset),)
    assert dataset.observations.dtype == np.uint8
    assert dataset.actions.dtype == np.int64


def test_same_seed_is_reproducible():
    first = collect_trajectories([42], max_steps=10)
    second = collect_trajectories([42], max_steps=10)

    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.next_observations, second.next_observations)
