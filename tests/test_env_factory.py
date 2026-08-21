from envs import make_env


def test_visual_environment_shape():
    env = make_env(seed=42)
    try:
        observation, _ = env.reset(seed=42)
        assert observation["image"].shape == (56, 56, 3)
        assert observation["image"].dtype.name == "uint8"
    finally:
        env.close()


def test_environment_can_step():
    env = make_env(seed=42)
    try:
        _, _ = env.reset(seed=42)
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
