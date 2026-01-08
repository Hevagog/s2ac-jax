import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym
from skrl.memories.jax import RandomMemory
from skrl.trainers.jax import SequentialTrainer

from agent.s2ac import (
    S2AC,
    S2AC_DEFAULT_CONFIG,
    Target_Critic_MLP,
    Policy_MLP,
    Critic_MLP,
)
from environments import simple_multimodal  # noqa: F401


def select_compute_device(preferred_platforms=("gpu", "cuda", "tpu", "cpu", None)):
    """Select the best available JAX device."""
    for platform in preferred_platforms:
        try:
            devices = jax.devices() if platform is None else jax.devices(platform)
        except RuntimeError:
            continue
        if devices:
            return devices[0]
    raise RuntimeError("No JAX devices available")


def create_models(env, device, hidden_sizes=(256, 256)):
    models = {}

    models["policy"] = Policy_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
        reduction="sum",
        hidden_sizes=hidden_sizes,
    )
    models["policy"].init_state_dict("policy")

    models["critic_1"] = Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_sizes=hidden_sizes,
    )
    models["critic_1"].init_state_dict("critic_1")

    models["critic_2"] = Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_sizes=hidden_sizes,
    )
    models["critic_2"].init_state_dict("critic_2")

    models["target_critic_1"] = Target_Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_sizes=hidden_sizes,
    )
    models["target_critic_1"].init_state_dict("target_critic_1")

    models["target_critic_2"] = Target_Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        hidden_sizes=hidden_sizes,
    )
    models["target_critic_2"].init_state_dict("target_critic_2")

    return models


def main():
    device = select_compute_device()
    print(f"[S2AC] Using JAX device: {device}")

    with jax.default_device(device):
        env = gym.make("SimpleMultiGoal-v0")
        env = wrap_env(env)

    env_device = getattr(env, "device", getattr(env, "_device", device))

    cfg_agent = S2AC_DEFAULT_CONFIG.copy()

    memory = RandomMemory(memory_size=50_000, num_envs=env.num_envs, device=env_device)

    models = create_models(env, env_device, hidden_sizes=(256, 256))

    agent = S2AC(
        models=models,
        memory=memory,
        cfg=cfg_agent,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env_device,
    )

    cfg = {
        "timesteps": 100_000,
        "headless": True,
    }

    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
