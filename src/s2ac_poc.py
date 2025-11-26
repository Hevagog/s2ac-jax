import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
from skrl.envs.wrappers.jax import wrap_env
import brax.envs
from skrl.memories.jax import RandomMemory
from skrl.trainers.jax import SequentialTrainer

from agent.s2ac import (
    S2AC,
    S2AC_DEFAULT_CONFIG,
    Target_Critic_MLP,
    Policy_MLP,
    Critic_MLP,
)


def select_compute_device(preferred_platforms=("gpu", "cuda", "tpu", "cpu", None)):
    for platform in preferred_platforms:
        try:
            devices = jax.devices() if platform is None else jax.devices(platform)
        except RuntimeError:
            continue
        if devices:
            return devices[0]
    raise RuntimeError("No JAX devices available")


device = select_compute_device()
print(f"[S2AC] Using JAX device: {device}")

with jax.default_device(device):
    env = brax.envs.create(
        "inverted_pendulum", episode_length=1000, batch_size=8192, backend="mjx"
    )


env = wrap_env(env, verbose=False)
if hasattr(env, "_device"):
    env._device = device
elif hasattr(env, "device"):
    env.device = device

memory = RandomMemory(memory_size=2000, num_envs=env.num_envs, device=env.device)

models = {}
models["policy"] = Policy_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="mean",
)
models["policy"].init_state_dict("policy")
models["critic"] = Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="mean",
)
models["critic"].init_state_dict("critic")
models["target_critic"] = Target_Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="mean",
)
models["target_critic"].init_state_dict("target_critic")

cfg_agent = S2AC_DEFAULT_CONFIG.copy()
cfg_agent["particles"] = 24
cfg_agent["svgd_steps"] = 3

agent = S2AC(
    models=models,
    memory=memory,
    cfg=cfg_agent,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
)

cfg = {"timesteps": 80_000, "headless": True}
trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)
trainer.train()

trainer.eval()
