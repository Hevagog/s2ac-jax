import jax
from skrl.envs.wrappers.jax import wrap_env
import brax.envs
from skrl.memories.jax import RandomMemory
import cv2

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


with jax.default_device(device):
    env = brax.envs.create(
        "inverted_pendulum", episode_length=3000, batch_size=1, backend="mjx"
    )

env = wrap_env(env, verbose=False)

env_device = getattr(env, "device", getattr(env, "_device", device))


memory = RandomMemory(memory_size=2000, num_envs=env.num_envs, device=env_device)

models = {}
models["policy"] = Policy_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["policy"].init_state_dict("policy")
models["critic_1"] = Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["critic_1"].init_state_dict("critic_1")
models["critic_2"] = Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["critic_2"].init_state_dict("critic_2")
models["target_critic_1"] = Target_Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["target_critic_1"].init_state_dict("target_critic_1")
models["target_critic_2"] = Target_Critic_MLP(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
    clip_actions=True,
    clip_log_std=True,
    min_log_std=-20,
    max_log_std=2,
    reduction="sum",
)
models["target_critic_2"].init_state_dict("target_critic_2")

cfg_agent = S2AC_DEFAULT_CONFIG.copy()
cfg_agent["particles"] = 6
cfg_agent["svgd_steps"] = 2

agent = S2AC(
    models=models,
    memory=memory,
    cfg=cfg_agent,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env_device,
)

agent.load("runs/25-12-13_23-27-57-285682_S2AC/checkpoints/best_agent.pickle")

key = jax.random.PRNGKey(42)
key, key_reset = jax.random.split(key)

states, infos = env.reset()

EVALUATION_STEPS = 200
frames = []

for i in range(EVALUATION_STEPS):
    key, key_act = jax.random.split(key)

    actions, _, _ = agent.act(
        states,
        timestep=i,
        timesteps=EVALUATION_STEPS,
    )

    states, rewards, dones, truncateds, infos = env.step(actions)
    frame = env._env.render(mode="rgb_array")[0]
    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if dones.any() or truncateds.any():
        print(f"Episode finished at step {i}")

height, width = frames[0].shape[:2]
video_writer = cv2.VideoWriter(
    "inverted_pendulum_5.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,  # 30 FPS
    (width, height),
)

for frame in frames:
    video_writer.write(frame)
video_writer.release()
