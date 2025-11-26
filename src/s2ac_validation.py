import jax
import gymnax
from skrl.envs.wrappers.jax import wrap_env
import brax.envs
from skrl.memories.jax import RandomMemory
from skrl.trainers.jax import SequentialTrainer
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
        "inverted_pendulum", episode_length=1000, batch_size=1, backend="mjx"
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
    reduction="sum",
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
    reduction="sum",
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
    reduction="sum",
)
models["target_critic"].init_state_dict("target_critic")

cfg_agent = S2AC_DEFAULT_CONFIG.copy()

agent = S2AC(
    models=models,
    memory=memory,
    cfg=cfg_agent,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
)

agent.load("runs/25-11-25_20-20-35-135279_S2AC/checkpoints/best_agent.pickle")

# cfg = {
#     "timesteps": 10_000,
#     "headless": False,
#     "stochastic_evaluation": False,
# }
# trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)
# trainer.eval()

key = jax.random.PRNGKey(42)
key, key_reset = jax.random.split(key)

states, infos = env.reset()

EVALUATION_STEPS = 1000
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
    # print(frame.shape)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if dones.any() or truncateds.any():
        print(f"Episode finished at step {i}")

height, width = 256, 256
video_writer = cv2.VideoWriter(
    "inverted_pendulum.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,  # 30 FPS
    (width, height),
)

for frame in frames:
    video_writer.write(frame)
video_writer.release()
