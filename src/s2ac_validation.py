import os
import argparse

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
from skrl.envs.wrappers.jax import wrap_env
import brax.envs
from skrl.memories.jax import RandomMemory
import cv2
import numpy as np

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., runs/.../checkpoints/best_agent.pickle)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="inverted_pendulum",
        help="Brax environment name (default: inverted_pendulum)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Episode length (default: 1000)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render and save video of the first episode",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="validation.mp4",
        help="Output video filename (default: validation.mp4)",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=None,
        help="Number of SVGD particles (default: use config default)",
    )
    parser.add_argument(
        "--svgd-steps",
        type=int,
        default=None,
        help="Number of SVGD steps (default: use config default)",
    )

    args = parser.parse_args()

    device = select_compute_device()
    print(f"[S2AC Validation] Using JAX device: {device}")
    print(f"[S2AC Validation] Environment: {args.env}")
    print(f"[S2AC Validation] Loading checkpoint: {args.checkpoint}")

    with jax.default_device(device):
        env = brax.envs.create(
            args.env,
            episode_length=args.episode_length,
            batch_size=1,
            backend="mjx",
        )

    env = wrap_env(env, verbose=False)
    env_device = getattr(env, "device", getattr(env, "_device", device))

    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env_device)

    models = create_models(env, env_device, hidden_sizes=(256, 256))

    cfg_agent = S2AC_DEFAULT_CONFIG.copy()
    if args.particles is not None:
        cfg_agent["particles"] = args.particles
    if args.svgd_steps is not None:
        cfg_agent["svgd_steps"] = args.svgd_steps

    print(
        f"[S2AC Validation] Particles: {cfg_agent['particles']}, SVGD steps: {cfg_agent['svgd_steps']}"
    )

    agent = S2AC(
        models=models,
        memory=memory,
        cfg=cfg_agent,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env_device,
    )

    agent.init()

    agent.load(args.checkpoint)
    agent.set_mode("eval")
    print("[S2AC Validation] Agent loaded and set to evaluation mode")

    episode_rewards = []
    episode_lengths = []
    frames = [] if args.render_video else None

    for episode in range(args.num_episodes):
        print(f"\n[Episode {episode + 1}/{args.num_episodes}]")

        key = jax.random.PRNGKey(42 + episode)
        key, key_reset = jax.random.split(key)

        states, infos = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(args.episode_length):
            actions, _, _ = agent.act(
                states,
                timestep=args.episode_length + 1,
                timesteps=args.episode_length,
            )

            states, rewards, dones, truncateds, infos = env.step(actions)

            episode_reward += float(rewards[0])
            episode_length += 1

            if args.render_video and episode == 0 and frames is not None:
                try:
                    frame = env._env.render(mode="rgb_array")[0]
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    if step == 0:
                        print(f"  Warning: Failed to render frame: {e}")
                        print("  Continuing without video rendering...")
                        frames = None

            if dones.any() or truncateds.any():
                print(f"  Episode finished at step {step + 1}")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")

    print(f"Episodes: {args.num_episodes}")
    print(
        f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(
        f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )

    if args.render_video and frames and len(frames) > 0:
        try:
            height, width = frames[0].shape[:2]
            video_writer = cv2.VideoWriter(
                args.output_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # 30 FPS
                (width, height),
            )

            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"\n[S2AC Validation] Video saved to: {args.output_video}")
        except Exception as e:
            print(f"\n[S2AC Validation] Failed to save video: {e}")
    elif args.render_video and not frames:
        print(
            "\n[S2AC Validation] Video rendering was requested but failed (no OpenGL context available)"
        )


if __name__ == "__main__":
    main()
