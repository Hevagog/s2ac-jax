import time
import jax
import jax.numpy as jnp
import pytest
from skrl.memories.jax import RandomMemory
from agent.s2ac import (
    S2AC,
    S2AC_DEFAULT_CONFIG,
    Policy_MLP,
    Critic_MLP,
    Target_Critic_MLP,
)
from gymnasium import spaces

# Enable x64
jax.config.update("jax_enable_x64", True)


class MockEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=jnp.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=jnp.float32)
        self.device = jax.devices()[0]


@pytest.fixture
def agent_setup():
    env = MockEnv(num_envs=16)
    models = {}
    models["policy"] = Policy_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        clip_actions=True,
    )
    models["critic_1"] = Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    models["critic_2"] = Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    models["target_critic_1"] = Target_Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    models["target_critic_2"] = Target_Critic_MLP(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    for name, model in models.items():
        model.init_state_dict(name)

    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)

    cfg = S2AC_DEFAULT_CONFIG.copy()
    cfg["particles"] = 16
    cfg["batch_size"] = 128

    agent = S2AC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    agent.init()

    # Fill memory
    obs = jnp.zeros((env.num_envs, *env.observation_space.shape))
    act = jnp.zeros((env.num_envs, *env.action_space.shape))
    rew = jnp.zeros((env.num_envs, 1))
    next_obs = jnp.zeros((env.num_envs, *env.observation_space.shape))
    terminated = jnp.zeros((env.num_envs, 1), dtype=bool)
    truncated = jnp.zeros((env.num_envs, 1), dtype=bool)
    infos = {}

    for _ in range(20):
        agent.record_transition(
            obs, act, rew, next_obs, terminated, truncated, infos, 0, 100
        )

    return agent, env


def test_act_speed(agent_setup):
    """Benchmark act method speed (check for JIT compilation)."""
    agent, env = agent_setup
    states = jnp.zeros((env.num_envs, *env.observation_space.shape))

    # Warmup
    start_time = time.time()
    agent.act(states, 0, 1000)
    jax.block_until_ready(agent.policy.state_dict.params)  # Sync
    warmup_time = time.time() - start_time

    # Run
    start_time = time.time()
    agent.act(states, 0, 1000)
    jax.block_until_ready(agent.policy.state_dict.params)  # Sync
    run_time = time.time() - start_time

    print(f"Act Warmup: {warmup_time:.4f}s, Run: {run_time:.4f}s")

    # Second run should be significantly faster if JIT is working
    # Note: This assertion might be flaky on some systems, but generally true for JAX
    assert run_time < warmup_time, "JIT compilation did not speed up execution"


def test_update_speed(agent_setup):
    """Benchmark update method speed (check for JIT compilation)."""
    agent, env = agent_setup

    # Warmup
    start_time = time.time()
    agent._update(100, 1000)
    jax.block_until_ready(agent.policy.state_dict.params)
    warmup_time = time.time() - start_time

    # Run
    start_time = time.time()
    agent._update(101, 1000)
    jax.block_until_ready(agent.policy.state_dict.params)
    run_time = time.time() - start_time

    print(f"Update Warmup: {warmup_time:.4f}s, Run: {run_time:.4f}s")

    assert run_time < warmup_time, "JIT compilation did not speed up execution"
