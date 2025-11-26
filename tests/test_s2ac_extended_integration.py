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
import gymnasium as gym
from gymnasium import spaces

# Enable x64
jax.config.update("jax_enable_x64", True)


class MockEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=jnp.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=jnp.float32)
        self.device = jax.devices()[0]


@pytest.fixture
def mock_env():
    return MockEnv(num_envs=2)


@pytest.fixture
def agent(mock_env):
    models = {}
    models["policy"] = Policy_MLP(
        observation_space=mock_env.observation_space,
        action_space=mock_env.action_space,
        device=mock_env.device,
        clip_actions=True,
    )
    models["critic"] = Critic_MLP(
        observation_space=mock_env.observation_space,
        action_space=mock_env.action_space,
        device=mock_env.device,
    )
    models["target_critic"] = Target_Critic_MLP(
        observation_space=mock_env.observation_space,
        action_space=mock_env.action_space,
        device=mock_env.device,
    )

    # Initialize models
    for name, model in models.items():
        model.init_state_dict(name)

    memory = RandomMemory(
        memory_size=100, num_envs=mock_env.num_envs, device=mock_env.device
    )

    cfg = S2AC_DEFAULT_CONFIG.copy()
    cfg["particles"] = 4  # Small number for testing
    cfg["actor_lr"] = 1e-2  # Increase LR to ensure updates are visible
    cfg["critic_lr"] = 1e-2

    agent = S2AC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=mock_env.observation_space,
        action_space=mock_env.action_space,
        device=mock_env.device,
    )

    agent.init()
    return agent


def test_agent_initialization(agent):
    """Test that the agent initializes correctly."""
    assert agent is not None
    assert agent.policy is not None
    assert agent.critic_model is not None
    assert agent.target_critic is not None
    assert agent._particles == 4


def test_agent_act_shape(agent, mock_env):
    """Test the act method returns correct shapes."""
    states = jnp.zeros((mock_env.num_envs, *mock_env.observation_space.shape))
    timestep = 0
    timesteps = 100

    actions, log_prob, outputs = agent.act(states, timestep, timesteps)

    assert actions.shape == (mock_env.num_envs, *mock_env.action_space.shape)
    # log_prob might be None or have shape depending on implementation, usually (num_envs, 1) or (num_envs,)
    if log_prob is not None:
        assert log_prob.shape[0] == mock_env.num_envs


def test_agent_update_changes_params(agent, mock_env):
    """Test that calling _update changes the parameters (learning is happening)."""
    # Fill memory with some dummy data
    key = jax.random.PRNGKey(42)

    # Add enough transitions to trigger update (batch_size is usually 256, let's lower it or add more)
    agent.cfg["batch_size"] = 10
    agent._batch_size = 10
    for i in range(20):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        obs = jax.random.normal(
            k1, (mock_env.num_envs, *mock_env.observation_space.shape)
        )
        act = jax.random.normal(k2, (mock_env.num_envs, *mock_env.action_space.shape))
        rew = jax.random.normal(k3, (mock_env.num_envs, 1))
        next_obs = jax.random.normal(
            k4, (mock_env.num_envs, *mock_env.observation_space.shape)
        )
        terminated = jnp.zeros((mock_env.num_envs, 1), dtype=bool)
        truncated = jnp.zeros((mock_env.num_envs, 1), dtype=bool)
        infos = {}

        agent.record_transition(
            obs, act, rew, next_obs, terminated, truncated, infos, 0, 100
        )

    # Get initial params
    initial_policy_params = jax.tree_util.tree_map(
        lambda x: x.copy(), agent.policy.state_dict.params
    )
    initial_critic_params = jax.tree_util.tree_map(
        lambda x: x.copy(), agent.critic_model.state_dict.params
    )

    # Perform update
    agent._update(timestep=100, timesteps=1000)

    # Check if params changed
    # Note: Depending on initialization and data, params might not change much, but they should change.
    # We check if at least one parameter has changed.

    diff_policy = jax.tree_util.tree_map(
        lambda p1, p2: jnp.max(jnp.abs(p1 - p2)),
        initial_policy_params,
        agent.policy.state_dict.params,
    )
    max_diff_policy = max([float(x) for x in jax.tree_util.tree_leaves(diff_policy)])

    diff_critic = jax.tree_util.tree_map(
        lambda p1, p2: jnp.max(jnp.abs(p1 - p2)),
        initial_critic_params,
        agent.critic_model.state_dict.params,
    )
    max_diff_critic = max([float(x) for x in jax.tree_util.tree_leaves(diff_critic)])

    assert max_diff_critic > 0, "Critic parameters did not change after update"
    assert max_diff_policy > 0, "Policy parameters did not change after update"
