from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointDynamics:
    """
    Simple point dynamics matching the S2AC paper.
    State: position (x, y).
    Action: velocity (dx, dy).
    """

    def __init__(self, dim: int = 2, sigma: float = 0.0):
        self.dim = dim
        self.sigma = sigma  # Stochasticity in dynamics
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Deterministic integrator dynamics: s_{t+1} = s_t + action."""
        mu_next = state + action
        if self.sigma > 0:
            state_next = mu_next + self.sigma * np.random.normal(size=self.s_dim)
        else:
            state_next = mu_next
        return state_next


class SimpleMultiGoalEnv(gym.Env):
    """
    Multi-goal point navigation environment for testing S2AC.

    This environment is designed to test multimodal policy learning:
    - G1 (Right): Single goal - low entropy optimal path
    - G2/G3 (Left): Two goals close together - high entropy optimal path

    MaxEnt RL agents should prefer the left side because it offers
    two equally good goals, maximizing entropy while achieving reward.

    Based on the S2AC paper (https://arxiv.org/abs/2405.00987) environment.
    """

    def __init__(
        self,
        goal_reward: float = 10.0,
        actuation_cost_coeff: float = 30.0,
        distance_cost_coeff: float = 1.0,
        init_sigma: float = 0.0,
        max_steps: int = 100,
        dynamics_sigma: float = 0.0,
    ):
        super().__init__()

        self.dynamics = PointDynamics(dim=2, sigma=dynamics_sigma)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.max_steps = max_steps
        self.xlim = (-7.0, 7.0)
        self.ylim = (-7.0, 7.0)
        self.vel_bound = 1.0  # Max velocity in each direction

        # Goals - matching the S2AC paper layout
        # G1 (Right): Unimodal path - single goal on the right
        # G2/G3 (Left): Bimodal path - two goals close together on the left
        self.goal_positions = np.array(
            [
                [5.0, 0.0],  # G1: Right (unimodal)
                [-4.0, 3.0],  # G2: Top Left (bimodal)
                [-4.0, -3.0],  # G3: Bottom Left (bimodal)
            ],
            dtype=np.float32,
        )
        self.num_goals = len(self.goal_positions)
        self.goal_threshold = 0.5  # Distance to consider goal reached
        self.goal_reward = goal_reward

        self.actuation_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff

        # Observation space: 2D position
        self.observation_space = spaces.Box(
            low=np.array([self.xlim[0], self.ylim[0]], dtype=np.float32),
            high=np.array([self.xlim[1], self.ylim[1]], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Action space: 2D velocity
        self.action_space = spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation = None
        self.ep_len = 0
        self._max_episode_steps = max_steps

        self.episode_observations = []
        self.number_of_hits_mode = np.zeros(self.num_goals)
        self.min_dist_index = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.init_sigma > 0:
            unclipped_observation = (
                self.init_mu
                + self.init_sigma * self.np_random.normal(size=2).astype(np.float32)
            )
        else:
            unclipped_observation = self.init_mu.copy()

        self.observation = np.clip(
            unclipped_observation,
            np.array([self.xlim[0], self.ylim[0]], dtype=np.float32),
            np.array([self.xlim[1], self.ylim[1]], dtype=np.float32),
        )
        self.ep_len = 0
        self.episode_observations = [self.observation.copy()]

        return self.observation.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        action = np.clip(action, -self.vel_bound, self.vel_bound)

        assert self.observation is not None, "Environment must be reset before step()"
        self.observation = self.dynamics.forward(self.observation, action)
        self.observation = np.clip(
            self.observation,
            np.array([self.xlim[0], self.ylim[0]], dtype=np.float32),
            np.array([self.xlim[1], self.ylim[1]], dtype=np.float32),
        )
        self.ep_len += 1

        reward = self.compute_reward(self.observation, action)

        distance_to_goals = [
            np.linalg.norm(self.observation - goal) for goal in self.goal_positions
        ]
        self.min_dist_index = int(np.argmin(distance_to_goals))
        min_dist = float(distance_to_goals[self.min_dist_index])

        terminated = bool(min_dist < self.goal_threshold)
        truncated = bool(self.ep_len >= self.max_steps)

        if terminated:
            reward += self.goal_reward
            self.number_of_hits_mode[self.min_dist_index] += 1

        self.episode_observations.append(self.observation.copy())

        info = {
            "min_dist": min_dist,
            "closest_goal": self.min_dist_index,
            "goal_reached": terminated,
        }

        return self.observation.copy(), float(reward), terminated, truncated, info

    def compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Compute reward matching S2AC paper formulation:
        reward = -actuation_cost - distance_cost

        This creates a dense reward signal that encourages moving toward goals
        while penalizing excessive actions.
        """
        # Penalize L2 norm of action (actuation cost)
        action_cost = np.sum(action**2) * self.actuation_cost_coeff

        # Penalize squared distance to closest goal
        goal_cost = self.distance_cost_coeff * np.min(
            [np.sum((observation - goal) ** 2) for goal in self.goal_positions]
        )

        reward = -(action_cost + goal_cost)
        return float(reward)

    def reset_rendering(self):
        self.episode_observations = []
        self.number_of_hits_mode = np.zeros(self.num_goals)

    def get_goal_stats(self) -> dict:
        total_hits = np.sum(self.number_of_hits_mode)
        if total_hits > 0:
            hit_ratios = self.number_of_hits_mode / total_hits
        else:
            hit_ratios = np.zeros(self.num_goals)
        return {
            "total_goals_reached": int(total_hits),
            "hits_per_goal": self.number_of_hits_mode.tolist(),
            "hit_ratios": hit_ratios.tolist(),
        }

    def render(self):
        pass


register(
    id="SimpleMultiGoal-v0",
    entry_point="environments.simple_multimodal:SimpleMultiGoalEnv",
    max_episode_steps=100,
)
