"""
Corridor Multimodal Environment for S2AC Validation.

This environment is designed to test multimodal policy learning with temporal
credit assignment. Unlike the simple multigoal environment, this one requires:
1. Sequential decision making (not single-step to goal)
2. Clear multimodal optimal solutions (upper vs lower corridor)
3. Suboptimal local optima that greedy policies might get stuck in
4. Measurable metrics for mode coverage

Environment Layout:
```
    +--------+--------+--------+
    | Start  |  Wall  |  Goal  |
    | (0,0)  |  ████  | (+5,0) |
    |        |  ████  |        |
    +--------+--------+--------+
             |        |
    Upper    |   ██   |  Upper
    Path ----+   ██   +--- Path
    (-1,+2)  |   ██   |  (+3,+2)
             |   ██   |
    +--------+---++---+--------+
    |        |   ||   |        |
    |   ██████   ||   ██████   |
    |        |   ||   |        |
    +--------+---++---+--------+
             |   ██   |
    Lower    |   ██   |  Lower
    Path ----+   ██   +--- Path
    (-1,-2)  |   ██   |  (+3,-2)
             |        |
    +--------+--------+--------+
```

The agent starts at (0, 0) and must reach the goal at (5, 0).
There's a central wall blocking the direct path, so the agent must go:
- Upper route: (0,0) -> (-1,+2) -> (+3,+2) -> (5,0)
- Lower route: (0,0) -> (-1,-2) -> (+3,-2) -> (5,0)

Both routes are equally optimal, so a maximum entropy agent should visit both.
A suboptimal agent might get stuck trying to go through the wall.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Any, Dict


class CorridorMultimodalEnv(gym.Env):
    """
    A corridor navigation environment with two equally optimal paths.

    Observation: [x, y] position (2D continuous)
    Action: [dx, dy] velocity (2D continuous, bounded to [-1, 1])

    The environment has:
    - A central wall blocking direct path to goal
    - Upper corridor requiring going through waypoint (~0, +2)
    - Lower corridor requiring going through waypoint (~0, -2)
    - Reward shaping that doesn't favor either path
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_steps: int = 50,
        goal_reward: float = 100.0,
        step_penalty: float = -0.1,
        wall_penalty: float = -5.0,
        distance_coeff: float = 0.5,
        actuation_coeff: float = 0.1,
        goal_threshold: float = 0.8,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.distance_coeff = distance_coeff
        self.actuation_coeff = actuation_coeff
        self.goal_threshold = goal_threshold
        self.render_mode = render_mode

        self.x_bounds = (-3.0, 7.0)
        self.y_bounds = (-4.0, 4.0)

        self.goal_pos = np.array([5.0, 0.0], dtype=np.float32)

        self.wall_x_range = (1.5, 3.5)
        self.wall_y_range = (-1.5, 1.5)

        self.upper_waypoint = np.array([2.5, 2.5], dtype=np.float32)
        self.lower_waypoint = np.array([2.5, -2.5], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([self.x_bounds[0], self.y_bounds[0]], dtype=np.float32),
            high=np.array([self.x_bounds[1], self.y_bounds[1]], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.position = None
        self.step_count = 0
        self.trajectory = []

        self.episode_stats = {
            "upper_visits": 0,
            "lower_visits": 0,
            "wall_collisions": 0,
            "success": False,
            "path_taken": None,  # "upper", "lower", or None
        }

    def _in_wall(self, pos: np.ndarray) -> bool:
        x, y = pos
        return (
            self.wall_x_range[0] <= x <= self.wall_x_range[1]
            and self.wall_y_range[0] <= y <= self.wall_y_range[1]
        )

    def _clip_to_bounds(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(
            pos,
            [self.x_bounds[0], self.y_bounds[0]],
            [self.x_bounds[1], self.y_bounds[1]],
        ).astype(np.float32)

    def _handle_wall_collision(
        self, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        if not self._in_wall(new_pos):
            return new_pos, False

        return old_pos.copy(), True

    def _compute_reward(
        self,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        action: np.ndarray,
        wall_collision: bool,
    ) -> float:
        reward = self.step_penalty

        if wall_collision:
            reward += self.wall_penalty

        # Distance-based reward (progress towards goal)
        old_dist = np.linalg.norm(old_pos - self.goal_pos)
        new_dist = np.linalg.norm(new_pos - self.goal_pos)
        reward += self.distance_coeff * (old_dist - new_dist)

        # Actuation cost
        reward -= self.actuation_coeff * np.sum(action**2)

        # Goal reward
        if new_dist < self.goal_threshold:
            reward += self.goal_reward

        return reward

    def _detect_path(self) -> Optional[str]:
        if len(self.trajectory) < 2:
            return None

        trajectory = np.array(self.trajectory)
        max_y = np.max(trajectory[:, 1])
        min_y = np.min(trajectory[:, 1])

        if max_y > 1.5 and min_y > -1.0:
            return "upper"
        elif min_y < -1.5 and max_y < 1.0:
            return "lower"
        return None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if options and "start_pos" in options:
            self.position = np.array(options["start_pos"], dtype=np.float32)
        else:
            noise = self.np_random.uniform(-0.1, 0.1, size=2).astype(np.float32)
            self.position = np.array([0.0, 0.0], dtype=np.float32) + noise

        self.step_count = 0
        self.trajectory = [self.position.copy()]

        self.episode_stats = {
            "upper_visits": 0,
            "lower_visits": 0,
            "wall_collisions": 0,
            "success": False,
            "path_taken": None,
        }

        return self.position.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        old_pos = self.position.copy()
        new_pos = old_pos + action
        new_pos = self._clip_to_bounds(new_pos)

        new_pos, wall_collision = self._handle_wall_collision(old_pos, new_pos)

        self.position = new_pos
        self.step_count += 1
        self.trajectory.append(self.position.copy())

        if wall_collision:
            self.episode_stats["wall_collisions"] += 1

        if self.position[1] > 2.0:
            self.episode_stats["upper_visits"] += 1
        elif self.position[1] < -2.0:
            self.episode_stats["lower_visits"] += 1

        reward = self._compute_reward(old_pos, new_pos, action, wall_collision)

        dist_to_goal = np.linalg.norm(self.position - self.goal_pos)
        success = dist_to_goal < self.goal_threshold
        truncated = self.step_count >= self.max_steps
        terminated = success

        if success:
            self.episode_stats["success"] = True
            self.episode_stats["path_taken"] = self._detect_path()

        info = {
            "distance_to_goal": dist_to_goal,
            "wall_collision": wall_collision,
            "step_count": self.step_count,
        }

        if terminated or truncated:
            info["episode_stats"] = self.episode_stats.copy()

        return self.position.copy(), float(reward), terminated, truncated, info

    def get_multimodality_metrics(
        self, num_episodes: int = 100, agent_fn=None
    ) -> Dict[str, Any]:
        """
        Evaluate multimodality of a policy.

        Args:
            num_episodes: Number of episodes to run
            agent_fn: Function that takes observation and returns action
                      If None, uses random policy

        Returns:
            Dictionary with metrics:
            - upper_ratio: Fraction of episodes using upper path
            - lower_ratio: Fraction of episodes using lower path
            - success_rate: Fraction of episodes reaching goal
            - entropy_estimate: Estimate of path distribution entropy
        """
        upper_count = 0
        lower_count = 0
        success_count = 0

        for _ in range(num_episodes):
            obs, _ = self.reset()
            done = False

            while not done:
                if agent_fn is not None:
                    action = agent_fn(obs)
                else:
                    action = self.action_space.sample()

                obs, _, terminated, truncated, info = self.step(action)
                done = terminated or truncated

            if "episode_stats" in info:
                stats = info["episode_stats"]
                if stats["success"]:
                    success_count += 1
                    if stats["path_taken"] == "upper":
                        upper_count += 1
                    elif stats["path_taken"] == "lower":
                        lower_count += 1

        total_paths = upper_count + lower_count
        upper_ratio = upper_count / max(total_paths, 1)
        lower_ratio = lower_count / max(total_paths, 1)

        if upper_ratio > 0 and lower_ratio > 0:
            entropy = -(
                upper_ratio * np.log(upper_ratio) + lower_ratio * np.log(lower_ratio)
            )
        else:
            entropy = 0.0

        return {
            "upper_ratio": upper_ratio,
            "lower_ratio": lower_ratio,
            "success_rate": success_count / num_episodes,
            "path_entropy": entropy,
            "max_entropy": np.log(2),  # Maximum entropy for binary choice
            "total_episodes": num_episodes,
            "total_successes": success_count,
        }

    def render(self):
        if self.render_mode == "human":
            print(f"Position: {self.position}, Step: {self.step_count}")
        return None


class CorridorMultimodalEnvSimple(CorridorMultimodalEnv):
    """
    Simplified version with smaller walls and easier navigation.
    Good for initial testing that the algorithm works.
    """

    def __init__(self, **kwargs):
        # Smaller wall
        super().__init__(**kwargs)
        self.wall_x_range = (2.0, 3.0)
        self.wall_y_range = (-1.0, 1.0)
        self.goal_threshold = 1.0
        self.max_steps = kwargs.get("max_steps", 30)


class CorridorMultimodalEnvHard(CorridorMultimodalEnv):
    """
    Harder version with narrow corridors and longer paths.
    Tests if algorithm handles more complex multimodal optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wider wall with narrow gaps
        self.wall_x_range = (0.5, 4.5)
        self.wall_y_range = (-2.5, 2.5)
        self.goal_threshold = 0.5
        self.max_steps = kwargs.get("max_steps", 80)


gym.register(
    id="CorridorMultimodal-v0",
    entry_point="src.environments.corridor_multimodal:CorridorMultimodalEnv",
)

gym.register(
    id="CorridorMultimodalSimple-v0",
    entry_point="src.environments.corridor_multimodal:CorridorMultimodalEnvSimple",
)

gym.register(
    id="CorridorMultimodalHard-v0",
    entry_point="src.environments.corridor_multimodal:CorridorMultimodalEnvHard",
)


if __name__ == "__main__":
    # Quick test
    env = CorridorMultimodalEnv()

    print("Testing CorridorMultimodal environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    metrics = env.get_multimodality_metrics(num_episodes=50)
    print("\nRandom policy metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    obs, _ = env.reset()
    total_reward = 0
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nSingle episode - Total reward: {total_reward:.2f}")
    if "episode_stats" in info:
        print(f"Episode stats: {info['episode_stats']}")
