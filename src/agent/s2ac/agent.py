from typing import Union, Tuple, Dict, Any, Optional

import copy
import numpy as np
import gymnasium
import jax
import jax.numpy as jnp
from jax import random
import jax.nn as jnn

from skrl import config
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam

from skrl.agents.jax import Agent
from .utils import action_score_from_Q, compute_logqL_closed_form, svgd_vector_field

# fmt: off
# [start-config-dict-jax]
S2AC_DEFAULT_CONFIG = {
    "particles": 16,
    "svgd_steps": 3,
    "svgd_step_size": 0.1,
    "kernel_sigma": 0.5,
    "alpha": 0.2,
    "action_scale": 1.0,
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "critic_target_update_interval": 1,

    
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": 250,  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


def _pytree_l2_norm(pytree):
    """Compute the L2 norm of a gradient pytree."""
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(pytree) if leaf is not None]
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    sq_sum = jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf)) for leaf in leaves]))
    return jnp.sqrt(sq_sum)


class S2AC(Agent):
    def __init__(
        self,
        models: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, Any]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Stein Soft Actor-Critic (S2AC)

        https://arxiv.org/abs/2405.00987

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.jax.Memory, list of skrl.memory.jax.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or Any, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(S2AC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self.policy = self.models.get("policy", None)
        self.critic_model = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        if self.policy is None or self.critic_model is None or self.target_critic is None:
            raise ValueError("S2AC agent requires 'policy', 'critic' and 'target_critic' models")

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic"] = self.critic_model
        self.checkpoint_modules["target_critic"] = self.target_critic

        self._policy_role = "policy"
        self._critic_role = "critic"
        self._target_role = "target_critic"

        self._particles = int(self.cfg["particles"])
        self._svgd_steps = int(self.cfg["svgd_steps"])
        self._svgd_step_size = float(self.cfg["svgd_step_size"])
        self._kernel_sigma = float(self.cfg["kernel_sigma"])
        self._alpha = float(self.cfg["alpha"])
        self._action_scale = float(self.cfg["action_scale"])
        self._batch_size = int(self.cfg["batch_size"])
        self._discount = float(self.cfg["discount"])
        self._tau = float(self.cfg["tau"])
        self._actor_lr = float(self.cfg["actor_lr"])
        self._critic_lr = float(self.cfg["critic_lr"])
        self._critic_target_update_interval = int(self.cfg["critic_target_update_interval"])

        # determine action dim
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self._action_dim = int(jnp.prod(jnp.asarray(self.action_space.shape)))
        elif isinstance(self.action_space, (tuple, list)):
            self._action_dim = int(jnp.prod(jnp.asarray(self.action_space)))
        else:
            self._action_dim = int(self.action_space)

        self._rng_key = random.PRNGKey(self.cfg.get("seed", 0))

        with jax.default_device(self.device):
            self.policy_optimizer = Adam(model=self.policy, lr=self._actor_lr)
            self.critic_optimizer = Adam(model=self.critic_model, lr=self._critic_lr)
        self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
        self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        if self.target_critic is not None:
            self.target_critic.freeze_parameters(True)
            self.target_critic.update_parameters(self.critic_model, polyak=1.0)

        self._update_steps = 0




    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor("states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor("actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor("rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor("next_states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor("terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor("truncated", size=1, dtype=jnp.int8)
            self._tensors_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        self.critic_model.apply = jax.jit(self.critic_model.apply, static_argnums=2)
        if self.target_critic is not None:
            self.target_critic.apply = jax.jit(self.target_critic.apply, static_argnums=2)

    def act(
        self, states: Union[jnp.ndarray, Any], timestep: int, timesteps: int
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict[str, Any]]:
        states = jnp.asarray(states)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        self._rng_key, subkey = random.split(self._rng_key)
        keys = random.split(subkey, states.shape[0])

        policy_params = self.policy.state_dict.params
        critic_params = self.critic_model.state_dict.params

        # returns: squashed_actions (batch, m, d), raw_actions (batch, m, d), log_prob (batch, m)
        particles, raw_actions, log_prob = self._svgd_rollout_batch(
            policy_params, self.critic_model, critic_params, states, keys
        )

        # Choose return action (mean of particles); alternative: sample one particle randomly
        actions = jnp.mean(particles, axis=1)  # (batch, d)
        mean_log_prob = jnp.mean(log_prob, axis=1, keepdims=True)  # (batch, 1)

        outputs = {"mean_actions": actions, "particles": particles}
        # For single env interface, caller expects action shape (d,) not (1,d); SKRL handles batching.
        return actions, mean_log_prob, outputs


    def record_transition(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: jnp.ndarray
        :param actions: Actions taken by the agent
        :type actions: jnp.ndarray
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: jnp.ndarray
        :param next_states: Next observations/states of the environment
        :type next_states: jnp.ndarray
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: jnp.ndarray
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: jnp.ndarray
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        next_states = jnp.asarray(next_states)
        terminated = jnp.asarray(terminated)
        truncated = jnp.asarray(truncated)

        if rewards.ndim == 0:
            rewards = rewards[None]
        if terminated.ndim == 0:
            terminated = terminated[None]
        if truncated.ndim == 0:
            truncated = truncated[None]

        states_np = np.asarray(jax.device_get(states))
        actions_np = np.asarray(jax.device_get(actions))
        rewards_np = np.asarray(jax.device_get(rewards))
        next_states_np = np.asarray(jax.device_get(next_states))
        terminated_np = np.asarray(jax.device_get(terminated))
        truncated_np = np.asarray(jax.device_get(truncated))

        super().record_transition(
            states_np,
            actions_np,
            rewards_np,
            next_states_np,
            terminated_np,
            truncated_np,
            infos,
            timestep,
            timesteps,
        )
        if self.memory is not None:
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        if self.memory is not None and len(self.memory) >= self._batch_size:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        if self.memory is None or len(self.memory) < self._batch_size:
            return
        
        # TODO: Add gradient steps?
        # sample a batch from memory
        (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

        sampled_states = jnp.asarray(sampled_states)
        sampled_actions = jnp.asarray(sampled_actions)
        sampled_rewards = jnp.asarray(sampled_rewards).squeeze(-1)
        sampled_next_states = jnp.asarray(sampled_next_states)
        sampled_terminated = jnp.asarray(sampled_terminated).squeeze(-1)
        sampled_truncated = jnp.asarray(sampled_truncated).squeeze(-1)
        dones = jnp.clip(sampled_terminated + sampled_truncated, 0.0, 1.0)

        policy_params = self.policy.state_dict.params
        critic_params = self.critic_model.state_dict.params
        target_model = self.target_critic if self.target_critic is not None else self.critic_model
        target_params = target_model.state_dict.params

        # sample next-state particles and log-probs
        self._rng_key, subkey = random.split(self._rng_key)
        next_keys = random.split(subkey, sampled_next_states.shape[0])
        next_particles, _, next_log_prob = self._svgd_rollout_batch(
            policy_params, self.critic_model, critic_params, sampled_next_states, next_keys
        )
        next_particles = jax.lax.stop_gradient(next_particles)
        next_log_prob = jax.lax.stop_gradient(next_log_prob)

        # target critic values: evaluate Q_target for each next particle
        target_q_values = self._critic_values_particles(target_model, target_params, sampled_next_states, next_particles)
        target_q_mean = jnp.mean(target_q_values, axis=1)
        entropy_term = jnp.mean(next_log_prob, axis=1)  # (batch,)

        targets = sampled_rewards + self._discount * (1.0 - dones) * (target_q_mean - self._alpha * entropy_term)
        targets = jax.lax.stop_gradient(targets)

        target_stats = {
            "target_q_mean": jnp.mean(target_q_mean),
            "target_q_std": jnp.std(target_q_mean),
            "entropy_term_mean": jnp.mean(entropy_term),
            "entropy_term_std": jnp.std(entropy_term),
            "reward_mean": jnp.mean(sampled_rewards),
            "reward_std": jnp.std(sampled_rewards),
            "done_fraction": jnp.mean(dones),
        }

        # === Critic update ===
        def critic_loss_fn(params):
            predicted = self._critic_values_single(self.critic_model, params, sampled_states, sampled_actions)
            loss = jnp.mean((predicted - targets) ** 2)
            metrics = {
                "prediction_mean": jnp.mean(predicted),
                "prediction_std": jnp.std(predicted),
            }
            return loss, metrics

        (critic_loss, critic_metrics), critic_grad = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params)
        if config.jax.is_distributed:
            critic_grad = self.critic_model.reduce_parameters(critic_grad)
        critic_grad_norm = _pytree_l2_norm(critic_grad)
        self.critic_optimizer = self.critic_optimizer.step(critic_grad, self.critic_model, self._critic_lr)

        critic_params = self.critic_model.state_dict.params

        # === Actor update ===
        self._rng_key, subkey = random.split(self._rng_key)
        actor_keys = random.split(subkey, sampled_states.shape[0])

        def actor_loss_fn(policy_p, keys):
            particles, _, log_prob = self._svgd_rollout_batch(
                policy_p, self.critic_model, critic_params, sampled_states, keys
            )
            q_values = self._critic_values_particles(self.critic_model, critic_params, sampled_states, particles)
            q_mean = jnp.mean(q_values, axis=1)
            entropy = jnp.mean(log_prob, axis=1)
            # We minimize alpha * H - E[Q] (so gradient ascent on Q + alpha H)
            loss = jnp.mean(self._alpha * entropy - q_mean)
            metrics = {
                "q_mean": jnp.mean(q_mean),
                "q_std": jnp.std(q_mean),
                "entropy_mean": jnp.mean(entropy),
                "entropy_std": jnp.std(entropy),
                "log_prob_mean": jnp.mean(log_prob),
                "log_prob_std": jnp.std(log_prob),
                "particle_std": jnp.mean(jnp.std(particles, axis=1)),
            }
            return loss, metrics

        (actor_loss, actor_metrics), actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(policy_params, actor_keys)
        if config.jax.is_distributed:
            actor_grad = self.policy.reduce_parameters(actor_grad)
        actor_grad_norm = _pytree_l2_norm(actor_grad)
        self.policy_optimizer = self.policy_optimizer.step(actor_grad, self.policy, self._actor_lr)

        self._update_steps += 1
        if self.target_critic is not None and self._update_steps % self._critic_target_update_interval == 0:
            self.target_critic.update_parameters(self.critic_model, polyak=self._tau)

        if self.write_interval > 0:
            log_values = {
                "Loss / Critic loss": critic_loss,
                "Loss / Policy loss": actor_loss,
                "Grad / Critic L2": critic_grad_norm,
                "Grad / Policy L2": actor_grad_norm,
                "Targets / Q mean": target_stats["target_q_mean"],
                "Targets / Q std": target_stats["target_q_std"],
                "Targets / Entropy mean": target_stats["entropy_term_mean"],
                "Targets / Entropy std": target_stats["entropy_term_std"],
                "Replay / Reward mean": target_stats["reward_mean"],
                "Replay / Reward std": target_stats["reward_std"],
                "Replay / Done fraction": target_stats["done_fraction"],
                "Policy / Q mean": actor_metrics["q_mean"],
                "Policy / Q std": actor_metrics["q_std"],
                "Policy / Entropy mean": actor_metrics["entropy_mean"],
                "Policy / Entropy std": actor_metrics["entropy_std"],
                "Policy / LogProb mean": actor_metrics["log_prob_mean"],
                "Policy / LogProb std": actor_metrics["log_prob_std"],
                "Policy / Particle std": actor_metrics["particle_std"],
                "Critic / Prediction mean": critic_metrics["prediction_mean"],
                "Critic / Prediction std": critic_metrics["prediction_std"],
                "Train / Update steps": float(self._update_steps),
            }
            for name, value in log_values.items():
                self.track_data(name, float(value))

    # === Helper functions ===
    def _critic_values_single(self, model: Model, params, states, actions):
        role = self._critic_role if model is self.critic_model else self._target_role
        values, _, _ = model.apply(params, {"states": states, "taken_actions": actions}, role)
        if values.ndim > 1 and values.shape[-1] == 1:
            values = jnp.squeeze(values, axis=-1)
        return values

    def _critic_values_particles(self, model: Model, params, states, particles):
        batch, num_particles, action_dim = particles.shape
        tiled_states = jnp.repeat(states[:, None, :], num_particles, axis=1).reshape(batch * num_particles, -1)
        flat_actions = particles.reshape(batch * num_particles, action_dim)
        role = self._critic_role if model is self.critic_model else self._target_role
        values, _, _ = model.apply(params, {"states": tiled_states, "taken_actions": flat_actions}, role)
        return values.reshape(batch, num_particles)

    def _svgd_rollout_batch(self, policy_params, critic_model: Model, critic_params, states, keys):
        def single(state, key):
            return self._svgd_rollout_single(policy_params, critic_model, critic_params, state, key)

        return jax.vmap(single)(states, keys)

    def _svgd_rollout_single(self, policy_params, critic_model: Model, critic_params, state, key):
        """
        Returns:
          squashed: (m, d) - actions after tanh and scaling (to send to env)
          raw_actions: (m, d) - pre-squash actions (needed for logqL and correction)
          log_prob: (m,) - log density (log q_L(u) + tanh-correction) per particle
        """
         
        state = jnp.reshape(state, (-1,))
        policy_inputs = {"states": state[None, :]}
        mean_actions, log_std, _ = self.policy.apply(policy_params, policy_inputs, self._policy_role)
        mean_actions = jnp.reshape(mean_actions, (self._action_dim,))
        log_std = jnp.reshape(log_std, (self._action_dim,))
        std = jnp.exp(log_std)

        eps = random.normal(key, (self._particles, self._action_dim))
        a0 = mean_actions[None, :] + std[None, :] * eps # (m, d)

        actions = a0
        all_a = []
        all_grad = []
        for _ in range(self._svgd_steps):
            # Compute gradients $\nabla_a Q(s, a)$ for each particle
            grad_q = action_score_from_Q(
                lambda params, s, a: self._critic_value_for_actions(critic_model, params, s, a),
                critic_params,
                state,
                actions,
                self._alpha,
            ) # (m, d)
            all_a.append(actions)
            all_grad.append(grad_q)
            phi = svgd_vector_field(actions, grad_q, self._kernel_sigma) # (m, d)
            actions = actions + self._svgd_step_size * phi

        # closed form log q_L from Appendix H (s2ac paper)
        logqL = compute_logqL_closed_form(
            a0,
            tuple(all_a),
            tuple(all_grad),
            mean_actions,
            log_std,
            self._svgd_step_size,
            self._kernel_sigma,
            self._alpha,
        ) # (m,)

        # tanh squash correction: log_prob = logqL + sum log |d tanh/du|
        # where log |d tanh/du| = log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        # actions here are pre-squash u
        logp_tanh = jnp.sum(2.0 * (jnp.log(2.0) - actions - jnn.softplus(-2.0 * actions)), axis=-1)
        log_prob = logqL + logp_tanh
        squashed = self._action_scale * jnp.tanh(actions)

        return squashed, actions, log_prob

    def _critic_value_for_actions(self, model: Model, params, state, actions):
        repeated_states = jnp.repeat(state[None, :], actions.shape[0], axis=0)
        role = self._critic_role if model is self.critic_model else self._target_role
        values, _, _ = model.apply(params, {"states": repeated_states, "taken_actions": actions}, role)
        return values.reshape(-1)
