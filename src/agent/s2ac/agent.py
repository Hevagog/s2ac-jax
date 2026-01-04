from typing import Union, Tuple, Dict, Any, Optional

import copy
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax
import optax

from skrl import config
from skrl.memories.jax import Memory
from skrl.models.jax import Model

from skrl.agents.jax import Agent
from .utils import (
    compute_logqL_closed_form,
    svgd_vector_field_s2ac,
    median_heuristic_sigma,
)
from .s2ac_cfg import S2AC_DEFAULT_CONFIG
from .optimizers import AdamW


class S2AC(Agent):
    """Stein Soft Actor-Critic (S2AC) agent using SVGD for policy optimization."""

    def __init__(
        self,
        models: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, jax.Device]] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
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

        self._num_particles = self.cfg.get("particles")
        self._svgd_steps = self.cfg.get("svgd_steps")
        self._svgd_step_size = self.cfg.get("svgd_step_size")
        self._kernel_sigma = self.cfg.get("kernel_sigma")
        self._kernel_sigma_adaptive = self.cfg.get("kernel_sigma_adaptive")
        self._kernel_sigma_max = self.cfg.get("kernel_sigma_max", 10.0)
        self._stop_grad_svgd_score = self.cfg.get("stop_grad_svgd_score")
        self._max_phi_norm = self.cfg.get("max_phi_norm", 10.0)
        self._u_clip_bound = self.cfg.get("u_clip_bound", 5.0)
        self._use_soft_q_backup = self.cfg.get("use_soft_q_backup")

        self._batch_size = self.cfg.get("batch_size")
        self._discount = self.cfg.get("discount")
        self._tau = self.cfg.get("tau")
        self._actor_lr = self.cfg.get("actor_lr")
        self._critic_lr = self.cfg.get("critic_lr")
        self._critic_target_update_interval = self.cfg.get(
            "critic_target_update_interval"
        )
        self._actor_update_frequency = self.cfg.get("actor_update_frequency")
        self._update_counter = 0

        self._entropy_floor = self.cfg.get("entropy_floor")
        self._entropy_floor_coef = self.cfg.get("entropy_floor_coef")

        self._auto_entropy_tuning = self.cfg.get("auto_entropy_tuning")
        self._alpha = self.cfg.get("alpha")
        self._target_entropy = self.cfg.get("target_entropy")
        self._log_alpha_bounds = self.cfg.get("log_alpha_bounds", (-2.0, 2.0))

        self._reward_scale = self.cfg.get("reward_scale")
        self._random_timesteps = self.cfg.get("random_timesteps")
        self._learning_starts = self.cfg.get("learning_starts")
        self._grad_norm_clip = self.cfg.get("grad_norm_clip")

        twin_cfg = bool(self.cfg.get("twin_critics"))
        has_critic2 = self.models.get("critic_2", None) is not None
        has_target2 = self.models.get("target_critic_2", None) is not None
        if twin_cfg:
            if has_critic2 or has_target2:
                if not (has_critic2 and has_target2):
                    raise ValueError(
                        "Twin critics requested/provided but critic_2/target_critic_2 are inconsistent"
                    )
                self._twin_critics = True
            else:
                self._twin_critics = False
        else:
            self._twin_critics = False

        self.policy = self.models.get("policy", None)
        self.critic_models = []
        if self._twin_critics:
            self.critic_models = [
                self.models.get("critic_1", self.models.get("critic", None)),
                self.models.get("critic_2", None),
            ]
            self.target_critic_models = [
                self.models.get(
                    "target_critic_1", self.models.get("target_critic", None)
                ),
                self.models.get("target_critic_2", None),
            ]
        else:
            self.critic_models = [
                self.models.get("critic", self.models.get("critic_1", None))
            ]
            self.target_critic_models = [
                self.models.get(
                    "target_critic", self.models.get("target_critic_1", None)
                )
            ]

        if self.policy is None:
            raise ValueError("Policy model is required")
        for i, c in enumerate(self.critic_models):
            if c is None:
                raise ValueError(f"Critic model {i} is required")
        for i, tc in enumerate(self.target_critic_models):
            if tc is None:
                raise ValueError(f"Target critic model {i} is required")

        self.critic_model = self.critic_models[0]
        self.target_critics = self.target_critic_models
        self._particles = self._num_particles

        if hasattr(self.action_space, "shape") and self.action_space.shape is not None:
            self._action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, (tuple, list)):
            self._action_dim = self.action_space[0]
        else:
            self._action_dim = self.action_space

        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            self._action_scale = jnp.array(
                (self.action_space.high - self.action_space.low) / 2.0,
                dtype=jnp.float32,
            )
            self._action_bias = jnp.array(
                (self.action_space.high + self.action_space.low) / 2.0,
                dtype=jnp.float32,
            )
        else:
            self._action_scale = jnp.ones(self._action_dim, dtype=jnp.float32)
            self._action_bias = jnp.zeros(self._action_dim, dtype=jnp.float32)

        if self._target_entropy is None:
            self._target_entropy = -float(self._action_dim)

        self._log_alpha = jnp.log(jnp.array(self._alpha, dtype=jnp.float32))
        self._rng_key = random.PRNGKey(0)

        self._tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
            "truncated",
        ]

        self._weight_decay = self.cfg.get("weight_decay", 1e-4)

        self.checkpoint_modules["policy"] = self.policy
        for i, c in enumerate(self.critic_models):
            self.checkpoint_modules[f"critic_{i}"] = c
        for i, tc in enumerate(self.target_critic_models):
            self.checkpoint_modules[f"target_critic_{i}"] = tc

    def _space_shape(self, space_like) -> Tuple[int, ...]:
        if space_like is None:
            raise ValueError("Space is required to infer tensor shapes")
        if hasattr(space_like, "shape") and space_like.shape is not None:
            return tuple(space_like.shape)
        if isinstance(space_like, int):
            return (space_like,)
        if isinstance(space_like, (tuple, list)):
            return tuple(space_like)
        raise TypeError(
            f"Unsupported space type for shape inference: {type(space_like)}"
        )

    def _memory_has_tensor(self, name: str) -> bool:
        if self.memory is None:
            return False
        try:
            _ = self.memory.tensors_keep_dimensions[name]
            return True
        except Exception:
            pass
        try:
            _ = self.memory.tensors[name]
            return True
        except Exception:
            return False

    def _ensure_memory_tensors(self) -> None:
        if self.memory is None:
            return
        if not hasattr(self.memory, "create_tensor"):
            return

        obs_shape = self._space_shape(self.observation_space)
        act_shape = self._space_shape(self.action_space)

        specs = [
            ("states", obs_shape, jnp.float32, False),
            ("actions", act_shape, jnp.float32, False),
            ("rewards", (1,), jnp.float32, False),
            ("next_states", obs_shape, jnp.float32, False),
            ("terminated", (1,), bool, False),
            ("truncated", (1,), bool, False),
        ]

        for name, size, dtype, keep_dimensions in specs:
            if not self._memory_has_tensor(name):
                self.memory.create_tensor(
                    name=name, size=size, dtype=dtype, keep_dimensions=keep_dimensions
                )

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().init(trainer_cfg)
        self.set_mode("train")
        self._ensure_memory_tensors()

        for critic, target in zip(self.critic_models, self.target_critic_models):
            target.update_parameters(critic, polyak=1.0)

        # If scale set to true, the learning rate would be applied twice, once by the optimizer and once by the step call.

        self.policy_optimizer = AdamW(
            model=self.policy,
            lr=self._actor_lr,
            weight_decay=self._weight_decay,
            grad_norm_clip=self._grad_norm_clip,
            scale=False,
        )
        if self._twin_critics:
            self.critic_optimizer_1 = AdamW(
                model=self.critic_models[0],
                lr=self._critic_lr,
                weight_decay=self._weight_decay,
                grad_norm_clip=self._grad_norm_clip,
                scale=False,
            )
            self.critic_optimizer_2 = AdamW(
                model=self.critic_models[1],
                lr=self._critic_lr,
                weight_decay=self._weight_decay,
                grad_norm_clip=self._grad_norm_clip,
                scale=False,
            )
        else:
            self.critic_optimizer = AdamW(
                model=self.critic_models[0],
                lr=self._critic_lr,
                weight_decay=self._weight_decay,
                grad_norm_clip=self._grad_norm_clip,
                scale=False,
            )

        if self._auto_entropy_tuning:
            self.alpha_optimizer = optax.adamw(
                learning_rate=self._actor_lr, weight_decay=0.0
            )
            self.alpha_opt_state = self.alpha_optimizer.init(self._log_alpha)

        self._setup_loss_functions()

        def _alpha_loss_fn(log_alpha, log_prob_mean):
            alpha = jnp.exp(log_alpha)
            return -alpha * (log_prob_mean + self._target_entropy)

        self._alpha_value_and_grad = jax.jit(jax.value_and_grad(_alpha_loss_fn))

    def _setup_loss_functions(self):
        # Critic loss function
        if self._twin_critics:

            def _critic_loss_fn_twin(params1, params2, states, actions, targets):
                pred1 = self._critic_values_single(
                    self.critic_models[0], params1, states, actions
                )
                pred2 = self._critic_values_single(
                    self.critic_models[1], params2, states, actions
                )
                loss = jnp.mean((pred1 - targets) ** 2 + (pred2 - targets) ** 2)
                metrics = {
                    "prediction_mean": 0.5 * (jnp.mean(pred1) + jnp.mean(pred2)),
                    "prediction_std": 0.5 * (jnp.std(pred1) + jnp.std(pred2)),
                }
                return loss, metrics

            self._critic_value_and_grad = jax.jit(
                jax.value_and_grad(_critic_loss_fn_twin, argnums=(0, 1), has_aux=True)
            )
        else:

            def _critic_loss_fn_single(params, states, actions, targets):
                pred = self._critic_values_single(
                    self.critic_models[0], params, states, actions
                )
                loss = jnp.mean((pred - targets) ** 2)
                metrics = {
                    "prediction_mean": jnp.mean(pred),
                    "prediction_std": jnp.std(pred),
                }
                return loss, metrics

            self._critic_value_and_grad = jax.jit(
                jax.value_and_grad(_critic_loss_fn_single, has_aux=True)
            )

        # Actor loss function
        if self._twin_critics:

            def _actor_loss_fn_twin(
                policy_params, critic1_params, critic2_params, states, keys, log_alpha
            ):
                alpha = jnp.exp(log_alpha)

                # SVGD Rollout -> get both u (unbounded) and scaled actions
                particles_u, particles_a, _ = self._svgd_rollout_batch(
                    policy_params,
                    (self.critic_models[0], self.critic_models[1]),
                    (critic1_params, critic2_params),
                    states,
                    keys,
                )
                particles_a = jax.lax.stop_gradient(particles_a)

                # Compute GRAD Q in u-space (attraction)
                # get scalar Q(u, s) callable for grads
                q_min_wrt_u = self._make_q_wrt_u_scalar(
                    (self.critic_models[0], self.critic_models[1]),
                    (critic1_params, critic2_params),
                )

                # vmap over batch and particles to compute grad dQ/du
                grad_q = jax.vmap(
                    jax.vmap(jax.grad(q_min_wrt_u), in_axes=(0, None)),
                    in_axes=(0, 0),
                )(particles_u, states)

                # Stein Force (operate in u-space)
                def compute_batch_force_u(p_batch_u, g_batch_u):
                    sigma = (
                        self._mean_heuristic_sigma(p_batch_u)
                        if self._kernel_sigma_adaptive
                        else self._kernel_sigma
                    )
                    sigma = jnp.maximum(sigma, 0.1)
                    if self._stop_grad_svgd_score:
                        g_batch_u = jax.lax.stop_gradient(g_batch_u)
                    return svgd_vector_field_s2ac(p_batch_u, g_batch_u, sigma, alpha)

                stein_forces = jax.vmap(compute_batch_force_u)(particles_u, grad_q)
                stein_forces_sg = jax.lax.stop_gradient(stein_forces)

                # Surrogate Loss: re-evaluate rollout (differentiable) to produce u outputs
                # Here we use u (the first returned element) so autodiff flows through u -> policy_params
                particles_u_for_grad, _, _ = self._svgd_rollout_batch(
                    policy_params,
                    (self.critic_models[0], self.critic_models[1]),
                    (critic1_params, critic2_params),
                    states,
                    keys,
                )
                surrogate_loss = -jnp.mean(
                    jnp.sum(particles_u_for_grad * stein_forces_sg, axis=-1)
                )

                # Metrics
                q1 = self._critic_values_particles(
                    self.critic_models[0], critic1_params, states, particles_a
                )
                q2 = self._critic_values_particles(
                    self.critic_models[1], critic2_params, states, particles_a
                )
                q_values = jnp.minimum(q1, q2)

                particle_std = jnp.mean(jnp.std(particles_a, axis=1), axis=-1)
                entropy_est = 0.5 * self._action_dim * (
                    1.0 + jnp.log(2 * jnp.pi)
                ) + jnp.log(particle_std + 1e-8)

                # Additional metrics
                stein_force_norm = jnp.mean(jnp.linalg.norm(stein_forces_sg, axis=-1))
                grad_q_norm = jnp.mean(jnp.linalg.norm(grad_q, axis=-1))

                if self._kernel_sigma_adaptive:
                    sigmas = jax.vmap(self._mean_heuristic_sigma)(particles_u)
                    mean_sigma = jnp.mean(sigmas)
                else:
                    mean_sigma = self._kernel_sigma if self._kernel_sigma else 0.0

                metrics = {
                    "q_mean": jnp.mean(q_values),
                    "q_std": jnp.std(q_values),
                    "entropy_mean": jnp.mean(entropy_est),
                    "entropy_std": jnp.std(entropy_est),
                    "log_prob_mean": -jnp.mean(entropy_est),
                    "log_prob_std": jnp.std(entropy_est),
                    "alpha": alpha,
                    "particle_std": jnp.mean(particle_std),
                    "stein_force_norm": stein_force_norm,
                    "grad_q_norm": grad_q_norm,
                    "kernel_sigma_mean": mean_sigma,
                    "entropy_floor_penalty": 0.0,
                }
                return surrogate_loss, metrics

            self._actor_value_and_grad = jax.jit(
                jax.value_and_grad(_actor_loss_fn_twin, argnums=0, has_aux=True)
            )
        else:
            # Single critic version
            def _actor_loss_fn_single(
                policy_params, critic_params, states, keys, log_alpha
            ):
                alpha = jnp.exp(log_alpha)

                particles_u, particles_a, _ = self._svgd_rollout_batch(
                    policy_params, self.critic_models[0], critic_params, states, keys
                )
                particles_a = jax.lax.stop_gradient(particles_a)

                q_wrt_u_scalar = self._make_q_wrt_u_scalar(
                    self.critic_models[0], critic_params
                )

                grad_q = jax.vmap(
                    jax.vmap(jax.grad(q_wrt_u_scalar), in_axes=(0, None)),
                    in_axes=(0, 0),
                )(particles_u, states)

                # Stein Force
                def compute_batch_force_u(p_batch_u, g_batch_u):
                    sigma = (
                        self._mean_heuristic_sigma(p_batch_u)
                        if self._kernel_sigma_adaptive
                        else self._kernel_sigma
                    )
                    sigma = jnp.maximum(sigma, 0.1)

                    if self._stop_grad_svgd_score:
                        g_batch_u = jax.lax.stop_gradient(g_batch_u)

                    return svgd_vector_field_s2ac(p_batch_u, g_batch_u, sigma, alpha)

                stein_forces = jax.vmap(compute_batch_force_u)(particles_u, grad_q)
                stein_forces_sg = jax.lax.stop_gradient(stein_forces)

                particles_u_for_grad, _, _ = self._svgd_rollout_batch(
                    policy_params, self.critic_models[0], critic_params, states, keys
                )
                surrogate_loss = -jnp.mean(
                    jnp.sum(particles_u_for_grad * stein_forces_sg, axis=-1)
                )

                q_values = self._critic_values_particles(
                    self.critic_models[0], critic_params, states, particles_a
                )
                particle_std = jnp.mean(jnp.std(particles_a, axis=1), axis=-1)
                entropy_est = 0.5 * self._action_dim * (
                    1.0 + jnp.log(2 * jnp.pi)
                ) + jnp.log(particle_std + 1e-8)

                # Additional metrics
                stein_force_norm = jnp.mean(jnp.linalg.norm(stein_forces_sg, axis=-1))
                grad_q_norm = jnp.mean(jnp.linalg.norm(grad_q, axis=-1))

                if self._kernel_sigma_adaptive:
                    sigmas = jax.vmap(self._mean_heuristic_sigma)(particles_u)
                    mean_sigma = jnp.mean(sigmas)
                else:
                    mean_sigma = self._kernel_sigma if self._kernel_sigma else 0.0

                metrics = {
                    "q_mean": jnp.mean(q_values),
                    "q_std": jnp.std(q_values),
                    "entropy_mean": jnp.mean(entropy_est),
                    "entropy_std": jnp.std(entropy_est),
                    "log_prob_mean": -jnp.mean(entropy_est),
                    "log_prob_std": jnp.std(entropy_est),
                    "alpha": alpha,
                    "particle_std": jnp.mean(particle_std),
                    "stein_force_norm": stein_force_norm,
                    "grad_q_norm": grad_q_norm,
                    "kernel_sigma_mean": mean_sigma,
                    "entropy_floor_penalty": 0.0,
                }
                return surrogate_loss, metrics

            self._actor_value_and_grad = jax.jit(
                jax.value_and_grad(_actor_loss_fn_single, argnums=0, has_aux=True)
            )

    def _role_for_critic_model(self, model: Model) -> str:
        for i, c in enumerate(self.critic_models):
            if c is model:
                return f"critic_{i + 1}" if self._twin_critics else "critic"
        for i, tc in enumerate(self.target_critic_models):
            if tc is model:
                return (
                    f"target_critic_{i + 1}" if self._twin_critics else "target_critic"
                )
        return "critic"

    def _make_q_wrt_u_scalar(self, critic_model, critic_params):
        """
        Return a callable q(u_vec, s) -> scalar that evaluates the critic(s)
        at the action produced by u_vec via action transform:
            a = action_scale * tanh(u_vec) + action_bias
        """
        # twin critics case
        if isinstance(critic_model, (tuple, list)):

            def q_wrt_u(u_vec, s):
                a_vec = self._action_scale * jnp.tanh(u_vec) + self._action_bias
                s_b = s[None, :]  # shape (1, S)
                a_b = a_vec[None, :]  # shape (1, A)
                v1, _, _ = critic_model[0].apply(
                    critic_params[0],
                    {"states": s_b, "taken_actions": a_b},
                    self._role_for_critic_model(critic_model[0]),
                )
                v2, _, _ = critic_model[1].apply(
                    critic_params[1],
                    {"states": s_b, "taken_actions": a_b},
                    self._role_for_critic_model(critic_model[1]),
                )
                return jnp.minimum(v1.reshape(()), v2.reshape(()))

        else:
            # single critic
            def q_wrt_u(u_vec, s):
                a_vec = self._action_scale * jnp.tanh(u_vec) + self._action_bias
                s_b = s[None, :]
                a_b = a_vec[None, :]
                v, _, _ = critic_model.apply(
                    critic_params,
                    {"states": s_b, "taken_actions": a_b},
                    self._role_for_critic_model(critic_model),
                )
                return v.reshape(())

        return q_wrt_u

    def act(self, states: jax.Array, timestep: int, timesteps: int) -> jax.Array:
        states = jnp.atleast_2d(states)
        if timestep < self._random_timesteps:
            self._rng_key, subkey = random.split(self._rng_key)
            actions = random.uniform(
                subkey,
                shape=(states.shape[0], self._action_dim),
                minval=-1.0,
                maxval=1.0,
            )
            return self._action_scale * actions + self._action_bias, None, {}

        policy_params = self.policy.state_dict.params
        if self._twin_critics:
            critic_params = (
                self.critic_models[0].state_dict.params,
                self.critic_models[1].state_dict.params,
            )
            critic_model = (self.critic_models[0], self.critic_models[1])
        else:
            critic_params = self.critic_models[0].state_dict.params
            critic_model = self.critic_models[0]

        self._rng_key, subkey = random.split(self._rng_key)
        keys = random.split(subkey, states.shape[0])

        _, particles, log_prob = self._svgd_rollout_batch(
            policy_params, critic_model, critic_params, states, keys
        )
        mode = "softmax" if self.training else "max"
        actions, selected_log_prob = self._select_action_from_particles(
            mode, states, particles, log_prob
        )
        return actions, selected_log_prob, {}

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
        if timestep >= self._learning_starts:
            self._update(timestep, timesteps)
        super().post_interaction(timestep, timesteps)

    def _clip_grads(self, grads):
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
        )
        if self._grad_norm_clip is not None and self._grad_norm_clip > 0:
            grads = optax.clip_by_global_norm(self._grad_norm_clip).update(grads, None)[
                0
            ]
        return grads

    def _update(self, timestep: int, timesteps: int) -> None:
        if self.memory is None or len(self.memory) < self._batch_size:
            return

        self._update_counter += 1

        (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[
            0
        ]

        sampled_states = jnp.asarray(sampled_states)
        sampled_actions = jnp.asarray(sampled_actions)
        sampled_rewards = jnp.asarray(sampled_rewards).squeeze() * self._reward_scale
        sampled_next_states = jnp.asarray(sampled_next_states)
        sampled_terminated = jnp.asarray(sampled_terminated).squeeze()

        self.track_data("Replay / Reward mean", float(jnp.mean(sampled_rewards)))
        self.track_data("Replay / Reward std", float(jnp.std(sampled_rewards)))
        self.track_data("Replay / Done fraction", float(jnp.mean(sampled_terminated)))

        policy_params = self.policy.state_dict.params
        if self._twin_critics:
            critic_params1 = self.critic_models[0].state_dict.params
            critic_params2 = self.critic_models[1].state_dict.params
            target_params1 = self.target_critic_models[0].state_dict.params
            target_params2 = self.target_critic_models[1].state_dict.params
        else:
            critic_params = self.critic_models[0].state_dict.params
            target_params = self.target_critic_models[0].state_dict.params

        current_alpha = (
            jnp.exp(self._log_alpha) if self._auto_entropy_tuning else self._alpha
        )

        self._rng_key, subkey = random.split(self._rng_key)
        next_keys = random.split(subkey, sampled_next_states.shape[0])

        if self._twin_critics:
            _, next_particles, next_log_prob = self._svgd_rollout_batch(
                policy_params,
                (self.critic_models[0], self.critic_models[1]),
                (critic_params1, critic_params2),
                sampled_next_states,
                next_keys,
            )
        else:
            _, next_particles, next_log_prob = self._svgd_rollout_batch(
                policy_params,
                self.critic_models[0],
                critic_params,
                sampled_next_states,
                next_keys,
            )
        next_particles = jax.lax.stop_gradient(next_particles)
        next_log_prob = jax.lax.stop_gradient(next_log_prob)

        if self._twin_critics:
            q1_next = self._critic_values_particles(
                self.target_critic_models[0],
                target_params1,
                sampled_next_states,
                next_particles,
            )
            q2_next = self._critic_values_particles(
                self.target_critic_models[1],
                target_params2,
                sampled_next_states,
                next_particles,
            )
            q_next = jnp.minimum(q1_next, q2_next)
        else:
            q_next = self._critic_values_particles(
                self.target_critic_models[0],
                target_params,
                sampled_next_states,
                next_particles,
            )

        q_next = jnp.where(jnp.isfinite(q_next), q_next, 0.0)

        if self._use_soft_q_backup:
            # Clip q_next / alpha to prevent overflow in logsumexp
            q_scaled = jnp.clip(q_next / (current_alpha + 1e-8), -50.0, 50.0)
            v_next = current_alpha * jax.scipy.special.logsumexp(
                q_scaled, axis=1
            ) - current_alpha * jnp.log(float(self._num_particles))
        else:
            v_next = jnp.mean(q_next - current_alpha * next_log_prob, axis=1)

        v_next = jnp.where(jnp.isfinite(v_next), v_next, 0.0)
        targets = sampled_rewards + self._discount * (1.0 - sampled_terminated) * v_next
        targets = jax.lax.stop_gradient(targets)

        # Critic Update
        if self._twin_critics:
            (critic_loss, critic_metrics), (grad1, grad2) = self._critic_value_and_grad(
                critic_params1, critic_params2, sampled_states, sampled_actions, targets
            )
            if config.jax.is_distributed:
                grad1 = self.critic_models[0].reduce_parameters(grad1)
                grad2 = self.critic_models[1].reduce_parameters(grad2)

            grad1 = self._clip_grads(grad1)
            grad2 = self._clip_grads(grad2)

            self.critic_optimizer_1 = self.critic_optimizer_1.step(
                grad1, self.critic_models[0], self._critic_lr
            )
            self.critic_optimizer_2 = self.critic_optimizer_2.step(
                grad2, self.critic_models[1], self._critic_lr
            )

            critic_params1 = self.critic_models[0].state_dict.params
            critic_params2 = self.critic_models[1].state_dict.params
        else:
            (critic_loss, critic_metrics), critic_grad = self._critic_value_and_grad(
                critic_params, sampled_states, sampled_actions, targets
            )
            if config.jax.is_distributed:
                critic_grad = self.critic_models[0].reduce_parameters(critic_grad)

            critic_grad = self._clip_grads(critic_grad)

            self.critic_optimizer = self.critic_optimizer.step(
                critic_grad, self.critic_models[0], self._critic_lr
            )
            critic_params = self.critic_models[0].state_dict.params

        # Actor Update (only every actor_update_frequency steps - delayed policy updates)
        actor_loss = 0.0
        actor_metrics = None
        actor_grad_norm = 0.0

        if self._update_counter % self._actor_update_frequency == 0:
            self._rng_key, subkey = random.split(self._rng_key)
            actor_keys = random.split(subkey, sampled_states.shape[0])
            log_alpha_for_actor = (
                self._log_alpha if self._auto_entropy_tuning else jnp.log(self._alpha)
            )

            if self._twin_critics:
                (actor_loss, actor_metrics), actor_grad = self._actor_value_and_grad(
                    policy_params,
                    critic_params1,
                    critic_params2,
                    sampled_states,
                    actor_keys,
                    log_alpha_for_actor,
                )
            else:
                (actor_loss, actor_metrics), actor_grad = self._actor_value_and_grad(
                    policy_params,
                    critic_params,
                    sampled_states,
                    actor_keys,
                    log_alpha_for_actor,
                )

            if config.jax.is_distributed:
                actor_grad = self.policy.reduce_parameters(actor_grad)

            actor_grad = self._clip_grads(actor_grad)

            self.policy_optimizer = self.policy_optimizer.step(
                actor_grad, self.policy, self._actor_lr
            )

            actor_grad_norm = self._pytree_l2_norm(actor_grad)

            # Alpha Update (only when actor updates)
            if self._auto_entropy_tuning:
                log_prob_mean = jax.lax.stop_gradient(actor_metrics["log_prob_mean"])
                log_prob_mean = jnp.where(
                    jnp.isfinite(log_prob_mean), log_prob_mean, -1.0
                )
                alpha_loss, alpha_grad = self._alpha_value_and_grad(
                    self._log_alpha, log_prob_mean
                )
                alpha_grad = jnp.where(jnp.isfinite(alpha_grad), alpha_grad, 0.0)
                updates, self.alpha_opt_state = self.alpha_optimizer.update(
                    alpha_grad, self.alpha_opt_state, params=self._log_alpha
                )
                self._log_alpha = optax.apply_updates(self._log_alpha, updates)

                # Lower bound prevents scores from exploding (scores = grad_q / alpha)
                self._log_alpha = jnp.clip(
                    self._log_alpha,
                    self._log_alpha_bounds[0],
                    self._log_alpha_bounds[1],
                )

        # Target network update
        if timestep % self._critic_target_update_interval == 0:
            for critic, target in zip(self.critic_models, self.target_critic_models):
                target.update_parameters(critic, polyak=self._tau)

        if self._twin_critics:
            critic_grad_norm = (
                self._pytree_l2_norm(grad1) + self._pytree_l2_norm(grad2)
            ) / 2.0
        else:
            critic_grad_norm = self._pytree_l2_norm(critic_grad)

        self.track_data("Reward / Batch Mean", float(jnp.mean(sampled_rewards)))
        self.track_data("Reward / Batch Max", float(jnp.max(sampled_rewards)))

        self.track_data("Loss / Critic loss", float(critic_loss))
        self.track_data("Loss / Actor loss", float(actor_loss))

        self.track_data("Coefficient / Alpha", float(current_alpha))

        self.track_data("Grad / Critic L2", float(critic_grad_norm))
        self.track_data("Grad / Actor L2", float(actor_grad_norm))

        if actor_metrics is not None:
            self.track_data("Loss / Alpha loss", float(alpha_loss))
            self.track_data(
                "Policy / Entropy mean", float(actor_metrics["entropy_mean"])
            )
            self.track_data("Policy / Entropy std", float(actor_metrics["entropy_std"]))
            self.track_data(
                "Policy / Log prob mean", float(actor_metrics["log_prob_mean"])
            )
            self.track_data(
                "Policy / Log prob std", float(actor_metrics["log_prob_std"])
            )
            self.track_data(
                "Policy / Particle std", float(actor_metrics["particle_std"])
            )
            self.track_data(
                "Policy / Stein Force Norm", float(actor_metrics["stein_force_norm"])
            )
            self.track_data("Policy / Grad Q Norm", float(actor_metrics["grad_q_norm"]))
            self.track_data(
                "Policy / Kernel Sigma Mean", float(actor_metrics["kernel_sigma_mean"])
            )

        self.track_data("Targets / Q mean", float(jnp.mean(targets)))
        self.track_data("Targets / Q std", float(jnp.std(targets)))
        self.track_data("Targets / Reward mean", float(jnp.mean(sampled_rewards)))
        self.track_data("Targets / Reward std", float(jnp.std(sampled_rewards)))

        self.track_data(
            "Critic / Prediction Mean", float(critic_metrics["prediction_mean"])
        )
        self.track_data(
            "Critic / Prediction Std", float(critic_metrics["prediction_std"])
        )

    def _critic_values_single(self, model: Model, params, states, actions):
        """Get Q-values for state-action pairs with robust squeeze."""
        role = self._role_for_critic_model(model)
        values, _, _ = model.apply(
            params, {"states": states, "taken_actions": actions}, role
        )
        if values.ndim > 1:
            return values.squeeze(-1)
        return values

    def _critic_value_for_actions(self, model: Model, params, state, actions):
        """
        Computes Q(s, a) where 'state' is (s_dim,) and 'actions' is (m, a_dim).
        """
        # Since SKRL models expect batch dimensions for "states" and "taken_actions",
        # we virtually broadcast state to match actions batch size.
        m = actions.shape[0]
        # (S) -> (M, S)
        if state.ndim == 1:
            state_broadcast = jnp.broadcast_to(state, (m, state.shape[0]))
        else:
            # If state passed as (1, S)
            state_broadcast = jnp.broadcast_to(state, (m, state.shape[1]))

        role = self._role_for_critic_model(model)
        values, _, _ = model.apply(
            params, {"states": state_broadcast, "taken_actions": actions}, role
        )

        if values.ndim > 1:
            return values.squeeze(-1)
        return values

    def _critic_values_particles(self, model: Model, params, states, particles):
        """
        states: (Batch, State_Dim)
        particles: (Batch, Num_Particles, Action_Dim)
        Returns: (Batch, Num_Particles)
        """
        role = self._role_for_critic_model(model)
        batch_size = states.shape[0]
        num_particles = particles.shape[1]
        state_dim = states.shape[1]

        # states_expanded: (Batch, Num_Particles, State_Dim)
        states_expanded = jnp.broadcast_to(
            states[:, None, :], (batch_size, num_particles, state_dim)
        )

        states_flat = states_expanded.reshape(-1, state_dim)
        particles_flat = particles.reshape(-1, particles.shape[-1])

        # Single forward pass through the model
        out, _, _ = model.apply(
            params, {"states": states_flat, "taken_actions": particles_flat}, role
        )

        # Reshape back to (Batch, Num_Particles)
        result = out.reshape(batch_size, num_particles)
        return result

    def _svgd_rollout_batch(
        self, policy_params, critic_model, critic_params, states, keys
    ):
        """
        Batched SVGD rollout using vmap.
        The vmap is applied once at call time for flexibility with different critic configs.
        """

        def single(state, key):
            return self._svgd_rollout_single(
                policy_params, critic_model, critic_params, state, key
            )

        return jax.vmap(single)(states, keys)

    def _svgd_rollout_single(
        self, policy_params, critic_model, critic_params, state, key
    ):
        """
        Perform SVGD rollout for a single state (vmapped over batch in _svgd_rollout_batch).
        Returns: (final_u, final_a, log_prob)
        """
        mean, log_std, _ = self.policy.apply(
            policy_params, {"states": state[None, ...]}, "policy"
        )
        mean = mean[0]
        log_std = log_std[0]
        std = jnp.exp(log_std)

        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=(self._particles, self._action_dim))
        u0 = mean + std * noise  # (P, A)

        q_wrt_u_scalar = self._make_q_wrt_u_scalar(critic_model, critic_params)

        # initial kernel sigma
        if (
            self._kernel_sigma_adaptive
            or self._kernel_sigma is None
            or self._kernel_sigma <= 0
        ):
            sigma0 = self._mean_heuristic_sigma(u0)
        else:
            sigma0 = self._kernel_sigma

        alpha = jnp.exp(self._log_alpha) if self._auto_entropy_tuning else self._alpha
        eps = self._svgd_step_size
        T = int(self._svgd_steps)

        u_clip_bound = self._u_clip_bound

        # step function for lax.scan: carry = (u, sigma), out = (u_pre, grad_q_pre, sigma_pre)
        def svgd_step(carry, _):
            u, sigma = carry  # u: (P, A)
            # compute grad_q (P, A)
            grad_q = jax.vmap(jax.grad(q_wrt_u_scalar), in_axes=(0, None))(u, state)

            grad_q = jnp.where(jnp.isfinite(grad_q), grad_q, 0.0)
            # optionally stop grad on score
            grad_q_for_phi = (
                jax.lax.stop_gradient(grad_q) if self._stop_grad_svgd_score else grad_q
            )
            phi = svgd_vector_field_s2ac(u, grad_q_for_phi, sigma, alpha)
            u_next = u + eps * phi
            # Clip particles to prevent explosion
            u_next = jnp.clip(u_next, -u_clip_bound, u_clip_bound)
            sigma_next = (
                self._mean_heuristic_sigma(u_next)
                if self._kernel_sigma_adaptive
                else sigma
            )
            out = (u, grad_q, sigma)  # store pre-update u, grad, sigma
            return (u_next, sigma_next), out

        # run scan
        (u_final, _), scan_outs = lax.scan(svgd_step, (u0, sigma0), None, length=T)
        # scan_outs is a tuple (u_traj, gradQ_traj, sigmas_traj) with shapes:
        # u_traj: (T, P, A), gradQ_traj: (T, P, A), sigmas_traj: (T,)
        u_traj, gradQ_traj, sigmas_traj = scan_outs

        # final scaled action
        final_a = self._action_scale * jnp.tanh(u_final) + self._action_bias

        # compute closed-form log q_L based on u-trajectory; a0 = initial u
        log_prob = compute_logqL_closed_form(
            a0=u0,  # initial u
            all_a_list=u_traj,  # (T, P, A)
            all_gradQ_list=gradQ_traj,  # (T, P, A)
            mu0=mean,
            logstd0=log_std,
            eps=eps,
            sigma_list=sigmas_traj,
            alpha=alpha,
        )

        return u_final, final_a, log_prob

    def _critic_values_for_actions_twin(self, models, params_tuple, state, actions):
        q1 = self._critic_value_for_actions(models[0], params_tuple[0], state, actions)
        q2 = self._critic_value_for_actions(models[1], params_tuple[1], state, actions)
        return jnp.minimum(q1, q2)

    def _mean_heuristic_sigma(
        self, actions: jnp.ndarray, h_min: float = 1e-3
    ) -> jnp.ndarray:
        return median_heuristic_sigma(actions, h_min, self._kernel_sigma_max)

    def _select_action_from_particles(
        self, mode: str, states: jax.Array, particles: jax.Array, log_prob: jax.Array
    ):
        batch, m, _ = particles.shape
        mode = (mode or "random").lower()

        if mode == "mean":
            return jnp.mean(particles, axis=1), jnp.mean(
                log_prob, axis=1, keepdims=True
            )

        if mode == "random":
            self._rng_key, subkey = random.split(self._rng_key)
            idx = random.randint(subkey, (batch,), minval=0, maxval=m)
            actions = particles[jnp.arange(batch), idx]
            selected_log_prob = log_prob[jnp.arange(batch), idx][:, None]
            return actions, selected_log_prob

        # Evaluate Q-values for selection (Max or Softmax)
        if self._twin_critics:
            q_vals = jnp.minimum(
                self._critic_values_particles(
                    self.critic_models[0],
                    self.critic_models[0].state_dict.params,
                    states,
                    particles,
                ),
                self._critic_values_particles(
                    self.critic_models[1],
                    self.critic_models[1].state_dict.params,
                    states,
                    particles,
                ),
            )
        else:
            q_vals = self._critic_values_particles(
                self.critic_models[0],
                self.critic_models[0].state_dict.params,
                states,
                particles,
            )

        if mode == "max":
            idx = jnp.argmax(q_vals, axis=1)
            actions = particles[jnp.arange(batch), idx]
            selected_log_prob = log_prob[jnp.arange(batch), idx][:, None]
            return actions, selected_log_prob

        if mode == "softmax":
            alpha = (
                jnp.exp(self._log_alpha) if self._auto_entropy_tuning else self._alpha
            )
            # pass logits directly to categorical for numerical stability
            logits = q_vals / (alpha + 1e-12)
            self._rng_key, subkey = random.split(self._rng_key)
            idx = random.categorical(subkey, logits, axis=1)
            actions = particles[jnp.arange(batch), idx]
            selected_log_prob = log_prob[jnp.arange(batch), idx][:, None]
            return actions, selected_log_prob

        return jnp.mean(particles, axis=1), jnp.mean(log_prob, axis=1, keepdims=True)

    def _pytree_l2_norm(self, tree):
        """Compute L2 norm of a pytree of arrays using optax for efficiency."""
        return optax.global_norm(tree)
