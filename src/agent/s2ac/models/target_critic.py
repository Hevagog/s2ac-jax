import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


class Target_Critic_MLP(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

    @nn.compact
    def __call__(self, inputs, role):
        states = jnp.atleast_2d(inputs["states"])
        actions = inputs.get("taken_actions")
        batch_size = states.shape[0]
        if actions is None:
            actions = jnp.zeros((batch_size, self.num_actions), dtype=states.dtype)
        else:
            actions = jnp.atleast_2d(actions)
            if actions.shape[0] != batch_size:
                if actions.shape[0] == 1:
                    # Use broadcast_to instead of repeat for better performance
                    actions = jnp.broadcast_to(actions, (batch_size, actions.shape[1]))
                else:
                    raise ValueError("Batch size mismatch between states and actions")
            actions = actions.astype(states.dtype)

        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        q = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        log_std_parameter = self.param(
            "log_std_parameter", lambda _: jnp.zeros(self.num_actions)
        )
        return jnp.squeeze(q, axis=-1), log_std_parameter, {}
