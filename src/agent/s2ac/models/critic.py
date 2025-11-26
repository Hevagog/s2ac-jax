import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


class Critic_MLP(GaussianMixin, Model):
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
        if actions is None:
            actions = jnp.zeros((states.shape[0], self.num_actions), dtype=states.dtype)
        else:
            actions = jnp.atleast_2d(actions)
            if actions.shape[0] != states.shape[0]:
                if actions.shape[0] == 1:
                    actions = jnp.repeat(actions, states.shape[0], axis=0)
                else:
                    raise ValueError("Batch size mismatch between states and actions")
            actions = actions.astype(states.dtype)

        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        q = nn.Dense(1)(x)
        log_std_parameter = self.param(
            "log_std_parameter", lambda _: jnp.zeros(self.num_actions)
        )
        return jnp.squeeze(q, axis=-1), log_std_parameter, {}
