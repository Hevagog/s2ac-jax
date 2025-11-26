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

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.Dense(64)(inputs["states"])
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        log_std_parameter = self.param(
            "log_std_parameter", lambda _: jnp.zeros(self.num_actions)
        )
        return nn.tanh(x), log_std_parameter, {}
