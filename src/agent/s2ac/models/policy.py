import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


class Policy_MLP(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,  # -20 causes log_prob explosion with auto entropy tuning
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
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(
            inputs["states"]
        )
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))(
            x
        )
        log_std = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        return mean, log_std, {}
