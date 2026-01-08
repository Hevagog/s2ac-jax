from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


class Policy_MLP(GaussianMixin, Model):
    """Gaussian MLP policy for S2AC.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        device: JAX device
        clip_actions: Whether to clip actions to action space bounds
        clip_log_std: Whether to clip log standard deviation
        min_log_std: Minimum log standard deviation
        max_log_std: Maximum log standard deviation
        reduction: Reduction method for log probability
        hidden_sizes: Tuple of hidden layer sizes (default: (256, 256))
    """

    hidden_sizes: Sequence[int] = (256, 256)

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
        reduction="sum",
        hidden_sizes: Sequence[int] = (256, 256),
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )
        object.__setattr__(self, "hidden_sizes", tuple(hidden_sizes))

    @nn.compact
    def __call__(self, inputs, role):
        x = inputs["states"]

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(
                hidden_size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
            )(x)
            x = nn.relu(x)

        mean = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))(
            x
        )
        log_std = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)

        return mean, log_std, {}
