from __future__ import annotations

import jax
import optax
from optax import GradientTransformation, OptState
import functools
from flax.struct import PyTreeNode, field

from skrl.models.jax import Model


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@functools.partial(jax.jit, static_argnames=("transformation"))
def _step(transformation, grad, state, state_dict):
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


@functools.partial(jax.jit, static_argnames=("transformation"))
def _step_with_scale(transformation, grad, state, state_dict, scale):
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    # custom scale
    # https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale
    params = jax.tree_util.tree_map(lambda params: scale * params, params)
    # apply transformation
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


class AdamW:
    def __new__(
        cls,
        model: Model,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_norm_clip: float = 0,
        scale: bool = True,
    ) -> Optimizer:
        if scale:
            transformation = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        else:
            # Manual chain for AdamW without LR scaling (handled in step)
            transformation = optax.chain(
                optax.scale_by_adam(),
                optax.add_decayed_weights(weight_decay),
            )

        # clip updates using their global norm
        if grad_norm_clip > 0:
            transformation = optax.chain(
                optax.clip_by_global_norm(grad_norm_clip), transformation
            )

        return Optimizer._create(
            transformation=transformation,
            state=transformation.init(model.state_dict.params),
        )


class Optimizer(PyTreeNode):
    transformation: GradientTransformation = field(pytree_node=False)
    state: OptState = field(pytree_node=True)

    @classmethod
    def _create(cls, *, transformation, state, **kwargs):
        return cls(transformation=transformation, state=state, **kwargs)

    def step(
        self, grad: jax.Array, model: Model, lr: float | None = None
    ) -> "Optimizer":
        if lr is None:
            optimizer_state, model.state_dict = _step(
                self.transformation, grad, self.state, model.state_dict
            )
        else:
            optimizer_state, model.state_dict = _step_with_scale(
                self.transformation, grad, self.state, model.state_dict, -lr
            )
        return self.replace(state=optimizer_state)
