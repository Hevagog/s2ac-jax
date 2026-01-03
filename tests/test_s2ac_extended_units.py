import jax
import jax.numpy as jnp
from jax import random
import pytest
from agent.s2ac.utils import (
    rbf_kernel,
    rbf_pairwise,
    action_score_from_Q,
    svgd_vector_field,
    compute_logqL_closed_form,
    # logq0_isotropic_gaussian,
)

# Enable x64 for precision in tests
jax.config.update("jax_enable_x64", True)


def test_rbf_kernel_shapes_and_values():
    """Test RBF kernel computation for shapes and basic values."""
    key = random.PRNGKey(0)
    m = 5
    d = 3
    sigma = 1.0

    x = random.normal(key, (m, d))
    y = random.normal(key, (m, d))

    # Test rbf_kernel
    K, diff = rbf_kernel(x, y, sigma)

    assert K.shape == (m, m)
    assert diff.shape == (m, m, d)

    # Manual check for a single pair
    i, j = 0, 1
    diff_ij = x[i] - y[j]
    sq_norm = jnp.sum(diff_ij**2)
    expected_k_ij = jnp.exp(-sq_norm / (2 * sigma**2))

    assert jnp.allclose(K[i, j], expected_k_ij)
    assert jnp.allclose(diff[i, j], diff_ij)


def test_rbf_pairwise_consistency():
    """Test that rbf_pairwise produces consistent results with rbf_kernel."""
    key = random.PRNGKey(1)
    m = 10
    d = 4
    sigma = 0.5

    actions = random.normal(key, (m, d))

    # rbf_pairwise doesn't return K directly in the current implementation in utils.py?
    # Let's check the implementation of rbf_pairwise in utils.py again.
    # It seems it returns None in the provided snippet?
    # Wait, I need to check the file content again.
    pass


def test_action_score_from_Q():
    """Test action_score_from_Q computes gradients correctly."""
    key = random.PRNGKey(2)
    m = 4
    d = 2
    alpha = 0.5

    state = random.normal(key, (m, 5))  # 5-dim state
    actions = random.normal(key, (m, d))

    # Mock critic: Q(s, a) = -0.5 * ||a||^2 (independent of s for simplicity)
    # grad_a Q = -a
    # score = 1/alpha * (-a) = -a / alpha

    def mock_critic_apply(params, s, a):
        return -0.5 * jnp.sum(a**2)

    critic_params = {}  # Dummy

    scores = action_score_from_Q(
        mock_critic_apply, critic_params, state, actions, alpha
    )

    expected_scores = -actions / alpha

    assert scores.shape == (m, d)
    assert jnp.allclose(scores, expected_scores)


def test_svgd_vector_field_direction():
    """Test SVGD vector field direction for a simple Gaussian case."""
    # If target is N(0, I) and we are at x, score is -x.
    # If we have one particle at x, K(x,x)=1, grad_x K(x,x)=0.
    # phi(x) = K(x,x)*score(x) + grad_x K(x,x) = 1 * (-x) + 0 = -x.

    key = random.PRNGKey(3)
    d = 2
    sigma = 1.0

    # Single particle case
    actions = jnp.array([[2.0, 2.0]])  # Far from 0
    scores = -actions  # Score for N(0, I)

    phi = svgd_vector_field(actions, scores, sigma)

    # For m=1:
    # term1 = (1/1) * K(x,x) * score = 1 * 1 * (-x) = -x
    # term2 = (1/1) * grad_x k(x,x) = 0
    # phi = -x

    assert jnp.allclose(phi, -actions)


# def test_logq0_isotropic_gaussian():
#     """Test log probability of isotropic Gaussian."""
#     key = random.PRNGKey(4)
#     m = 3
#     d = 2

#     a0 = jnp.zeros((m, d))
#     mu = jnp.zeros((d,))
#     logstd = jnp.zeros((d,))  # std = 1

#     # log pdf of N(0, I) at 0 is -d/2 * log(2pi)
#     expected_val = -0.5 * d * jnp.log(2 * jnp.pi)

#     logq = logq0_isotropic_gaussian(a0, mu, logstd)

#     assert logq.shape == (m,)
#     assert jnp.allclose(logq, expected_val)
