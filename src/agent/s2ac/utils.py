import jax
import jax.numpy as jnp
from jax import jit, vmap

EPS = 1e-9


def rbf_kernel(x, y, sigma):
    """
    RBF kernel matrix K_ij = exp(-||x_i - y_j||^2 / (2 * sigma^2)).
    Returns (K, diffs) where diffs is x[:,None,:] - y[None,:,:] (m,n,d).
    """
    diffs = x[:, None, :] - y[None, :, :]  # (m, n, d)
    sq = jnp.sum(diffs**2, axis=-1)  # (m, n)
    inv_denom = 1.0 / (2.0 * (sigma**2) + EPS)
    K = jnp.exp(-sq * inv_denom)
    return K, diffs


def rbf_pairwise(actions, sigma):
    """
    Pairwise kernel among actions (square matrix).
    Returns (K, diffs, sq_norms).

    """
    diffs = actions[:, None, :] - actions[None, :, :]  # (m, m, d)
    sq_norms = jnp.sum(diffs**2, axis=-1)  # (m, m)
    inv_denom = 1.0 / (2.0 * (sigma**2) + EPS)
    K = jnp.exp(-sq_norms * inv_denom)
    return K, diffs, sq_norms


def _get_sqdist_matrix_optimized(x):
    """
    Efficient squared-distance matrix: (m,m).
    Uses a single matmul and broadcasting.
    """
    # x: (m, d)
    # Compute ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
    x_sq = jnp.sum(x * x, axis=1)  # (m,)
    # Use broadcasting: x_sq[:, None] + x_sq[None, :] is (m, m)
    sq = x_sq[:, None] + x_sq[None, :] - 2.0 * (x @ x.T)
    return jnp.maximum(sq, 0.0)


@jit
def median_heuristic_sigma(actions, h_min=1e-3, h_max=10.0):
    """
    Median heuristic for RBF bandwidth (returns bandwidth sigma).
    Includes upper bound to prevent explosion when particles diverge.
    """
    m = actions.shape[0]
    dist_sq = _get_sqdist_matrix_optimized(actions)
    # mask out diagonal by adding a large number there
    mask = 1.0 - jnp.eye(m)
    maxval = jnp.max(dist_sq)

    maxval = jnp.where(jnp.isfinite(maxval), maxval, 1e6)
    median_sq = jnp.median(dist_sq + (1.0 - mask) * maxval)

    median_sq = jnp.where(jnp.isfinite(median_sq), median_sq, 1.0)
    # paper uses something like h = median_sq / (2 log m) ; we return sigma = sqrt(h)
    h = jnp.maximum(median_sq / (2.0 * jnp.log(m + 1.0) + EPS), h_min * h_min)
    sigma = jnp.sqrt(h)

    return jnp.clip(sigma, h_min, h_max)


@jit
def svgd_vector_field(actions, scores, sigma):
    """
    SVGD vector field phi evaluated at each particle actions[i].
    - actions: (m, d)
    - scores:  (m, d)  (score = grad log target)
    Returns phi (m,d).

    Formula:
      phi_i = (1/m) * sum_j [ K_ij * scores_j + grad_{x_j} K_ij ]
    where grad_{x_j} K_ij = (x_i * sum_j K_ij - (K @ x)_i) / sigma^2
    """
    m = actions.shape[0]
    inv_m = 1.0 / (m + EPS)
    sigma_sq = sigma * sigma
    inv_sigma_sq = 1.0 / (sigma_sq + EPS)
    inv_2sigma_sq = 0.5 * inv_sigma_sq

    # Compute squared distances efficiently
    actions_sq = jnp.sum(actions * actions, axis=1)  # (m,)
    sq_dists = actions_sq[:, None] + actions_sq[None, :] - 2.0 * (actions @ actions.T)
    sq_dists = jnp.maximum(sq_dists, 0.0)

    # Kernel matrix
    K = jnp.exp(-sq_dists * inv_2sigma_sq)  # (m, m)

    # term1 = (1/m) * K @ scores
    term1 = (K @ scores) * inv_m

    # repulsive term: (1/m) * sum_j grad_{x_i} K_ij
    # grad_{x_i} K_ij = (x_i - x_j) * K_ij / sigma^2
    # sum_j = (x_i * sum_j K_ij - (K @ x)_i) / sigma^2
    K_sum = jnp.sum(K, axis=1, keepdims=True)  # (m, 1)
    K_actions = K @ actions  # (m, d)
    grad_k_sum = actions * K_sum - K_actions  # (m, d)
    term2 = grad_k_sum * (inv_sigma_sq * inv_m)

    return term1 + term2


def _q_scalar_closure(critic_apply_fn, critic_params, critic_reduce):
    """
    Returns a scalar-q callable q_scalar(a, s) -> scalar, used by jax.grad.
    Assumes critic_apply_fn(params, s, a) -> scalar-or-vector.

    """

    def q_scalar(a, s):
        q_out = critic_apply_fn(critic_params, s, a)
        q_out = jnp.asarray(q_out)
        if q_out.ndim == 0:
            return q_out
        return critic_reduce(q_out)

    return q_scalar


def action_grad_from_Q(
    critic_apply_fn, critic_params, state, actions, critic_reduce=jnp.min
):
    """
    Compute ∇_a Q(s,a). Supports:
      - `state` 1D (single state): will map across actions -> returns (m,d)
      - `state` 2D (batch, s_dim): must be same batch length as actions (m,...).

    """
    q_scalar = _q_scalar_closure(critic_apply_fn, critic_params, critic_reduce)
    grad_fn = jax.grad(q_scalar, argnums=0)

    state_axis = 0 if state.ndim > 1 else None
    return jax.vmap(grad_fn, in_axes=(0, state_axis))(actions, state)


def action_score_from_Q(
    critic_apply_fn, critic_params, state, actions, alpha, critic_reduce=jnp.min
):
    """
    Score used by S2AC: score = (1/alpha) * ∇_a Q(s,a)

    """
    grad_q = action_grad_from_Q(
        critic_apply_fn, critic_params, state, actions, critic_reduce=critic_reduce
    )
    return grad_q / (alpha + EPS)


def _logqL_step(a_curr, gradQ_curr, sigma_kernel, eps, alpha_internal=1.0):
    """
    Compute the incremental log q change for one SVGD step.
    Based on Theorem 3.3 from the S2AC paper.

    Formula: Δlog q_i = -(ε/mσ²) * Σ_{j≠i} K(a_j,a_i) * [(a_i-a_j)^T ∇Q_j + (α/σ²)||a_i-a_j||² - dα]

    Note: The SVGD updates DECREASE entropy (increase log_prob), so this term is SUBTRACTED
    from log q_0 in the accumulation.
    """
    m, d = a_curr.shape
    sigma_sq = sigma_kernel * sigma_kernel
    inv_sigma_sq = 1.0 / (sigma_sq + 1e-6)

    # Pairwise squared dists: ||a_i - a_j||^2
    a_sq = jnp.sum(a_curr * a_curr, axis=1)
    dist_sq = jnp.maximum(
        a_sq[:, None] + a_sq[None, :] - 2.0 * (a_curr @ a_curr.T), 0.0
    )

    K = jnp.exp(-dist_sq * (0.5 * inv_sigma_sq))

    # dot_terms_ij = (a_i - a_j)^T * gradQ_j
    # = a_i^T gradQ_j - a_j^T gradQ_j
    term_a = a_curr @ gradQ_curr.T  # (m, m): term_a[i,j] = a_i^T gradQ_j
    term_b_vec = jnp.sum(a_curr * gradQ_curr, axis=1)  # (m,): term_b[j] = a_j^T gradQ_j
    dot_terms = (
        term_a - term_b_vec[None, :]
    )  # (m, m): dot_terms[i,j] = (a_i - a_j)^T gradQ_j

    mask = 1.0 - jnp.eye(m)

    # M[i,j] = K(a_j, a_i) * [(a_i-a_j)^T ∇Q_j + (α/σ²)||a_i-a_j||² - dα]
    # Note: K is symmetric, so K[i,j] = K[j,i]
    M = K * (
        dot_terms + (alpha_internal * inv_sigma_sq) * dist_sq - (d * alpha_internal)
    )
    M = M * mask

    # Sum over j for each i: this gives the change in log q_i
    per_i_sum = jnp.sum(M, axis=1)  # Sum over j (columns)
    coeff = -eps / ((m + 1e-6) * (sigma_sq + 1e-6))
    return coeff * per_i_sum


@jit
def compute_logqL_closed_form(
    a0, all_a_list, all_gradQ_list, mu0, logstd0, eps, sigma_list, alpha
):
    """
    Compute approximated log q_L for final particles.

    - a0: (m,d) initial u particles (u0)
    - all_a_list: list or stack of length T, each (m,d) (u at each step)
    - all_gradQ_list: list or stack (T, m, d)
    - mu0, logstd0: (d,) or (1,d) initial gaussian params (for u0)
    - eps: scalar step size
    - sigma_list: (T,) array of kernel sigmas used at each step, or scalar
    - alpha: scalar temperature
    Returns logqL: (m,) per-particle log-likelihood estimate.

    """
    # log q0 (initial isotropic gaussian in u-space)
    d = a0.shape[-1]
    mu0 = jnp.reshape(mu0, (1, d))
    logstd0 = jnp.reshape(logstd0, (1, d))
    var = jnp.exp(2.0 * logstd0)
    inv_var = 1.0 / (var + EPS)

    # Precompute constants
    log_2pi = jnp.log(2.0 * jnp.pi)

    logq0 = jnp.sum(
        -0.5 * ((a0 - mu0) ** 2) * inv_var - 0.5 * log_2pi - logstd0,
        axis=-1,
    )

    # stack lists -> (T, m, d)
    traj_a = jnp.stack(all_a_list)  # (T, m, d)
    traj_gradQ = jnp.stack(all_gradQ_list)  # (T, m, d)
    T = traj_a.shape[0]

    # ensure sigma_list shape matches T - handle scalar vs array case
    sigma_arr = jnp.atleast_1d(sigma_list)
    # If scalar was passed (now shape (1,)), broadcast to (T,)
    sigma_arr = jnp.where(
        sigma_arr.shape[0] == 1, jnp.broadcast_to(sigma_arr, (T,)), sigma_arr
    )

    # per-step logqL increments using vmap
    def step_fn(a_step, gq_step, sigma_step):
        return _logqL_step(a_step, gq_step, sigma_step, eps, alpha_internal=alpha)

    per_step = vmap(step_fn, in_axes=(0, 0, 0))(traj_a, traj_gradQ, sigma_arr)  # (T, m)

    per_step = jnp.clip(per_step, -10.0, 10.0)

    accum = jnp.sum(per_step, axis=0)  # (m,)

    logqL = logq0 + accum
    # Upper bound of 0.0 ensures valid probability; -20.0 lower bound prevents -inf
    logqL = jnp.clip(logqL, -20.0, 0.0)

    return logqL


@jit
def compute_tanh_jacobian_correction(u: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log determinant of the Jacobian for tanh squashing.

    When transforming from u-space to a-space via a = tanh(u),
    we need to correct the log probability:

    log p(a) = log p(u) - Σ_d log |da_d/du_d|
             = log p(u) - Σ_d log(1 - tanh²(u_d))

    Using the numerically stable form:
    log(1 - tanh²(u)) = log(sech²(u)) = 2 * log(sech(u))
                      = 2 * (log(2) - u - softplus(-2u))

    Args:
        u: (m, d) or (d,) array of unbounded actions

    Returns:
        correction: (m,) or () array of log-det-jacobian corrections (to subtract)
    """
    log_1_minus_tanh_sq = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    # Sum over action dimensions
    return jnp.sum(log_1_minus_tanh_sq, axis=-1)


@jit
def compute_logqL_closed_form_with_tanh(
    u0, u_final, all_u_list, all_gradQ_list, mu0, logstd0, eps, sigma_list, alpha
):
    """
    Compute approximated log q_L for final particles WITH tanh Jacobian correction.

    This is the correct log probability in action space (a = tanh(u)).

    - u0: (m,d) initial u particles
    - u_final: (m,d) final u particles (before tanh)
    - all_u_list: list or stack of length T, each (m,d) (u at each SVGD step)
    - all_gradQ_list: list or stack (T, m, d)
    - mu0, logstd0: (d,) or (1,d) initial gaussian params
    - eps: scalar step size
    - sigma_list: (T,) array of kernel sigmas used at each step, or scalar
    - alpha: scalar temperature

    Returns logqL: (m,) per-particle log-likelihood estimate in ACTION SPACE.
    """
    # Get log q in u-space (without tanh correction)
    logq_u = compute_logqL_closed_form(
        u0, all_u_list, all_gradQ_list, mu0, logstd0, eps, sigma_list, alpha
    )

    # Apply tanh Jacobian correction for final particles
    # log p(a) = log p(u) - log |det(da/du)| = log p(u) - Σ log(1 - tanh²(u))
    tanh_correction = compute_tanh_jacobian_correction(u_final)

    logq_a = logq_u - tanh_correction
    return logq_a


@jit
def svgd_vector_field_s2ac(actions, grad_q, sigma, alpha, max_phi_norm=10.0):
    """
    SVGD vector field for S2AC.
    Wrapper around the optimized svgd_vector_field.
    Includes clipping to prevent exploding updates.

    """
    # The score function is grad_a Q(a) / alpha
    # IMPORTANT: grad_q should be detached before calling this if you don't want
    # gradients to flow through the score function (standard SVGD).
    grad_q = jnp.where(jnp.isfinite(grad_q), grad_q, 0.0)
    inv_alpha = 1.0 / (alpha + EPS)
    scores = grad_q * inv_alpha

    scores = jnp.clip(scores, -100.0, 100.0)
    # Reuse the optimized calculation
    phi = svgd_vector_field(actions, scores, sigma)

    phi_norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
    phi = jnp.where(
        phi_norm > max_phi_norm, phi * (max_phi_norm / (phi_norm + EPS)), phi
    )
    return phi
