import jax.numpy as jnp
from jax import jit
import jax

EPS = 1e-8


def rbf_kernel(x, y, sigma):
    """
    Computes the RBF kernel matrix between x and y.
    References:

    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a/

    Parameters:
        sigma: variance parameter for the RBF kernel

    """
    # x: (..., d), y: (..., d)
    diff = (
        x[..., None, :] - y[..., None, :, :]
    )  # (batch?, m, m, d) depends on broadcasting
    sq = jnp.sum(diff**2, axis=-1)
    K = jnp.exp(-sq / (2.0 * sigma**2))
    return K, diff  # diff shaped so that grad wrt appropriate particles is easy


@jit
def rbf_pairwise(actions, sigma):
    """
    Compute pairwise RBF kernel K_{j,i} = k(a_j, a_i) and pairwise diffs (a_i - a_j)
    actions: (m, d)
    returns:
        K: (m, m)   where K[j, i] = exp(-||a_j - a_i||^2 / (2 sigma^2))
        diffs_i_minus_j: (m, m, d) where [i, j, :] = a_i - a_j
        sq_norms: (m, m) squared norms ||a_i - a_j||^2 with indexing [i,j]
    Note: indexing is chosen to make the inner sums in theorem natural (sum over j != i).
    """
    # actions[..., None, :] - actions[None, ..., :] yields shape (m, m, d) with [i, j, :] = a_i - a_j
    diffs = actions[:, None, :] - actions[None, :, :]  # (m, m, d) indexed [i, j, :]
    sq_norms = jnp.sum(diffs**2, axis=-1)  # (m, m) with element [i,j] = ||a_i - a_j||^2
    K = jnp.exp(-sq_norms / (2.0 * (sigma**2) + EPS))
    return K, diffs, sq_norms


@jit
def svgd_vector_field(actions, scores, sigma):
    """
    Clean SVGD vector field:
      actions: (m, d)
      scores:  (m, d)    where scores[j] = ∇_{a_j} log p(a_j)  (here ∇ Q / alpha)
      sigma:   kernel bandwidth
    returns:
      phi: (m, d)
    """
    m, d = actions.shape
    # pairwise diffs [i,j,:] = a_i - a_j
    diffs = actions[:, None, :] - actions[None, :, :]  # (m, m, d)
    sq = jnp.sum(diffs**2, axis=-1)  # (m, m)
    K = jnp.exp(-sq / (2.0 * (sigma**2) + EPS))  # (m, m) where K[i,j] = k(a_i, a_j)

    # term1: (1/m) sum_j k(x_j, x_i) * score_j  -> using K^T so K_T[i, j] = k(x_j, x_i)
    term1 = (K.T @ scores) / m  # (m, d)

    # term2: (1/m) sum_j ∇_{x_j} k(x_j, x_i)
    # For RBF: ∇_{x_j} k(x_j, x_i) = - (x_j - x_i) / sigma^2 * k(x_j, x_i)
    # Build (x_j - x_i) arranged as [j,i,:] from actions: actions[:, None, :] - actions[None, :, :] gives [i,j,:] = a_i - a_j
    a_j_minus_ai = (
        actions[:, None, :] - actions[None, :, :]
    ) * -1.0  # now [j, i, :] = a_j - a_i
    # K[j,i] is k(a_j, a_i) -> use K.T to index as [j,i]
    K_for_grad = K.T[..., None]  # (m, m, 1) with [j,i,0] = k(a_j,a_i)
    G = -a_j_minus_ai * K_for_grad / (sigma**2 + EPS)  # (m, m, d) with [j,i,:] shape
    # sum over j to get per i vector
    term2 = jnp.sum(G, axis=1) / m  # (m, d) sum over neighbors j for each target i

    phi = term1 + term2
    return phi


def action_score_from_Q(
    critic_apply_fn, critic_params, state, actions, alpha, critic_reduce=jnp.min
):
    """
    Compute action-space score (1/alpha * grad_a Q(s,a)) for each action in `actions`.

    Parameters
    ----------
    critic_apply_fn : callable
        Function (params, state, action) -> scalar-or-vector Q(s,a).
        Often a wrapper around model.apply or model.act. If it returns a vector
        (e.g. multiple critic heads), we reduce it to a scalar using `critic_reduce`.
    critic_params : pytree
        Parameters of the critic passed to critic_apply_fn.
    state : array (state_dim,) or (m, state_dim)
        Single environment state (will be tiled) or tiled states matching `actions`.
    actions : array (m, d)
        Particles / actions for which to compute the score.
    alpha : float
        Temperature used in S2AC; we return grad(Q)/alpha.
    critic_reduce : callable, optional
        Reducer applied to multi-headed critic outputs to produce a scalar.
        Default: jnp.min  (safest, like twin critics).

    Returns
    -------
    grad_q : array (m, d)
        The action gradients (1/alpha * grad_a Q(s, a)) for each particle/action.
    """
    m = actions.shape[0]

    # Ensure state has batch dimension matching actions.
    if state.ndim == 1:
        state_tiled = jnp.repeat(state[None, :], m, axis=0)
    else:
        state_tiled = state

    # Define scalar-valued Q for a single action and state
    def q_scalar(a, s):
        # critic_apply_fn is expected to accept (params, state, action) and return either scalar or vector.
        q_out = critic_apply_fn(critic_params, s, a)
        q_out = jnp.asarray(q_out)
        # If scalar already, return it; if vector, reduce (e.g., min across heads).
        # Note: using `critic_reduce` (e.g., jnp.min) yields a scalar
        if q_out.ndim == 0:
            return q_out
        else:
            return critic_reduce(q_out)

    # gradient w.r.t. first arg (the action). jax.grad default argnums=0 but explicit is clearer.
    grad_fn = jax.grad(q_scalar, argnums=0)

    # vmap across actions and states
    grad_q = jax.vmap(grad_fn)(actions, state_tiled)  # shape (m, d)

    # return grad divided by alpha (since objective uses Q/alpha)
    return grad_q / (alpha + 1e-12)


@jit
def logq0_isotropic_gaussian(a0, mu, logstd):
    """
    Compute log q0(a0) for isotropic (per-dim) Gaussian param:
      a0: (m, d)
      mu: (d,)  or (1, d)
      logstd: (d,) or (1, d)
    returns:
      logq0: (m,) per particle
    """
    d = a0.shape[-1]
    mu = jnp.reshape(mu, (1, d))
    logstd = jnp.reshape(logstd, (1, d))
    var = jnp.exp(2.0 * logstd)
    # log pdf per-dim: -0.5 * ( (a-mu)^2 / var ) - 0.5*log(2π) - logstd
    # sum over dims
    lp = jnp.sum(
        -0.5 * ((a0 - mu) ** 2) / (var + EPS) - 0.5 * jnp.log(2.0 * jnp.pi) - logstd,
        axis=-1,
    )
    return lp  # shape (m,)


@jit
def compute_logqL_closed_form(
    a0, all_a_list, all_gradQ_list, mu0, logstd0, eps, sigma_kernel, alpha
):
    """
    Compute approximation of log q_L for each final particle using Theorem 3.3 (Appendix H).

    Args:
      a0: (m, d) initial particles used to compute logq0
      all_a_list: list length L of arrays (m, d) corresponding to a_l for l=0..L-1 (particles at each step)
                  NOTE: paper's formula sums l=0..L-1 where a_l denotes particles before step l->l+1.
      all_gradQ_list: list length L of arrays (m, d) where all_gradQ_list[l][j] = ∇_{a_{l,j}} Q(s, a_{l,j})
      mu0, logstd0: initial Gaussian parameters for q0 (mu shape (d,), logstd shape (d,))
      eps: scalar step size ϵ
      sigma_kernel: kernel bandwidth σ
      alpha: temperature α

    Returns:
      logqL_particles: (m,) vector of approximated log q_L evaluated at each final particle a_L[i]
                       Note: the approximation shares the same size m for outputs; it corresponds to each final particle's log-likelihood.
    """
    m, d = a0.shape
    logq0 = logq0_isotropic_gaussian(a0, mu0, logstd0)  # (m,)

    # accumulate the inner sums over steps and neighbors
    accum = jnp.zeros((m,))  # will accumulate per-particle scalar sum over l and j
    coeff = eps / (m * (sigma_kernel**2) + EPS)

    # loop over SVGD steps (L small in practice, so Python loop is fine; we jit the whole function)
    for a_l, gradQ_l in zip(all_a_list, all_gradQ_list):
        # a_l: (m, d), gradQ_l: (m, d) with index j corresponding to particle j
        K, diffs, sq_norms = rbf_pairwise(
            a_l, sigma_kernel
        )  # K shape (m,m) with [i,j] = k(a_i, a_j)
        # NOTE indexing: our K[i,j] = k(a_i, a_j) (i is "target" particle index, j is neighbor index)
        # The formula in the paper uses k(a_{l,j}, a_{l,i}) and sum over j != i. That is K_T in prior code.
        # We can align to paper by transposing: K_paper = K.T so that K_paper[j,i] = K[i,j].
        K_paper = K.T  # now K_paper[j, i] = k(a_{l,j}, a_{l,i})

        # diffs has diffs[i,j,:] = a_i - a_j, but the paper uses (a_i - a_j)·grad_{a_j}Q(s,a_j), sum over j != i
        # For each pair (i, j) we need:
        #   term_dot = (a_i - a_j)^T gradQ_l[j]
        # We can compute matrix of shape (m, m) where E[i,j] = (a_i - a_j)·gradQ_l[j]
        # Build gradQ expanded: shape (m, m, d) where [i,j,:] use gradQ_l[j]
        gradQ_expanded = jnp.broadcast_to(
            gradQ_l[None, :, :], (m, m, d)
        )  # [i,j,:] copies gradQ_l[j]
        # diffs is [i, j, :] = a_i - a_j  (already what we need)
        dot_terms = jnp.sum(
            diffs * gradQ_expanded, axis=-1
        )  # (m, m) where [i,j] = (a_i - a_j)^T gradQ_l[j]

        # norm squared term ||a_i - a_j||^2 is sq_norms [i, j]
        # Note we must exclude diagonal j == i; in formula sum_{j != i}
        mask = 1.0 - jnp.eye(m)  # (m,m) zeros on diag, ones elsewhere

        # Compute per-pair contribution matrix M[i,j] = K_paper[j, i] * ( dot_terms[i,j] + (alpha/sigma^2) * sq_norms[i,j] - d*alpha )
        # Careful with indexing: K_paper[j,i] = K.T[j,i] = K[i,j] in our original K; equivalently K_paper = K.T
        M = K_paper * (
            dot_terms + (alpha / (sigma_kernel**2 + EPS)) * sq_norms - (d * alpha)
        )

        # Zero out diagonal contributions
        M = M * mask

        # Sum over j for each i: per_i_sum = sum_j M[i, j]
        per_i_sum = jnp.sum(M, axis=1)  # (m,) sum over j

        accum = accum + per_i_sum

    logqL = logq0 + coeff * accum
    return logqL  # shape (m,)
