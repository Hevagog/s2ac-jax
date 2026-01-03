import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jacfwd, jacrev, vmap
import pytest

# Import functions & classes from your repo (adjust import paths as needed)
from agent.s2ac.utils import (
    svgd_vector_field,
    compute_logqL_closed_form,
    action_score_from_Q,
)
from agent.s2ac import S2AC  # adjust if your agent module path differs
from skrl.memories.jax import RandomMemory  # or your memory class
from skrl.models.jax import Model  # used to create mock models
from functools import partial

EPS = 1e-8


# ---------------------------
# Helper: numerical logdet for change-of-variable for small maps
# ---------------------------
def numerical_logdet_transform(a0, phi_fn, eps):
    """
    Given a0 (m,d) and phi_fn mapping (m,d)->(m,d) (vector field acting on all particles),
    compute log|det dT/dx| where T(x)=x + eps * phi(x) viewed as a flattened transform
    acting on the concatenated vector (m*d,).
    Returns scalar logdet.
    """
    flat = a0.reshape(-1)

    def T_flat(x_flat):
        x = x_flat.reshape(a0.shape)
        phi = phi_fn(x)
        return (x + eps * phi).reshape(-1)

    J = jacfwd(T_flat)(flat)  # (m*d, m*d)
    # compute logabsdet; jacfwd returns an array; convert to numpy for stable logdet
    Jnp = jnp.asarray(J)
    sign, logabsdet = jnp.linalg.slogdet(Jnp)
    return float(logabsdet)


# ---------------------------
# Test A — Closed-form log q_L vs explicit jacobian logdet (L=1)
# ---------------------------
def test_closed_form_matches_change_of_variable_first_order():
    """
    For one SVGD step (L=1) with small eps, compute exact log q_L via:
      log q_L = log q0 - log|det(I + eps * dPhi/dx)|
    Compare:
      - closed-form approximation from compute_logqL_closed_form (the paper's algebra)
      - numerical jacobian logdet via jacfwd on flattened map, but take first-order approx:
        use expansion logdet(I + eps A) ≈ eps * tr(A)  (for small eps)
    This test checks that closed-form equals first-order logdet for a small eps
    (i.e., they agree on linear term).
    """
    key = random.PRNGKey(123)
    m = 6
    d = 3
    # simple initial gaussian params
    mu0 = jnp.zeros((d,))
    logstd0 = jnp.zeros((d,))
    key, sub = random.split(key)
    a0 = random.normal(sub, (m, d)) * 0.5

    # define a simple Q(s,a) that yields gradQ easily (quadratic)
    # Q(a) = -0.5 * (a - b)^T diag(c) (a - b)  -> grad = -diag(c) (a - b)
    b = jnp.linspace(-0.5, 0.5, d)
    c = jnp.ones((d,)) * 0.75

    def gradQ_fn(a):
        return -(a - b) * c[None, :]  # shape (m,d)

    # build phi via SVGD formula using gradQ / alpha (alpha=1)
    sigma = 0.8
    gradQ = gradQ_fn(a0)
    phi = svgd_vector_field(a0, gradQ, sigma)

    # compute closed-form from our implementation for L=1
    all_a_list = (a0,)
    all_gradQ_list = (gradQ,)
    eps = 1e-3  # small eps for first-order validity
    alpha = 1.0
    logqL_closed = compute_logqL_closed_form(
        a0, all_a_list, all_gradQ_list, mu0, logstd0, eps, sigma, alpha
    )

    # Using Jacobian: compute trace(A) where A = dPhi_flat/dx_flat (i.e., derivative of flattened phi wrt flattened x)
    # We'll compute per-particle trace via constructing phi_flat and jacfwd-jvp trick:
    flat = a0.reshape(-1)

    def phi_flat(x_flat):
        x = x_flat.reshape(a0.shape)
        # inside phi computation we need gradQ for x; but gradQ defined for our simple quadratic:
        g = -(x - b[None, :]) * c[None, :]
        return svgd_vector_field(x, g, sigma).reshape(-1)

    # compute diagonal trace quickly via jvp with standard basis vectors but that's expensive;
    # Instead compute full jacobian and its trace for this small test
    J = jacfwd(phi_flat)(flat)
    tr = float(jnp.trace(J))

    # First-order predicted change in logdet: -eps * trace(dPhi/dx)
    # compute log q0 per particle
    var = jnp.exp(2.0 * logstd0)
    logq0 = jnp.sum(
        -0.5 * ((a0 - mu0) ** 2) / var - 0.5 * jnp.log(2.0 * jnp.pi) - logstd0, axis=-1
    )

    logqL_first_order = logq0 - eps * tr / 1.0  # alpha factor included in gradQ already

    # Compare closed-form mean to first-order mean
    mean_closed = float(jnp.mean(logqL_closed))
    mean_first = float(jnp.mean(logqL_first_order))
    diff = abs(mean_closed - mean_first)

    print("mean_closed", mean_closed, "mean_first", mean_first, "diff", diff)
    assert diff < 1e-2


# ---------------------------
# Test B — SVGD reduces Kernelized Stein Discrepancy (KSD) to target Gaussian
# ---------------------------


# ---------------------------
# Test B — SVGD reduces Kernelized Stein Discrepancy (KSD) to target Gaussian
# ---------------------------
def compute_empirical_ksd(particles, score_fn, kernel_sigma):
    """
    Empirical KSD estimator for target with score function score_fn(x).
    KSD^2 = 1/m^2 sum_{i,j} u_p(x_i, x_j) where
    u_p(x, y) = s_p(x)^T k(x,y) s_p(y) + s_p(x)^T grad_y k(x,y)
                 + grad_x k(x,y)^T s_p(y) + trace(grad_x grad_y k(x,y))
    For RBF kernel with bandwidth sigma we implement the expression directly.
    """
    m = particles.shape[0]
    diffs = particles[:, None, :] - particles[None, :, :]
    sq = jnp.sum(diffs**2, axis=-1)
    K = jnp.exp(-sq / (2.0 * (kernel_sigma**2) + EPS))
    s = score_fn(particles)  # (m,d)

    # term1: s(x)^T k(x,y) s(y)  -> matrix M1[i,j]
    M1 = jnp.einsum("id,jd->ij", s, s) * K  # (m,m)

    # term2: s(x)^T grad_y k(x,y)
    # grad_y k(x,y) = - (y-x)/sigma^2 * k
    grad_y = (
        -diffs / (kernel_sigma**2 + EPS) * K[..., None]
    )  # (m,m,d) where [i,j,:] = grad_y k(x_i, x_j)
    M2 = jnp.einsum("id,ijd->ij", s, grad_y)

    # term3: grad_x k(x,y)^T s(y) = -grad_y^T s(y) but we implement explicitly
    grad_x = -grad_y  # because grad_x k(x,y) = -grad_y k(x,y)
    M3 = jnp.einsum("ijd,jd->ij", grad_x, s)

    # term4: trace grad_x grad_y k(x,y) for RBF we can compute: trace = (d / sigma^2 - ||x-y||^2 / sigma^4) * k(x,y)
    d = particles.shape[1]
    M4 = ((d / (kernel_sigma**2 + EPS)) - (sq / ((kernel_sigma**4) + EPS))) * K

    u = M1 + M2 + M3 + M4
    ksd2 = jnp.sum(u) / (m * m)
    return float(ksd2)


def test_svgd_reduces_ksd_to_gaussian():
    key = random.PRNGKey(1)
    m = 128
    d = 2
    mu = jnp.array([0.2, -0.3])
    cov = jnp.array([[0.5, 0.02], [0.02, 0.4]])
    cov_inv = jnp.linalg.inv(cov)

    def score_fn(x):
        # x shape (m,d)
        return -((x - mu) @ cov_inv.T)

    key, sub = random.split(key)
    particles = random.normal(sub, (m, d)) * 1.5 + jnp.array([2.0, 1.5])

    sigma = (
        float(
            jnp.median(
                jnp.sqrt(
                    jnp.sum(
                        (particles[:, None, :] - particles[None, :, :]) ** 2, axis=-1
                    )
                )[jnp.triu_indices(m, k=1)]
            )
        )
        + 1e-3
    )
    ksd_before = compute_empirical_ksd(particles, score_fn, sigma)

    eps = 0.03
    n_steps = 400
    for _ in range(n_steps):
        sc = score_fn(particles)
        phi = svgd_vector_field(particles, sc, sigma)
        particles = particles + eps * phi
        # optionally update sigma occasionally: skip to keep deterministic here

    ksd_after = compute_empirical_ksd(particles, score_fn, sigma)
    print("ksd_before", ksd_before, "ksd_after", ksd_after)
    assert ksd_after < ksd_before


# ---------------------------
# Test C — S2AC agent smoke + numeric checks (fast)
# ---------------------------
class DummyState:
    def __init__(self, params):
        self.params = params

    def __repr__(self):
        return f"DummyState(params={self.params})"

    def replace(self, params):
        return DummyState(params)


jax.tree_util.register_pytree_node(
    DummyState,
    lambda s: ((s.params,), None),
    lambda _, children: DummyState(children[0]),
)


def make_dummy_policy_and_critic(action_dim, state_dim):
    """
    Build extremely small deterministic models for policy and critic using simple param structures.
    We will pass them into a lightweight S2AC agent instance for a short training run.
    These are *not* full nn modules — we create Models with callable apply wrappers expected by the agent.
    """

    class DummyModel:
        def __init__(self):
            # emulate SKRL .state_dict.params container
            self.state_dict = DummyState({})

        def apply(self, params, inputs, role=None):
            # policy: if role == 'policy' expect inputs{'states': (1, state_dim)}
            if role == "policy":
                s = inputs["states"].reshape(-1)
                # mean = linear in states, logstd = small constant
                mean = jnp.zeros((1, action_dim)) + 0.0
                logstd = jnp.full((1, action_dim), -1.0)  # std ~ 0.367
                return mean, logstd, {}
            else:
                # critic: expect inputs {"states": (N, state_dim), "taken_actions": (N, action_dim)}
                st = inputs["states"]
                ac = inputs["taken_actions"]
                # return a scalar Q = negative squared norm (to encourage staying near zero)
                vals = -jnp.sum((ac - 0.0) ** 2, axis=-1, keepdims=True)
                return vals, {}, {}

        def act(self, inputs, role=None, params=None):
            return self.apply(params, inputs, role=role)

        def freeze_parameters(self, freeze: bool) -> None:
            pass

        def update_parameters(self, critic_model: Model, polyak: float) -> None:
            pass

        def set_mode(self, mode: str) -> None:
            pass

    policy = DummyModel()
    critic = DummyModel()
    target_critic = DummyModel()
    return policy, critic, target_critic


def test_s2ac_agent_smoke_run(tmp_path):
    # small env-like shapes
    state_dim = 3
    action_dim = 2

    policy, critic, target = make_dummy_policy_and_critic(action_dim, state_dim)
    models = {"policy": policy, "critic": critic, "target_critic": target}

    # Minimal memory using SKRL RandomMemory
    memory = RandomMemory(memory_size=256, device="cpu")

    agent = S2AC(
        models=models,
        memory=memory,
        observation_space=(state_dim,),
        action_space=(action_dim,),
        cfg={
            "particles": 8,
            "svgd_steps": 2,
            "svgd_step_size": 0.05,
            "kernel_sigma": 0.5,
            "alpha": 0.1,
            "batch_size": 16,
        },
    )

    agent.init()
    # fill memory with random transitions
    rng = np.random.RandomState(0)
    for _ in range(64):
        s = rng.randn(state_dim).astype(np.float32)
        a = np.tanh(rng.randn(action_dim).astype(np.float32))
        r = float(rng.randn(1))
        ns = rng.randn(state_dim).astype(np.float32)
        term = 0
        trunc = 0
        agent.record_transition(s, a, r, ns, term, trunc, {}, 0, 0)

    # run a few update steps; ensure no exceptions and losses finite
    agent.set_mode("train")
    for i in range(4):
        agent._update(i, 100)
    # If we reach here assert no NaNs in models' params or last losses tracked
    # Check last tracked metrics (if track_data stores them)
    # At minimum make sure critic.apply still returns finite values
    test_state = jnp.zeros((state_dim,))
    policy_params = policy.state_dict.params
    critic_params = critic.state_dict.params
    # run a single SVGD rollout to verify shapes & finite outputs
    key = random.PRNGKey(0)
    particles, raw_actions, log_probs = agent._svgd_rollout_single(
        policy_params, critic, critic_params, test_state, key
    )
    assert particles.shape == (agent._particles, action_dim)
    assert raw_actions.shape == (agent._particles, action_dim)
    assert log_probs.shape == (agent._particles,)
    assert jnp.all(jnp.isfinite(particles))
    assert jnp.all(jnp.isfinite(log_probs))
