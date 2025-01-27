from collections import namedtuple

from jax.flatten_util import ravel_pytree
from jax import random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform

AMHState = namedtuple(
    "AMHState",
    [
        "i",  # Iteration
        "z",  # Current Point
        "potential_energy",  # Current potential energy
        "mean_accept_prob",  # Running mean of acceptance probabilities
        "adapt_state",  # Mean & Covariance matrix estimate + log of step size
        "rng_key",  # Random number generator state
    ],
)

AMHAdaptState = namedtuple("AMHAdaptState", ["mean", "covariance", "log_step_size"])

class AMH(numpyro.infer.mcmc.MCMCKernel):
    """
    AMH kernel for adaptive random walk-based Markov Chain Monte Carlo.

    Attributes
    ----------
    sample_field : str
        The field name in `AMHState` that contains the current sample.
    """

    sample_field = "z"

    def __init__(
        self, model=None, potential_fn=None, target_accept_prob=0.234, eps=1e-6, init_strategy=init_to_uniform
    ):
        """
        Initialize the AMH kernel.

        Parameters
        ----------
        model : callable, optional
            The model to initialize the kernel. Either `model` or `potential_fn` must be specified, but not both.
        potential_fn : callable, optional
            A potential function representing the negative log-posterior density.
        target_accept_prob : float, optional
            Target acceptance rate for adaptation, default is 0.3.
        eps : float, optional
            Regularization for proposal covariance, default is 1e-6.
        init_strategy : callable, optional
            A per-site initialization function, default is `init_to_uniform`.

        Raises
        ------
        ValueError
            If both `model` and `potential_fn` are specified or neither is specified.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._target_accept_prob = target_accept_prob
        self._eps = eps
        self._postprocess_fn = None
        self._init_strategy = init_strategy
        self._num_warmup = 0

    @property
    def model(self):
        return self._model

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the ARWMH kernel state.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key.
        num_warmup : int
            Number of warmup iterations.
        init_params : dict or None
            Initial parameters for the chain.
        model_args : tuple
            Arguments for the model.
        model_kwargs : dict, optional
            Keyword arguments for the model.

        Returns
        -------
        ARWMHState
            Initial state of the AMH algorithm.

        """
        self._num_warmup = num_warmup

        if self._model is not None:
            # Get callable potential function and initial values for parameters
            params_info, potential_fn_gen, self._postprocess_fn, _ = initialize_model(
                rng_key, self._model, init_strategy=self._init_strategy, dynamic_args=True, 
                model_args=model_args, model_kwargs=model_kwargs or {}
            )
            init_params = params_info[0]
            self._potential_fn = potential_fn_gen(*model_args, **(model_kwargs or {}))

        if self._potential_fn and init_params is None:
            raise ValueError("Valid value of `init_params` must be provided with `potential_fn`.")

        potential_energy = self._potential_fn(init_params)
        
        mu_hat = ravel_pytree(init_params)[0]
        sigma_hat = jnp.eye(len(mu_hat))
        log_lambda = 0.0
        adapt_state = AMHAdaptState(mu_hat, sigma_hat, log_lambda)

        init_state = AMHState(
            i=jnp.array(0),
            z=init_params,
            potential_energy=potential_energy,
            mean_accept_prob=jnp.array(0.0),
            adapt_state=adapt_state,
            rng_key=rng_key,
        )

        return init_state

    def sample(self, state, model_args, model_kwargs):
        """
        Generate the next sample using the adaptive random walk kernel.

        Parameters
        ----------
        state : ARWMHState
            Current state of the ARWMH algorithm.
        model_args : tuple
            Arguments for the model.
        model_kwargs : dict, optional
            Keyword arguments for the model.

        Returns
        -------
        AMHState
            Updated state after sampling.
        """
        i, z, potential_energy, mean_accept_prob, adapt_state, rng_key = state
        mu_hat, sigma_hat, log_lambda = adapt_state

        z_flat, unravel_fn = ravel_pytree(z)
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)

        proposal_cov = sigma_hat + jnp.eye(len(z_flat)) * self._eps
        z_step_flat = dist.MultivariateNormal(loc=0.0, covariance_matrix=proposal_cov).sample(key_proposal)
        z_proposal_flat = z_flat + z_step_flat * jnp.exp(log_lambda / 2)
        z_proposal = unravel_fn(z_proposal_flat)
        potential_energy_proposal = self._potential_fn(z_proposal)
        potential_energy_proposal = jnp.where(jnp.isnan(potential_energy_proposal), jnp.inf, potential_energy_proposal)

        accept_prob = jnp.clip(jnp.exp(potential_energy - potential_energy_proposal), min=0, max=1)
        is_accepted = dist.Uniform().sample(key_accept) < accept_prob

        z_new_flat = jnp.where(is_accepted, z_proposal_flat, z_flat)
        z_new = unravel_fn(z_new_flat)
        potential_energy_new = jnp.where(is_accepted, potential_energy_proposal, potential_energy)

        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        # learning rate
        gamma = 1/n

        mean_accept_prob_new = mean_accept_prob + gamma * (accept_prob - mean_accept_prob)

        # Adaptation
        delta = z_new_flat - mu_hat
        mu_new = mu_hat + gamma * delta
        sigma_new = sigma_hat + gamma * (jnp.outer(delta, delta) - sigma_hat)

        log_lambda_new = log_lambda + gamma * (accept_prob - self._target_accept_prob)

        adapt_state_new = AMHAdaptState(mu_new, sigma_new, log_lambda_new)

        return AMHState(
            i=itr,
            z=z_new,
            potential_energy=potential_energy_new,
            mean_accept_prob=mean_accept_prob_new,
            adapt_state=adapt_state_new,
            rng_key=rng_key,
        )

    def get_diagnostics_str(self, state):
        """
        Return diagnostics string for progress monitoring.

        Parameters
        ----------
        state : AMHState
            Current state of the AMH algorithm.

        Returns
        -------
        str
            A string containing diagnostics information.
        """
        return f"Acceptance rate: {state.mean_accept_prob:.2f}, Step size: {jnp.exp(state.adapt_state.log_step_size / 2):.3f}"


def sample_adapt_state(rng_key, dim=1, x_rad=5., log_step_rad=4.):
    """
    Generates an initial adaption state for Adaptive Metropolis-Hastings (AMH) algorithm by sampling random values for the mean, covariance, and log step size.

    Parameters:
    rng_key: A JAX random key seeded for reproducibility.
    dim: Integer specifying the dimensionality of the random variables (default is 1).
    x_rad: Float determining the scaling factor for mean and covariance matrix (default is 5.0).
    log_step_rad: Float scaling factor for the log step size (default is 4.0).

    Returns:
    An instance of AMHAdaptState containing:
    - mean: A sampled mean vector of shape (dim,).
    - cov: A positive semi-definite covariance matrix of shape (dim, dim).
    - lam: A sampled log step size (scalar).
    """
    k1, k2, k3 = random.split(rng_key, 3)
    
    mean = x_rad * dist.Uniform(-jnp.ones((dim,)), jnp.ones((dim,))).sample(k1)

    A = x_rad * dist.Uniform(low=jnp.zeros((dim, dim)), high=jnp.ones((dim, dim))).sample(k2) 
    cov = A.T @ A
    
    lam = log_step_rad * dist.Uniform(-1, 1).sample(k3)

    return AMHAdaptState(mean, cov, lam)
    

def sample_neigbour(rng_key, adapt_state, eps=1e-2):
    """
    Samples a neighboring adapted state for Adapting Metropolis-Hastings algorithm.

    This function perturbs the current adaptation state by adding small random variations
    to the mean, covariance, and lambda values. The variations are sampled from a uniform
    distribution scaled by a small epsilon value. The covariance matrix perturbation
    ensures positive semidefiniteness by applying perturbation to its Cholesky factor.

    Parameters:
    rng_key: PRNGKey
        Random number key used for sampling.
    adapt_state: tuple
        Contains the current adaptation state (mean, covariance matrix, lambda value).
    eps: float, optional
        Perturbation scale for the uniform distribution. Default is 1e-2.

    Returns:
    AMHAdaptState
        The perturbed adaptation state with updated mean, covariance, and lambda value.
    """
    mean1, cov1, lam1 = adapt_state
    k1, k2, k3 = random.split(rng_key, 3)

    mean = mean1 + eps * dist.Uniform(-jnp.ones_like(mean1), jnp.ones_like(mean1)).sample(k1)

    A1 = cholesky(cov1)
    A = A1 + eps * dist.Uniform(low=-jnp.ones_like(cov1), high=jnp.ones_like(cov1)).sample(k2) 
    cov = A.T @ A
    
    lam = lam1 + eps * dist.Uniform(-1, 1).sample(k3)

    return AMHAdaptState(mean, cov, lam)


def state_dist(s1, s2):
    """
    Computes the distance between two states based on their log step size and covariance.

    Args:
        s1: The first state object containing log_step_size and covariance attributes.
        s2: The second state object containing log_step_size and covariance attributes.

    Returns:
        The Frobenius norm of the difference between the two covariance matrices scaled by their respective log step sizes.
    """
    
    C1 = jnp.exp(s1.log_step_size)*s1.covariance
    C2 = jnp.exp(s2.log_step_size)*s2.covariance

    return jnp.linalg.norm(C1 - C2)


def sample_Px(rng_key, kernel, x, adapt_state, n_samples=1000, batch_size=100):
    """
    Samples from the posterior distribution using a stochastic MCMC sampling kernel.

    Parameters:
    rng_key: PRNGKey
        Random key for reproducibility of stochastic processes.
    kernel: MCMCKernel
        The kernel object that defines the MCMC sampling algorithm and its properties.
    x: array-like
        Initial state or starting point for the Markov Chain.
    adapt_state: PyTree or custom object
        State or configuration object for adaptive MCMC algorithms to maintain adaptation information during sampling.
    n_samples: int, optional (default=1000)
        Total number of posterior samples to draw.
    batch_size: int, optional (default=100)
        Number of samples to generate in each batch. The total number of samples (`n_samples`) should ideally be divisible by `batch_size`.

    Returns:
    samples: array-like
        Array containing `n_samples` samples drawn from the posterior distribution.
    """
    def single_Px(rkey):
        input_state = AMHState(
                i=0,
                z=x,
                potential_energy=kernel._potential_fn(x),
                mean_accept_prob=0,
                adapt_state=adapt_state,
                rng_key=rkey
        )
        next_state = kernel.sample(state=input_state, model_args=(), model_kwargs={})
        x_next = next_state.z
            
        return x_next

    n_batches = n_samples // batch_size
    batch_keys = random.split(rng_key, n_batches)

    sampled_batches = [
        vmap(single_Px)(random.split(batch_key, batch_size))
        for batch_key in batch_keys
    ]
    samples = jnp.concatenate(sampled_batches, axis=0)

    return samples
    