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
    Adaptive Random Walk Metropolis-Hastings (ARWMH) kernel for MCMC sampling.
	
    Attributes
    ----------
    sample_field : str
        The field in the state that represents the MCMC sample ('z').
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
    Samples an initial adaptive state for AMH kernel.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        A random key for reproducible random number generation.
    dim : int, optional
        Dimensionality of the target distribution (default is 1).
    x_rad : float, optional
        Scaling factor for the mean and covariance components (default is 5).
    log_step_rad : float, optional
        Scaling factor for the log step size (default is 4).

    Returns
    -------
    AMHAdaptState
        A tuple representing the adaptive state, which includes:
        - mean: array of shape (dim,)
        - cov: array of shape (dim, dim)
        - lam: float, log step size parameter
    """
    k1, k2, k3 = random.split(rng_key, 3)
    
    mean = x_rad * dist.Uniform(-jnp.ones((dim,)), jnp.ones((dim,))).sample(k1)

    A = x_rad * dist.Uniform(low=jnp.zeros((dim, dim)), high=jnp.ones((dim, dim))).sample(k2) 
    cov = A.T @ A
    
    lam = log_step_rad * dist.Uniform(-1, 1).sample(k3)

    return AMHAdaptState(mean, cov, lam)
    

def sample_neigbour(rng_key, adapt_state, eps=1e-2):
    """
    Samples a neighboring adaptive state by perturbing the given state.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        A random key for reproducible random number generation.
    adapt_state : AMHAdaptState
        The current adaptive state, represented as a tuple of (mean, cov, lam).
    eps : float, optional
        Perturbation scale for the mean, covariance, and log step size (default is 1e-2).

    Returns
    -------
    AMHAdaptState
        A new adaptive state with perturbed values:
        - mean: array of the same shape as the input mean
        - cov: array of the same shape as the input covariance matrix
        - lam: float, perturbed log step size
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
    Computes the distance between two adaptive states.

    Parameters
    ----------
    s1 : AMHAdaptState
        The first adaptive state, represented as a tuple of (mean, cov, lam).
    s2 : AMHAdaptState
        The second adaptive state, represented as a tuple of (mean, cov, lam).

    Returns
    -------
    float
        The distance between the two states, computed as the sum of:
        - L2 norm of the difference between the means
        - L2 norm of the difference between the covariance matrices
        - L2 norm of the difference between the log step sizes
    """
    
    C1 = jnp.exp(s1.log_step_size)*s1.covariance
    C2 = jnp.exp(s2.log_step_size)*s2.covariance

    return jnp.linalg.norm(C1 - C2)


def sample_Px(rng_key, kernel, x, adapt_state, n_samples=1000):

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

    keys = random.split(rng_key, n_samples)
    return vmap(single_Px)(keys)
    