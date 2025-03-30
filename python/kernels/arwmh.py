from collections import namedtuple
from math import gamma

import jax
from jax.flatten_util import ravel_pytree
from jax import random, vmap, jit
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions.util import cholesky_update
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform, init_to_median
from numpyro.util import identity

ARWMHState = namedtuple(
    "ARWMHState",
    [
        "i",  # Iteration
        "z",  # Current Point
        "potential_energy",  # Current potential energy
        "mean_accept_prob",  # Running mean of acceptance probabilities
        "adapt_state",  # Mean & Covariance matrix estimate + log of step size
        "rng_key",  # Random number generator state
    ],
)

ARWMHAdaptState = namedtuple("ARWMHAdaptState", ["loc", "scale", "log_step_size"])


class ARWMH(MCMCKernel):
    """
    ARWMH kernel for adaptive random walk-based Markov Chain Monte Carlo.

    Attributes
    ----------
    sample_field : str
        The field name in `ARWMHState` that contains the current sample.
    """

    sample_field = "z"

    def __init__(
            self, model=None, potential_fn=None, alpha=1/2, target_accept_prob=0.234, eps=1e-6, init_strategy=init_to_median
    ):
        """
        Initialize the ARWMH kernel.

        Parameters
        ----------
        model : callable, optional
            The model to initialize the kernel. Either `model` or `potential_fn` must be specified, but not both.
        potential_fn : callable, optional
            A potential function representing the negative log-posterior density.
        alpha: float, optional
            Parameter controlling the learning rate: gamma = 1 / n**alpha, default is 1/2.
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
        self._alpha = alpha
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
            Initial state of the ARWMH algorithm.

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
        sigma_hat_sqrt = jnp.eye(len(mu_hat))
        log_lambda = 0.0
        adapt_state = ARWMHAdaptState(mu_hat, sigma_hat_sqrt, log_lambda)

        init_state = ARWMHState(
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
        ARWMHState
            Updated state after sampling.
        """
        i, z, potential_energy, mean_accept_prob, adapt_state, rng_key = state
        mu_hat, sigma_hat_sqrt, log_lambda = adapt_state

        z_flat, unravel_fn = ravel_pytree(z)
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)

        dim = len(z_flat)
        prop_base = dist.Normal().sample(key_proposal, sample_shape=(dim,))
        prop_scale = sigma_hat_sqrt * jnp.exp(log_lambda) + jnp.eye(dim) * self._eps
        z_proposal_flat = z_flat + jnp.dot(prop_scale, prop_base)

        # z_step_flat = dist.MultivariateNormal(loc=0.0, covariance_matrix=proposal_cov).sample(key_proposal)
        # z_proposal_flat = z_flat + z_step_flat * jnp.exp(log_lambda / 2)
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
        gamma = 1 / n ** self._alpha

        mean_accept_prob_new = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        # Adaptation
        delta = z_new_flat - mu_hat
        mu_new = mu_hat + gamma * delta
        cholesky = cholesky_update(jnp.sqrt(1 - gamma) * sigma_hat_sqrt, delta, gamma)
        sigma_sqrt_new = jnp.where(jnp.any(jnp.isnan(cholesky)), sigma_hat_sqrt, cholesky)
        # sigma_new = sigma_hat + gamma * (jnp.outer(delta, delta) - sigma_hat)

        log_lambda_new = log_lambda + gamma * (accept_prob - self._target_accept_prob)

        adapt_state_new = ARWMHAdaptState(mu_new, sigma_sqrt_new, log_lambda_new)

        return ARWMHState(
            i=itr,
            z=z_new,
            potential_energy=potential_energy_new,
            mean_accept_prob=mean_accept_prob_new,
            adapt_state=adapt_state_new,
            rng_key=rng_key,
        )

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def get_diagnostics_str(self, state):
        """
        Return diagnostics string for progress monitoring.

        Parameters
        ----------
        state : ARWMHState
            Current state of the ARWMH algorithm.

        Returns
        -------
        str
            A string containing diagnostics information.
        """
        return f"Acceptance rate: {state.mean_accept_prob:.2f}, Step size: {jnp.exp(state.adapt_state.log_step_size):.3f}"

    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000):
        @jit
        def single_Pnx(x, key):
            def single_Px(i, val_tuple):
                x, key, pot_energy = val_tuple
                input_state = ARWMHState(
                    i=0,
                    z=x,
                    potential_energy=pot_energy,
                    mean_accept_prob=0.,
                    adapt_state=adapt_state,
                    rng_key=key
                )
                next_state = self.sample(state=input_state, model_args=(), model_kwargs={})
                return (next_state.z, next_state.rng_key, next_state.potential_energy)

            pot_energy = self._potential_fn(x)
            x_new, key, pot_energy = jax.lax.fori_loop(0, n, single_Px, (x, key, pot_energy))
            return x_new

        n_points = x.shape[0]
        r_keys = random.split(rng_key, (n_points, n_samples))

        samples = vmap(vmap(single_Pnx, in_axes=(None, 0)))(x, r_keys)

        # r_keys = random.split(rng_key, n_samples)
        # if isinstance(x, dict):
        #     samples = vmap(single_Pnx, in_axes=(None, 0))(x, r_keys)
        # else:
        #     samples = vmap(
        #         vmap(single_Pnx, in_axes=(None, 0), out_axes=0),
        #         in_axes=(0, None), out_axes=0
        #     )(x, r_keys)
        return samples

    def get_init_adapt_state(self, rng_key, init_params, model_args=(), model_kwargs={}):
        """Return the first adapt state after initialization."""
        num_warmup = 0
        init_state = self.init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return init_state.adapt_state
