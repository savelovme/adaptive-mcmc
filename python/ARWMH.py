from collections import namedtuple
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform


# Define state for ARWMH
ARWMHState = namedtuple(
    "ARWMHState",
    [
        "i",  # Iteration
        "z",  # Current Point
        "potential_energy",  # Current potential energy
        "mean_accept_prob",  # Running mean of acceptance probabilities
        "adapt_state",  # Mean & Covariance matrix estimate; step size
        "rng_key",  # Random number generator state
    ],
)

AdaptState = namedtuple("AdaptState", ["mean", "covariance", "step_size"])

class ARWMH(numpyro.infer.mcmc.MCMCKernel):
    """
    Adaptive Random Walk Metropolis-Hastings
    """
    sample_field = "z"  # Field representing the MCMC sample

    def __init__(
        self, model=None, potential_fn=None, target_accept_prob=0.3, gamma=0.05, eps=1e-6, init_strategy=init_to_uniform
    ):
        """
        :param model: Optional model to initialize the kernel.
        :param potential_fn: A callable representing the negative log-posterior density.
        :param target_accept_prob: Target acceptance rate.
        :param gamma: Adaptation parameter for step size.
	    :param eps: Regularization for proposal covariance.
        :param callable init_strategy: a per-site initialization function.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._step_size = 1.0
        self._target_accept_prob = target_accept_prob
        self._gamma=gamma
        self._eps=eps
        self._postprocess_fn = None
        self._init_strategy = init_strategy

    @property
    def model(self):
        return self._model

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the ARWMH kernel state.

        :param rng_key: Random number generator key.
        :param num_warmup: Number of warmup iterations.
        :param init_params: Initial parameters for the chain.
        :param model_args: Arguments for the model.
        :param model_kwargs: Keyword arguments for the model.
        :return: Initial ARWMHState.
        """
        self._num_warmup = num_warmup

        if self._model is not None:
            # Get callable potential function initial values for parameters w.r.t. init_strategy
            (
                params_info,
                potential_fn_gen,
                self._postprocess_fn,
                _,
            ) = initialize_model(
                rng_key,
                self._model,
                init_strategy=self._init_strategy,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            init_params = params_info[0]

            model_kwargs = {} if model_kwargs is None else model_kwargs
            self._potential_fn = potential_fn_gen(*model_args, **model_kwargs)

        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with `potential_fn`."
            )

        potential_energy = self._potential_fn(init_params)

    	# Flatten a pytree structure of arrays down to a 1D array.
        mu_hat = ravel_pytree(init_params)[0]
        step_size = 1.0
        sigma_hat = jnp.eye(len(mu_hat)) * step_size**2
        adapt_state = AdaptState(mu_hat, sigma_hat, step_size)

        init_state = ARWMHState(
            i=jnp.array(0),
            z=init_params,
            potential_energy=potential_energy,
            mean_accept_prob=jnp.array(0.0),
            adapt_state=adapt_state,
            rng_key=rng_key,
        )

        return jax.device_put(init_state)


    def sample(self, state, model_args, model_kwargs):
        """
        Generate the next sample using the adaptive random walk kernel.

        :param state: Current ARWMHState.
        :param model_args: Arguments for the model.
        :param model_kwargs: Keyword arguments for the model.
        :return: Updated ARWMHState.
        """
        i, z, potential_energy, mean_accept_prob, adapt_state, rng_key = state
        mu_hat, sigma_hat, step_size = adapt_state

        # Flatten a pytree of arrays down to a 1D array.
        z_flat, unravel_fn = ravel_pytree(z)

        # Split RNG for proposal and acceptance
        rng_key, key_proposal, key_accept = jax.random.split(rng_key, 3)

        # Propose a new sample
        proposal_cov =  sigma_hat * step_size**2 + jnp.eye(len(z_flat)) * self._eps
        z_step_flat = dist.MultivariateNormal(loc=0.0, covariance_matrix=proposal_cov).sample(key_proposal)
        z_proposal_flat = z_flat + z_step_flat
        z_proposal = unravel_fn(z_proposal_flat)
        potential_energy_proposal = self._potential_fn(z_proposal)
        potential_energy_proposal = jnp.where(jnp.isnan(potential_energy_proposal), jnp.inf, potential_energy_proposal)

        # Compute acceptance probability
        accept_prob = jnp.clip(jnp.exp(potential_energy - potential_energy_proposal), min=0, max=1)
        is_accepted = dist.Uniform().sample(key_accept) < accept_prob

        # Update state based on acceptance
        z_new_flat = jnp.where(is_accepted, z_proposal_flat, z_flat)
        z_new = unravel_fn(z_new_flat)
        potential_energy_new = jnp.where(is_accepted, potential_energy_proposal, potential_energy)

	    # Update iteration counter
        itr = i + 1
        
        # Restart n after warmup
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        eta = 1.0 / n

        # Update running mean of acceptance probabilities
        mean_accept_prob_new = mean_accept_prob + eta * (accept_prob - mean_accept_prob)

        # Adaptation: update mean and covariance
        delta = z_new_flat - mu_hat
        mu_new = mu_hat + eta * delta
        sigma_new = sigma_hat + eta * (jnp.outer(delta, delta) - sigma_hat)
        
        # Step Size Adaptation (only during warmup)
        step_size_new = step_size * jnp.exp(self._gamma * (mean_accept_prob_new - self._target_accept_prob))
        step_size_new = jnp.clip(step_size_new, 1e-3, 1e1)
        step_size_new = jnp.where(i < self._num_warmup, step_size_new, step_size)

        adapt_state_new = AdaptState(mu_new, sigma_new, step_size_new)

        return ARWMHState(
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

        :param state: Current ARWMHState.
        :return: Diagnostics string.
        """
        return f"Acceptance rate: {state.mean_accept_prob:.2f}, Step size: {state.adapt_state.step_size:.3f}"
