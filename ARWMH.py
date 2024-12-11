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
        "accept_prob",  # Acceptance probability of the last step
        "mean_accept_prob",  # Running mean of acceptance probabilities
        "adapt_state",  # Mean estimate & Covariance matrix estimate
        "rng_key",  # Random number generator state
    ],
)

class ARWMH(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "z"  # Field representing the MCMC sample

    def __init__(
        self, model=None, potential_fn=None, step_size=1.0, init_strategy=init_to_uniform
    ):
        """
        :param model: Optional model to initialize the kernel.
        :param potential_fn: A callable representing the negative log-posterior density.
        :param step_size: Adaptation update
        :param callable init_strategy: a per-site initialization function.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._step_size = step_size
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
            (
                params_info,
                potential_fn_gen,
                self._postprocess_fn,
                model_trace,
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
        
        mu_hat = ravel_pytree(init_params)[0]
        sigma_hat = jnp.eye(len(mu_hat)) * self._step_size
        adapt_state = (mu_hat, sigma_hat)

        init_state = ARWMHState(
            i=jnp.array(0),
            z=init_params,
            potential_energy=potential_energy,
            accept_prob=jnp.zeros(()),
            mean_accept_prob=jnp.zeros(()),
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
        i, z, potential_energy, _, mean_accept_prob, adapt_state, rng_key = state
        mu_hat, sigma_hat = adapt_state

        # Flatten a pytree of arrays down to a 1D array.
        z_flat, unravel_fn = ravel_pytree(z)
        
        # Split RNG for proposal and acceptance
        rng_key, key_proposal, key_accept = jax.random.split(rng_key, 3)

        # Propose a new sample
        z_proposal_flat = dist.MultivariateNormal(z_flat, self._step_size * sigma_hat + jnp.eye(len(z_flat)) * 1e-6).sample(key_proposal)
        z_proposal = unravel_fn(z_proposal_flat)
        potential_energy_proposal = self._potential_fn(z_proposal)

        # Compute acceptance probability
        accept_prob = jnp.clip(jnp.exp(potential_energy - potential_energy_proposal), min=None, max=1)
        is_accepted = dist.Uniform().sample(key_accept) < accept_prob

        # Update state based on acceptance
        z_new_flat = jnp.where(is_accepted, z_proposal_flat, z_flat)
        z_new = unravel_fn(z_new_flat)
        potential_energy_new = jnp.where(is_accepted, potential_energy_proposal, potential_energy)

        # # do not update adapt_state after warmup phase
        # adapt_state = jax.lax.cond(
        #     i < self._num_warmup,
        #     (i, accept_prob, (x,), adapt_state),
        #     lambda args: self._wa_update(*args),
        #     adapt_state,
        #     identity,
        # )

        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)

        # Update mean and covariance (adaptation step)
        eta = 1 / n
        delta = z_new_flat - mu_hat
        mu_new = mu_hat + eta * delta
        sigma_new = sigma_hat + eta *(jnp.outer(delta, delta) - sigma_hat)
        adapt_state_new = (mu_new, sigma_new)

        # Update running mean of acceptance probabilities
        mean_accept_prob_new = mean_accept_prob + eta * (accept_prob - mean_accept_prob)

        return ARWMHState(
            i=i + 1,
            z=z_new,
            potential_energy=potential_energy_new,
            accept_prob=accept_prob,
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
        return f"Acceptance rate: {state.mean_accept_prob:.2f}"
