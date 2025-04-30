from collections import namedtuple

import jax
from jax.flatten_util import ravel_pytree
from jax import random, vmap, jit
import jax.numpy as jnp
from jax.lax.linalg import cholesky, triangular_solve

import numpyro.distributions as dist
from numpyro.distributions.util import cholesky_update
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform, init_to_median
from numpyro.util import identity


ASSSState = namedtuple(
    "ASSSState",
    [
        "i",  # Iteration
        "z",  # Current point
        "potential_energy",  # Current potential energy
        # "diverging" # Whether the new sample potential energy is diverging from the current one
        "adapt_state",  # Mean & Scale matrix estimates
        "as_change",
        "rng_key",  # Random number generator state
    ],
)

ASSSAdaptState = namedtuple("ASSSAdaptState", ["loc", "scale"])


def _stereographic_project(x, loc, scale):
    """
    Project x from R^d to S^d using mu and Sigma.
    scale is assumed to be lower-triangular
    """
    x_shifted = x - loc
    x_rescaled = triangular_solve(scale, x_shifted, left_side=True, lower=True)
    norm_sq = jnp.sum(x_rescaled ** 2)
    z_1d = 2 * x_rescaled / (norm_sq + 1)
    z_d1 = (norm_sq - 1) / (norm_sq + 1)
    z = jnp.concatenate([z_1d, z_d1[None]])
    return z


def _stereographic_inverse(z, loc, scale):
    """
    Project z from S^d back to R^d.
    scale is assumed to be lower-triangular
    """
    z_1d = z[:-1]
    z_d1 = z[-1]
    x_base = z_1d / (1 - z_d1)
    x = jnp.dot(scale, x_base) + loc
    return x


def _shrinkage(rng_key, z, v, t_pe, transformed_pe_fn, eps=1e-6, max_iterations=50):
    key_init, key_loop = random.split(rng_key)
    theta = dist.Uniform(0, 2 * jnp.pi).sample(key_init)

    theta_min = theta - 2 * jnp.pi
    theta_max = theta

    def cond_fn(val):
        r_key, theta, theta_min, theta_max, iteration = val

        z_theta = z * jnp.cos(theta) + v * jnp.sin(theta)
        pe_theta = transformed_pe_fn(z_theta)
        pe_theta = jnp.where(jnp.isnan(pe_theta), jnp.inf, pe_theta)

        return jnp.logical_and(
            iteration < max_iterations,
            jnp.logical_or(pe_theta > t_pe, (1.0 - z_theta[-1]) < eps).squeeze()
        )

    def body_fn(val):
        r_key, theta, theta_min, theta_max, iteration = val
        key_sample, r_key_next = random.split(r_key)
        theta_min_new = jnp.where(theta < 0.0, theta, theta_min)
        theta_max_new = jnp.where(theta >= 0.0, theta, theta_max)

        theta_new = dist.Uniform(theta_min_new, theta_max_new).sample(key_sample)

        return (r_key_next, theta_new, theta_min_new, theta_max_new, iteration + 1)

    val = (key_loop, theta, theta_min, theta_max, 0)

    r_key_final, theta_final, _, _, iteration_final = jax.lax.while_loop(
        cond_fn, body_fn, val
    )

    theta_final = jnp.where(iteration_final >= max_iterations, 0.0, theta_final)
    z_new = z * jnp.cos(theta_final) + v * jnp.sin(theta_final)
    return z_new

class ASSS(MCMCKernel):
    """
    Adaptive Stereographic Slice Sampler kernel for MCMC.
    """

    sample_field = "z"

    def __init__(
        self, model=None, potential_fn=None, lr_decay=2/3, eps=1e-6, init_strategy=init_to_uniform
    ):
        """
        Initialize the ASSS kernel.

        Parameters
        ----------
        model : callable, optional
            The model to initialize the kernel. Either `model` or `potential_fn` must be specified.
        potential_fn : callable, optional
            A potential function representing the negative log-posterior density.
        lr_decay: float, optional
            Parameter controlling the learning rate: gamma = 1 / n^lr_decay, default is 2/3.
        eps : float, optional
            Regularization term, default is 1e-6.
        init_strategy : callable, optional
            A per-site initialization function, default is `init_to_uniform`.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._lr_decay = lr_decay
        self._eps = eps
        self._postprocess_fn = None
        self._init_strategy = init_strategy
        self._num_warmup = 0

    @property
    def model(self):
        return self._model

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the ASSS kernel state.

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
        ASSSState
            Initial state of the ASSS algorithm.
        """
        self._num_warmup = num_warmup

        if self._model is not None:
            params_info, potential_fn_gen, self._postprocess_fn, _ = initialize_model(
                rng_key, self._model, init_strategy=self._init_strategy, dynamic_args=True,
                model_args=model_args, model_kwargs=model_kwargs or {}
            )
            init_params = params_info[0]
            self._potential_fn = potential_fn_gen(*model_args, **(model_kwargs or {}))

        if self._potential_fn and init_params is None:
            raise ValueError("Valid value of `init_params` must be provided with `potential_fn`.")

        potential_energy = self._potential_fn(init_params)

        # Flatten init_params and initialize adaptation state
        loc, unravel_fn = ravel_pytree(init_params)

        d = len(loc)
        scale = jnp.eye(d)
        adapt_state = ASSSAdaptState(loc, scale)

        init_state = ASSSState(
            i=jnp.array(0),
            z=init_params,
            potential_energy=potential_energy,
            adapt_state=adapt_state,
            as_change=jnp.array(0.0),
            rng_key=rng_key,
        )
        return init_state

    def sample(self, state, model_args, model_kwargs):
        """
        Generate the next sample using the adaptive stereographic slice sampler.

        Parameters
        ----------
        state : ASSSState
            Current state of the ASSS algorithm.
        model_args : tuple
            Arguments for the model.
        model_kwargs : dict, optional
            Keyword arguments for the model.

        Returns
        -------
        ASSSState
            Updated state after sampling.
        """
        i, x, potential_energy, adapt_state, _, rng_key = state
        loc, scale = adapt_state

        x_flat, unravel_fn = ravel_pytree(x)
        rng_key, key_v, key_t, key_shrink = random.split(rng_key, 4)

        dim = loc.shape[-1]

        sigma_sqrt = (scale + self._eps * jnp.eye(dim)) * dim**0.5
        # sigma_sqrt_inv = triangular_solve(scale, jnp.eye(dim), lower=True) / dim**0.5

        # Define transformed potential energy
        def transformed_pe(z):
            x_flat = _stereographic_inverse(z, loc, sigma_sqrt)
            pe_transformed = self._potential_fn(unravel_fn(x_flat)) + dim * jnp.log(1. - z[-1])
            return pe_transformed

        z = _stereographic_project(x_flat, loc, sigma_sqrt)
        pe_transformed = transformed_pe(z)

        # Sample velocity v orthogonal to z
        v = random.normal(key_v, (dim+1,))
        v -= jnp.dot(v, z) * z  # Project to tangent space
        v /= jnp.linalg.norm(v)  # Normalize to S^d

        # Set level for potential energy
        u_t = dist.Uniform().sample(key_t)
        t_pe = pe_transformed - jnp.log(u_t)

        z_new = _shrinkage(key_shrink, z, v, t_pe, transformed_pe, self._eps)

        x_new_flat = _stereographic_inverse(z_new, loc, sigma_sqrt)
        x_new = unravel_fn(x_new_flat)
        pe_new = self._potential_fn(x_new)
        pe_new = jnp.where(jnp.isnan(pe_new), jnp.inf, pe_new)

        # Adaptation
        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        gamma = 1 / n ** self._lr_decay # Learning rate

        delta = x_new_flat - loc
        loc_new = loc + gamma * delta
        # cov_new = cov + gamma * (jnp.outer(delta, delta) - cov)
        cholesky = cholesky_update(jnp.sqrt(1 - gamma) * scale, delta, gamma)
        scale_new = jnp.where(jnp.any(jnp.isnan(cholesky)), scale, cholesky)

        adapt_state_new = ASSSAdaptState(loc_new, scale_new)

        loc_diffs = jnp.linalg.vector_norm(loc_new - loc)
        scale_diffs = jnp.linalg.matrix_norm(scale_new - scale)

        return ASSSState(
            i=itr,
            z=x_new,
            potential_energy=pe_new,
            adapt_state=adapt_state_new,
            as_change=(loc_diffs + scale_diffs),
            rng_key=rng_key,
        )

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def get_diagnostics_str(self, state):
        return f"Iteration: {state.i}, Potential Energy: {state.potential_energy:.2f}"

    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000, jit_inner=True):
        """
        Return samples from P^n(x, .)
        """

        def single_Pnx(x, key):
            def single_Px(i, val_tuple):
                x, key, pot_energy = val_tuple
                input_state = ASSSState(
                    i=0,
                    z=x,
                    potential_energy=pot_energy,
                    adapt_state=adapt_state,
                    rng_key=key
                )
                next_state = self.sample(state=input_state, model_args=(), model_kwargs={})
                return (next_state.z, next_state.rng_key, next_state.potential_energy)

            pot_energy = self._potential_fn(x)

            x_new, key_new, pot_energy_new = jax.lax.fori_loop(
                0, n, single_Px, (x, key, pot_energy)
            )

            return x_new

        if jit_inner:
            single_Pnx = jit(single_Pnx)

        n_points = x.shape[0]
        r_keys = random.split(rng_key, (n_points, n_samples))

        samples = vmap(vmap(single_Pnx, in_axes=(None, 0)))(x, r_keys)
        # samples = vmap(vmap(single_Pnx), in_axes=(None, 1), out_axes=1)(x, r_keys)

        return samples

    def get_init_adapt_state(self, rng_key, init_params, model_args=(), model_kwargs={}):
        """Return the first adapt state after initialization."""
        num_warmup = 0
        init_state = self.init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return init_state.adapt_state
