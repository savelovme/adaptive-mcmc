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
        "z",  # Current point on the sphere (S^d)
        "potential_energy",  # Current potential energy
        "adapt_state",  # Mean & Covariance matrix estimate
        "rng_key",  # Random number generator state
    ],
)

ASSSAdaptState = namedtuple("ASSSAdaptState", ["loc", "scale"])


class ASSS(MCMCKernel):
    """
    Adaptive Stereographic Slice Sampler kernel for MCMC.
    """

    sample_field = "z"

    def __init__(
        self, model=None, potential_fn=None, alpha=1/2, eps=1e-6, x_rad=1e6, init_strategy=init_to_median
    ):
        """
        Initialize the ASSS kernel.

        Parameters
        ----------
        model : callable, optional
            The model to initialize the kernel. Either `model` or `potential_fn` must be specified.
        potential_fn : callable, optional
            A potential function representing the negative log-posterior density.
        alpha: float, optional
            Parameter controlling the learning rate: gamma = 1 / n**alpha, default is 1/2.
        eps : float, optional
            Regularization term, default is 1e-6.
        x_rad : float, optional
            Max x value per dimension, default is 1e6.
        init_strategy : callable, optional
            A per-site initialization function, default is `init_to_uniform`.
        """
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        self._model = model
        self._potential_fn = potential_fn
        self._alpha = alpha
        self._eps = eps
        self._x_rad = x_rad
        self._postprocess_fn = None
        self._init_strategy = init_strategy
        self._num_warmup = 0

    @property
    def model(self):
        return self._model

    def _stereographic_project(self, x, mu, sigma_inv_sqrt):
        """Project x from R^d to S^d using mu and Sigma."""
        x_shifted = x - mu
        x_scaled = jnp.dot(sigma_inv_sqrt, x_shifted)
        norm_sq = jnp.sum(x_scaled ** 2)
        z_1d = 2 * x_scaled / (norm_sq + 1)
        z_d1 = (norm_sq - 1) / (norm_sq + 1)
        z = jnp.concatenate([z_1d, z_d1[None]])
        return z

    def _stereographic_inverse(self, z, mu, sigma_sqrt):
        """Project z from S^d back to R^d."""
        z_1d = z[:-1]
        z_d1 = z[-1]
        x_base = z_1d / (1 - z_d1)
        x = jnp.dot(sigma_sqrt, x_base) + mu
        # x_clipped = jnp.clip(x, -self._x_rad, self._x_rad)
        return x

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
        i, x, potential_energy, adapt_state, rng_key = state
        loc, scale = adapt_state

        x_flat, unravel_fn = ravel_pytree(x)
        rng_key, key_v, key_t, key_shrink = random.split(rng_key, 4)

        dim = len(loc)

        sigma_sqrt = scale * dim**0.5
        sigma_sqrt_inv = triangular_solve(scale, jnp.eye(dim), lower=True) / dim**0.5

        z = self._stereographic_project(x_flat, loc, sigma_sqrt_inv)
        pe_transformed = potential_energy + dim * jnp.log(1 - z[-1])

        # Sample velocity v orthogonal to z
        v = random.normal(key_v, (dim+1,))
        v -= jnp.dot(v, z) * z  # Project to tangent space
        v /= jnp.linalg.norm(v)  # Normalize to S^d

        # Set level for potential energy
        u_t = dist.Uniform().sample(key_t)
        t_pe = pe_transformed - jnp.log(u_t)

        def shrinkage(r_key, z, v, t_pe):
            key_init, key_loop = random.split(r_key)
            theta = dist.Uniform(0, 2 * jnp.pi).sample(key_init)

            theta_min = theta - 2 * jnp.pi
            theta_max = theta

            def cond_fn(val):
                r_key, theta, theta_min, theta_max = val

                z_theta = z * jnp.cos(theta) + v * jnp.sin(theta)
                x_theta = self._stereographic_inverse(z_theta, loc, sigma_sqrt)
                pe_theta = self._potential_fn(unravel_fn(x_theta)) + dim * jnp.log(1 - z_theta[-1])
                pe_theta = jnp.where(jnp.isnan(pe_theta), jnp.inf, pe_theta)

                return jnp.logical_or(pe_theta >= t_pe, (1.0 - z_theta[-1]) < self._eps).squeeze()

            def body_fn(val):
                r_key, theta, theta_min, theta_max = val
                key_sample, r_key_next = random.split(r_key)
                theta_min_new = jnp.where(theta < 0.0, theta, theta_min)
                theta_max_new = jnp.where(theta >= 0.0, theta, theta_max)

                theta_new = dist.Uniform(theta_min_new, theta_max_new).sample(key_sample)

                return (r_key_next, theta_new, theta_min_new, theta_max_new)

            val = (key_shrink, theta, theta_min, theta_max)
            # while cond_fn(val):
            #     val = body_fn(val)
            # theta_final = val[1]

            _, theta_final, _, _ = jax.lax.while_loop(
                cond_fn, body_fn, val
            )
            z_new = z * jnp.cos(theta_final) + v * jnp.sin(theta_final)
            return z_new

        z_new = shrinkage(key_shrink, z, v, t_pe)

        x_new_flat = self._stereographic_inverse(z_new, loc, sigma_sqrt)
        x_new = unravel_fn(x_new_flat)
        pe_new = self._potential_fn(x_new)
        pe_new = jnp.where(jnp.isnan(pe_new), jnp.inf, pe_new)

        # Adaptation
        itr = i + 1
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        gamma = 1 / n ** self._alpha # Learning rate

        delta = x_new_flat - loc
        mu_new = loc + gamma * delta
        # cov_new = cov + gamma * (jnp.outer(delta, delta) - cov)
        cholesky = cholesky_update(jnp.sqrt(1 - gamma) * scale, delta, gamma)
        scale_new = jnp.where(jnp.any(jnp.isnan(cholesky)), scale, cholesky)

        adapt_state_new = ASSSAdaptState(mu_new, scale_new)

        return ASSSState(
            i=itr,
            z=x_new,
            potential_energy=pe_new,
            adapt_state=adapt_state_new,
            rng_key=rng_key,
        )

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def get_diagnostics_str(self, state):
        return f"Iteration: {state.i}, Potential Energy: {state.potential_energy:.2f}"

    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000):
        """
        Return samples from P^n(x, .)
        """

        @jit
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

        n_points = x.shape[0]
        r_keys = random.split(rng_key, (n_points, n_samples))

        samples = vmap(vmap(single_Pnx, in_axes=(None, 0)))(x, r_keys)

        return samples

    def get_init_adapt_state(self, rng_key, init_params, model_args=(), model_kwargs={}):
        """Return the first adapt state after initialization."""
        num_warmup = 0
        init_state = self.init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return init_state.adapt_state
