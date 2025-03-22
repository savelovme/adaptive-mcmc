from collections import namedtuple

import jax.lax
from jax.flatten_util import ravel_pytree
from jax import random, vmap, jit
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform
from numpyro.util import identity
from numpyro.infer import NUTS, SA

from numpyro.infer.sa import SA as SA_numpyro, SAState, SAAdaptState

class SA(SA_numpyro):
    """
    SA class from NumPyro.
    """
    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000):
        """
        rng_key: PRNGKey
        x: Array of shape (n_points, d)
        adapt_state: SAAdaptState
        n: int Power
        n_samples: int Number of samples

        Returns: Array of shape (n_points, n_samples, d)
        """

        @jit
        def single_Pnx(x, key):

            def single_Px(i, val_tuple):
                x, key, pot_energy = val_tuple

                input_state = SAState(
                    i=0,
                    z=x,
                    potential_energy=pot_energy,
                    mean_accept_prob=0,
                    adapt_state=adapt_state,
                    rng_key=key
                )
                next_state = self.sample(state=input_state, model_args=(), model_kwargs={})

                return (next_state.z, next_state.rng_key, next_state.potential_energy)


            pot_energy = self._potential_fn(x)
            x, key, pot_energy = jax.lax.fori_loop(0, n, single_Px, (x, key, pot_energy))
            return x

        # Generate separate keys for each sample
        r_keys = random.split(rng_key, n_samples)  # Shape: (n_samples, 2)

        # Use vmap to apply sampling across n_samples and x
        samples = vmap(
            vmap(single_Pnx, in_axes=(None, 0), out_axes=0),  # Apply over r_keys
            in_axes=(0, None), out_axes=0  # Apply over x
        )(x, r_keys)

        return samples  # Shape: (n_points, n_samples, d)


# def sample_adapt_state(rng_key, dim=1, x_rad=5., log_step_rad=4.):
#     """
#     Generates an initial adaption state for Adaptive Metropolis-Hastings (AMH) algorithm by sampling random values for the mean, covariance, and log step size.
#
#     Parameters:
#     rng_key: A JAX random key seeded for reproducibility.
#     dim: Integer specifying the dimensionality of the random variables (default is 1).
#     x_rad: Float determining the scaling factor for mean and covariance matrix (default is 5.0).
#     log_step_rad: Float scaling factor for the log step size (default is 4.0).
#
#     Returns:
#     An instance of AMHAdaptState containing:
#     - mean: A sampled mean vector of shape (dim,).
#     - cov: A positive semi-definite covariance matrix of shape (dim, dim).
#     - lam: A sampled log step size (scalar).
#     """
#     k1, k2, k3 = random.split(rng_key, 3)
#
#     mean = x_rad * dist.Uniform(-jnp.ones((dim,)), jnp.ones((dim,))).sample(k1)
#
#     A = x_rad * dist.Uniform(low=jnp.zeros((dim, dim)), high=jnp.ones((dim, dim))).sample(k2)
#     cov = A.T @ A
#
#     lam = log_step_rad * dist.Uniform(-1, 1).sample(k3)
#
#     return AMHAdaptState(mean, cov, lam)
    

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
    mean1, cov1, lambda1 = adapt_state
    k1, k2, k3 = random.split(rng_key, 3)

    mean = mean1 + eps * dist.Uniform(-jnp.ones_like(mean1), jnp.ones_like(mean1)).sample(k1)

    A1 = cholesky(cov1)
    A = A1 + eps * dist.Uniform(low=-jnp.ones_like(cov1), high=jnp.ones_like(cov1)).sample(k2) 
    cov = A.T @ A
    
    lambda_new = lambda1 + eps * dist.Uniform(-1, 1).sample(k3)

    return AMHAdaptState(mean, cov, lambda_new)


def state_dist(s1, s2):
    """
    Computes the distance between two states based on their log step size and covariance.

    Args:
        s1: The first state object containing log_step_size and covariance attributes.
        s2: The second state object containing log_step_size and covariance attributes.

    Returns:
    """
    
    C1 = jnp.exp(s1.log_step_size)*s1.covariance
    C2 = jnp.exp(s2.log_step_size)*s2.covariance

    return jnp.linalg.norm(C1 - C2)

#
# def sample_Px(rng_key, kernel, x, adapt_state, n_samples=1000):
#     """
#     rng_key: PRNGKey (array of shape (2,))
#     kernel: AMH
#     x: Array of shape (n_points, d)
#     adapt_state: AMHAdaptState
#     n_samples: int Number of samples
#
#     Returns: Array of shape (n_points, n_samples, d)
#     """
#
#     @jit
#     def single_Px(x, key):
#         """
#         x: Array of shape (d,)
#         key:  PRNGKey (array of shape (2,))
#
#         Returns: Array of shape (d,)
#         """
#         input_state = AMHState(
#             i=0,
#             z=x,
#             potential_energy=kernel._potential_fn(x),
#             mean_accept_prob=0,
#             adapt_state=adapt_state,
#             rng_key=key
#         )
#         next_state = kernel.sample(state=input_state, model_args=(), model_kwargs={})
#         x_next = next_state.z
#
#         return x_next
#
#     # Generate separate keys for each sample
#     r_keys = random.split(rng_key, n_samples)  # Shape: (n_samples, 2)
#
#     # Use vmap to apply sampling across n_samples and x
#     samples = vmap(
#         vmap(single_Px, in_axes=(None, 0), out_axes=0),  # Apply over r_keys
#         in_axes=(0, None), out_axes=0  # Apply over x
#     )(x, r_keys)
#
#     return samples  # Shape: (n_points, n_samples, d)
#
#
# def sample_Pnx(rng_key, kernel, x, adapt_state, n=1, n_samples=1000):
#     """
#     rng_key: PRNGKey (array of shape (2,))
#     kernel: AMH
#     x: Array of shape (n_points, d)
#     adapt_state: AMHAdaptState
#     n: int power
#     n_samples: int Number of samples
#
#     Returns: Array of shape (n_points, n_samples, d)
#     """
#
#     def single_Px(i, val):
#         x, key, pot_energy = val
#
#         input_state = AMHState(
#             i=0,
#             z=x,
#             potential_energy=pot_energy,
#             mean_accept_prob=0,
#             adapt_state=adapt_state,
#             rng_key=key
#         )
#         next_state = kernel.sample(state=input_state, model_args=(), model_kwargs={})
#
#         return (next_state.z, next_state.rng_key, next_state.potential_energy)
#
#     @jit
#     def single_Pnx(x, key):
#         pot_energy = kernel._potential_fn(x)
#         x, key, pot_energy = jax.lax.fori_loop(0, n, single_Px, (x, key, pot_energy))
#         return x
#
#     # Generate separate keys for each sample
#     r_keys = random.split(rng_key, n_samples)  # Shape: (n_samples, 2)
#
#     # Use vmap to apply sampling across n_samples and x
#     samples = vmap(
#         vmap(single_Pnx, in_axes=(None, 0), out_axes=0),  # Apply over r_keys
#         in_axes=(0, None), out_axes=0  # Apply over x
#     )(x, r_keys)
#
#     return samples  # Shape: (n_points, n_samples, d)