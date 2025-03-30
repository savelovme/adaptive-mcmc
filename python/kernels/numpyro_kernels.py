from collections import namedtuple

import jax.lax
from jax.flatten_util import ravel_pytree
from jax import random, vmap, jit
import jax.numpy as jnp

from numpyro.infer.initialization import init_to_uniform
from numpyro.util import identity

from numpyro.infer.sa import SAState, SAAdaptState, SA as SA_numpyro
from numpyro.infer.hmc import HMCState, NUTS as NUTS_numpyro
from numpyro.infer.hmc_util import HMCAdaptState


class SA(SA_numpyro):
    def __init__(self, model=None, potential_fn=None, adapt_state_size=None,
                 dense_mass=True, init_strategy=init_to_uniform):
        super().__init__(model=model,
                         potential_fn=potential_fn,
                         adapt_state_size=adapt_state_size,
                         dense_mass=dense_mass,
                         init_strategy=init_strategy)

    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000):
        @jit
        def single_Pnx(x, key):
            def single_Px(i, val_tuple):
                x, key, pot_energy = val_tuple
                input_state = SAState(
                    i=0,
                    z=x,
                    potential_energy=pot_energy,
                    accept_prob=jnp.zeros(()),
                    mean_accept_prob=jnp.zeros(()),
                    diverging=jnp.array(False),
                    adapt_state=adapt_state,
                    rng_key=key
                )
                next_state = self.sample(state=input_state, model_args=(), model_kwargs={})
                return (next_state.z, next_state.rng_key, next_state.potential_energy)

            if isinstance(x, dict):
                x_flat, unravel_fn = ravel_pytree(x)
            else:
                x_flat, unravel_fn = x, identity

            pot_energy = self._potential_fn(x)
            x_new, key_new, pot_energy_new = jax.lax.fori_loop(
                0, n, single_Px, (x_flat, key, pot_energy)
            )
            # val = (x_flat, key, pot_energy)
            # for i in range(n):
            #     val = single_Px(i, val)
            # x_new, key_new, pot_energy_new = val

            return unravel_fn(x_new) if isinstance(x, dict) else x_new

        r_keys = random.split(rng_key, n_samples)
        if isinstance(x, dict):
            samples = vmap(single_Pnx, in_axes=(None, 0))(x, r_keys)
        else:
            samples = vmap(
                vmap(single_Pnx, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None), out_axes=0
            )(x, r_keys)
        return samples

    def get_init_adapt_state(self, rng_key, init_params, model_args=(), model_kwargs={}):
        """Return the first adapt state after initialization."""
        num_warmup = 0
        init_state = self.init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return init_state.adapt_state


class NUTS(NUTS_numpyro):
    def __init__(self, model=None, potential_fn=None, step_size=1.0,
                 adapt_step_size=True, adapt_mass_matrix=True, dense_mass=False,
                 target_accept_prob=0.8, max_tree_depth=10, init_strategy=init_to_uniform):
        super().__init__(
            model=model,
            potential_fn=potential_fn,
            step_size=step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            init_strategy=init_strategy
        )

    def sample_Pnx(self, rng_key, x, adapt_state, n=1, n_samples=1000):
        @jit
        def single_Pnx(x, key):
            def single_Px(i, val_tuple):
                x, key, pot_energy = val_tuple
                input_state = HMCState(
                    i=0,
                    z=x,
                    z_grad=jnp.zeros_like(x),
                    potential_energy=pot_energy,
                    energy=None,
                    r=None,
                    trajectory_length=None,
                    num_steps=jnp.array(0),
                    accept_prob=jnp.zeros(()),
                    mean_accept_prob=jnp.zeros(()),
                    diverging=jnp.array(False),
                    adapt_state=adapt_state,
                    rng_key=key
                )
                next_state = self.sample(state=input_state, model_args=(), model_kwargs={})
                return (next_state.z, next_state.rng_key, next_state.potential_energy)

            if isinstance(x, dict):
                x_flat, unravel_fn = ravel_pytree(x)
            else:
                x_flat, unravel_fn = x, identity

            pot_energy = self._potential_fn(x)
            x_new, key_new, pot_energy_new = jax.lax.fori_loop(
                0, n, single_Px, (x, key, pot_energy)
            )
            return unravel_fn(x_new) if isinstance(x, dict) else x_new

        r_keys = random.split(rng_key, n_samples)
        if isinstance(x, dict):
            samples = vmap(single_Pnx, in_axes=(None, 0))(x, r_keys)
        else:
            samples = vmap(
                vmap(single_Pnx, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None), out_axes=0
            )(x, r_keys)
        return samples

    def get_init_adapt_state(self, rng_key, init_params, model_args=(), model_kwargs={}):
        """Return the first adapt state after initialization."""
        num_warmup = 0
        init_state = self.init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        return init_state.adapt_state
