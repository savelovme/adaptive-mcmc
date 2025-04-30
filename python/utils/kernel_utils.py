import jax
import jax.numpy as jnp
from functools import partial

from numpyro.util import fori_collect


def ns_logscale(n_pow=6):
    return jnp.concat([
        jnp.arange(0 if p < 1 else 10 ** (p - 1), 10 ** p, 10 ** (max(0, p - 2))) + 10 ** (max(0, p - 2))
        for p in range(n_pow + 1)
    ])

def concat_trees(trees):
    treedef = jax.tree.structure(trees[0])
    leaves_list = [jax.tree.leaves(tree) for tree in trees]
    concatenated_leaves = [jnp.concatenate(leaves) for leaves in zip(*leaves_list)]
    return jax.tree.unflatten(treedef, concatenated_leaves)

def collect_states_logscale(rng_key, sampler, model_data: dict, n_pow=6):
    last_state = sampler.init(rng_key, num_warmup=0, init_params={}, model_args=(), model_kwargs=model_data)

    step_fn = partial(sampler.sample, model_args=(), model_kwargs=model_data)

    collections = []
    for p in range(n_pow + 1):
        lower_idx = 0 if p < 1 else 10 ** (p - 1)
        upper_idx = 10 ** p
        states, last_state = fori_collect(
            0, upper_idx - lower_idx, step_fn, last_state,
            thinning=10 ** (max(0, p - 2)), progbar=False, return_last_val=True
        )
        collections.append(states)

    states = concat_trees(collections)
    # states = states._replace(z=sampler.postprocess_fn(args=(), kwargs=model_data)(states.z))

    return states