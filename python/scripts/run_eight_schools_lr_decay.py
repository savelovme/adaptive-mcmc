import jax
from posteriordb import PosteriorDatabase
import pickle
import numpyro
import os, sys
import numpyro.distributions as dist
import numpyro.infer as infer
from jax import random
import jax.numpy as jnp

sys.path.append("/Users/mikhail/Master/adaptive-mcmc/python")

from kernels import ARWMH, ASSS, NUTS
from utils.kernel_utils import collect_states_logscale


pdb_path = "/Users/mikhail/Master/posteriordb/posterior_database"
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("eight_schools-eight_schools_centered")

p_data = posterior.data.values()
data = {key: jnp.array(val) for key, val in p_data.items() if type(val) is list}


def model(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', len(sigma)):
        with numpyro.handlers.reparam(config={'theta': infer.reparam.TransformReparam()}):
            theta = numpyro.sample(
                'theta',
                dist.TransformedDistribution(dist.Normal(0,1), dist.transforms.AffineTransform(mu, tau))
            )
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)


kernel_rwm = ARWMH(model)
kernel_sss = ASSS(model)
kernel_nuts = NUTS(model)


def run_kernel(rng_seed, kernel_str, lr_decay):

    rng_key = random.PRNGKey(rng_seed)

    if kernel_str == "rwm":
        sampler = ARWMH(model, lr_decay=lr_decay)
    elif kernel_str == "sss":
        sampler = ASSS(model, lr_decay=lr_decay)
    elif kernel_str == "nuts":
        sampler = NUTS(model)

    states = collect_states_logscale(rng_key, sampler, data, n_pow=6)

    if lr_decay == 1:
        decay_str = "1"
    elif lr_decay == 2 / 3:
        decay_str = "2_3"
    elif lr_decay == 1 / 2:
        decay_str = "1_2"

    out_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/lr_decay/eight_schools/{kernel_str}/{decay_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f"{out_dir}/run{rng_seed}.pkl", "wb") as f:
        pickle.dump(states, f)


if __name__ == "__main__":
    for kernel_str in ["rwm", "sss"]:
        for rng_seed in range(100):
            for lr_decay in [1, 2 / 3, 1 / 2]:
                run_kernel(rng_seed, kernel_str, lr_decay)
        print(f"{kernel_str} ready!")
