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


pdb_path = "/Users/mikhail/Master/posteriordb/posterior_database"
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("eight_schools-eight_schools_noncentered")

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

def run_kernel(rng_seed, kernel_str, sample_params):

    rng_key = random.PRNGKey(rng_seed)

    if kernel_str == "rwm":
        sampler = ARWMH(model)
    elif kernel_str == "sss":
        sampler = ASSS(model)
    elif kernel_str == "nuts":
        sampler = NUTS(model)

    mcmc = infer.MCMC(sampler, progress_bar=False, **sample_params)
    mcmc.run(rng_key,
             **data,
             extra_fields=("potential_energy", "adapt_state")
    )
    run_results = (mcmc.get_samples(), mcmc.get_extra_fields())

    out_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/w_eval/eight_schools/{kernel_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f"{out_dir}/run{rng_seed}.pkl", "wb") as f:
        pickle.dump(mcmc, f)


if __name__ == "__main__":
    for kernel_str in ["rwm", "sss", "nuts"]:
        if kernel_str == "rwm":
            sample_params = dict(num_warmup=50000, num_samples=500000, thinning=50)
        elif kernel_str == "sss":
            sample_params = dict(num_warmup=25000, num_samples=250000, thinning=25)
        elif kernel_str == "nuts":
            sample_params = dict(num_warmup=10000, num_samples=100000, thinning=10)
        # sample_params = dict(num_warmup=0, num_samples=int(1e6), thinning=1)
        for rng_seed in range(100):
                run_kernel(rng_seed, kernel_str, sample_params)
        print(f"{kernel_str} ready!")
