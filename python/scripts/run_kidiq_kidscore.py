from posteriordb import PosteriorDatabase
import pickle
import os, sys
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from jax import random
import jax.numpy as jnp
import jax

sys.path.append("/Users/mikhail/Master/adaptive-mcmc/python")

from kernels import ARWMH, ASSS, NUTS
from utils.kernel_utils import ns_logscale, collect_states_logscale

pdb_path = "/Users/mikhail/Master/posteriordb/posterior_database"
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("kidiq-kidscore_momhsiq")

p_data = posterior.data.values()
columns=['kid_score', 'mom_hs', 'mom_iq']

data = {
    col: jnp.array(p_data[col])
    for col in columns
}

def model(mom_iq, mom_hs, kid_score=None):

    # Priors
    beta = numpyro.sample("beta", dist.ImproperUniform(dist.constraints.real_vector, (), event_shape=(3,)))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(2.5))

    # Linear model
    ones = jnp.ones_like(mom_hs)
    X = jnp.stack([ones, mom_hs, mom_iq], axis=1)
    mu = jnp.matmul(X, beta)

    # Observed variable
    numpyro.sample("kid_score_obs", dist.Normal(mu, sigma), obs=kid_score)


def run_kernel(rng_seed, kernel_str, sample_params, lr_decay):

    rng_key = random.PRNGKey(rng_seed)

    if kernel_str == "rwm":
        sampler = ARWMH(model, lr_decay=lr_decay)
    elif kernel_str == "sss":
        sampler = ASSS(model, lr_decay=lr_decay)
    elif kernel_str == "nuts":
        sampler = NUTS(model)
    mcmc = infer.MCMC(sampler, progress_bar=False, **sample_params)

    # mcmc.run(rng_key,
    #          **data,
    #          extra_fields=("potential_energy", "adapt_state", "as_change")
    # )
    # run_results = (mcmc.get_samples(), mcmc.get_extra_fields())
    states = collect_states_logscale(rng_key, sampler, data, n_pow=6)

    if lr_decay == 1:
        decay_str = "1"
    elif lr_decay == 2/3:
        decay_str = "2_3"
    elif lr_decay == 1/2:
        decay_str = "1_2"

    out_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/lr_decay/kidiq_kidscore/{kernel_str}/{decay_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f"{out_dir}/run{rng_seed}.pkl", "wb") as f:
        pickle.dump(states, f)

if __name__ == "__main__":
    for kernel_str in ["rwm", "sss"]: #, "nuts"]:
        # if kernel_str in ["rwm", "sss"]:
        #     sample_params = dict(num_warmup=10000, num_samples=100000, thinning=10)
        # elif kernel_str == "nuts":
        #     sample_params = dict(num_warmup=1000, num_samples=10000, thinning=1)
        sample_params = dict(num_warmup=0, num_samples=int(1e6), thinning=1)

        for rng_seed in range(100):
            for lr_decay in [1, 2/3, 1/2]:
                run_kernel(rng_seed, kernel_str, sample_params, lr_decay)

        print(f"{kernel_str} ready!")
