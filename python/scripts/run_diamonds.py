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
from utils.kernel_utils import ns_logscale, collect_states_logscale

pdb_path = "/Users/mikhail/Master/posteriordb/posterior_database"
my_pdb = PosteriorDatabase(pdb_path)

posterior = my_pdb.posterior("diamonds-diamonds")

p_data = posterior.data.values()
data = {key: jnp.array(val) for key, val in p_data.items() if type(val) is list}

def model(Y, X):
    # Data transformation
    N, K = X.shape
    Kc = K - 1
    means_X = jnp.mean(X[:, 1:], axis=0)  # Means of columns excluding the first (intercept)
    Xc = jnp.column_stack([X[:, 0], X[:, 1:] - means_X])  # Center the predictors

    # Priors
    b = numpyro.sample("b", dist.Normal(loc=0, scale=1), sample_shape=(Kc, ))
    Intercept = numpyro.sample("Intercept", dist.StudentT(df=3, loc=8, scale=10))
    sigma = numpyro.sample("sigma", dist.FoldedDistribution(dist.StudentT(df=3, loc=0, scale=10)))
    # sigma = numpyro.deterministic("sigma", jnp.abs(sigma_base))

    # Likelihood
    # mu = numpyro.deterministic("mu", Intercept + jnp.dot(Xc[:, 1:], b)) # Linear predictor without intercept from Xc
    mu = Intercept + jnp.dot(Xc[:, 1:], b)
    numpyro.sample("Y", dist.Normal(mu, sigma), obs=Y)
#
# kernel_rwm = ARWMH(model)
# kernel_sss = ASSS(model)
# kernel_nuts = NUTS(model)

def run_kernel(rng_seed, kernel_str, sample_params, lr_decay):

    if lr_decay == 1:
        decay_str = "1"
    elif lr_decay == 2 / 3:
        decay_str = "2_3"
    elif lr_decay == 1 / 2:
        decay_str = "1_2"

    out_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/lr_decay/diamonds/{kernel_str}/{decay_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fname = f"{out_dir}/run{rng_seed}.pkl"
    if not os.path.exists(fname):

        rng_key = random.PRNGKey(rng_seed)

        if kernel_str == "rwm":
            sampler = ARWMH(model, lr_decay=lr_decay)
        elif kernel_str == "sss":
            sampler = ASSS(model, lr_decay=lr_decay)
        elif kernel_str == "nuts":
            sampler = NUTS(model)
        mcmc = infer.MCMC(sampler, progress_bar=False, **sample_params)

        states = collect_states_logscale(rng_key, sampler, data, n_pow=6)

        # mcmc.run(rng_key,
        #          **data,
        #          extra_fields=("potential_energy", "adapt_state", "as_change")
        # )
        # run_results = (mcmc.get_samples(), mcmc.get_extra_fields())
        # run_results_subset = jax.tree.map(lambda x: x[ns_logscale(6)], run_results)

        with open(fname, "wb") as f:
            pickle.dump(states, f)

if __name__ == "__main__":
    for kernel_str in ["rwm", "sss"]: # ["rwm", "sss", "nuts"]:
        # if kernel_str == "rwm":
        #     sample_params = dict(num_warmup=1000000, num_samples=10000000, thinning=1000)
        #
        # elif kernel_str == "sss":
        #     sample_params = dict(num_warmup=500000, num_samples=5000000, thinning=500)
        #
        # elif kernel_str == "nuts":
        #         sample_params = dict(num_warmup=1000, num_samples=10000, thinning=1)
        sample_params = dict(num_warmup=0, num_samples=int(1e6), thinning=1)

        for rng_seed in range(100):
            for lr_decay in [1, 2/3, 1/2]:
                run_kernel(rng_seed, kernel_str, sample_params, lr_decay)

        print(f"{kernel_str} ready!")
