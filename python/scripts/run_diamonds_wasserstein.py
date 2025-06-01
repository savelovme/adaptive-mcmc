import jax
from posteriordb import PosteriorDatabase
import pickle
import numpyro
import os, sys
import numpyro.distributions as dist
import numpyro.infer as infer
from jax import random
import jax.numpy as jnp

sys.path.append(f"{os.environ['MCMC_WORKDIR']}/python")

from kernels import ARWMH, ASSS, NUTS

pdb_path = f"{os.environ['MCMC_WORKDIR']}/posteriordb/posterior_database"
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
    out_dir = f"{os.environ['MCMC_WORKDIR']}/python/mcmc_runs/w_eval/diamonds/{kernel_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f"{out_dir}/run{rng_seed}.pkl", "wb") as f:
        pickle.dump(mcmc, f)

if __name__ == "__main__":
    for kernel_str in ["rwm", "sss", "nuts"]:
        if kernel_str == "rwm":
            sample_params = dict(num_warmup=1000000, num_samples=10000000, thinning=1000)

        elif kernel_str == "sss":
            sample_params = dict(num_warmup=500000, num_samples=5000000, thinning=500)

        elif kernel_str == "nuts":
                sample_params = dict(num_warmup=1000, num_samples=10000, thinning=1)

        for rng_seed in range(100):
            for lr_decay in [1, 2/3, 1/2]:
                run_kernel(rng_seed, kernel_str, sample_params, lr_decay)

        print(f"{kernel_str} ready!")
