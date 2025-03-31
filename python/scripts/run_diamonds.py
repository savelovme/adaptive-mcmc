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

kernel_rwm = ARWMH(model)
kernel_sss = ASSS(model)
kernel_nuts = NUTS(model)

def run_kernel(rng_seed, kernel_str, sample_params):

    rng_key = random.PRNGKey(rng_seed)

    if kernel_str == "rwm":
        mcmc = infer.MCMC(kernel_rwm, progress_bar=False, **sample_params)
    elif kernel_str == "sss":
        mcmc = infer.MCMC(kernel_sss, progress_bar=False, **sample_params)
    elif kernel_str == "nuts":
        mcmc = infer.MCMC(kernel_nuts, progress_bar=False, **sample_params)

    mcmc.run(rng_key,
             **data,
             extra_fields=("potential_energy", "adapt_state")
    )

    out_dir = f"../mcmc_runs/diamonds/{kernel_str}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f"{out_dir}/run{rng_seed}.pkl", "wb") as f:
        pickle.dump(mcmc, f)

if __name__ == "__main__":
    for kernel_str in ["rwm", "sss", "nuts"]:
        if kernel_str in ["rwm", "sss"]:
            sample_params = dict(num_warmup=1000000, num_samples=1000000, thinning=100)
        elif kernel_str == "nuts":
            sample_params = dict(num_warmup=1000, num_samples=10000, thinning=1)

        for rng_seed in range(100):
            run_kernel(rng_seed, kernel_str, sample_params)

        print(f"{kernel_str} ready!")
