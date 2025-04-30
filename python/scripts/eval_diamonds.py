import jax
from posteriordb import PosteriorDatabase
import pickle
import numpyro
import os, sys
import numpyro.distributions as dist
import numpyro.infer as infer
from jax import random
import jax.numpy as jnp
import pandas as pd

sys.path.append("/Users/mikhail/Master/adaptive-mcmc/python")

from kernels import ARWMH, ASSS, NUTS
from utils.kernel_utils import ns_logscale, collect_states_logscale
from utils.evaluation import pth_moment_rmse, wasserstein_dist11_p, mmd_heuristic


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

reference_draws_df = pd.concat([
    pd.DataFrame(chain_draw).assign(chain_id=i)
    for i, chain_draw in enumerate(posterior.reference_draws())
])

reference_draws = {
    "b": jnp.array(reference_draws_df[[f"b[{i}]" for i in range(1,25)]]),
    "Intercept": jnp.array(reference_draws_df["Intercept"]),
    "sigma": jnp.array(reference_draws_df["sigma"])
}

ref_intercepts = jnp.array(reference_draws_df["Intercept"])
ref_bs = jnp.array(reference_draws_df[[f"b[{i}]" for i in range(1,25)]])
ref_sigmas =  jnp.array(reference_draws_df["sigma"])

y = jnp.concat(
    [
        ref_intercepts[:, None],
        ref_bs,
        jnp.log(ref_sigmas)[:, None],
    ],
    axis=1
)

def eval_rows(runs_dir):

    for rng_seed in range(100):

        with open(f"{runs_dir}/run{rng_seed}.pkl", "rb") as f:
            mcmc = pickle.load(f)

        posterior_samples = mcmc.get_samples()

        intercepts, bs, sigmas = jax.tree.leaves(posterior_samples)

        x = jnp.concat(
            [
                intercepts[:, None],
                bs,
                jnp.log(sigmas)[:, None],
            ],
            axis=1
        )

        eval_row = {
            "rng_seed": rng_seed,
            "rmse_means": pth_moment_rmse(x, y, p=1).item(),
            "wasserstein": wasserstein_dist11_p(x, y),
            "mmd": mmd_heuristic(x, y).item(),
        }
        yield eval_row

if __name__ == "__main__":

    for kernel_str in ["rwm", "sss", "nuts"]:

        runs_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/w_eval/diamonds/{kernel_str}"

        eval_df = pd.DataFrame.from_records(data=eval_rows(runs_dir), nrows=100)

        eval_df.to_csv(f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/w_eval/diamonds/eval_{kernel_str}.csv")