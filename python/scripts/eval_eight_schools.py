import jax
import pandas as pd
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
from utils.evaluation import pth_moment_rmse, wasserstein_dist11_p, mmd2_unbiased, mmd_heuristic

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


reference_draws_df = pd.concat([pd.DataFrame(chain_draw).assign(chain_id=i) for i, chain_draw in enumerate(posterior.reference_draws())])

ref_mus =  jnp.array(reference_draws_df["mu"])
ref_taus = jnp.array(reference_draws_df["tau"])
ref_thetas = jnp.array(reference_draws_df[[f"theta[{i+1}]" for i in range(8)]])
ref_thetas_base = (ref_thetas - ref_mus[:, None]) / ref_taus[:, None]

y = jnp.concat(
    [
        ref_mus[:, None],
        jnp.log(ref_taus)[:, None],
        ref_thetas_base,
    ],
    axis=1
)


def eval_rows(runs_dir):

    for rng_seed in range(100):

        with open(f"{runs_dir}/run{rng_seed}.pkl", "rb") as f:
            mcmc = pickle.load(f)

        posterior_samples = mcmc.get_samples()
        mus, taus, thetas, thetas_base = jax.tree.leaves(posterior_samples)

        x = jnp.concat(
            [
                mus[:, None],
                jnp.log(taus)[:, None],
                thetas_base,
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

        runs_dir = f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/w_eval/eight_schools/{kernel_str}"

        eval_df = pd.DataFrame.from_records(data=eval_rows(runs_dir), nrows=100)

        eval_df.to_csv(f"/Users/mikhail/Master/adaptive-mcmc/python/mcmc_runs/w_eval/eight_schools/eval_{kernel_str}.csv")
