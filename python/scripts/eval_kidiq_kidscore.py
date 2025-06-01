import pandas as pd
from posteriordb import PosteriorDatabase
import pickle
import os, sys
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from jax import random
import jax.numpy as jnp
import jax

os.path.join(os.environ["MCMC_WORKDIR"], "python")
sys.path.append(f"{os.environ['MCMC_WORKDIR']}/python")

from kernels import ARWMH, ASSS, NUTS
from utils.evaluation import pth_moment_rmse, wasserstein_dist11_p, mmd2_unbiased, mmd_heuristic


pdb_path = f"{os.environ['MCMC_WORKDIR']}/posteriordb/posterior_database"
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


reference_draws_df = pd.concat([
    pd.DataFrame(chain_draw).assign(chain_id=i)
    for i, chain_draw in enumerate(posterior.reference_draws())
])

ref_beta = jnp.array(reference_draws_df[["beta[1]", "beta[2]", "beta[3]"]])
ref_sigma = jnp.array(reference_draws_df["sigma"])

y = jnp.concat([ref_beta, jnp.log(ref_sigma)[:, None]], axis=1)

def eval_rows(runs_dir):

    for rng_seed in range(100):

        with open(f"{runs_dir}/run{rng_seed}.pkl", "rb") as f:
            mcmc = pickle.load(f)

        posterior_samples = mcmc.get_samples()

        betas, sigmas = jax.tree.leaves(posterior_samples)

        x = jnp.concat(
            [
                betas,
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

        runs_dir = f"{os.environ['MCMC_WORKDIR']}/python/mcmc_runs/w_eval/kidiq_kidscore/{kernel_str}"

        eval_df = pd.DataFrame.from_records(data=eval_rows(runs_dir), nrows=100)

        eval_df.to_csv(f"{os.environ['MCMC_WORKDIR']}/python/mcmc_runs/w_eval/kidiq_kidscore/eval_{kernel_str}.csv")