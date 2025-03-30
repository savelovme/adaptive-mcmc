from time import time
import pandas as pd
import jax.numpy as jnp
from jax import random

from utils import wasserstein_dist11_p, wasserstein_sinkhorn, max_sliced_wasserstein

def run_algo(x, y, algo):
    if algo == "hungarian":
        algo_func = lambda x, y: wasserstein_dist11_p(x, y)
    elif algo == "sinkhorn":
        algo_func = lambda x, y: wasserstein_sinkhorn(x, y)
    elif algo == "sinkhorn_eps1e-2":
        algo_func = lambda x, y: wasserstein_sinkhorn(x, y, epsilon=1e-2)
    elif algo == "sinkhorn_eps1e-3":
        algo_func = lambda x, y: wasserstein_sinkhorn(x, y, epsilon=1e-3)
    elif algo == "sinkhorn_eps1e-4":
        algo_func = lambda x, y: wasserstein_sinkhorn(x, y, epsilon=1e-4)
    elif algo == "max_sliced_dir100":
        algo_func = lambda x, y: max_sliced_wasserstein(x, y, random.PRNGKey(0), n_directions=100).item()
    elif algo == "max_sliced_dir10000":
        algo_func = lambda x, y: max_sliced_wasserstein(x, y, random.PRNGKey(0), n_directions=10000).item()

    start = time()
    dist = algo_func(x, y)
    runtime = time() - start

    return dist, runtime

if __name__ == "__main__":

    start_time_global = time()

    combined_df = pd.read_csv("../diamonds-diamonds.csv")

    references = jnp.array(combined_df.where(combined_df["source"] == "reference").dropna().drop(columns="source"))
    samples = jnp.array(combined_df.where(combined_df["source"] == "samples").dropna().drop(columns="source"))

    algo_data = {"algo": [], "n": [], "d": [], "dist": [], "runtime": []}


    for algo in ["hungarian", "sinkhorn_eps1e-2", "sinkhorn_eps1e-3", "sinkhorn_eps1e-4",
                 "max_sliced_dir100", "max_sliced_dir10000"]:

        start_time_algo = time()

        for d in [5, 10, 25]:
            for n in [30, 100, 300, 1000, 3000, 10000]:
                x = references[-n:, -d:]
                y = samples[-n:, -d:]

                dist, runtime = run_algo(x, y, algo)

                algo_data["n"].append(n)
                algo_data["d"].append(d)
                algo_data["algo"].append(algo)
                algo_data["dist"].append(dist)
                algo_data["runtime"].append(runtime)

        elapsed_time_algo = time() - start_time_algo
        if elapsed_time_algo > 7200:
            elapsed_time_str = f"{elapsed_time_algo / 3600:.2f} h"
        elif elapsed_time_algo > 120:
            elapsed_time_str = f"{elapsed_time_algo / 60:.2f} min"
        else:
            elapsed_time_str = f"{elapsed_time_algo:.2f} s"

        print(f"{algo} done, took: {elapsed_time_str}")


    df = pd.DataFrame(algo_data)
    df.to_pickle("wasserstein_comparison.pkl")

    total_elapsed_time = time() - start_time_global
    if total_elapsed_time > 7200:
        total_elapsed_time_str = f"{total_elapsed_time / 3600:.2f} h"
    elif total_elapsed_time > 120:
        total_elapsed_time_str = f"{total_elapsed_time / 60:.2f} min"
    else:
        total_elapsed_time_str = f"{total_elapsed_time:.2f} s"

print("Finished in ", total_elapsed_time)