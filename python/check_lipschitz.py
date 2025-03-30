# import argparse
# import os
# import glob
# import pickle
# import time
#
# import jax.numpy as jnp
# from jax import random, vmap
# from jax.scipy.interpolate import RegularGridInterpolator
#
# import numpyro.distributions as dist
#
# from AMH import AMH, sample_adapt_state, sample_neigbour, sample_Px, state_dist
#
# def sample_lipschitz_func(rng_key, domain=(-10, 10), step=0.1):
#     """
#     Generates a random 1-Lipschitz finction with given domain and grid size
#     """
#     points = jnp.arange(*domain, step)
#     N, = points.shape
#
#     u = dist.Uniform(-1, 1).sample(rng_key, sample_shape=(N,))
#     u -= u.mean()
#     u /= jnp.max(jnp.abs(u))
#
#     values = jnp.tri(N,N) @ u * step
#
#     return RegularGridInterpolator((points,), values, fill_value=0)
#
#
# def compute_lipschitz(rng_key, kernel, eps=0.001, domain=(-10, 10), step=0.01, n_samples=500):
#
#     x = jnp.arange(*domain, step)
#     key_func, rng_key = random.split(rng_key)
#     func = sample_lipschitz_func(key_func, domain, 10*step)
#
#     key_state1, key_state2, key_kernel1, key_kernel2 = random.split(rng_key, 4)
#
#     s1 = sample_adapt_state(key_state1)
#     s2 = sample_neigbour(key_state2, s1, eps)
#
#     P1x = sample_Px(key_kernel1, kernel, x, s1, n_samples=n_samples)
#     P2x = sample_Px(key_kernel2, kernel, x, s2, n_samples=n_samples)
#
#     P1fx = vmap(func)(P1x).mean(axis=0)
#     P2fx = vmap(func)(P2x).mean(axis=0)
#
#     diff = P1fx - P2fx
#
#     diff_norm = jnp.abs(jnp.diff(diff)/jnp.diff(x)).max()
#
#     return diff_norm, state_dist(s1, s2)
#
#
# def run(N, rng_seed, bs):
#     mixing_dist = dist.Categorical(probs=jnp.array([1 / 3, 2 / 3]))
#     component_dist = dist.Normal(loc=jnp.array([-5, 5]), scale=jnp.array([1, 1]))
#     mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
#
#     potential_fn = lambda x: -1 * mixture.log_prob(x)
#     kernel = AMH(potential_fn=potential_fn)
#
#     # Directory to save chunks
#     os.makedirs("lipschitz_chunks", exist_ok=True)
#
#     start_time_global = time.time()
#
#     rng_keys = random.split(random.PRNGKey(rng_seed), N)
#
#     for idx in range(0, N, bs):
#
#         batch_data = {
#             "rng_key": rng_keys[idx:idx + bs],
#             "diff_norm": [],
#             "states_dist": []
#         }
#
#         start_time = time.time()
#         for rng_key in rng_keys[idx:idx + bs]:
#             diff_norm, states_dist = compute_lipschitz(rng_key, kernel)
#
#             batch_data["diff_norm"].append(diff_norm)
#             batch_data["states_dist"].append(states_dist)
#
#         chunk_path = f"lipschitz_chunks/batch_{idx // bs}.pkl"
#         batch_data["diff_norm"] = jnp.asarray(batch_data["diff_norm"])
#         batch_data["states_dist"] = jnp.asarray(batch_data["states_dist"])
#         with open(chunk_path, "wb") as f:
#             pickle.dump(batch_data, f)
#
#         elapsed_time = time.time() - start_time
#         if elapsed_time > 7200:
#             elapsed_time_str = f"{elapsed_time / 3600:.2f} h"
#         elif elapsed_time > 120:
#             elapsed_time_str = f"{elapsed_time / 60:.2f} min"
#         else:
#             elapsed_time_str = f"{elapsed_time:.2f} s"
#         print(f"{min(idx + bs, N)} / {N} ready. Time elapsed for last iteration: {elapsed_time_str}.")
#
#     chunk_files = glob.glob("lipschitz_chunks/*.pkl")
#
#     final_data = {"rng_key": [], "diff_norm": [], "states_dist": []}
#
#     for file_path in chunk_files:
#         with open(file_path, "rb") as f:
#             batch_data = pickle.load(f)
#             final_data["rng_key"].append(batch_data["rng_key"])
#             final_data["diff_norm"].append(batch_data["diff_norm"])
#             final_data["states_dist"].append(batch_data["states_dist"])
#
#     # Final concatenation
#     final_data["rng_key"] = jnp.concatenate(final_data["rng_key"], axis=0)
#     final_data["diff_norm"] = jnp.concatenate(final_data["diff_norm"], axis=0)
#     final_data["states_dist"] = jnp.concatenate(final_data["states_dist"], axis=0)
#
#     # Save the consolidated file
#     file_path = "lipschitz.pkl"
#     with open(file_path, "wb") as f:
#         pickle.dump(final_data, f)
#
#     # Remove the chunk files after processing
#     for file_path in chunk_files:
#         os.remove(file_path)
#
#     total_elapsed_time = time.time() - start_time_global
#     if total_elapsed_time > 7200:
#         total_elapsed_time_str = f"{total_elapsed_time / 3600:.2f} h"
#     elif total_elapsed_time > 120:
#         total_elapsed_time_str = f"{total_elapsed_time / 60:.2f} min"
#     else:
#         total_elapsed_time_str = f"{total_elapsed_time:.2f} s"
#
#     print(f"Data written to {file_path}. Took {total_elapsed_time_str}.")
#
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description="Process two parameters: N and rng_seed.")
#
#     parser.add_argument(
#         "--N",
#         type=int,
#         required=True,
#         help="Number of experiments"
#     )
#     parser.add_argument(
#         "--rng_seed",
#         type=int,
#         required=True,
#         help="Random number generator seed"
#     )
#
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=10,
#         help="Size of batches for processing (default: 10)"
#     )
#
#     # Parse the arguments
#     args = parser.parse_args()
#
#     # Access the arguments
#     N = args.N
#     rng_seed = args.rng_seed
#     bs = args.batch_size
#
#     run(N, rng_seed, bs)
#
