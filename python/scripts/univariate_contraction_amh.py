import jax
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from jax import random, numpy as jnp, vmap, jit
from kernels_ import ARWMH, ARWMHAdaptState
# from evaluation import wasserstein_1d
from metrics import compute_wasserstein_contraction


svg_dir = "../../img/svg/"

normal_dist = dist.Normal(0, 1)
potential_fn = lambda x: -1 * normal_dist.log_prob(x)

kernel_normal = ARWMH(potential_fn=potential_fn)


mixing_dist = dist.Categorical(probs=jnp.array([1 / 2, 1 / 2]))
component_dist = dist.Normal(loc=jnp.array([-1, 1]), scale=jnp.array([.1, .1]))
mixture = dist.MixtureSameFamily(mixing_dist, component_dist)

potential_fn = lambda x: -1 * mixture.log_prob(x)

kernel_mixture = ARWMH(potential_fn=potential_fn)


def get_adapt_state(cov_matrix):
    return ARWMHAdaptState(
        mean=jnp.array([0]),
        covariance=cov_matrix,
        log_step_size=jnp.array(0)
    )

def run_kernel(kernel, svg_out_name="univariate-contraction-decrease.svg"):

    s1 = get_adapt_state(jnp.array([[1.]]))
    s2 = get_adapt_state(jnp.array([[1e-1]]))
    s3 = get_adapt_state(jnp.array([[1e1]]))

    rng_key = random.PRNGKey(0)

    X = jnp.arange(-10, 10, 0.1).reshape(-1, 1)
    x = X.squeeze()

    def get_max_taus(n_list, adapt_state):

        @jit
        def max_tau_fn(n):
            sample_fn = lambda rng_key, x, n_samples: kernel.sample_Pnx(rng_key, x, adapt_state, n=n, n_samples=n_samples)
            tau, _, _ = compute_wasserstein_contraction(sample_fn, rng_key, X, n_eval_batches=500)
            return tau

        return [max_tau_fn(n) for n in n_list]

    n_list = [1, 10, 20, 30, 40, 50]

    max_taus_s1 = get_max_taus(n_list, s1)
    max_taus_s2 = get_max_taus(n_list, s2)
    max_taus_s3 = get_max_taus(n_list, s3)

    fig, ax1 = plt.subplots()
    # plt.title(r"$\sigma^2=1, n=5$")

    ax1.plot(n_list, max_taus_s2, ".-", color='orange', label=r"$\sigma^2 = 0.1$")
    # ax1.fill_between(n_list, max_taus_ci_lower_s2, max_taus_ci_upper_s2, alpha=0.3, color="orange", label="90% CI")

    ax1.plot(n_list, max_taus_s1, ".-", color='blue', label=r"$\sigma^2 = 1$")
    # ax1.fill_between(n_list, max_taus_ci_lower_s1, max_taus_ci_upper_s1, alpha=0.3, color="blue", label="90% CI")

    ax1.plot(n_list, max_taus_s3, ".-", color='red', label=r"$\sigma^2 = 10$")
    # ax1.fill_between(n_list, max_taus_ci_lower_s3, max_taus_ci_upper_s3, alpha=0.3, color="red", label="90% CI")

    ax1.set_ylabel(r"contraction estimate $ \tau(P_{\sigma^2}^n)$")

    ax1.set_xlabel("power $n$")

    ax1.set_xticks(n_list)
    # ax1.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
    # ax1.semilogy()

    ax1.legend(loc="lower left")

    fig.savefig(svg_dir + svg_out_name, format="svg")

run_kernel(kernel_normal, svg_out_name="univariate-contraction-decrease-dual.svg")
run_kernel(kernel_mixture, svg_out_name="mixture-contraction-decrease-dual.svg")