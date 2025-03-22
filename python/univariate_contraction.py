
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from jax import random, numpy as jnp, vmap, jit
from AMH import AMH, AMHAdaptState
# from evaluation import wasserstein_1d
from metrics import compute_wasserstein_contraction


svg_dir = "../img/svg/"

def run_normal():
    normal_dist = dist.Normal(0, 1)
    potential_fn = lambda x: -1 *normal_dist.log_prob(x)

    kernel = AMH(potential_fn=potential_fn)

    def get_adapt_state(cov_matrix):
        return AMHAdaptState(
            mean=jnp.array([0]),
            covariance=cov_matrix,
            log_step_size=jnp.array(0)
        )

    s1 = get_adapt_state(jnp.array([[1.]]))
    s2 = get_adapt_state(jnp.array([[1e-1]]))
    s3 = get_adapt_state(jnp.array([[1e1]]))

    rng_key = random.PRNGKey(0)

    X = jnp.arange(-10, 10, 0.1).reshape(-1, 1)
    x = X.squeeze()

    N = 10000

    def get_max_taus(n_list, adapt_state, tau_samples=100, ci=0.9):
        rng_keys = random.split(rng_key, tau_samples)
        n_vals = jnp.array(n_list)

        @jit
        def taus_dual_fn(rng_key, n):
            sample_fn = lambda rng_key, x, n_samples: kernel.sample_Pnx(rng_key, x, adapt_state, n, n_samples)
            tau, model, params = compute_wasserstein_contraction(sample_fn, rng_key, X)
            return tau

        max_taus_dual = vmap(
            vmap(taus_dual_fn, in_axes=(None, 0), out_axes=0),  # Apply over r_keys
            in_axes=(0, None), out_axes=0  # Apply over n
        )(rng_keys, n_vals)

        taus_mean = max_taus_dual.mean(axis=0)

        taus_ci_lower = jnp.quantile(max_taus_dual, (1 - ci) / 2, axis=0)
        taus_ci_upper = jnp.quantile(max_taus_dual, (1 + ci) / 2, axis=0)

        return taus_mean, taus_ci_lower, taus_ci_upper

    # def get_max_taus(n_list, adapt_state, tau_samples=100, ci=0.9):
    #     rng_keys = random.split(rng_key, tau_samples)
    #     n_vals = jnp.array(n_list)
    #
    #     def max_tau_fn(rng_key, n):
    #         Pnx = kernel.sample_Pnx(rng_key, X, adapt_state, n, n_samples=N).squeeze()
    #
    #         W_dists_n = vmap(wasserstein_1d)(Pnx[:-2], Pnx[2:])
    #         diffs = jnp.abs(x[:-2] - x[2:])
    #
    #         taus_n = W_dists_n / diffs
    #
    #         return taus_n.max()
    #
    #     @jit
    #     def tau_w_ci(n):
    #         max_taus = vmap(max_tau_fn, in_axes=(0, None))(rng_keys, n)
    #         return jnp.stack([
    #             max_taus.mean(axis=0),
    #             jnp.quantile(max_taus, (1 - ci) / 2, axis=0),
    #             jnp.quantile(max_taus, (1 + ci) / 2, axis=0)
    #         ])
    #
    #     max_taus_mean, max_taus_ci_lower, max_taus_ci_upper = vmap(tau_w_ci, out_axes=1)(n_vals)
    #
    #     return max_taus_mean, max_taus_ci_lower, max_taus_ci_upper

    n_list = [1, 5, 10, 25, 50, 100]

    max_taus_s1, max_taus_ci_lower_s1, max_taus_ci_upper_s1 = get_max_taus(n_list, s1)
    max_taus_s2, max_taus_ci_lower_s2, max_taus_ci_upper_s2 = get_max_taus(n_list, s2)
    max_taus_s3, max_taus_ci_lower_s3, max_taus_ci_upper_s3 = get_max_taus(n_list, s3)

    fig, ax1 = plt.subplots()
    # plt.title(r"$\sigma^2=1, n=5$")

    ax1.plot(n_list, max_taus_s2, color='orange', label=r"$\sigma^2 = 0.1$")
    ax1.fill_between(n_list, max_taus_ci_lower_s2, max_taus_ci_upper_s2, alpha=0.3, color="orange", label="90% CI")

    ax1.plot(n_list, max_taus_s1, color='blue', label=r"$\sigma^2 = 1$")
    ax1.fill_between(n_list, max_taus_ci_lower_s1, max_taus_ci_upper_s1, alpha=0.3, color="blue", label="90% CI")

    ax1.plot(n_list, max_taus_s3, color='red', label=r"$\sigma^2 = 10$")
    ax1.fill_between(n_list, max_taus_ci_lower_s3, max_taus_ci_upper_s3, alpha=0.3, color="red", label="90% CI")

    ax1.set_ylabel(r"contraction estimate $ \tau(P_{\sigma^2}^n)$, log scale")

    ax1.set_xlabel("power $n$")

    ax1.set_xticks(n_list)
    # ax1.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
    ax1.semilogy()

    ax1.legend(loc="lower left")

    fig.savefig(svg_dir + "univariate-contraction-decrease.svg", format="svg")


def run_mixture():
    mixing_dist = dist.Categorical(probs=jnp.array([1 / 2, 1 / 2]))
    component_dist = dist.Normal(loc=jnp.array([-1, 1]), scale=jnp.array([.1, .1]))
    mixture = dist.MixtureSameFamily(mixing_dist, component_dist)

    potential_fn = lambda x: -1 * mixture.log_prob(x)

    kernel = AMH(potential_fn=potential_fn)

    def get_adapt_state(cov_matrix):
        return AMHAdaptState(
            mean=jnp.array([0]),
            covariance=cov_matrix,
            log_step_size=jnp.array(0)
        )

    s1 = get_adapt_state(jnp.array([[1.]]))
    s2 = get_adapt_state(jnp.array([[1e-1]]))
    s3 = get_adapt_state(jnp.array([[1e1]]))

    rng_key = random.PRNGKey(0)

    X = jnp.arange(-10, 10, 0.1).reshape(-1, 1)
    x = X.squeeze()

    N = 10000

    def get_max_taus(n_list, adapt_state, tau_samples=100, ci=0.9):
        rng_keys = random.split(rng_key, tau_samples)
        n_vals = jnp.array(n_list)

        def max_tau_fn(rng_key, n):
            Pnx = kernel.sample_Pnx(rng_key, X, adapt_state, n, n_samples=N).squeeze()

            W_dists_n = vmap(wasserstein_1d)(Pnx[:-2], Pnx[2:])
            diffs = jnp.abs(x[:-2] - x[2:])

            taus_n = W_dists_n / diffs

            return taus_n.max()

        @jit
        def tau_w_ci(n):
            max_taus = vmap(max_tau_fn, in_axes=(0, None))(rng_keys, n)
            return jnp.stack([
                max_taus.mean(axis=0),
                jnp.quantile(max_taus, (1 - ci) / 2, axis=0),
                jnp.quantile(max_taus, (1 + ci) / 2, axis=0)
            ])

        max_taus_mean, max_taus_ci_lower, max_taus_ci_upper = vmap(tau_w_ci, out_axes=1)(n_vals)

        return max_taus_mean, max_taus_ci_lower, max_taus_ci_upper

    n_list = [1, 5, 10, 25, 50, 100]

    max_taus_s1, max_taus_ci_lower_s1, max_taus_ci_upper_s1 = get_max_taus(n_list, s1)
    max_taus_s2, max_taus_ci_lower_s2, max_taus_ci_upper_s2 = get_max_taus(n_list, s2)
    max_taus_s3, max_taus_ci_lower_s3, max_taus_ci_upper_s3 = get_max_taus(n_list, s3)

    fig, ax1 = plt.subplots()
    # plt.title(r"$\sigma^2=1, n=5$")

    ax1.plot(n_list, max_taus_s2, color='orange', label=r"$\sigma^2 = 0.1$")
    ax1.fill_between(n_list, max_taus_ci_lower_s2, max_taus_ci_upper_s2, alpha=0.3, color="orange", label="90% CI")

    ax1.plot(n_list, max_taus_s1, color='blue', label=r"$\sigma^2 = 1$")
    ax1.fill_between(n_list, max_taus_ci_lower_s1, max_taus_ci_upper_s1, alpha=0.3, color="blue", label="90% CI")

    ax1.plot(n_list, max_taus_s3, color='red', label=r"$\sigma^2 = 10$")
    ax1.fill_between(n_list, max_taus_ci_lower_s3, max_taus_ci_upper_s3, alpha=0.3, color="red", label="90% CI")

    ax1.set_ylabel(r"contraction estimate $ \tau(P_{\sigma^2}^n)$, log scale")

    ax1.set_xlabel("power $n$")

    ax1.set_xticks(n_list)
    # ax1.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
    ax1.semilogy()

    ax1.legend(loc="lower left")

    fig.savefig(svg_dir + "mixture-contraction-decrease.svg", format="svg")


# run_normal()
run_mixture()