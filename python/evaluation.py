from jax import random, vmap, jit
import jax.numpy as jnp

from scipy.stats import wasserstein_distance_nd as wasserstein_scipy

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from ott.geometry import pointcloud, costs
from ott.solvers import linear


@jit
def pth_moment_rmse(x, y, p=2.0):
    """
    Compute RMSE between estimates of p-th moment from samples.

    Parameters
    ----------
    x : jnp.ndarray
        Array of shape (n, d).
    y : jnp.ndarray
        Array of shape (m, d).
    p : float
        The moment to compute, default is 2.0.

    Returns
    -------
    float
        Scalar RMSE of p-th moment estimates.

    """

    pth_moment_x = jnp.mean(x**p, axis=0)
    pth_moment_y = jnp.mean(y**p, axis=0)
    
    rmse = jnp.mean(((pth_moment_x - pth_moment_y))**2)**0.5
        
    return rmse


def wasserstein_dist11_p(u_values, v_values, ord=2.0):
    """
    Compute Wasserstein-p distance via optimal 1-1 coupling between samples.

    Parameters
    ----------
    u_values : jnp.ndarray
        Array of shape (n, d).
    v_values : jnp.ndarray
        Array of shape (n, d).
    ord : float
        Order of the used norm, default is 2.0.

    Returns
    -------
    float
        Scalar value representing Wasserstein-p distance.

    """

    cost_matrix = distance_matrix(u_values, v_values, p=ord)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    opt_cost = cost_matrix[row_ind, col_ind].mean().item()
    return opt_cost


def wasserstein_sinkhorn(u_values, v_values, cost_fn=costs.Euclidean(), epsilon=None):
    """
    Compute entropy regularized Wasserstein distance approximation using Sinkhorn's algorithm.

    Parameters
    ----------
    u_values : jnp.ndarray
        Array of shape (n, d).
    v_values : jnp.ndarray
        Array of shape (m, d).
    cost_fn : costs.CostFunction, optional
        Cost function to use, default is Euclidean.
    epsilon : float or None, optional
        Entropy regularization parameter.

    Returns
    -------
    float
        Scalar value representing the approximated Wasserstein distance.

    """

    geom = pointcloud.PointCloud(x=u_values, y=v_values, cost_fn=cost_fn, epsilon=epsilon)

    ot_solve_fn = jit(linear.solve)
    ot = ot_solve_fn(geom)

    assert ot.converged
        
    return ot.ent_reg_cost.item()


def wasserstein_sinkhorn_unbiased(u_values, v_values, cost_fn=costs.Euclidean(), epsilon=None):
    """
    Compute unbiased Wasserstein distance approximation using Sinkhorn's algorithm.

    Parameters
    ----------
    u_values : jnp.ndarray
        Array of shape (n, d).
    v_values : jnp.ndarray
        Array of shape (m, d).
    cost_fn : costs.CostFunction, optional
        Cost function to use, default is Euclidean.
    epsilon : float or None, optional
        Entropy regularization parameter.

    Returns
    -------
    float
        Scalar value representing the unbiased approximated Wasserstein distance.

    """

    Wuv = wasserstein_sinkhorn(u_values, v_values, cost_fn=cost_fn, epsilon=epsilon)
    Wuu = wasserstein_sinkhorn(u_values, u_values, cost_fn=cost_fn, epsilon=epsilon)
    Wvv = wasserstein_sinkhorn(v_values, v_values, cost_fn=cost_fn, epsilon=epsilon)

    return Wuv - (Wuu + Wvv)/2


def wasserstein_1d(mu, nu, p=1.0):
    """
    Compute the Wasserstein-p distance between two 1D arrays.

    Parameters
    ----------
    mu : jnp.ndarray
        1D array.
    nu : jnp.ndarray
        1D array.
    p : float, optional
        Order of the Wasserstein distance, default is 1.0.

    Returns
    -------
    float
        The Wasserstein distance in 1D.

    """

    # Compute the absolute differences
    diff = jnp.abs(jnp.sort(mu, axis=-1) - jnp.sort(nu, axis=-1))
        
    # Compute the p-th root of the sum of differences to the power p
    return jnp.mean(diff ** p, axis=-1) ** (1.0 / p)


# @jit
def max_sliced_wasserstein(mu, nu, rng_key, p=1.0, n_directions=1000):
    """
    Approximate the Wasserstein distance using the max-sliced approach.

    Parameters
    ----------
    mu : jnp.ndarray
        Array of shape (n, d) representing points sampled from the first distribution.
    nu : jnp.ndarray
        Array of shape (n, d) representing points sampled from the second distribution.
    rng_key : random.PRNGKey
        JAX PRNG key for randomness.
    p : float, optional
        Order of the Wasserstein distance, default is 1.0.
    n_directions : int, optional
        Number of random directions to project onto, default is 1000.

    Returns
    -------
    float
        An approximation of the Wasserstein distance.

    """
    n_dim = mu.shape[1]
    
    # Generate random directions on the unit sphere
    directions = random.normal(rng_key, (n_directions, n_dim))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    
    # Vectorizing projection and 1D Wasserstein computation over directions
    project_and_compute = vmap(lambda dir: wasserstein_1d(
        jnp.dot(mu, dir),
        jnp.dot(nu, dir),
        p=p
    ))
    
    distances = project_and_compute(directions)
    
    return jnp.max(distances)
    

def gaussian_kernel(x, y, gamma):
    """
    Compute the Gaussian kernel between vectors x and y.

    Parameters
    ----------
    x : jnp.ndarray
        Array of shape (n, d).
    y : jnp.ndarray
        Array of shape (m, d).
    gamma : float
        Bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    jnp.ndarray
        Jax array of shape (n, m).

    """

    eucl_dist2 = jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    
    return jnp.exp(-gamma * eucl_dist2)


@jit
def mmd2_unbiased(x, y, gamma=1.0):
    """
    Compute the unbiased MMD^2 estimator between two sets of samples.

    Parameters
    ----------
    x : jnp.ndarray
        Array of shape (n, d).
    y : jnp.ndarray
        Array of shape (m, d).
    gamma : float, optional
        Bandwidth parameter for the Gaussian kernel, default is 1.0.

    Returns
    -------
    float
        Scalar value representing the MMD^2.

    """
    n = x.shape[0]
    m = y.shape[0]
    
    # Compute kernel matrices
    Kxx = gaussian_kernel(x, x, gamma)
    Kyy = gaussian_kernel(y, y, gamma)
    Kxy = gaussian_kernel(x, y, gamma)

    # Diagonal elements set to 0 for the unbiased estimator
    Kxx = Kxx.at[jnp.arange(n), jnp.arange(n)].set(0)
    Kyy = Kyy.at[jnp.arange(m), jnp.arange(m)].set(0)
    
    # Compute each term in the MMD formula
    mmd2 = jnp.sum(Kxx) / (n * (n - 1)) + jnp.sum(Kyy) / (m * (m - 1)) - 2 * jnp.sum(Kxy) / (n * m)
    
    return mmd2