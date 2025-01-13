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
    Compute estimate of p-th moment from samples

    :param x: Array of shape (n, d)
    :param y: Array of shape (m, d)
    :param p: float 
    :return: Scalar
    """

    pth_moment_x = jnp.mean(x**p, axis=0)
    pth_moment_y = jnp.mean(y**p, axis=0)
    
    rmse = (((pth_moment_x - pth_moment_y))**2).mean()**0.5
        
    return rmse


def wasserstein_dist11_p(u_values, v_values, p=2.0):
    """
    Compute Wasserstein-p distance via optimal 1-1 coupling between samples.

    :param u_values: Array of shape (n, d)
    :param v_values: Array of shape (n, d)
    :param p: float 
    :return: Scalar value representing Wasserstein-p distance
    """

    cost_matrix = distance_matrix(u_values, v_values, p=p)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    opt_cost = cost_matrix[row_ind, col_ind].mean().item()
    return opt_cost


ot_solve_fn = jit(linear.solve)

def wasserstein_distance_ot(u_values, v_values, cost_fn=costs.Euclidean()):
    """
    Compute Wasserstein distance approximation using Sinkhorn's algorithm.
    """
    geom = pointcloud.PointCloud(x=u_values, y=v_values, cost_fn=cost_fn)
    ot = ot_solve_fn(geom)

    assert ot.converged
        
    return ot.primal_cost.item()


def wasserstein_1d(mu, nu, p=2.0):
    """
    Compute the Wasserstein-p distance between two 1D arrays.
    
    Parameters:
    - mu, nu: 1D arrays.
    
    Returns:
    - float: The Wasserstein distance in 1D.
    """

    # Compute the absolute differences
    diff = jnp.abs(jnp.sort(mu) - jnp.sort(nu))
    
    # Raise to the power of p
    diff_p = diff ** p
    
    # Compute the p-th root of the sum of differences to the power p
    return jnp.power(jnp.mean(diff_p), 1.0 / p)


@jit
def max_sliced_wasserstein(mu, nu, rng_key, p=2.0, n_directions=1000):
    """
    Approximate the Wasserstein distance using the max-sliced approach with JAX.

    Parameters:
    - mu, nu: jnp.arrays of shape (n_samples, n_dimensions) representing 
              points sampled from two distributions.
    - key: JAX PRNG key for randomness.
    - n_directions: Number of random directions to project onto.

    Returns:
    - float: An approximation of the Wasserstein distance.
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
    
    :param x: Array of shape (n, d)
    :param y: Array of shape (m, d)
    :param gamma: Bandwidth parameter for the Gaussian kernel
    :return: Jax array of shape (n, m)
    """

    eucl_dist2 = jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    
    return jnp.exp(-gamma * eucl_dist2)


@jit
def mmd2_unbiased(x, y, gamma=1.0):
    """
    Compute the unbiased MMD^2 estimator between two sets of samples.

    :param x: Array of shape (n, d)
    :param y: Array of shape (m, d)
    :param gamma: Bandwidth parameter for the Gaussian kernel
    :return: Scalar value representing the MMD^2
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