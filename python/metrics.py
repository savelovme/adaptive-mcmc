from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, vmap, jit
import flax.linen as nn
import optax


# from AMH import sample_Px

#
# def spectral_norm(W, num_power_iters=10, eps=1e-12):
#     """
#     Apply spectral normalization to weight matrix W.
#
#     Args:
#         W: Weight matrix of shape (out_features, in_features).
#         num_power_iters: Number of power iterations for approximating spectral norm.
#         eps: Small constant to avoid division by zero.
#
#     Returns:
#         W_sn: Spectrally normalized weight matrix.
#     """
#     # Flatten weight to (out_features, in_features)
#     W_shape = W.shape
#     W = W.reshape(W_shape[0], -1)
#
#     rng_key = random.fold_in(random.PRNGKey(0), W[0,0])
#     # Initialize singular vectors u, v
#     u = jax.random.normal(rng_key, (W.shape[0],))  # Random unit vector
#     u /= jnp.linalg.norm(u)
#
#     # Power iteration for dominant singular value
#     for _ in range(num_power_iters):
#         v = jnp.dot(W.T, u)
#         v /= jnp.linalg.norm(v) + eps
#         u = jnp.dot(W, v)
#         u /= jnp.linalg.norm(u) + eps
#
#     # Compute spectral norm (largest singular value)
#     sigma = jnp.dot(u, jnp.dot(W, v))
#
#     # sigma = jnp.linalg.matrix_norm(W, ord=2)
#
#     # Normalize weight matrix
#     W_sn = W / (sigma + eps)
#
#     # Reshape back to original shape
#     return W_sn.reshape(W_shape)
#
#
# class SpectralNormDense(nn.Module):
#     features: int
#     use_bias: bool = True
#
#     @nn.compact
#     def __call__(self, x):
#         W = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
#         W_sn = spectral_norm(W)  # Apply spectral normalization
#         x = jnp.dot(x, W_sn)
#
#         if self.use_bias:
#             b = self.param('bias', nn.initializers.zeros, (self.features,))
#             x += b
#
#         return x
#
#
# class LipschitzNN(nn.Module):
#     dim: int  # Input dimension
#     num_features: int = 64
#
#     @nn.compact
#     def __call__(self, x):
#         x = SpectralNormDense(self.num_features)(x)
#         x = nn.leaky_relu(x)
#
#         x = SpectralNormDense(self.num_features)(x)
#         x = nn.leaky_relu(x)
#
#         x = SpectralNormDense(1)(x)  # Single output for scalar function
#         return x.squeeze()




# # Define the Lipschitz Neural Network
# class LipschitzNN(nn.Module):
#     """
#     Neural network with enforced 1-Lipschitz constraint through spectral normalization.
#     """
#     num_features: int = 32  # Number of features in hidden layers
#     dim: int = 2       # Input dimension
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(self.num_features, kernel_init=nn.initializers.lecun_normal())(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.num_features, kernel_init=nn.initializers.lecun_normal())(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.num_features, kernel_init=nn.initializers.lecun_normal())(x)
#         x = nn.tanh(x)
#         x = nn.Dense(1, kernel_init=nn.initializers.lecun_normal())(x)
#         return x.squeeze()


# def apply_spectral_norm(params): # , spectral_norm_states, n_iterations=10, eps=1e-12):
#     """
#     Apply spectral normalization to all weight matrices in params.
#
#     Args:
#         params: Model parameters
#         spectral_norm_states: Current spectral norm vectors
#         n_iterations: Number of power iterations
#         eps: Small value for numerical stability
#
#     Returns:
#         new_params: Updated parameters
#         new_spectral_norm_states: Updated spectral norm states
#     """
#
#     def normalize_layer(param):
#         if param.ndim == 2:
#             sigma = jnp.linalg.matrix_norm(param, ord=2)
#             W_normalized = param / jnp.maximum(sigma, 1.0)
#             return W_normalized
#         return param
#
#     updated = jax.tree.map(normalize_layer, params) #, spectral_norm_states)
#
#     # # Unzip the pytree: extract the first elements (updated parameters)
#     # new_params = jax.tree_map(lambda t: t[0], updated)
#     # # And extract the second elements (updated spectral norm states)
#     # new_spectral_norm_states = jax.tree_map(lambda t: t[1], updated)
#
#     return updated

class LipschitzOperator(nn.Module):

    @nn.compact
    def __call__(self, x):
        W = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], 1))
        W_norm = W / jnp.sqrt(W.T @ W)
        x = jnp.dot(x, W_norm)

        return x.squeeze()

# def init_lipschitz_model(rng, input_dim):
#     """
#     Initialize the LipschitzNN model with parameters and spectral norm states.
#
#     Args:
#         rng: JAX PRNG key
#         input_dim: Dimension of the input data
#
#     Returns:
#         model: Initialized LipschitzNN instance
#         params: Model parameters
#         spectral_norm_states: Dictionary of spectral norm vectors
#     """
#     model = LipschitzNN(dim=input_dim)
#     params = model.init(rng, jnp.ones((1, input_dim)))  # Initialize model parameters
#
#     # params = apply_spectral_norm(params)
#     print("Model initialized!")
#
#     return model, params


# Distance Matrix Computation
def distance_matrix(x, y, p=2.0):
    """
    Compute pairwise distances between points in x and y using p-norm.
    
    Args:
        x: Array of shape (n_points_x, dim)
        y: Array of shape (n_points_y, dim)
        p: The power for the p-norm distance (default: 2.0 for Euclidean)
    
    Returns:
        Array of shape (n_points_x, n_points_y) containing pairwise distances
    """
    x = jnp.expand_dims(x, 1)  # Shape: (n_points_x, 1, dim)
    y = jnp.expand_dims(y, 0)  # Shape: (1, n_points_y, dim)
    return jnp.power(jnp.sum(jnp.abs(x - y) ** p, axis=-1), 1/p)


# Wasserstein Contraction Coefficient
def compute_wasserstein_contraction(
    sample_Px: Callable,
    rng_key,
    X,
    sample_batch_size=1000,
    n_train_batches=10,
    n_eval_batches=100,
    alpha=10,
    max_steps=100,
    lr=0.01,
):
    """
    Compute the Wasserstein contraction coefficient tau(P) numerically.

    Args:
        sample_Px:  Sample function with arguments rng_key, x, n_samples
        rng_key: JAX PRNG key
        X: Array of shape (n_points, d), initial points to evaluate contraction
        sample_batch_size: int Batch size for sampling (per point)
        n_train_batches: int Number of batches for training
        n_eval_batches: int Number of batches for final eval
        max_steps: Number of training max_steps
        lr: Learning rate
        alpha: Parameter for smooth max

    Returns:
        tau: Estimated Wasserstein contraction coefficient
    """

    n_points, dim = X.shape
    threshold = 1e-6

    # Pre-compute distance matrix
    dists = jnp.linalg.norm(X[:, None] - X[None, :], axis=-1)
    quantile = 2 * dim / n_points
    lower_bound = jnp.maximum(2 * jnp.quantile(dists, quantile), threshold)
    upper_bound = jnp.sqrt(dim) * lower_bound + threshold
    mask = (lower_bound <= dists) & (dists <= upper_bound)

    # Initialize model
    rng_key, key_func = random.split(rng_key)
    model = LipschitzOperator()
    params = model.init(rng_key, jnp.ones((1, dim)))

    # Loss function with batched sampling
    def loss_fn(params, rng_key):
        f = lambda x: model.apply(params, x)

        # Scan over batches
        def sample_body(carry, _):
            rng_key, Pf_batches = carry
            rng_key, subkey = random.split(rng_key)

            batch = sample_Px(subkey, X, sample_batch_size)
            Pf_batch = vmap(lambda s: jnp.mean(f(s)))(batch)  # Shape: (n_points,)
            return (rng_key, Pf_batches + Pf_batch), None

        (rng_key, Pf_batches), _ = jax.lax.scan(
            sample_body,
            (rng_key, jnp.zeros((n_points,))),
            None,
            length=n_train_batches
        )

        Pf = Pf_batches / n_train_batches
        diffs = jnp.abs(Pf[:, None] - Pf[None, :])
        safe_dists = jnp.where(mask, dists, 1.0)
        ratios = diffs / safe_dists
        safe_ratios = jnp.where(mask, ratios, 0.0)
        smooth_max = jax.nn.logsumexp(alpha * safe_ratios) / alpha
        return -smooth_max

    # Optimization step with scan
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def scan_step(carry, _):
        iter, rng_key, params, opt_state, continue_flag = carry
        rng_key, subkey = random.split(rng_key)

        # Only compute if we should continue
        def compute_step(params, opt_state, subkey):
            loss, grads = value_and_grad(loss_fn, argnums=0)(params, subkey)
            grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            grad_norm = jax.tree_util.tree_reduce(
                lambda acc, x: acc + jnp.sum(x ** 2), grads, 0.0
            )
            return new_params, opt_state, grad_norm

        # If continue_flag is True, compute; otherwise keep current state
        new_params, new_opt_state, grad_norm = jax.lax.cond(
            continue_flag,
            lambda p: compute_step(p, opt_state, subkey),
            lambda p: (p, opt_state, 0.0),
            params
        )

        # Update continue flag based on grad_norm and iteration count
        new_continue_flag = jnp.logical_and(
            grad_norm > threshold,
            iter < max_steps
        )

        return (iter + 1, rng_key, new_params, new_opt_state, new_continue_flag), None

    # Run optimization with scan and early stopping
    initial_carry = (0, rng_key, params, opt_state, True)
    final_carry, _ = jax.lax.scan(
        scan_step,
        initial_carry,
        None,
        length=max_steps
    )
    _, _, params, opt_state, _ = final_carry

    f = lambda x: model.apply(params, x)

    # Final evaluation with batches
    def eval_body(carry, _):
        rng_key, Pf_batches = carry
        rng_key, subkey = random.split(rng_key)

        batch = sample_Px(subkey, X, sample_batch_size)
        Pf_batch = vmap(lambda s: jnp.mean(f(s)))(batch)  # Shape: (n_points,)
        return (rng_key, Pf_batches + Pf_batch), None

    (rng_key, Pf_batches), _ = jax.lax.scan(
        eval_body,
        (rng_key, jnp.zeros((n_points,))),
        None,
        length=n_eval_batches
    )

    Pf = Pf_batches / n_eval_batches

    diffs = jnp.abs(Pf[:, None] - Pf[None, :])
    ratios = jnp.where(mask, diffs / dists, 0.0)
    tau = jnp.max(ratios)

    return tau, model, params

# Kernel Distance Computation
def compute_kernel_distance(sample_Px, sample_Qx, rng_key, X, n_samples=1000, steps=100, lr=0.01, alpha=10):
    """
    Compute the kernel distance rho_d(P, Q) using a 1-Lipschitz neural network.

    Args:
        sample_Px: Callable Sample function with arguments rng_key, x, n_samples
        sample_Qx: Callable Sample function with arguments rng_key, x, n_samples
        rng_key: JAX PRNG key
        kernel_P, kernel_Q: MCMC kernels
        adapt_state_P, adapt_state_Q: Adaptation states
        X: Array of shape (n_points, d)
        n_samples: Number of samples per point
        max_steps: Optimization max_steps
        lr: Learning rate
        alpha: Parameter for smooth max

    Returns:
        rho: Estimated kernel distance
    """
    dists = distance_matrix(X, X, p=2.0)  # Shape: (n_points, n_points)
    threshold = 1e-6
    nonzero_dists = dists[dists > threshold]

    dim = X.shape[-1]
    N = len(X)
    quantile = 2 * dim / N

    lower_bound = 2 * jnp.percentile(nonzero_dists, quantile) if nonzero_dists.size > 0 else threshold
    upper_bound = 2 * lower_bound
    mask = (lower_bound < dists) & (dists <= upper_bound)

    # Initialize the 1-Lipschitz neural network
    model, params = init_lipschitz_model(rng_key, dim)

    # Sampling functions
    sample_P = lambda key, x: sample_Px(rng_key=key, x=x, n_samples=n_samples)
    sample_Q = lambda key, x: sample_Qx(rng_key=key, x=x, n_samples=n_samples)

    # Define loss function to maximize kernel distance
    @jit
    def loss_fn(params, samples_P, samples_Q):
        f = lambda x: model.apply(params, x)
        Pf = vmap(lambda s: jnp.mean(f(s)))(samples_P)  # Shape: (n_points,)
        Qf = vmap(lambda s: jnp.mean(f(s)))(samples_Q)  # Shape: (n_points,)

        dPf = Pf - Qf
        diffs = jnp.abs(dPf[:, None] - dPf[None, :]) # Shape: (n_points, n_points)

        safe_dists = jnp.where(mask, dists, 1.0)
        ratios = diffs / safe_dists
        safe_ratios = jnp.where(mask, ratios, 0.0)

        smooth_max = jax.nn.logsumexp(alpha * safe_ratios, where=mask) / alpha
        return -smooth_max  # Maximize the ratio by minimizing the negative

    # Set up the optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Optimization step
    @jit
    def step(params, opt_state, samples_P, samples_Q):
        loss, grads = value_and_grad(loss_fn, argnums=0)(params, samples_P, samples_Q)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)  # Clip gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = apply_spectral_norm(params)
        return params, opt_state, loss, grads

    # Training loop
    for i in range(steps):
        if i % 10 == 0:
            rng_key, subkey = random.split(rng_key)
            samples_P = sample_P(subkey, X)
            samples_Q = sample_Q(subkey, X)

        params, opt_state, loss, grads = step(params, opt_state, samples_P, samples_Q)
        if i % 10 == 0:
            grad_norm = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x ** 2), grads, 0.0)
            print(f"Step {i}, Loss: {loss}, Grad norm: {grad_norm}")

    # Compute final kernel distance
    f = lambda x: model.apply(params, x)
    Pf = vmap(lambda s: jnp.mean(f(s)))(samples_P)
    Qf = vmap(lambda s: jnp.mean(f(s)))(samples_Q)

    dPf = Pf - Qf
    diffs = jnp.abs(dPf[:, None] - dPf[None, :])

    ratios = jnp.where(mask, diffs / dists, 0.0)
    rho = jnp.max(ratios)

    return rho, f
