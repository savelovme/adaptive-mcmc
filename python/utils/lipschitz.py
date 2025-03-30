from typing import Callable

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, vmap, jit
import flax.linen as nn
import optax


def spectral_norm(W):
    """
    Apply spectral normalization to weight matrix W.

    Args:
        W: Weight matrix of shape (out_features, in_features)

    Returns:
        W_sn: Spectrally normalized weight matrix.
    """
    num_power_iters = 10
    eps = 1e-10

    # Flatten weight to (out_features, in_features)
    W_shape = W.shape
    W = W.reshape(W_shape[0], -1)

    rng_key = random.fold_in(random.PRNGKey(0), W[0,0])
    # Initialize singular vectors u, v
    u = jax.random.normal(rng_key, (W.shape[0],))  # Random unit vector
    u /= jnp.linalg.norm(u)
    v = jnp.zeros((W_shape[1],))

    # Power iteration for dominant singular value
    # for _ in range(num_power_iters):
    #     v = jnp.dot(W.T, u)
    #     v /= jnp.linalg.norm(v) + eps
    #     u = jnp.dot(W, v)
    #     u /= jnp.linalg.norm(u) + eps

    def body_fun(i, val):
        u, v = val

        v = jnp.dot(W.T, u)
        v /= jnp.linalg.norm(v) + eps
        u = jnp.dot(W, v)
        u /= jnp.linalg.norm(u) + eps

        return (u, v)

    u,v = jax.lax.fori_loop(lower=0, upper=num_power_iters, body_fun=body_fun, init_val=(u,v))

    # Compute spectral norm (largest singular value)
    sigma = jnp.dot(u, jnp.dot(W, v))

    # sigma = jnp.linalg.matrix_norm(W, ord=2)

    # Normalize weight matrix
    W_sn = W / jnp.clip(sigma, 1.0)

    # Reshape back to original shape
    return W_sn.reshape(W_shape)


class SpectralNormDense(nn.Module):
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        W = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        W_sn = spectral_norm(W)  # Apply spectral normalization
        x = jnp.dot(x, W_sn)

        if self.use_bias:
            b = self.param('bias', nn.initializers.zeros, (self.features,))
            x += b

        return x


class LipschitzNN(nn.Module):
    dim: int  # Input dimension
    num_features: int = 32

    @nn.compact
    def __call__(self, x):
        x = SpectralNormDense(self.num_features)(x)
        x = nn.leaky_relu(x)

        x = SpectralNormDense(self.num_features)(x)
        x = nn.leaky_relu(x)

        x = SpectralNormDense(1)(x)  # Single output for scalar function
        return x.squeeze()


# class LipschitzOperator(nn.Module):
#
#     @nn.compact
#     def __call__(self, x):
#         W = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], 1))
#         W_norm = W / jnp.sqrt(W.T @ W)
#         x = jnp.dot(x, W_norm)
#
#         return x.squeeze()



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
    lr=0.1,
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
    threshold = 1e-10

    # Pre-compute distance matrix
    dists = jnp.linalg.norm(X[:, None] - X[None, :], axis=-1)
    quantile = 2 * dim / n_points
    lower_bound = jnp.maximum(2 * jnp.quantile(dists, quantile), threshold)
    upper_bound = jnp.sqrt(dim) * lower_bound + threshold
    mask = (lower_bound <= dists) & (dists <= upper_bound)

    # Initialize model
    rng_key, key_func = random.split(rng_key)
    model = LipschitzNN(dim)
    params = model.init(rng_key, jnp.ones((1, dim)))

    # Loss function with batched sampling
    def loss_fn(params, rng_key):
        f = jit(lambda x: model.apply(params, x))

        rng_keys = random.split(rng_key, n_train_batches)

        def Pf_batch(r_key):
            batch = sample_Px(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
            Pf = vmap(lambda s: jnp.mean(f(s)))(batch)  # Shape: (n_points,)

            return Pf

        Pf = vmap(Pf_batch)(rng_keys).mean(axis=0)

        diffs = jnp.abs(Pf[:, None] - Pf[None, :])
        safe_dists = jnp.where(mask, dists, 1.0)
        ratios = diffs / safe_dists
        safe_ratios = jnp.where(mask, ratios, 0.0)
        smooth_max = jax.nn.logsumexp(alpha * safe_ratios) / alpha
        return -smooth_max

    # Optimization step with scan
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Optimization step
    @jit
    def step(params, opt_state, rng_key):
        # Compute loss and gradients only w.r.t. params (argnums=0)
        loss, grads = value_and_grad(loss_fn, argnums=0)(params, rng_key)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)  # Clip gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, grads

    # Training loop
    def body_fun(val):
        iter, rng_key, params, opt_state, grad_norm = val

        iter += 1
        rng_key, subkey = random.split(rng_key)

        params, opt_state, loss, grads = step(params, opt_state, subkey)
        grad_norm = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x ** 2), grads, 0.0)

        new_val = (iter, rng_key, params, opt_state, grad_norm)
        return new_val

    def cond_fun(val):
        iter, rng_key, params, opt_state, grad_norm = val

        return jnp.logical_and(iter < max_steps, grad_norm > threshold)

    init_val_loop = (0, rng_key, params, opt_state, 1.0)
    # val = init_val_loop
    # while(cond_fun(val)):
    #     val = body_fun(val)
    # iter, rng_key, params, opt_state, grad_norm = val
    iter, rng_key, params, opt_state, grad_norm = jax.lax.while_loop(cond_fun, body_fun, init_val_loop)


    # Final evaluation with batches
    f = jit(lambda x: model.apply(params, x))

    rng_keys = random.split(rng_key, n_eval_batches)

    def Pf_batch(r_key):
        batch = sample_Px(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
        Pf = vmap(lambda s: jnp.mean(f(s)))(batch)  # Shape: (n_points,)

        return Pf

    Pf = vmap(Pf_batch)(rng_keys).mean(axis=0)

    diffs = jnp.abs(Pf[:, None] - Pf[None, :])
    ratios = jnp.where(mask, diffs / dists, 0.0)
    tau = jnp.max(ratios)

    return tau, model, params

# Kernel Distance Computation
def compute_kernel_distance(
    sample_Px: Callable,
    sample_Qx: Callable,
    rng_key,
    X,
    sample_batch_size=1000,
    n_train_batches=10,
    n_eval_batches=100,
    alpha=10,
    max_steps=100,
    lr=0.1,
):
    """
    Compute the kernel distance rho_d(P, Q) using a 1-Lipschitz neural network.

    Args:
        sample_Px: Callable Sample function with arguments rng_key, x, n_samples
        sample_Qx: Callable Sample function with arguments rng_key, x, n_samples
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
    threshold = 1e-10

    # Pre-compute distance matrix
    dists = jnp.linalg.norm(X[:, None] - X[None, :], axis=-1)
    quantile = 2 * dim / n_points
    lower_bound = jnp.maximum(2 * jnp.quantile(dists, quantile), threshold)
    upper_bound = jnp.sqrt(dim) * lower_bound + threshold
    mask = (lower_bound <= dists) & (dists <= upper_bound)

    # Initialize model
    rng_key, key_func = random.split(rng_key)
    model = LipschitzNN(dim)
    params = model.init(rng_key, jnp.ones((1, dim)))

    # Loss function with batched sampling
    def loss_fn(params, rng_key):
        f = jit(lambda x: model.apply(params, x))

        rng_keys = random.split(rng_key, n_train_batches)

        def dPf_batch(r_key):
            batch_Px = sample_Px(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
            batch_Qx = sample_Qx(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
            Pf = vmap(lambda s: jnp.mean(f(s)))(batch_Px)  # Shape: (n_points,)
            Qf = vmap(lambda s: jnp.mean(f(s)))(batch_Qx)  # Shape: (n_points,)

            dPf = Pf - Qf

            return dPf

        dPf = vmap(dPf_batch)(rng_keys).mean(axis=0)

        diffs = jnp.abs(dPf[:, None] - dPf[None, :])
        safe_dists = jnp.where(mask, dists, 1.0)
        ratios = diffs / safe_dists
        safe_ratios = jnp.where(mask, ratios, 0.0)
        smooth_max = jax.nn.logsumexp(alpha * safe_ratios) / alpha
        return -smooth_max

    # Optimization step with scan
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Optimization step
    @jit
    def step(params, opt_state, rng_key):
        # Compute loss and gradients only w.r.t. params (argnums=0)
        loss, grads = value_and_grad(loss_fn, argnums=0)(params, rng_key)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)  # Clip gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, grads

    # Training loop
    def body_fun(val):
        iter, rng_key, params, opt_state, grad_norm = val

        iter += 1
        rng_key, subkey = random.split(rng_key)

        params, opt_state, loss, grads = step(params, opt_state, subkey)
        grad_norm = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x ** 2), grads, 0.0)

        new_val = (iter, rng_key, params, opt_state, grad_norm)
        return new_val

    def cond_fun(val):
        iter, rng_key, params, opt_state, grad_norm = val

        return jnp.logical_and(iter < max_steps, grad_norm > threshold)

    init_val_loop = (0, rng_key, params, opt_state, 1.0)
    # val = init_val_loop
    # while(cond_fun(val)):
    #     val = body_fun(val)
    # iter, rng_key, params, opt_state, grad_norm = val
    iter, rng_key, params, opt_state, grad_norm = jax.lax.while_loop(cond_fun, body_fun, init_val_loop)


    # Final evaluation with batches
    f = jit(lambda x: model.apply(params, x))

    rng_keys = random.split(rng_key, n_eval_batches)

    def dPf_batch(r_key):
        batch_Px = sample_Px(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
        batch_Qx = sample_Qx(r_key, X, sample_batch_size)  # Shape: (n_points, sample_batch_size, dim)
        Pf = vmap(lambda s: jnp.mean(f(s)))(batch_Px)  # Shape: (n_points,)
        Qf = vmap(lambda s: jnp.mean(f(s)))(batch_Qx)  # Shape: (n_points,)

        dPf = Pf - Qf

        return dPf

    dPf = vmap(dPf_batch)(rng_keys).mean(axis=0)

    diffs = jnp.abs(dPf[:, None] - dPf[None, :])
    ratios = jnp.where(mask, diffs / dists, 0.0)
    tau = jnp.max(ratios)

    return tau, model, params