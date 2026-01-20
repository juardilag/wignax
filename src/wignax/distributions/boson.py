import jax
import jax.numpy as jnp
from typing import Optional, Union, Tuple

KeyArray = jax.Array

def sample_coherent(
    key: KeyArray,
    n_samples: int,
    n_modes: int = 1,
    alpha: Union[complex, jax.Array] = 0.0
) -> jax.Array:
    """
    Samples initial states from the Wigner distribution of a Coherent State.

    Generates random samples for 'n_modes' independent bosonic modes.
    The noise corresponds to vacuum fluctuations in the Wigner representation:
    <|delta_alpha|^2> = 1/2.

    Args:
        key: A JAX PRNGKey for reproducibility.
        n_samples: Number of independent trajectories (Monte Carlo samples).
        n_modes: Number of bosonic modes (e.g., number of cavities or phonons).
        alpha: The mean field amplitude(s). Can be a single complex number
               or an array of shape (n_modes,). Default is 0.0 (Vacuum).

    Returns:
        jnp.ndarray: Array of shape (n_samples, n_modes) with complex type.
    """
    k1, k2 = jax.random.split(key)
    
    # Standard Wigner width for vacuum is sigma = 0.5 for both Re and Im parts
    # Re(alpha) ~ N(Re(alpha_0), 1/4)
    # Im(alpha) ~ N(Im(alpha_0), 1/4)
    # 0.5 * N(0,1) gives std_dev = 0.5, variance = 0.25. Correct.
    
    noise_real = 0.5 * jax.random.normal(k1, shape=(n_samples, n_modes))
    noise_imag = 0.5 * jax.random.normal(k2, shape=(n_samples, n_modes))
    
    noise = noise_real + 1j * noise_imag
    
    # Broadcast alpha across samples
    return alpha + noise


def sample_quadratures(
    key: KeyArray,
    n_samples: int,
    n_modes: int = 1,
    x0: Union[float, jax.Array] = 0.0,
    p0: Union[float, jax.Array] = 0.0
) -> Tuple[jax.Array, jax.Array]:
    """
    Samples initial x and p quadratures directly.
    
    Useful for optomechanics or continuous variable systems where H(x, p) 
    is used instead of H(a, adag).
    
    Args:
        key: JAX PRNGKey.
        n_samples: Number of trajectories.
        n_modes: Number of modes.
        x0: Initial mean position.
        p0: Initial mean momentum.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - x: Shape (n_samples, n_modes)
            - p: Shape (n_samples, n_modes)
    """
    k1, k2 = jax.random.split(key)

    # Vacuum variance for x and p is 1/2.
    # Std Dev = sqrt(1/2) = 1/sqrt(2) approx 0.707
    scale = 1.0 / jnp.sqrt(2.0)

    noise_x = scale * jax.random.normal(k1, shape=(n_samples, n_modes))
    noise_p = scale * jax.random.normal(k2, shape=(n_samples, n_modes))

    return (x0 + noise_x), (p0 + noise_p)