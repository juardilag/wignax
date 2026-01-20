import jax
import jax.numpy as jnp
from typing import Union

KeyArray = jax.Array

def sample_spin_discrete(
    key: KeyArray,
    n_samples: int,
    n_spins: int = 1,
    initial_z: Union[float, jax.Array] = -1.0
) -> jax.Array:
    """
    Generates initial spin vectors using Discrete Truncated Wigner Approximation (DTWA).
    
    Instead of Gaussian noise, this samples the transverse components from a 
    discrete distribution (+1 or -1) to respect the eigenvalues of Pauli matrices.
    
    Args:
        key: JAX PRNGKey.
        n_samples: Number of trajectories.
        n_spins: Number of spins (e.g., atoms in the lattice).
        initial_z: The initial polarization of the spin (-1.0 for Down, +1.0 for Up).
                   Can be broadcastable to (n_spins,).

    Returns:
        jnp.ndarray: Array of shape (n_samples, n_spins, 3).
                     Last dimension corresponds to [Sx, Sy, Sz].
    """
    k1, k2 = jax.random.split(key)
    
    shape = (n_samples, n_spins)
    
    # 1. Sample Transverse Components (Quantum Fluctuations)
    # Bernoulli(0.5) gives 0 or 1.
    # We map: 0 -> -1, 1 -> +1.
    # This simulates the uncertainty of Sx and Sy when the spin is pointing along Z.
    sx = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=shape).astype(jnp.float32) - 1.0
    sy = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=shape).astype(jnp.float32) - 1.0
    
    # 2. Set Longitudinal Component (Mean Field value)
    # In TWA, the variable aligned with the mean field has zero noise initially.
    sz = jnp.full(shape, initial_z, dtype=jnp.float32)
    
    # 3. Stack into a single tensor (n_samples, n_spins, 3)
    # axis=-1 ensures the last dimension holds the vector components
    s_init = jnp.stack([sx, sy, sz], axis=-1)
    
    return s_init