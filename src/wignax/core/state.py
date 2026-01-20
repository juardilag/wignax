import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class PhaseSpaceState:
    """
    A JAX-compatible container for TWA phase-space trajectories.
    
    Attributes:
        samples (jnp.ndarray): The bundle of trajectories.
                               Shape: (n_samples, n_modes)
                               Type:  Complex (complex64 or complex128)
    """
    def __init__(self, samples):
        # We ensure it is always a JAX array
        self.samples = jnp.asarray(samples)

    @property
    def n_samples(self):
        """Number of independent trajectories (the 'swarm' size)."""
        return self.samples.shape[0]

    @property
    def n_modes(self):
        """Number of quantum modes."""
        return self.samples.shape[1]

    def __repr__(self):
        return f"PhaseSpaceState(n_samples={self.n_samples}, n_modes={self.n_modes})"

    # --- JAX PyTree Registration Methods ---
    def tree_flatten(self):
        """Destroys the class to arrays"""
        children = (self.samples,)  # The arrays JAX needs to trace/differentiate
        aux_data = None             # Static data (strings, ints) that don't change
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuilds the class from the arrays."""
        (samples,) = children
        return cls(samples)