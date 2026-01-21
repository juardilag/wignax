"""
Wignax: Phase-space quantum many-body dynamics with JAX.

Wignax is a library for simulating quantum systems using the Truncated Wigner 
Approximation (TWA). It leverages JAX for automatic differentiation, 
hardware acceleration (GPU/TPU), and massively parallel trajectory simulations.
"""

# 1. Metadata
__version__ = "0.1.0"
__author__ = "Juan E. Ardila-Garcia"
__license__ = "Apache-2.0"

# 2. Expose the Core Container
from .core.state import PhaseSpaceState

# 3. Expose the Solver
from .dynamics.solver import solve

# 4. Expose the Physics/Symplectic Engine
# We group these under 'symplectic' or just expose them directly.
# Exposing them directly is usually more convenient for users.
from .utils.symplectic import (
    generating_eom_bosons,
    generating_eom_spins,
    generating_eom_quadratures
)

# 5. Expose Distributions (Initial States)
# Users will use these often, so keeping them accessible is good.
from .distributions.boson import sample_coherent, sample_quadratures
from .distributions.spin import sample_spin_discrete

# 6. Define the Public API
# This controls what happens if someone types "from wignax import *"
__all__ = [
    "PhaseSpaceState",
    "solve",
    "generating_eom_bosons",
    "generating_eom_spins",
    "generating_eom_quadratures",
    "sample_coherent",
    "sample_quadratures",
    "sample_spin_discrete",
]