import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any, Union

Time = float
State = Any  # Can be Array, Tuple, or PyTree
Args = Any   # User defined arguments (parameters)
DriftFunc = Callable[[Time, State, Args], State]


def generating_eom_bosons(
    hamiltonian_func: Callable[[Time, State, Args], float]
) -> DriftFunc:
    """
    Generates the Equation of Motion (Drift) for Bosonic systems using complex variables.

    Implements the symplectic flow for complex phase space variables \alpha:
        d(alpha)/dt = -i * dH/d(alpha*)

    JAX Implementation Note:
        When JAX computes the gradient of a real-valued function H with respect to 
        a complex variable z, it returns dH/dz* (the Wirtinger derivative conjugate).
        Therefore, grad(H) returns exactly the term we need.

    Args:
        hamiltonian_func: A function H(t, alpha, args) -> float (Real Energy).
                          'alpha' can be a JAX array or any PyTree of complex numbers.

    Returns:
        drift: A function f(t, alpha, args) -> d_alpha/dt that computes the time derivative.
    """
    # Differentiate H w.r.t the 2nd argument (alpha)
    grad_H = jax.grad(hamiltonian_func, argnums=1)

    def drift(t: Time, alpha: State, args: Args) -> State:
        # 1. Compute the gradient dH/d(alpha*)
        dH_dalpha_star = grad_H(t, alpha, args)
        
        # 2. Apply Symplectic Structure: alpha_dot = -i * (dH/d_alpha*)
        # We use tree_map to handle arbitrary structures (lists, dicts, single arrays)
        d_alpha_dt = jax.tree_util.tree_map(lambda g: -1j * g, dH_dalpha_star)
        
        return d_alpha_dt

    return drift


def generating_eom_spins(
    hamiltonian_func: Callable[[Time, State, Args], float]
) -> DriftFunc:
    """
    Generates the Equation of Motion for Classical Spin systems.

    Implements the precession equation for a spin vector S in an effective magnetic field:
        dS/dt = S x B_eff
              = S x (-dH/dS)  (Standard Physics Convention)
              = S x grad(H)   (If using the cyclic property of cross product)
    
    Actually, for H = -S.B, grad(H) = -B. 
    EOM is dS/dt = S x B.
    Substituting B = -grad(H), we get dS/dt = S x (-grad H) = (grad H) x S.
    
    Equivalent form often used in TWA: dS/dt = S x grad_S(H).

    Args:
        hamiltonian_func: A function H(t, S, args) -> float.
                          S must be an array of shape (..., 3) or a PyTree of such arrays.

    Returns:
        drift: A function f(t, S, args) -> dS/dt.
    """
    grad_H = jax.grad(hamiltonian_func, argnums=1)

    def drift(t: Time, S: State, args: Args) -> State:
        # 1. Compute the "Torque Field" or Gradient vector
        Omega_field = grad_H(t, S, args)
        
        # 2. Compute Cross Product: dS/dt = S x Omega_field
        # We assume S is an array where the last dimension is 3 (x, y, z)
        # If S is a PyTree (e.g. multiple spins), we map the cross product over leaves.
        dS_dt = jax.tree_util.tree_map(
            lambda s, o: jnp.cross(s, o), 
            S, 
            Omega_field
        )
        
        return dS_dt

    return drift


def generating_eom_quadratures(
    hamiltonian_func: Callable[[Time, Tuple[State, State], Args], float]
) -> DriftFunc:
    """
    Generates the Equation of Motion for Real Quadratures (x, p).

    Implements standard Hamilton's Equations:
        dx/dt = + dH/dp
        dp/dt = - dH/dx

    Structure Expectation:
        The state must be a Tuple (x, p).
        The Hamiltonian must accept a Tuple (x, p) as the second argument.

    Args:
        hamiltonian_func: A function H(t, (x, p), args) -> float.

    Returns:
        drift: A function f(t, (x, p), args) -> (dx_dt, dp_dt).
    """
    # Differentiate w.r.t the state tuple (x, p)
    grad_H = jax.grad(hamiltonian_func, argnums=1)

    def drift(t: Time, state: Tuple[State, State], args: Args) -> Tuple[State, State]:
        # 1. Compute Gradients
        # grad_H returns a tuple: (dH/dx, dH/dp)
        grads = grad_H(t, state, args)
        
        dh_dx, dh_dp = grads
        
        # 2. Apply Symplectic Swap
        # dx/dt = dh_dp
        # dp/dt = -dh_dx
        dx_dt = dh_dp
        dp_dt = jax.tree_util.tree_map(lambda g: -g, dh_dx)
        
        return (dx_dt, dp_dt)

    return drift