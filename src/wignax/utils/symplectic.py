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
    Generates EOMs for Bosons using Real-Symplectic gradients.
    
    Instead of trusting complex autodiff (which can swap conjugates), 
    we treat alpha = x + iy and enforce Hamilton's Real Equations:
        dx/dt = + dH/dy  * (1/2)
        dy/dt = - dH/dx  * (1/2)
    
    The factor 1/2 comes from the Wirtinger calculus relation:
    d/dt(alpha) = -i dH/d(alpha*)
    """
    
    def drift(t: Time, alpha: State, args: Args) -> State:
        # 1. Define a Real-View wrapper of the Hamiltonian locally
        def H_real(x, y):
            # Reconstruct alpha
            a = x + 1j * y
            return hamiltonian_func(t, a, args)

        # 2. Compute Real Gradients (Guaranteed correct direction)
        # We differentiate H w.r.t Real and Imag parts
        dH_dx = jax.grad(lambda x: H_real(x, alpha.imag))(alpha.real)
        dH_dy = jax.grad(lambda y: H_real(alpha.real, y))(alpha.imag)
        
        # 3. Apply Symplectic Flow (Hamilton's Equations)
        # Flow: dx/dt = +1/2 dH/dy
        #       dy/dt = -1/2 dH/dx
        # Factor 1/2 is necessary because d/dz* = 1/2 (d/dx + i d/dy)
        
        dt_x =  0.5 * dH_dy
        dt_y = -0.5 * dH_dx
        
        return dt_x + 1j * dt_y

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
            lambda s, o: jnp.cross(s, -2.0*o), 
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