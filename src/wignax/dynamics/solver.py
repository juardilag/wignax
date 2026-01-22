import jax
import jax.numpy as jnp
import diffrax
from tqdm.auto import tqdm
from typing import Callable, Tuple, Any, Optional

from wignax.core.state import PhaseSpaceState

Time = float
DriftFunc = Callable[[Time, Any, Any], Any]  

def solve(
    drift_func: DriftFunc,
    initial_state: PhaseSpaceState,
    t_eval: jax.Array,
    args: Any = None,
    batch_size: int = 1024,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves the Closed System TWA equations using direct complex arithmetic.

    Args:
        drift_func: Deterministic EOM f(t, y, args).
        initial_state: PhaseSpaceState object.
        t_eval: Time points to save.
        args: Static arguments.
        batch_size: Trajectories per batch.

    Returns:
        mean_traj: Average path (len(t_eval), ...).
        var_traj: Variance (len(t_eval), ...).
    """
    
    # 1. CONFIGURE SOLVER
    # We use Dopri5. Diffrax handles complex numbers by treating them 
    # as a single variable. The PIDController norm might be slightly 
    # less efficient than on real vectors, but usually works fine.
    term = diffrax.ODETerm(drift_func)
    solver = diffrax.Dopri5()
    
    # Standard controller. If this fails on complex norms, 
    # we can try stepsize_controller=diffrax.ConstantStepSize()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)

    # 2. BATCH RUNNER (JIT)
    @jax.jit
    def run_single_batch(state_batch_samples):
        
        # A. Define the single trajectory solver
        def solve_one(y0):
            sol = diffrax.diffeqsolve(
                term, 
                solver, 
                t0=t_eval[0], 
                t1=t_eval[-1], 
                dt0=0.01,
                y0=y0, 
                args=args, 
                saveat=diffrax.SaveAt(ts=t_eval),
                stepsize_controller=stepsize_controller,
                max_steps=100000
            )
            return sol.ys # Shape (Time, ModeShape...)

        # B. Vectorize
        batch_results = jax.vmap(solve_one)(state_batch_samples)

        # C. Reduce (Online Statistics)
        # Sum of the values (Complex)
        batch_sum = jnp.sum(batch_results, axis=0)
        
        # Sum of the squares of the magnitude (Real)
        # Necessary for variance calculation: E[|x|^2]
        batch_sq_sum = jnp.sum(jnp.abs(batch_results)**2, axis=0)
        
        return batch_sum, batch_sq_sum

    # 3. EXECUTION LOOP
    n_total = initial_state.n_samples
    num_batches = int(jnp.ceil(n_total / batch_size))
    
    # Initialize Accumulators
    # We infer shape from the first sample
    sample_shape = jax.eval_shape(lambda: initial_state.samples[0])
    acc_shape = (len(t_eval),) + sample_shape.shape
    
    total_sum = jnp.zeros(acc_shape, dtype=sample_shape.dtype) # Complex
    total_sq_sum = jnp.zeros(acc_shape, dtype=jnp.float32)     # Real

    print(f"Wignax Simulation: {n_total} trajectories (Direct Complex Mode)")

    for i in tqdm(range(num_batches), desc="Simulating"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        actual_batch_size = end_idx - start_idx
        
        # Slice batch
        batch_samples = jax.lax.dynamic_slice_in_dim(
            initial_state.samples, start_idx, actual_batch_size
        )

        b_sum, b_sq_sum = run_single_batch(batch_samples)
        
        total_sum = total_sum + b_sum
        total_sq_sum = total_sq_sum + b_sq_sum

    # 4. FINALIZE
    mean_traj = total_sum / n_total
    
    # Variance of the magnitude (standard for TWA population plots)
    # Var = E[|x|^2] - |E[x]|^2
    var_traj = (total_sq_sum / n_total) - jnp.abs(mean_traj)**2
    
    return mean_traj, var_traj