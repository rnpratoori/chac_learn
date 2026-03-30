from firedrake import *
import numpy as np
import pyvista as pv



class CHSolver:
    """
    Reusable Cahn-Hilliard Allen-Cahn solver that builds forms once and reuses the solver object.
    This eliminates the UFL expression rebuilding overhead.
    Works in any spatial dimension (1D, 2D, 3D) — all UFL forms are dimension-agnostic.
    """
    
    def __init__(self, W, dt, M, lmbda, lmbda_eta=5e-2, L=-100.0):
        """
        Initialize the solver with problem parameters.
        
        Args:
            W: Mixed function space (V * V * V)
            dt: Time step size
            M: Mobility coefficient
            lmbda: Interface width parameter for c
            lmbda_eta: Interface width parameter for eta
            L: Kinetic coefficient for Allen-Cahn
        """
        self.dt = dt
        self.M = M
        self.lmbda = lmbda
        self.lmbda_eta = lmbda_eta
        self.L = L
        
        # Create functions once - these will be reused
        self.u = Function(W, name="Solution")
        self.u_ = Function(W, name="Solution_Old")
        
        # Get sub-functions
        c, mu, eta = split(self.u)
        c_, mu_, eta_ = split(self.u_)
        
        # Test functions
        v = TestFunction(W)
        c_test, mu_test, eta_test = split(v)
        
        # Placeholders for dfdc and dfdeta - will be updated each timestep
        V = W.sub(0)
        self.dfdc_f = Function(V, name="dfdc")
        self.dfdeta_f = Function(V, name="dfdeta")
        
        # Build form ONCE using the structure from ch_ac.py
        # All forms use dimension-agnostic UFL operators (inner, dot, grad, dx)
        F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + \
             (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
        
        F1 = inner(mu, mu_test) * dx - \
             inner(self.dfdc_f, mu_test) * dx - \
             lmbda**2 * dot(grad(c), grad(mu_test)) * dx
        
        F2 = (inner(eta, eta_test) - inner(eta_, eta_test)) * dx - \
             (dt/2) * (lmbda_eta * L * dot(grad(eta), grad(eta_test)) + \
                       L * inner(self.dfdeta_f, eta_test)) * dx
        
        F = F0 + F1 + F2
        
        # Create solver ONCE with direct linear solver
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        
        problem = NonlinearVariationalProblem(F, self.u)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
        
        print("CHSolver initialized - forms built once, solver ready for reuse", flush=True)
    
    def solve_step(self, u_old, dfdc_f, dfdeta_f, u_target):
        """
        Solve one timestep.
        
        Args:
            u_old: Previous solution (Function)
            dfdc_f: Neural network prediction for df/dc (Function)
            dfdeta_f: Neural network prediction for df/deta (Function)
            u_target: Target solution Function to update
            
        Returns:
            Updated solution (Function)
        """
        # Update data in existing Functions (no form rebuilding!)
        self.u_.assign(u_old)
        self.dfdc_f.assign(dfdc_f)
        self.dfdeta_f.assign(dfdeta_f)
        
        # Solve (reuses compiled form and solver)
        self.solver.solve()
        
        # Copy result to target
        u_target.assign(self.u)
        
        return u_target
    
    def get_dfdc_function(self):
        """Return the dfdc Function for use with adjoint."""
        return self.dfdc_f


def solve_one_step(u_old, dfdc_f, u, c, mu, c_test, mu_test, dt, M, lmbda):
    """
    Original solve function (kept for backward compatibility).
    Consider using CHSolver class for better performance.
    """
    u_ = Function(u.function_space(), name="Solution_Old")
    u_.assign(u_old)
    c_ = u_.sub(0)
    mu_ = u_.sub(1)

    F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
    F1 = inner(mu, mu_test) * dx - inner(dfdc_f, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx
    F = F0 + F1

    # Solve nonlinear/linear system for u (holds global Function `u`)
    solve(F == 0, u, solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    })
    return u

def load_target_data(num_timesteps, V, comm=None, rank=None, data_index=1):
    """
    Load target data from VTK files produced by ch_ac.py.

    Because both ch_ac.py and learn_dfdc.py use the exact same BoxMesh, the
    .vtu point ordering is guaranteed to match the Firedrake DOF ordering,
    so data can be assigned directly without any coordinate remapping.

    A mesh-size assertion is run once on the first file to catch any
    accidental mesh mismatch early.
    """
    print(f"Loading target from PVD (pyvista) using index {data_index}...", flush=True)
    c_target_list = []
    eta_target_list = []

    # One-time mesh equality check on the first timestep file
    first_reader = pv.get_reader(f"ch_ac_{data_index}/ch_ac_{data_index}_0.vtu")
    first_data = first_reader.read()
    n_vtu_points = first_data.n_points
    n_dofs = V.dof_count
    assert n_vtu_points == n_dofs, (
        f"Mesh mismatch: .vtu file has {n_vtu_points} points but the Firedrake "
        f"function space has {n_dofs} DOFs. Ensure both ch_ac.py and learn_dfdc.py "
        f"use the same BoxMesh(N, N, N, L, L, L)."
    )
    print(f"Mesh check passed: {n_dofs} DOFs match .vtu point count.", flush=True)

    for i in range(num_timesteps):
        reader = pv.get_reader(f"ch_ac_{data_index}/ch_ac_{data_index}_{i}.vtu")
        data = reader.read()

        c_arr = data.point_data["Volume Fraction"].astype(np.float64)
        eta_arr = data.point_data["Crystallinity"].astype(np.float64)

        f_c = Function(V, name=f"target_c_{i}")
        f_eta = Function(V, name=f"target_eta_{i}")

        # Direct assignment — same mesh topology guarantees index correspondence
        f_c.dat.data[:] = c_arr
        f_eta.dat.data[:] = eta_arr

        c_target_list.append(f_c)
        eta_target_list.append(f_eta)

    return c_target_list, eta_target_list, None