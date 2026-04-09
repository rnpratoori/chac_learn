from firedrake import *

import numpy as np

# Add MPI communicator
comm = COMM_WORLD
rank = comm.rank

# Model parameters
lmbda = 5e-2
chi_aa = 1.0
chi_ac = 1.0
N1 = 4
N2 = 4
M = 1
Weta = 1
z = 0.5
z0 = 1
lmbda_eta = 5e-2
L_kinetic = -100  # Allen-Cahn kinetic coefficient (renamed to avoid clash with domain L)

# Mesh / domain parameters (always square)
N_mesh = 100   # cells per axis
L_domain = 1.0  # domain length per axis

# Simulation parameters
dt = 5e-5
T = dt * 200
outfile = VTKFile("ch_ac_1.pvd")

# Create 2D mesh
mesh = RectangleMesh(N_mesh, N_mesh, L_domain, L_domain)

# Define function space
V = FunctionSpace(mesh, "Lagrange", 1)
W = V * V * V

# Define functions
u = Function(W, name="Solution")
u_ = Function(W, name="Solution_Old")
c, mu, eta = split(u)
c_, mu_, eta_ = split(u_)

v = TestFunction(W)
c_test, mu_test, eta_test = split(v)

# ---- Initial condition -------------------------------------------------------
x, y = SpatialCoordinate(mesh)

# 2D sinusoidal IC for c (concentration)
ic_c = 0.5 + 0.2 * sin(np.pi * x) * sin(np.pi * y)
u_.sub(0).interpolate(ic_c)

# Initial condition for mu
u_.sub(1).assign(0.0)

# 2D circular disc nucleus for eta (crystallinity)
center = L_domain / 2.0
r = lmbda_eta * 4.0  # radius of nucleus
dist = sqrt((x - center)**2 + (y - center)**2)

# Use conditional(condition, true_value, false_value) for the step function
ic_eta = conditional(le(dist, r), 1.0, 0.0)
u_.sub(2).interpolate(ic_eta)

u.assign(u_)

# ---- Weak form --------------------------------------------------------------
c_var = variable(c)
eta_var = variable(eta)
f_mix = c_var * ln(c_var) / N1 + (1 - c_var) * ln(1 - c_var) / N2 + chi_aa * c_var * (1 - c_var)
f_cr = Weta * (eta_var**4 / 4 - (z + z0) * eta_var**3 / 3 + z * z0 * eta_var**2 / 2)
f_cp = chi_ac * c_var * (1 - c_var)
f = f_mix + c_var * f_cr + f_cp
dfdc = diff(f, c_var)
dfdeta = diff(f, eta_var)

F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + \
     (dt / 2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
F1 = inner(mu, mu_test) * dx - \
     inner(dfdc, mu_test) * dx - \
     lmbda**2 * dot(grad(c), grad(mu_test)) * dx
F2 = (inner(eta, eta_test) - inner(eta_, eta_test)) * dx - \
     (dt / 2) * (lmbda_eta * L_kinetic * dot(grad(eta), grad(eta_test)) +
                 L_kinetic * inner(dfdeta, eta_test)) * dx
F = F0 + F1 + F2

# Create nonlinear problem
problem = NonlinearVariationalProblem(F, u)

# ---- Output / time loop -----------------------------------------------------
t = 0.0
n = 0
outfile.write(
    project(c_, V, name="Volume Fraction"),
    project(eta_, V, name="Crystallinity"),
    time=t
)

while t < T:
    if rank == 0:
        print(f"Solving for t = {t:.6e} ...", flush=True)
    solve(F == 0, u, solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "convergence_criteria": "incremental",
        "pc_factor_mat_solver_type": "mumps"
    })
    u_.assign(u)
    t += dt
    n += 1
    outfile.write(
        project(c_, V, name="Volume Fraction"),
        project(eta_, V, name="Crystallinity"),
        time=t
    )
