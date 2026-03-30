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
L = -100


# Simulation parameters
dt = 5e-5
T = dt*200
N = T/dt
outfile = VTKFile("ch_ac_1.pvd")

# Create mesh
mesh = IntervalMesh(100, 1)

# Define function space
V = FunctionSpace(mesh, "Lagrange", 1)
W = V*V*V

# Define functions
u = Function(W, name="Solution")
u_ = Function(W, name="Solution_Old")
c, mu, eta = split(u)
c_, mu_, eta_ = split(u_)

v = TestFunction(W)
c_test, mu_test, eta_test = split(v)

# Initial condition
rng = np.random.default_rng(11)
num_dofs = u.sub(0).dat.data.shape[0]
ic = np.zeros((num_dofs, 3))
ic[:, 0] = [0.5 + 0.2 * sin(pi*i/4) for i in range(num_dofs)]
ic[:, 1] = 0  # Initial condition for mu

# Initial condition for eta (crystallinity) - Step function nucleus
# Get x-coordinates of the DOFs for the V space
mesh_coords = mesh.coordinates.dat.data_ro
if mesh_coords.ndim == 1:
    x_coords = mesh_coords
else:
    x_coords = mesh_coords[:, 0]  # For 1D mesh, extract first column

# Calculate center of the domain and desired width
center_x = (mesh.coordinates.dat.data.max() + mesh.coordinates.dat.data.min()) / 2.0
nucleus_width_factor = 4.0 # User requested 4 times the interface width
width = lmbda_eta * nucleus_width_factor # Total width of the step function

# Define the start and end of the step function
start_x = center_x - width / 2.0
end_x = center_x + width / 2.0

# Create a step function for eta: 1.0 within the band, 0.0 otherwise
ic[:, 2] = 0.0 # Initialize all to 0
ic[ (x_coords >= start_x) & (x_coords <= end_x), 2 ] = 1.0

u_.sub(0).dat.data[:] = ic[:, 0]  # First component (c)
u_.sub(1).dat.data[:] = ic[:, 1]  # Second component (mu)
u_.sub(2).dat.data[:] = ic[:, 2]  # Third component (eta)
u.assign(u_)

# Define residuals
c = variable(c)
eta = variable(eta)
f_mix = c*ln(c)/N1 + (1-c)*ln(1-c)/N2 + chi_aa*c*(1-c)
f_cr = Weta*(eta**4/4 - (z+z0)*eta**3/3 + z*z0*eta**2/2)
f_cp = chi_ac*c*(1-c)
f = f_mix + c*f_cr + f_cp
dfdc = diff(f, c)
dfdeta = diff(f, eta)

F0 = (inner(c, c_test) - inner(c_, c_test)) * dx + (dt/2) * M * dot(grad(mu + mu_), grad(c_test)) * dx
F1 = inner(mu, mu_test) * dx - inner(dfdc, mu_test) * dx - lmbda**2 * dot(grad(c), grad(mu_test)) * dx
F2 = (inner(eta, eta_test) - inner(eta_, eta_test)) * dx - (dt/2) * ( lmbda_eta * L * dot(grad(eta), grad(eta_test)) + L * inner(dfdeta, eta_test) ) * dx
F = F0 + F1 + F2

# Create nonlinear problem
problem = NonlinearVariationalProblem(F, u)

# Output
t = 0.0
n = 0
outfile.write(project(c_, V, name="Volume Fraction"), project(eta_, V, name="Crystallinity"), time=t)

while (t < T):
    if rank == 0:
        print("Solving for t = ", t, "...")
    solve(F == 0, u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "convergence_criteria": "incremental", "pc_factor_mat_solver_type": "mumps"})
    u_.assign(u)
    t += dt
    n += 1
    outfile.write(project(c_, V, name="Volume Fraction"), project(eta_, V, name="Crystallinity"), time=t)
