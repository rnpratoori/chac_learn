# chac_learn: Discovering Continuum Thermodynamics via Differentiable Phase-Field Models

A physics-informed machine learning framework for discovering free energy functionals directly from spatiotemporal simulation or Molecular Dynamics (MD) data, by embedding neural networks inside a differentiable finite element solver.

> **Research context:** This codebase supports the manuscript *"Discovering Continuum Thermodynamics from Molecular Dynamics via Differentiable Phase-Field Models"* (target: *Journal of Computational Physics*, *npj Computational Materials*, *Computer Methods in Applied Mechanics and Engineering*, *Acta Materialia*).

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Method Overview](#method-overview)
3. [Branch Overview](#branch-overview)
4. [Repository Structure](#repository-structure)
5. [Dependencies and Installation](#dependencies-and-installation)
6. [Quickstart](#quickstart)
7. [Detailed Usage](#detailed-usage)
   - [Step 1 — Generate reference data](#step-1--generate-reference-data)
   - [Step 2 — Train the model](#step-2--train-the-model)
   - [Step 3 — Reproduce plots](#step-3--reproduce-plots)
8. [Command-line Arguments](#command-line-arguments)
9. [Physics and Governing Equations](#physics-and-governing-equations)
10. [Neural Network Architecture](#neural-network-architecture)
11. [Loss Functions](#loss-functions)
12. [File Formats and Data](#file-formats-and-data)
13. [Outputs](#outputs)
14. [Experiment Tracking with WandB](#experiment-tracking-with-wandb)
15. [Known Limitations and Future Work](#known-limitations-and-future-work)

---

## Scientific Background

Phase-field models are widely used in materials science to simulate mesoscale microstructural evolution (e.g., phase separation, solidification, crystallization). Their predictive power depends critically on the accuracy of the free energy functional *f(c, η)*, where *c* is concentration and *η* is a crystalline order parameter.

Deriving free energy functionals from first principles or from Molecular Dynamics (MD) trajectories is a long-standing challenge. Traditional coarse-graining approaches require restrictive assumptions about the functional form of the energy. This framework removes those assumptions by directly learning *∂f/∂c* and *∂f/∂η* from spatiotemporal field data.

---

## Method Overview

The framework couples three components:

1. **A Firedrake FEM solver** for the coupled Cahn-Hilliard / Allen-Cahn (CH-AC) system — a set of coupled nonlinear PDEs that govern concentration and crystallinity evolution.

2. **A neural network (MLP)** that replaces the analytical free energy derivatives *∂f/∂c* and *∂f/∂η*. The NN takes a (c, η) point as input and predicts the two partial derivatives.

3. **`firedrake-adjoint`** for exact gradient propagation backward through the entire PDE time-integration loop. This allows the NN weights to be trained end-to-end by comparing the simulated fields to target (MD or synthetic) data.

**Key innovations:**

- **Adjoint-based learning:** exact gradients through O(100) PDE solves per epoch with no approximation.
- **Integrability constraint:** enforces ∂(∂f/∂c)/∂η = ∂(∂f/∂η)/∂c, guaranteeing that the learned derivatives are consistent with a scalar potential.
- **Spectral (FFT) loss:** compares fields in Fourier space to capture morphological features and characteristic length scales rather than pointwise values.

---

## Branch Overview

The repository has three branches, each targeting a different spatial dimensionality. All branches share the same neural network architecture, training loop logic, and loss formulation — only the mesh, initial conditions, and a few numerical details differ.

| Branch | Mesh | Status |
|---|---|---|
| `main` | 1D `IntervalMesh(100, 1)` | Stable reference implementation |
| `2d-simulation` | 2D `RectangleMesh(N, N, L, L)` | Active development — includes GPU support and auto-weighting |
| `3d-simulation` | 3D `BoxMesh(N, N, N, L, L, L)` | Experimental — mesh size is very coarse by default |

### `main` — 1D baseline

The foundational branch. The domain is a 1D interval [0, 1] with 100 uniform cells. It is the simplest case and serves as the validation testbed against known analytical Flory-Huggins potentials. Key characteristics:

- **Mesh:** `IntervalMesh(100, 1)` — 100 cells, 101 DOFs per field.
- **Initial condition for *c*:** sinusoidal perturbation `0.5 + 0.2·sin(πx/4)` evaluated at DOF positions.
- **Initial condition for *η*:** step-function nucleus of width `4λ_η` centred on the domain.
- **FFT loss:** uses `torch.fft.fft` (1D FFT); optional truncation of high-frequency modes via `--truncation-modes`.
- **GPU:** not active (hardcoded to CPU in `training_utils.py`).
- **η loss weight:** set manually via `--eta-loss-weight`.
- **DOF mapping:** requires a coordinate-based index remapping when loading VTU target data into Firedrake, because the VTU point ordering is not guaranteed to match the Firedrake DOF ordering in parallel.

---

### `2d-simulation` — 2D extension (active)

Extends the framework to a 2D square domain. This is the primary development branch and contains several improvements that have not yet been back-ported to `main`.

- **Mesh:** `RectangleMesh(N_mesh, N_mesh, L_domain, L_domain)` — configurable via `--N-mesh` and `--L-domain` arguments.
- **Initial condition for *c*:** random noise around mean 0.5 with amplitude 0.05 (fixed seed 42); produces a realistic spinodal decomposition starting point.
- **Initial condition for *η*:** 2D circular disc nucleus (`conditional(dist ≤ r, 1, 0)`) using UFL `SpatialCoordinate` — avoids the NumPy coordinate-loop used in `main`.
- **FFT loss:** upgraded to `torch.fft.fftn` (n-dimensional FFT); the same code path works for 1D, 2D, and 3D fields without modification.
- **GPU support:** active — `setup_device()` selects CUDA if available; all PyTorch tensors are moved to the selected device.
- **Auto-weighting of η loss:** a calibration pass over the first few timesteps is run before training begins. It measures the raw *c* and *η* FFT losses under the initial NN predictions and sets `eta_loss_weight` automatically so that the two loss terms start at comparable magnitudes (with a minimum of 1.0). This removes the need to tune `--eta-loss-weight` by hand.
- **VTU loading:** simplified — because `ch_ac.py` and `learn_dfdc.py` both create the identical `RectangleMesh`, the VTU point order matches the Firedrake DOF order exactly. The coordinate-remapping logic present in `main` is replaced by a direct assignment plus a one-time mesh-size assertion.
- **Compute parameters:** uses `dt = 1e-4` (2× larger than `main`) and 100 timesteps by default, keeping wall-clock time manageable on 2D meshes. N1 = N2 = 3 instead of 4.
- **HPC compatibility:** environment variable exports (`OMP_NUM_THREADS`, etc.) are moved to the very top of `learn_dfdc.py`, before any library imports, to prevent threading conflicts with MPI on cluster environments.

---

### `3d-simulation` — 3D extension (experimental)

Extends the solver to a 3D cubic domain. This branch is in an early experimental state; the default mesh is intentionally very coarse.

- **Mesh:** `BoxMesh(N_mesh, N_mesh, N_mesh, L_domain, L_domain, L_domain)` — default `N_mesh = 1` (effectively a single hexahedral element); must be increased for any meaningful simulation.
- **Initial condition for *c*:** 3D sinusoidal perturbation `0.5 + 0.2·sin(πx)·sin(πy)·sin(πz)` using UFL operators.
- **Initial condition for *η*:** 3D spherical nucleus (`conditional(dist ≤ r, 1, 0)`) using `SpatialCoordinate(mesh)` with three coordinates `(x, y, z)`.
- **FFT loss:** same `torch.fft.fftn` upgrade as `2d-simulation`.
- **GPU support:** CUDA detection code is present but commented out — left as CPU-only for now due to memory pressure on 3D meshes.
- **VTU loading:** same direct-assignment approach as `2d-simulation` (no coordinate remapping), with the mesh-equality assertion updated to reference `BoxMesh`.
- **Physics parameters:** same defaults as `main` (N1 = N2 = 4, dt = 5×10⁻⁵, 200 timesteps); only the mesh geometry changes.

---

### Choosing a branch

```
Validating against a known analytical potential → main (1D, fastest)
Spinodal decomposition / 2D microstructure patterns → 2d-simulation
Exploring 3D geometry or preparing for MD coupling → 3d-simulation
```

---

## Repository Structure

```
chac_learn/
│
├── ch_ac.py                  # Generate 1D CH-AC reference data (standalone solver)
├── simulation.py             # CHSolver class + target data loader (used by training)
├── learn_dfdc.py             # Main training script
├── training_utils.py         # Argument parsing, device setup, checkpointing
├── plotting.py               # Visualization utilities (Matplotlib, Plotly)
├── checkpoint.py             # Save/load model checkpoints
├── reproduce_plots.py        # Post-processing: regenerate plots from .npz archive
│
├── models/
│   └── dfdc.py               # FEDerivative neural network (MLP, PyTorch)
│
├── ch_ac_1/                  # Reference simulation data (200 VTU timesteps)
│   └── ch_ac_1_*.vtu
│
├── ch_learn_adjoint/         # Training simulation outputs (VTU + PVD)
│
├── post_processing_data.npz  # Archived training history + NN surface evaluations
├── ch_learn_dfdc.pth         # Saved model checkpoint
├── MANUSCRIPT_OUTLINE.md     # Research manuscript outline
└── .gitignore
```

---

## Dependencies and Installation

This project runs inside the **Firedrake** virtual environment. Firedrake must be installed separately following the [official instructions](https://www.firedrakeproject.org/download.html). All other Python packages are installed into the same environment.

### Core requirements

| Library | Role |
|---|---|
| [Firedrake](https://www.firedrakeproject.org/) | FEM solver and mesh I/O |
| [firedrake-adjoint](https://www.firedrakeproject.org/adjoint.html) | Adjoint AD through PDE time loop |
| [PyTorch](https://pytorch.org/) | Neural network, AD, optimization |
| [NumPy](https://numpy.org/) | Arrays, FFT |
| [SciPy](https://scipy.org/) | Numerical integration (`cumulative_trapezoid`) |
| [Matplotlib](https://matplotlib.org/) | Static plots (loss curves, field comparisons) |
| [Plotly](https://plotly.com/python/) | Interactive 3D surface and space-time plots |
| [PyVista](https://pyvista.org/) | Reading VTU files into NumPy arrays |
| [WandB](https://wandb.ai/) | Experiment tracking (optional but recommended) |

### Installing additional packages (inside the Firedrake venv)

```bash
source /path/to/firedrake/bin/activate
pip install torch numpy scipy matplotlib plotly pyvista wandb
```

---

## Quickstart

```bash
# Activate Firedrake environment
source /path/to/firedrake/bin/activate

# 1. Generate 1D reference data (creates ch_ac_1/ directory with 200 VTU files)
python ch_ac.py

# 2. Train the model (default: 10000 epochs, cosine LR schedule, all defaults)
python learn_dfdc.py

# 3. Reproduce post-processing plots from saved .npz archive
python reproduce_plots.py
```

---

## Detailed Usage

### Step 1 — Generate reference data

`ch_ac.py` runs a standalone 1D Cahn-Hilliard / Allen-Cahn solve and writes 200 VTU snapshots to `ch_ac_1/`.

```bash
python ch_ac.py
```

**What it does:**
- Creates a 1D mesh with 100 cells on the interval [0, 1].
- Sets up a Flory-Huggins free energy with the parameters listed in [Physics and Governing Equations](#physics-and-governing-equations).
- Advances the solution for 200 implicit-explicit time steps (dt = 5×10⁻⁵).
- Writes `c` (Volume Fraction) and `η` (Crystallinity) to `ch_ac_1/ch_ac_1_N.vtu` for N = 0…199.
- Writes a ParaView collection file `ch_ac_1.pvd`.

The generated data acts as the synthetic "ground truth" that the NN will learn from, in place of MD data.

---

### Step 2 — Train the model

`learn_dfdc.py` is the main entry point for training.

```bash
# Default run (resume from checkpoint if one exists)
python learn_dfdc.py

# Fresh run ignoring any existing checkpoint
python learn_dfdc.py --no-resume

# Custom hyperparameters
python learn_dfdc.py \
    --epochs 5000 \
    --learning-rate 5e-4 \
    --scheduler cosine \
    --warmup-epochs 200 \
    --eta-loss-weight 2.0 \
    --integrability-weight 0.5 \
    --integrability-start-epoch 500 \
    --seed 42

# Disable WandB logging
python learn_dfdc.py --no-wandb

# Quick debug run (2 epochs only)
python learn_dfdc.py --profile
```

**Training loop per epoch:**
1. The FE solver advances the CH-AC system one full simulation using the current NN predictions for *∂f/∂c* and *∂f/∂η*.
2. An FFT-based loss compares simulated fields against target snapshots in Fourier space.
3. An integrability loss penalizes the thermodynamic inconsistency of the predicted derivatives.
4. `firedrake-adjoint` computes adjoint sensitivities through all PDE solves.
5. PyTorch backpropagation updates the NN weights via the Adam optimizer.
6. The learning rate scheduler (cosine or plateau) adjusts the step size.
7. A checkpoint is saved periodically.

---

### Step 3 — Reproduce plots

After training, `reproduce_plots.py` reads `post_processing_data.npz` and generates interactive Plotly HTML files.

```bash
python reproduce_plots.py
```

**Output directory:** `reproduced_plots/` containing:
- `dfdc_phase_space_*.html` — Predicted vs. target *∂f/∂c* surface over (c, η) phase space with epoch slider.
- `dfdeta_phase_space_*.html` — Same for *∂f/∂η*.
- `free_energy_phase_space_*.html` — Reconstructed free energy *f(c, η)* (integrated from NN derivatives).
- `simulation_spacetime_*.html` — Space-time 3D plots of *c(x, t)* and *η(x, t)*.

---

## Command-line Arguments

All arguments are passed to `learn_dfdc.py`. Run `python learn_dfdc.py --help` for the full list.

### Training

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 10000 | Number of training epochs |
| `--learning-rate` | 1e-3 | Initial learning rate |
| `--resume-lr` | — | Override learning rate when resuming from checkpoint |
| `--seed` | 12 | Random seed for reproducibility |
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `plateau`, or `none` |
| `--warmup-epochs` | 100 | Epochs for linear warmup before cosine decay |
| `--no-resume` | — | Ignore existing checkpoint, start fresh |
| `--no-wandb` | — | Disable Weights & Biases logging |
| `--profile` | — | Debug mode — runs only 2 epochs |
| `--output-dir` | — | Custom output directory for artifacts |
| `--cpu` | — | Force CPU (PyTorch); GPU is not yet default |
| `--data-index` | 1 | Reference data directory index (loads `ch_ac_<N>/`) |

### Loss weighting

| Argument | Default | Description |
|---|---|---|
| `--truncation-modes` | 0 | Number of high-frequency FFT modes to discard (0 = all modes) |
| `--eta-loss-weight` | 1.0 | Multiplier on the *η* (crystallinity) FFT loss term |
| `--integrability-weight` | 1.0 | Multiplier on the integrability constraint loss |
| `--integrability-start-epoch` | 0 | Epoch at which the integrability loss is activated |

### Physics parameters

| Argument | Default | Description |
|---|---|---|
| `--chi` | 1.0 | Flory-Huggins χ_aa interaction parameter |
| `--chi-ac` | 1.0 | Coupling parameter χ_ac between c and η |
| `--N1` | 4.0 | Degree of polymerization (species 1) |
| `--N2` | 4.0 | Degree of polymerization (species 2) |
| `--Weta` | 1.0 | Barrier height for the double-well crystallinity potential |
| `--z` | 0.5 | Shape parameter for the crystallinity potential |
| `--z0` | 1.0 | Reference crystallinity for the coupling term |
| `--T` | 0.1 | Total simulation time |
| `--dt` | 5e-5 | Time step |
| `--M` | 1.0 | Mobility coefficient (Cahn-Hilliard) |
| `--lmbda` | 5e-2 | Interface width parameter λ (concentration) |
| `--lmbda-eta` | 5e-2 | Interface width parameter λ_η (crystallinity) |
| `--L` | -100 | Allen-Cahn kinetic coefficient |

---

## Physics and Governing Equations

### Cahn-Hilliard equation (conserved dynamics)

The concentration field *c* evolves as:

```
∂c/∂t = ∇·(M ∇μ)
μ = ∂f/∂c − λ ∇²c
```

where *μ* is the chemical potential, *M* is the mobility, and *λ* is the gradient energy coefficient controlling interface width.

### Allen-Cahn equation (non-conserved dynamics)

The crystalline order parameter *η* evolves as:

```
∂η/∂t = L (∂f/∂η − λ_η ∇²η)
```

where *L < 0* is the kinetic coefficient (negative for relaxation dynamics).

### Free energy functional (Flory-Huggins / double-well)

The bulk free energy used to generate reference data:

```
f(c, η) = f_mix(c) + c · f_cr(η) + f_cp(c, η)
```

- **f_mix(c):** Flory-Huggins mixing free energy with parameters χ, N1, N2.
- **f_cr(η):** Double-well crystallinity potential with barrier height W_η.
- **f_cp(c, η):** Coupling term with parameters χ_ac, z, z0.

The neural network learns the partial derivatives of this functional — not the functional itself — and the free energy surface is reconstructed post hoc by numerical integration.

### Initial conditions

- Concentration *c*: sinusoidal perturbation around *c = 0.5*.
- Crystallinity *η*: step function creating a crystalline nucleus in the center of the domain.

### Finite element discretization

- **Mesh:** 1D `IntervalMesh` with 100 cells (extendable to 2D/3D).
- **Function spaces:** Lagrange P1 elements for both *c* and *η*.
- **Time integration:** Implicit-explicit (IMEX) scheme.
- **Linear solver:** Direct LU factorization (MUMPS).

---

## Neural Network Architecture

**File:** [models/dfdc.py](models/dfdc.py)

The `FEDerivative` model is a compact multilayer perceptron (MLP):

```
Input:  (c, η)  — 2 scalar features
Hidden: Linear(2 → 200) + LeakyReLU
Output: (∂f/∂c, ∂f/∂η)  — 2 scalar outputs
```

**Constraint on *∂f/∂c*:** The output for *∂f/∂c* is mean-subtracted (zero mean enforced) at inference time. Because *c* is a conserved field, adding a constant to *∂f/∂c* does not affect the dynamics; the zero-mean constraint removes this degeneracy.

**Why this architecture works:** The target free energy derivatives are smooth functions of *(c, η)* in the physically relevant range [0, 1] × [0, 1], so a single hidden layer with sufficient width (200 neurons) is adequate. Deeper networks were found to be unnecessary for this problem class.

---

## Loss Functions

The total loss is a weighted sum of three terms:

### 1. FFT data fidelity loss (concentration)

```
L_c = ‖FFT(c_sim) − FFT(c_target)‖²
```

Computes the squared difference of Fourier-transformed concentration fields, summed over all timesteps. High-frequency modes can optionally be truncated via `--truncation-modes` to focus on dominant morphological features.

### 2. FFT data fidelity loss (crystallinity)

```
L_η = η_weight × ‖FFT(η_sim) − FFT(η_target)‖²
```

Same as above for the crystallinity field, weighted by `--eta-loss-weight`.

### 3. Integrability (thermodynamic consistency) loss

```
L_int = int_weight × ‖∂(∂f/∂c)/∂η − ∂(∂f/∂η)/∂c‖²
```

Evaluated on a grid of *(c, η)* points sampled from [0, 1]². Second-order mixed derivatives are computed by automatic differentiation through the NN. This loss activates at epoch `--integrability-start-epoch` and guarantees that the discovered *(∂f/∂c, ∂f/∂η)* pair is the gradient of a single scalar potential.

### Total loss

```
L = L_c + L_η + L_int
```

---

## File Formats and Data

### Reference simulation data (`ch_ac_1/`)

- **Format:** VTK Unstructured Grid (`.vtu`, XML)
- **Files:** `ch_ac_1_0.vtu` through `ch_ac_1_199.vtu` (200 timesteps)
- **Fields per file:**
  - `Volume Fraction` — concentration *c* ∈ [0, 1]
  - `Crystallinity` — order parameter *η* ∈ [0, 1]
- **Readable with:** ParaView, PyVista, or the `load_target_data()` function in `simulation.py`

### Model checkpoint (`ch_learn_dfdc.pth`)

PyTorch checkpoint containing:
- `model_state_dict` — MLP weights
- `optimizer_state_dict` — Adam first/second moment estimates
- `scheduler_state_dict` — LR scheduler state
- `epoch_losses`, `epoch_losses_c`, `epoch_losses_eta`, `epoch_losses_int` — loss histories up to the saved epoch
- `epoch_numbers` — corresponding epoch indices

### Post-processing archive (`post_processing_data.npz`)

NumPy compressed archive written by `learn_dfdc.py` at the end of training:

| Key | Shape / Type | Description |
|---|---|---|
| `preds_collection` | list | Simulated fields at final timestep, per epoch |
| `epochs_collection` | list | Epoch indices |
| `target_final_global` | array | Reference *c* and *η* at final timestep |
| `all_epochs_comparison_data` | object array | Timestep-by-timestep comparison data |
| `all_nn_outputs` | array | NN *∂f/∂c*, *∂f/∂η* on 50×50 (c, η) grid, per epoch |
| `c_values_nn` | array (50,) | *c* grid coordinates |
| `eta_values_nn` | array (50,) | *η* grid coordinates |
| `epoch_losses` | array | Total loss per epoch |
| `epoch_losses_c` | array | Concentration FFT loss per epoch |
| `epoch_losses_eta` | array | Crystallinity FFT loss per epoch |
| `epoch_losses_int` | array | Integrability loss per epoch |
| `epoch_numbers` | array | Epoch indices |
| Physics constants | scalars | χ, χ_ac, N1, N2, W_η, z, z0, T, dt, M, λ, λ_η, L |

---

## Outputs

| File / Directory | Generated by | Description |
|---|---|---|
| `ch_ac_1/` | `ch_ac.py` | 200 VTU snapshots of reference simulation |
| `ch_ac_1.pvd` | `ch_ac.py` | ParaView collection pointing to reference VTU files |
| `ch_learn_adjoint/` | `learn_dfdc.py` | VTU snapshots from the final training epoch |
| `ch_learn_adjoint.pvd` | `learn_dfdc.py` | ParaView collection for training run outputs |
| `ch_learn_dfdc.pth` | `learn_dfdc.py` | Latest model checkpoint |
| `post_processing_data.npz` | `learn_dfdc.py` | Full training history and NN surface archive |
| `lve_dfdc.png` | `learn_dfdc.py` | Loss vs. epoch plot (all loss components) |
| `lve_true.png` | `learn_dfdc.py` | Loss computed against analytical ground truth |
| `reproduced_plots/*.html` | `reproduce_plots.py` | Interactive Plotly visualizations |
| `wandb/` | WandB SDK | Experiment logs and metadata (gitignored) |

---

## Experiment Tracking with WandB

Training metrics, hyperparameters, and model artifacts are logged to [Weights & Biases](https://wandb.ai) by default. To use WandB:

1. Install: `pip install wandb`
2. Log in: `wandb login`
3. Run training (WandB is active unless `--no-wandb` is passed).

Logged quantities per epoch:
- `loss/total`, `loss/c`, `loss/eta`, `loss/integrability`
- `learning_rate`
- All command-line hyperparameters (logged once at init)

Model checkpoints are saved as WandB artifacts when `wandb` is active.

---

## Known Limitations and Future Work

- **1D only (current):** Reference data generation and the solver are implemented for 1D. Extension to 2D/3D meshes is a planned next step.
- **CPU-only training:** PyTorch computations run on CPU. GPU acceleration is not yet integrated because the adjoint loop runs on the Firedrake/PETSc side.
- **Synthetic data only (current):** The pipeline has been validated on Firedrake-generated synthetic data. Coupling to real MD trajectories (mapping atom positions to continuum fields) is described in the manuscript outline but not yet implemented.
- **Single hidden layer:** The MLP architecture is minimal by design for smooth potentials. Deeper or wider networks may be needed for more complex free energy landscapes.
- **Noise handling:** The loss formulation does not explicitly account for MD noise. Regularization strategies for noisy targets are a planned addition.

---

**Keywords:** Molecular Dynamics · Phase-Field Modeling · Scale Bridging · Adjoint Methods · Differentiable Physics · Thermodynamics Discovery · Cahn-Hilliard · Allen-Cahn · Finite Element Method · Firedrake · Physics-Informed Machine Learning
