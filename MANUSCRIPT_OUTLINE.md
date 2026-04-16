# Manuscript Outline: Learning Thermodynamic Driving Forces in Coupled Phase-Field Models

**Target Journals:** *Journal of Computational Physics*, *npj Computational Materials*, *Computer Methods in Applied Mechanics and Engineering*.

---

## 1. Title Ideas
*   **Primary:** Learning Thermodynamic Driving Forces in Coupled Phase-Field Models via Differentiable Physics.
*   **Alternative 1:** Data-Driven Discovery of Free Energy Functionals for Cahn-Hilliard/Allen-Cahn Systems using Adjoint Methods.
*   **Alternative 2:** Physics-Informed Neural Networks for Recovering Thermodynamics in Phase Separation and Crystallization.

## 2. Abstract
*   **Problem Statement:** Accurate phase-field modeling of complex materials (e.g., polymers, alloys) is limited by the need for phenomenological free energy functionals that are difficult to derive.
*   **Methodology:** We propose a framework that embeds a Neural Network (NN) within a differentiable Finite Element solver. The NN parameterizes the free energy derivatives ($df/dc$ and $df/d\eta$).
*   **Key Innovation:** Use of the discrete adjoint method for exact gradient propagation through time, combined with an integrability constraint to ensure thermodynamic consistency.
*   **Results:** The framework successfully recovers the underlying thermodynamics of a coupled Cahn-Hilliard/Allen-Cahn system from spatiotemporal snapshots, maintaining morphological accuracy and physical validity.

## 3. Introduction
*   **Background:** The role of phase-field models in materials science for simulating microstructural evolution.
*   **The Challenge:** The "Free Energy Problem"—how to define the driving forces for multi-physics systems where analytical forms are unknown or overly simplified.
*   **Objectives:** To demonstrate a machine learning approach that is not a "black box" but is instead "physics-embedded," ensuring the learned model obeys conservation laws and thermodynamic principles.

## 4. Mathematical and Computational Framework
*   **Governing Equations:** 
    *   Cahn-Hilliard for conserved concentration ($c$).
    *   Allen-Cahn for non-conserved crystallinity ($\eta$).
    *   The coupled weak form used in the Firedrake implementation.
*   **Neural Network Architecture:** 
    *   Input: Local state $(c, \eta)$.
    *   Output: Predicted derivatives $(\frac{\partial f}{\partial c}, \frac{\partial f}{\partial \eta})$.
    *   Implementation details: MLP with double-precision weights.
*   **The Differentiable Solver:** 
    *   Spatial discretization: Finite Element Method (Lagrange P1 elements).
    *   Temporal discretization: Semi-implicit Euler scheme.
    *   Adjoint Logic: Utilizing `firedrake-adjoint` to compute sensitivities of the loss function with respect to NN parameters.

## 5. Training Formulation and Loss Functions
*   **Spectral (FFT) Loss:** 
    *   Mathematical formulation of the Fourier-space loss.
    *   Justification: capturing length scales and morphological features better than L2 loss in real space.
*   **Thermodynamic Consistency (Integrability):** 
    *   Explanation of the mixed partials constraint: $\frac{\partial}{\partial \eta}(\frac{\partial f}{\partial c}) = \frac{\partial}{\partial c}(\frac{\partial f}{\partial \eta})$.
    *   Implementation as a regularization term in the loss function.
*   **Calibration:** Automatic selection of `eta_loss_weight` to balance the scales of conserved and non-conserved variables.

## 6. Results and Discussion
*   **Experimental Setup:** 2D simulation details (100x100 mesh, spinodal decomposition with nucleation).
*   **Convergence Analysis:** Reduction of weighted vs. true loss over training epochs.
*   **Morphological Accuracy:** Visual comparison of $c$ and $\eta$ evolution between ground truth and neural-network-driven simulations.
*   **Landscape Recovery:** Visualization of the learned $df/dc$ and $df/d\eta$ surfaces compared to the analytical Flory-Huggins/Double-well forms.

## 7. Conclusion
*   Summary of the framework's ability to discover physics from data.
*   Discussion on scalability to 3D systems and robustness to noisy experimental data (e.g., TEM/AFM images).

---
**Keywords:** Phase-Field Modeling, Cahn-Hilliard, Allen-Cahn, Adjoint Methods, Differentiable Physics, Machine Learning, Thermodynamics.
