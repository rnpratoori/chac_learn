# Manuscript Outline: Discovering Continuum Thermodynamics from Molecular Dynamics via Differentiable Phase-Field Models

**Target Journals:** *Journal of Computational Physics*, *npj Computational Materials*, *Computer Methods in Applied Mechanics and Engineering*, *Acta Materialia*.

---

## 1. Title Ideas
*   **Primary:** Discovering Continuum Thermodynamics from Molecular Dynamics via Differentiable Phase-Field Models.
*   **Alternative 1:** Bridging Scales: Extracting Free Energy Functionals from Atomistic Simulations using Adjoint-Based Machine Learning.
*   **Alternative 2:** Data-Driven Discovery of Thermodynamics for Unknown Systems: From MD Trajectories to Consistent Phase-Field Potentials.

## 2. Abstract
*   **Problem Statement:** Deriving coarse-grained free energy functionals for complex systems studied via Molecular Dynamics (MD) remains a significant challenge. Traditional methods often require restrictive assumptions about the functional form of the energy.
*   **Methodology:** We present a framework that embeds Neural Networks (NN) within a differentiable Finite Element solver to learn thermodynamics directly from spatiotemporal data. The NN parameterizes the free energy derivatives ($df/dc$ and $df/d\eta$).
*   **Key Innovation:** Use of the discrete adjoint method for exact gradient propagation, combined with an integrability constraint to ensure the discovered potential is thermodynamically consistent. 
*   **Scope:** The framework is validated on synthetic 1D/2D benchmarks and subsequently applied to 3D datasets derived from MD simulations of phase separation and evolution.
*   **Results:** We demonstrate that the model successfully recovers underlying thermodynamic driving forces for unknown systems, providing a direct link between atomistic trajectories and continuum-scale physics.

## 3. Introduction
*   **Background:** The necessity of phase-field models in capturing meso-scale microstructural evolution.
*   **The Scale-Bridging Gap:** While MD provides high-fidelity atomistic insights, extracting the corresponding continuum free energy functional is non-trivial.
*   **The Challenge:** Limitations of traditional coarse-grained approaches and the "Free Energy Problem" in multi-physics systems.
*   **Objectives:** To demonstrate a "physics-embedded" learning approach that discovers the thermodynamics of unknown MD systems while strictly obeying conservation laws and thermodynamic principles (Integrability).

## 4. Mathematical and Computational Framework
*   **Governing Equations:** Coupled Cahn-Hilliard (conserved) and Allen-Cahn (non-conserved) systems.
*   **MD to Continuum Mapping:** Brief description of how MD snapshots (atom positions/types) are mapped to continuous concentration ($c$) and order parameter ($\eta$) fields.
*   **Neural Network Architecture:** MLP inputting $(c, \eta)$ and outputting $(\partial f/\partial c, \partial f/\partial \eta)$ with double-precision sensitivity.
*   **The Differentiable Solver:** Firedrake-based Finite Element discretization with `firedrake-adjoint` for sensitivity analysis through the temporal loop.

## 5. Training Formulation and Loss Functions
*   **Spectral (FFT) Loss:** Fourier-space comparison to capture morphological features and characteristic length scales in MD-derived patterns.
*   **Thermodynamic Consistency (Integrability):** Enforcing $\frac{\partial}{\partial \eta}(\frac{\partial f}{\partial c}) = \frac{\partial}{\partial c}(\frac{\partial f}{\partial \eta})$ to guarantee the existence of a scalar potential.
*   **Noise Robustness:** Strategies for handling the inherent fluctuations and noise present in MD-derived data.

## 6. Results and Discussion
*   **Synthetic Validation (1D/2D):** Verification against known analytical potentials (Flory-Huggins/Double-well).
*   **MD Case Study (3D):** 
    *   Description of the MD system (e.g., binary alloy or polymer blend).
    *   Discovery of the "unknown" free energy landscape from 3D MD evolution.
*   **Consistency Analysis:** Convergence of the integrability loss and visual validation of the reconstructed potential surface.
*   **Predictive Capability:** Using the learned NN to simulate evolution beyond the training time-horizon and comparing against held-out MD snapshots.

## 7. Conclusion
*   Summary of the framework’s ability to bridge the gap between atomistic simulations and continuum modeling.
*   Implications for the rapid development of phase-field models for novel materials where only MD data is available.

---
**Keywords:** Molecular Dynamics, Phase-Field Modeling, Scale Bridging, Adjoint Methods, Differentiable Physics, Thermodynamics Discovery.
