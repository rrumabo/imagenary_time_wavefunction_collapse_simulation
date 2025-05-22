# Entropy-Driven Wavefunction Collapse (Imaginary Time)

This project implements a numerical simulation of deterministic wavefunction collapse in **imaginary time**, based on an entropy-minimization framework. Inspired by a thermodynamic variational principle, the wavefunction evolves under a gradient flow that minimizes a free-energy functional:

\[
F[\psi] = \langle \psi | \hat{H}_0 | \psi \rangle - T S[\psi]
\]

where \( \hat{H}_0 \) is the kinetic Hamiltonian and \( S[\psi] \) is the Shannon entropy of the probability density \( |\psi(x)|^2 \).

## 💡 Features

- Imaginary-time evolution using split-step method
- Entropy-based localization of quantum states
- Norm-preserving dynamics with adaptive μ(τ)
- Diagnostic plots: entropy, energy, expectation values
- Heatmap visualization of wavefunction collapse
- Configurable parameters and batch sweeps

## 📦 Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

Install with:

```bash
pip install numpy matplotlib pandas scipy
