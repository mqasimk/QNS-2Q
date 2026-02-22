# QNS-2Q: Two-Qubit Quantum Noise Spectroscopy

Expert context for the **QNS-2Q** repository, a modular Python framework for characterizing and mitigating noise in two-qubit quantum systems.

## Project Overview

**QNS-2Q** implements protocols for **Quantum Noise Spectroscopy (QNS)** to reconstruct power spectral densities (PSDs) of dephasing noise and uses this information to optimize robust control pulses for CZ and identity gates.

### Core Technical Stack
- **Language:** Python 3.12
- **Numerical Engines:** JAX (primary for optimization and simulation), NumPy, SciPy.
- **Quantum Simulation:** QuTiP, qutip-qip.
- **Visualization:** Matplotlib.

## Repository Structure & Key Modules

### 1. Simulation & Physics Core (`src/`)
- `trajectories.py`: Core logic for generating temporally-correlated noise trajectories using JAX `vmap` and FFT-based methods. Defines the system Hamiltonian and propagators.
- `observables.py`: Defines POVM operators and expectation value calculations for two-qubit systems.
- `spectra_input.py`: Analytical definitions of noise spectra (Lorentzian, Gaussian, etc.) used for simulation and as targets for reconstruction.

### 2. The 4-Stage Pipeline
1.  **Data Generation (`qns_experiments.py`):** Simulates QNS experiments with various pulse sequences (CPMG, CDD) to produce time-domain observables.
2.  **Reconstruction (`reconstruct_spectra.py`, `spectral_inversion.py`):** Inverts experimental data via least-squares to estimate noise spectra ($S_{11}, S_{22}, S_{12}, S_{12,12}$).
3.  **Optimization:**
    - `cz_optimize.py`: Optimizes CZ gate sequences and coupling strength $J$ using JAX-based gradient descent and overlap integrals.
    - `id_optimize.py`: Optimizes identity gate (dynamical decoupling) sequences to minimize gate infidelity.
4.  **Analysis & Plotting:** `cz_plots.py`, `id_plots.py`, `qns_plots.py`, and `cz_pulse_plot.py` generate publication-quality figures.

## Strategic Workflows

### Running an Optimization Cycle
1.  **Generate Data:** Run `python src/qns_experiments.py` to create a `DraftRun` directory with simulated observables.
2.  **Reconstruct:** Run `python src/reconstruct_spectra.py` to produce `specs.npz`.
3.  **Optimize:** Run `python src/cz_optimize.py` or `python src/id_optimize.py`. These scripts load `specs.npz` and perform pulse timing optimization.

### Key Configuration
Most scripts use `dataclass`-based configurations (e.g., `CZOptConfig`, `QNSExperimentConfig`). Modifications should be made within the `main()` block of the respective scripts before execution.

## Critical Notes
- **JAX 64-bit:** All scripts enable `jax_enable_x64` for numerical precision.
- **SPAM:** The framework includes specific logic for SPAM-robust spectroscopy.
- **Legacy Code:** Files suffixed with `_legacy.py` are preserved for reference but superseded by the main versions.
