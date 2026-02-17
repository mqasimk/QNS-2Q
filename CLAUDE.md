# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QNS-2Q is a Python framework for two-qubit Quantum Noise Spectroscopy (QNS). It simulates QNS experiments to reconstruct power spectral densities of dephasing noise, then optimizes robust control pulses (CZ and identity gates) for the reconstructed noise environment.

## Running the Pipeline

All scripts are standalone and run directly from the project root. There is no build system or test framework — each script is executed individually. The virtual environment is at `./venv`.

```bash
# Stage 1: Generate QNS experiment data
python src/qnsExps.py          # -> DraftRun_NoSPAM_Feature/{results,params}.npz

# Stage 2: Reconstruct noise spectra from observables
python src/reconSpectra.py     # -> DraftRun_NoSPAM_Feature/specs.npz

# Stage 3a: Optimize CZ gate sequences
python src/CZopt_v2.py         # -> infs_{known,opt}_cz_v2.npz + PDF plots

# Stage 3b: Optimize identity (DD) sequences across M values
python src/ID_opt_v4.py        # -> infs_{known,opt}_id_v4_M{1..256}.npz + plots

# Stage 4: Generate publication-quality figures
python src/ID_opt_plots.py     # -> publication PDFs
python src/CZopt_plots.py
python src/CZopt_pulse_plot.py
```

Configuration is done by editing dataclass instances in each script's `main()` function (e.g., `QNSExperimentConfig`, `SpectraReconConfig`, `CZOptConfig`, `Config`). Output directories are controlled by the `data_folder` field.

## Architecture

### Data Flow

```
spectraIn.py (noise spectrum definitions)
       ↓
qnsExps.py (experiment simulation)
  uses: trajectories.py (pulses, noise, Hamiltonian, propagators)
        observables.py  (POVM measurements, correlation functions)
       ↓
{results.npz, params.npz}
       ↓
reconSpectra.py + fixedAS.py (spectral inversion)
       ↓
specs.npz
       ↓
CZopt_v2.py / ID_opt_v4.py (gate optimization)
       ↓
infs_*.npz + PDF plots
       ↓
*_plots.py (publication figures via plot_utils.py)
```

### Core Modules

- **`trajectories.py`** — Pulse sequence definitions (CPMG, CDD), noise trajectory generation, Hamiltonian construction, and time-domain propagator computation. Central dependency for both experiment and optimization stages.
- **`observables.py`** — POVM operators with SPAM error modeling, JAX-accelerated expectation values (Pauli and two-qubit correlations), and correlation functions computed via joblib parallelism.
- **`spectraIn.py`** — JAX-JIT compiled spectral density functions (`S_11`, `S_22`, `S_1212`, `S_1_2`, `S_1_12`, `S_2_12`). Noise model combines Lorentzian and Gaussian components. Modify this file to change the noise model.
- **`fixedAS.py`** — Spectral reconstruction via least-squares inversion of measurement matrices. Helper `ff()` computes Fourier transforms of pulse sequences.
- **`plot_utils.py`** — Shared plotting functions used across all `*_plots.py` scripts. Handles infidelity vs gate time, filter functions, spectra overlays.

### Optimization Scripts

- **`CZopt_v2.py`** — CZ gate optimization. Builds sequence libraries (CPMG, CDD, mqCDD), computes overlap integrals for infidelity, optimizes coupling strength J via scipy.optimize. Current active version.
- **`ID_opt_v4.py`** — Identity gate (dynamical decoupling) optimization with M-repetition scaling. Time-domain infidelity minimization with parametric pulse timing optimization. Current active version.
- **`CZopt.py`**, **`ID_opt_v2.py`**, **`ID_opt_v3.py`** — Previous versions kept for reference.

## Key Technical Conventions

- **JAX 64-bit mode** is enabled in every script: `jax.config.update("jax_enable_x64", True)`. This is required for numerical precision.
- **All numerical computation** uses `jax.numpy` (not bare numpy) for JIT compilation and vectorization (`vmap`).
- Dataclass-based configuration: each pipeline stage uses a frozen/immutable dataclass for parameters.
- Output data uses NumPy `.npz` format; plots use PDF.
- GPU memory is managed via batch slicing in `solver_prop()` (trajectories.py).

## Dependencies

`numpy`, `scipy`, `matplotlib`, `jax`, `jaxlib`, `qutip`, `qutip-qip`, `joblib` — Python 3.12.
