# QNS-2Q: Two-Qubit Quantum Noise Spectroscopy

A modular Python framework for characterizing and mitigating noise in two-qubit quantum systems through **Quantum Noise Spectroscopy (QNS)** and optimized control pulse design.

This codebase accompanies the paper *"Noise-tailored two-qubit gates"* and implements a complete simulation pipeline: from generating QNS experiment data and reconstructing noise power spectral densities, to optimizing robust CZ and dynamical decoupling gate sequences tailored to the reconstructed noise environment.

---

## Features

- **JAX-Accelerated Simulation** -- High-performance numerical integration using `jax` with JIT compilation, vectorization (`vmap`), and automatic differentiation for pulse optimization. All scripts enable 64-bit precision (`jax_enable_x64`).
- **Modular Experiment Design** -- Flexible dataclass-based configuration for defining pulse sequences (CPMG, CDD1, CDD3, mqCDD) and experiment parameters.
- **SPAM Error Mitigation** -- Full support for analyzing and mitigating State Preparation and Measurement (SPAM) errors through confusion-matrix correction and parametric bootstrapping.
- **Spectral Reconstruction** -- Least-squares inversion to reconstruct six spectral components (three self-spectra and three complex cross-spectra) from time-domain correlation observables, with full error propagation.
- **Gate Optimization** -- Dedicated optimization engines for CZ gates (with coupling strength $J$ tuning) and identity/dynamical decoupling gates (with M-repetition scaling), using both library sequences and JAX-based random sequence optimization.
- **Publication-Quality Figures** -- Matplotlib-based plotting utilities with Okabe-Ito color palette for generating publication-ready figures.

---

## Installation

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/mqasimk/QNS-2Q.git
cd QNS-2Q

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** JAX 64-bit mode (`jax_enable_x64`) is enabled in every script for numerical precision. Ensure your JAX installation supports this configuration. For GPU acceleration, install the appropriate `jaxlib` variant for your CUDA version.

---

## Project Structure

```text
QNS-2Q/
├── README.md
├── requirements.txt
├── paper/
│   └── main_v6.tex                   # LaTeX source for the accompanying paper
├── src/
│   ├── spectra_input.py              # Analytical noise PSD definitions (Lorentzian + Gaussian)
│   ├── trajectories.py               # Pulse sequences, noise trajectories, Hamiltonian, propagators
│   ├── observables.py                # POVM operators, expectation values, correlation functions
│   ├── spectral_inversion.py         # Least-squares inversion for spectral reconstruction
│   │
│   ├── qns_experiments.py            # Stage 1: QNS experiment simulation
│   ├── single_qubit_qns.py           # Single-qubit QNS experiment variant
│   ├── reconstruct_spectra.py        # Stage 2: Spectral reconstruction from observables
│   │
│   ├── cz_optimize.py                # Stage 3a: CZ gate pulse optimization
│   ├── id_optimize.py                # Stage 3b: Identity gate (DD) optimization
│   │
│   ├── cz_plots.py                   # Stage 4: CZ gate publication figures
│   ├── cz_pulse_plot.py              # Stage 4: CZ pulse sequence visualization
│   ├── id_plots.py                   # Stage 4: Identity gate publication figures
│   ├── qns_plots.py                  # Stage 4: QNS spectra comparison plots
│   ├── plot_utils.py                 # Shared plotting utilities
│   │
│   ├── cz_optimize_legacy.py         # Legacy: previous CZ optimization version
│   ├── id_optimize_v2_legacy.py      # Legacy: previous ID optimization v2
│   ├── id_optimize_v3_legacy.py      # Legacy: previous ID optimization v3
│   ├── opt_plot.py                   # Auxiliary plotting scripts
│   └── opt_window.py                 # Window optimization helper
└── tests/
    ├── conftest.py                   # Shared test fixtures and path setup
    ├── test_spectra_input.py         # Tests for noise spectrum definitions
    ├── test_trajectories.py          # Tests for pulse sequences and evolution
    ├── test_observables.py           # Tests for quantum observable calculations
    └── test_spectral_inversion.py    # Tests for spectral reconstruction
```

---

## Pipeline Overview

The simulation pipeline has four stages. Each stage reads outputs from the previous stage and produces data files and/or figures.

```text
┌─────────────────────────┐     ┌──────────────────────────┐
│  Stage 1: QNS Experiments│────>│  Stage 2: Reconstruct    │
│  qns_experiments.py      │     │  Spectra                 │
│                          │     │  reconstruct_spectra.py  │
│  Output: results.npz     │     │  Output: specs.npz       │
│          params.npz       │     │          comparison plots │
└─────────────────────────┘     └──────────┬───────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                ┌─────────▼──────────┐          ┌──────────▼──────────┐
                │  Stage 3a: CZ Gate │          │  Stage 3b: ID/DD    │
                │  Optimization      │          │  Gate Optimization  │
                │  cz_optimize.py    │          │  id_optimize.py     │
                │                    │          │                     │
                │  Output:           │          │  Output:            │
                │  infs_*_cz_*.npz   │          │  infs_*_id_*.npz    │
                └─────────┬──────────┘          └──────────┬──────────┘
                          │                                 │
                          └────────────────┬────────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Stage 4: Publication│
                                │  Figures             │
                                │  cz_plots.py         │
                                │  id_plots.py         │
                                │  cz_pulse_plot.py    │
                                │  qns_plots.py        │
                                └──────────────────────┘
```

---

## Usage

All scripts are standalone runners. Configuration is done by editing dataclass instances in each script's `main()` function. Scripts should be run from the project root.

### Stage 1: Generate QNS Experiment Data

Simulates QNS experiments using configurable pulse sequences (CPMG, CDD1, CDD3) and computes correlation function observables ($C_{12,0}$, $C_{12,12}$, $C_{a,0}$, $C_{a,b}$) with error bars.

```bash
python src/qns_experiments.py
```

**Configuration:** Edit `QNSExperimentConfig` in the script.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `T` | Total experiment time per block | `160 * tau` |
| `M` | Number of blocks | `1` |
| `t_grain` | Time discretization grain | `10000` |
| `w_grain` | Frequency discretization grain | `10000` |
| `truncate` | Number of harmonics | `20` |
| `n_shots` | Noise realizations per experiment | `10000` |
| `a_sp` | State preparation fidelities | `[1., 1.]` |

**Output:** `DraftRun_NoSPAM_Feature/results.npz` and `params.npz`

### Stage 2: Reconstruct Noise Spectra

Loads correlation observables from Stage 1 and solves the inverse problem via least-squares to reconstruct six spectral components: $S_{11}$, $S_{22}$, $S_{12,12}$ (real-valued self-spectra) and $S_{1,2}$, $S_{1,12}$, $S_{2,12}$ (complex-valued cross-spectra).

```bash
python src/reconstruct_spectra.py
```

**Configuration:** Set `data_folder` in `main()` to point to the Stage 1 output directory.

**Output:** `specs.npz` containing reconstructed spectra + PDF comparison plots

### Stage 3: Optimize Gate Sequences

Loads reconstructed spectra and optimizes pulse timings to minimize gate infidelity. Two independent optimizations are available:

**CZ Gate Optimization** -- Builds pulse sequence libraries, computes overlap integrals, and optimizes the coupling strength $J$ via `scipy.optimize`.

```bash
python src/cz_optimize.py
```

**Configuration:** Edit `CZOptConfig` -- set `data_folder`, `max_pulses`, gate time factors.

**Identity Gate (DD) Optimization** -- Time-domain infidelity minimization with parametric pulse timing optimization across M-repetition values.

```bash
python src/id_optimize.py
```

**Configuration:** Edit `Config` -- set `data_folder`, `max_pulses_per_rep`, gate time factors.

**Output:** `infs_{known,opt}_*.npz` + infidelity vs. gate time PDF plots

### Stage 4: Generate Publication Figures

```bash
python src/cz_plots.py          # CZ gate infidelity figures
python src/id_plots.py           # Identity gate infidelity figures
python src/cz_pulse_plot.py      # CZ pulse sequence comparison
python src/qns_plots.py          # QNS spectra SPAM mitigation comparison
```

---

## Output Directory Structure

Each pipeline run generates an output directory (e.g., `DraftRun_NoSPAM_Feature/`) with the following structure:

```text
DraftRun_NoSPAM_Feature/
├── params.npz                         # Experiment configuration parameters
├── results.npz                        # Correlation function observables (Stage 1)
├── specs.npz                          # Reconstructed spectral components (Stage 2)
├── infs_known_cz_v2.npz              # CZ known sequence infidelities (Stage 3a)
├── infs_opt_cz_v2.npz                # CZ optimized sequence infidelities (Stage 3a)
├── infs_known_id_v4.npz              # ID known sequence infidelities (Stage 3b)
├── infs_opt_id_v4.npz                # ID optimized sequence infidelities (Stage 3b)
├── plotting_data/
│   ├── plotting_data_cz_v2.npz       # CZ plotting data with best sequences
│   └── plotting_data_id_v4.npz       # ID plotting data
└── figures/
    ├── reconstruction/
    │   └── spectral_reconstruction_pub.pdf
    └── publication/
        ├── infidelity_vs_gatetime_pub.pdf
        └── pulse_sequence_comparison_pub.pdf
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers the core mathematical and utility functions:

- **`test_spectra_input.py`** -- Noise spectrum properties (symmetry, non-negativity, Cauchy-Schwarz)
- **`test_trajectories.py`** -- Pulse sequences, control matrices, density matrix validity, Hamiltonian properties, propagator unitarity
- **`test_observables.py`** -- Operator unitarity, POVM structure, concurrence metrics, bootstrap error estimation
- **`test_spectral_inversion.py`** -- Filter functions, error propagation, spectral inversion consistency

---

## Core Modules

### `spectra_input.py`
Defines analytical noise power spectral densities using Lorentzian (`L`) and Gaussian (`Gauss`) basis functions. Provides self-spectra ($S_{11}$, $S_{22}$, $S_{12,12}$) and complex cross-spectra ($S_{1,2}$, $S_{1,12}$, $S_{2,12}$) with configurable time-delay parameters.

### `trajectories.py`
Generates temporally-correlated noise trajectories from spectral functions and simulates quantum evolution. Includes pulse sequence generators (CPMG, CDD1, CDD3), Hamiltonian construction for dephasing noise, and ensemble-averaged density matrix propagation.

### `observables.py`
Computes quantum observables from simulated density matrices, including single-qubit and two-qubit expectation values with SPAM error mitigation via confusion-matrix correction. Provides correlation functions ($C_{12,0}$, $C_{12,12}$, $C_{a,b}$, $C_{a,0}$) used for spectral reconstruction.

### `spectral_inversion.py`
Implements least-squares spectral reconstruction from correlation observables. Computes filter functions (Fourier transforms of toggle functions), constructs and inverts the measurement matrix, and propagates observation errors through the inversion.

---

## Citation

If you use this code in your research, please cite the associated paper and this repository.

- **Author:** Muhammad Qasim Khan
- **Affiliation:** Dartmouth College
- **Paper:** *Noise-tailored two-qubit gates*
