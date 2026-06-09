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
│   ├── main_v7.tex                   # LaTeX source for the accompanying paper
│   └── aps_v2.bib                    # Bibliography
├── src/
│   ├── run_paths.py                  # Regime selection (QNS2Q_REGIME) + canonical run-folder paths
│   ├── spectra_input.py              # Noise PSD definitions (bland | featured), selected by QNS2Q_REGIME
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
│   └── plot_utils.py                 # Shared plotting utilities
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

All scripts are standalone runners. Configuration is done by editing the config object in each script's `main()` function. **Run all scripts from inside `src/`** — Stages 1–2 locate the repo-root run folders via a relative `..` path, so running from elsewhere will not find them.

The active noise model *and* output folder are selected by the `QNS2Q_REGIME` environment variable (`bland` or `featured`, default `featured`), resolved centrally in `run_paths.py`. Switch the whole pipeline with one variable — no source edits:

> **Layout note (updated).** The code is now a package under `src/qns2q/`
> (`characterize/` + `control/` arms, shared `noise/`+`model/`, `viz/`), run from the
> **repo root** via thin `scripts/run_*.py` entry points. The per-stage `python <file>.py`
> commands below map to `python scripts/run_<stage>.py` (e.g. `run_experiments`,
> `run_reconstruct`, `run_cz`, `run_idle`, `run_id_plots`, …). Paths resolve from
> `qns2q.paths.project_root()`, so no `cd src/` is needed. **CLAUDE.md is authoritative.**

```bash
export QNS2Q_REGIME=featured   # or: bland   (run from the repo root)
```

### Stage 1: Generate QNS Experiment Data

Simulates QNS experiments using configurable pulse sequences (CPMG, CDD1, CDD3) and computes correlation function observables ($C_{12,0}$, $C_{12,12}$, $C_{a,0}$, $C_{a,b}$) with error bars.

```bash
python scripts/run_experiments.py
```

**Configuration:** Edit `QNSExperimentConfig` in the script.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `T` | Total experiment time per block | `160 * tau` |
| `M` | Number of blocks | `10` |
| `t_grain` | Time discretization grain | `3000` |
| `w_grain` | Frequency discretization grain | `500` |
| `truncate` | Number of harmonics | `20` |
| `n_shots` | Noise realizations per experiment | `10000` |
| `a_sp` | State preparation fidelities | `[1., 1.]` |

**Output:** `DraftRun_NoSPAM_<regime>/results.npz` and `params.npz`

### Stage 2: Reconstruct Noise Spectra

Loads correlation observables from Stage 1 and solves the inverse problem via least-squares to reconstruct six spectral components: $S_{11}$, $S_{22}$, $S_{12,12}$ (real-valued self-spectra) and $S_{1,2}$, $S_{1,12}$, $S_{2,12}$ (complex-valued cross-spectra).

```bash
python reconstruct_spectra.py
```

**Configuration:** Set `data_folder` in `main()` to point to the Stage 1 output directory.

**Output:** `specs.npz` containing reconstructed spectra + PDF comparison plots

### Stage 3: Optimize Gate Sequences

Loads the noise spectra (the reconstructed `specs.npz`, or the ground-truth `simulated_spectra.npz`) and optimizes pulse timings to minimize gate infidelity. Two independent optimizations are available:

**CZ Gate Optimization** -- Builds pulse sequence libraries, computes overlap integrals, and optimizes the coupling strength $J$ via `scipy.optimize`. As shipped, `main()` runs against the ground-truth spectra, so generate them first with `python spectra_input.py` (writes `simulated_spectra.npz`); set `use_simulated=False` to use the reconstructed `specs.npz` from Stage 2 instead.

```bash
python spectra_input.py   # writes simulated_spectra.npz (needed only for the default CZ config)
python cz_optimize.py
```

**Configuration:** Edit `CZOptConfig` -- set `data_folder`, `max_pulses`, gate time factors.

**Identity Gate (DD) Optimization** -- Time-domain infidelity minimization with parametric pulse timing optimization across M-repetition values.

```bash
python id_optimize.py
```

**Configuration:** Edit `Config` -- set `data_folder`, `max_pulses_per_rep`, gate time factors.

**Output:** `infs_{known,opt}_*.npz` + infidelity vs. gate time PDF plots

### Stage 4: Generate Publication Figures

```bash
python cz_plots.py          # CZ gate infidelity figures
python id_plots.py          # Identity gate infidelity figures
python cz_pulse_plot.py     # CZ pulse sequence comparison
python qns_plots.py         # QNS spectra SPAM mitigation comparison
```

---

## Output Directory Structure

Each pipeline run generates an output directory named for the active regime (e.g., `DraftRun_NoSPAM_featured/` or `DraftRun_NoSPAM_bland/`) with the following structure:

```text
DraftRun_NoSPAM_<regime>/
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
