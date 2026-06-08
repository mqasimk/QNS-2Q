# QNS-2Q: Two-Qubit Quantum Noise Spectroscopy

**QNS-2Q** is a modular Python framework for characterizing and mitigating noise in two-qubit quantum systems. It implements protocols for **Quantum Noise Spectroscopy (QNS)** to reconstruct power spectral densities (PSDs) of dephasing noise and uses this information to optimize robust control pulses (CZ and identity gates).

The codebase features a full simulation pipeline:

1. **Data Generation** — Simulating QNS experiments with configurable pulse sequences.
2. **Reconstruction** — Inverting experimental observables to estimate noise spectra ($S_{11}, S_{22}, S_{12}, S_{12,12}$, etc.).
3. **Optimization** — Using JAX-based gradient descent to design high-fidelity gate sequences tailored to the reconstructed noise environment.

## Features

- **JAX-Accelerated Simulation** — Uses `jax` and `jax.numpy` for high-performance numerical integration, JIT compilation, vectorization (`vmap`), and automatic differentiation during pulse optimization.
- **Modular Experiment Design** — Flexible dataclass-based configuration for defining pulse sequences (CPMG, CDD, mqCDD) and experiment parameters.
- **SPAM Robustness** — Includes protocols for analyzing noise in the presence of State Preparation and Measurement (SPAM) errors.
- **Spectra Reconstruction** — Tools to reconstruct single-qubit, cross-correlated, and multi-qubit noise spectra from time-domain observables via least-squares inversion.
- **Gate Optimization** — Dedicated engines for optimizing both CZ gate and identity (dynamical decoupling) sequences, including coupling strength $J$ optimization and M-repetition scaling.

## Requirements

Python 3.12 with the following dependencies:

```bash
pip install numpy scipy matplotlib jax jaxlib qutip qutip-qip joblib
```

> **Note:** The code enables JAX 64-bit mode (`jax_enable_x64`) in every script for numerical precision. Ensure your JAX installation supports this configuration. A preconfigured virtual environment with JAX+CUDA and all dependencies is provided at `./venv` (activate with `source venv/bin/activate`).

## Project Structure

```text
QNS-2Q/
├── README.md
└── src/
    ├── run_paths.py              # Regime selection (QNS2Q_REGIME) + canonical run-folder paths
    ├── spectra_input.py          # Noise spectra (bland | featured regime), selected by QNS2Q_REGIME
    ├── trajectories.py           # Pulse sequences, noise trajectories, Hamiltonian, propagators
    ├── observables.py            # POVM operators, expectation values, correlation functions
    │
    ├── qns_experiments.py        # Stage 1: QNS experiment simulation
    ├── single_qubit_qns.py       # Single-qubit QNS experiment framework
    ├── reconstruct_spectra.py    # Stage 2: Spectral reconstruction from observables
    ├── spectral_inversion.py     # Least-squares inversion for reconstruction
    │
    ├── cz_optimize.py            # Stage 3a: CZ gate pulse optimization
    ├── id_optimize.py            # Stage 3b: Identity gate (DD) optimization
    │
    ├── cz_plots.py               # Stage 4: CZ publication figures
    ├── cz_pulse_plot.py          # Stage 4: CZ pulse sequence visualization
    ├── id_plots.py               # Stage 4: Identity gate publication figures
    ├── qns_plots.py              # Stage 4: QNS spectra comparison plots
    ├── plot_utils.py             # Shared plotting utilities
    │
    ├── cz_optimize_legacy.py     # Previous CZ optimization version
    ├── id_optimize_v2_legacy.py  # Previous ID optimization versions
    ├── id_optimize_v3_legacy.py  #
    ├── opt_plot.py               # Auxiliary plotting scripts
    └── opt_window.py             #
```

## Workflow

The pipeline has four stages. **All scripts must be run from inside `src/`** — Stages 1–2 locate the repo-root run folders via a relative `..` path, so running from elsewhere will not find them. Configuration is done by editing the config object in each script's `main()` function.

The active noise model *and* output folder are selected by the `QNS2Q_REGIME` environment variable (`bland` or `featured`, default `featured`), resolved centrally in `run_paths.py`. A whole pipeline switches with one variable — no source edits:

```bash
cd src/
export QNS2Q_REGIME=featured   # or: bland
```

### Stage 1: Generate QNS Experiment Data

Simulates QNS experiments using configurable pulse sequences (CPMG, CDD) and computes observables (overlap integrals, correlation functions).

```bash
python qns_experiments.py
```

- **Config**: `QNSExperimentConfig` — total time, number of blocks, input spectra
- **Output**: `DraftRun_NoSPAM_<regime>/{results,params}.npz`

### Stage 2: Reconstruct Noise Spectra

Loads observables from Stage 1 and solves the inverse problem via least-squares to reconstruct the noise power spectral densities.

```bash
python reconstruct_spectra.py
```

- **Config**: `SpectraReconConfig` — set `data_folder` to the Stage 1 output directory
- **Output**: `specs.npz` containing reconstructed spectra + comparison plots

### Stage 3: Optimize Gate Sequences

Loads the noise spectra (the reconstructed `specs.npz`, or the ground-truth `simulated_spectra.npz`) and optimizes pulse timings to minimize gate infidelity. Two independent optimizations are available:

**CZ Gate Optimization** — Builds sequence libraries, computes overlap integrals, and optimizes coupling strength $J$ via `scipy.optimize`. As shipped, `main()` runs against the ground-truth spectra, so generate them first with `python spectra_input.py` (writes `simulated_spectra.npz`); set `use_simulated=False` in the config to use the reconstructed `specs.npz` from Stage 2 instead.

```bash
python spectra_input.py   # writes simulated_spectra.npz (needed only for the default CZ config)
python cz_optimize.py
```

**Identity Gate (DD) Optimization** — Time-domain infidelity minimization with parametric pulse timing optimization across M-repetition values.

```bash
python id_optimize.py
```

- **Config**: `CZOptConfig` / `Config` — data folder, max pulses, gate time factors
- **Output**: `infs_{known,opt}_*.npz`, `plotting_data/*.npz` + PDF plots

### Stage 4: Generate Publication Figures

```bash
python id_plots.py
python cz_plots.py
python cz_pulse_plot.py
```

## Citation

If you use this code in your research, please cite the associated papers or this repository.

- **Author**: Muhammad Qasim Khan
- **Affiliation**: Dartmouth College
