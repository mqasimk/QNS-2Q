
# QNS-2Q: Two-Qubit Quantum Noise Spectroscopy

**QNS-2Q** is a modular Python framework designed for characterizing and mitigating noise in two-qubit quantum systems. It implements protocols for **Quantum Noise Spectroscopy (QNS)** to reconstruct power spectral densities (PSDs) of dephasing noise and uses this information to optimize robust control pulses (specifically CZ gates).

The codebase features a full simulation pipeline:
1.  **Data Generation**: Simulating QNS experiments with configurable pulse sequences.
2.  **Reconstruction**: Inverting experimental observables to estimate noise spectra ($S_{11}, S_{22}, S_{12}, S_{12,12}$, etc.).
3.  **Optimization**: Using JAX-based gradient descent to design high-fidelity gate sequences tailored to the reconstructed noise environment.

## Features

* **JAX-Accelerated Simulation**: Utilizes `jax` and `jax.numpy` for high-performance numerical integration and automatic differentiation during pulse optimization.
* **Modular Experiment Design**: Flexible configuration for defining pulse sequences (CPMG, CDD, etc.) and experiment parameters via `QNSExperimentConfig`.
* **SPAM Robustness**: Includes protocols for analyzing noise even in the presence of State Preparation and Measurement (SPAM) errors.
* **Spectra Reconstruction**: Tools to reconstruct single-qubit, cross-correlated, and multi-qubit noise spectra from time-domain observables.
* **Gate Optimization**: A dedicated engine (`CZopt_v2.py`) for optimizing CZ gate parameters (pulse timings and coupling strength $J$) to minimize infidelity.

## Installation & Requirements

The project requires Python 3.x and the following dependencies:

```bash
pip install numpy scipy matplotlib jax jaxlib

```

**Note:** The code explicitly enables JAX 64-bit mode (`jax_enable_x64`) for precision. Ensure your JAX installation supports this configuration.

## Project Structure

```text
mqasimk/qns-2q/
├── README.md           # Project documentation
└── src/
    ├── 1qQNS.py        # Framework for single-qubit QNS experiments
    ├── qnsExps.py      # Main script for running 2-qubit QNS experiment batches
    ├── reconSpectra.py # Reconstruction of spectra from experimental observables
    ├── CZopt_v2.py     # Optimization of CZ gate sequences using reconstructed spectra
    ├── observables.py  # (Dependency) Definitions of observable functions
    ├── spectraIn.py    # (Dependency) Input spectra definitions (S_11, S_22, etc.)
    └── ...

```

## Workflow & Usage

The typical workflow involves three stages: **Experiment**, **Reconstruction**, and **Optimization**.

### 1. Running QNS Experiments

Use `src/qnsExps.py` to simulate the acquisition of experimental data. This script defines a set of experiments (using sequences like CPMG, CDD) and calculates the resulting overlap integrals (observables).

* **Configuration**: Modify `QNSExperimentConfig` in the script to change parameters like total time (), number of blocks (), or the input spectra.
* **Output**: Saves results to a directory (default: `DraftRun_NoSPAM_Feature`) as `results.npz` and `params.npz`.

```bash
python src/qnsExps.py

```

### 2. Reconstructing Spectra

Use `src/reconSpectra.py` to process the data generated in step 1. It loads the observables and solves the inverse problem to reconstruct the noise spectra.

* **Configuration**: Set the `data_folder` variable in `main()` to point to the output of the experiment step.
* **Output**: Generates `specs.npz` containing the reconstructed spectra (, etc.) and comparison plots (`reconstruct.pdf`).

```bash
python src/reconSpectra.py

```

### 3. Pulse Sequence Optimization

Use `src/CZopt_v2.py` to find optimal control sequences for a CZ gate. This script loads the reconstructed spectra from step 2 and minimizes the gate infidelity.

* **Methodology**:
* Constructs libraries of known sequences (CDD, mqCDD).
* Performs gradient-based optimization on random pulse sequences.
* Optimizes the inter-qubit coupling strength .


* **Configuration**: Adjust `CZOptConfig` to point to the correct data folder.
* **Output**: Saves optimized sequences and infidelity plots to the data directory.

```bash
python src/CZopt_v2.py

```

## Citation

If you use this code in your research, please cite the associated papers or this repository.

* **Author**: Muhammad Qasim Khan (Q)
* **Affiliation**: Dartmouth College
