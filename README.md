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

**Requirements:** Python 3.12 (see `pyproject.toml`/`requirements.txt` for the exact
third-party versions this repo is validated against -- they are pinned exactly,
not with `>=`, because this repo's job is to reproduce a specific paper's numbers).

```bash
# Clone the repository
git clone https://github.com/mqasimk/QNS-2Q.git
cd QNS-2Q

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (either works; they declare the same pinned versions)
pip install -r requirements.txt        # matches every command in this README/CLAUDE.md as-is
# or:
pip install -e .                       # also makes `import qns2q` work without PYTHONPATH=src
pip install -e ".[gpu]"                # same, plus the CUDA 12 JAX plugin
```

> **Note:** JAX 64-bit mode (`jax_enable_x64`) is enabled in every script for numerical precision. Ensure your JAX installation supports this configuration. For GPU acceleration without the `pip install -e ".[gpu]"` above, install the matching `jax[cuda12]` version by hand (see `requirements.txt`).

See `DEPENDENCY_MAP.md` for the Python import graph (which module depends on
which, and the "attribute contract" edges a plain grep for imports would miss)
-- useful the first time you're deciding where to even start reading this codebase.

---

## Project Structure

The code is a package under `src/qns2q/`, mirroring the paper's two-arm pipeline
(**`characterize/`** = QNS → reconstruction, **`control/`** = gate design), with thin
per-stage entry points in **`scripts/`** that are run from the repo root. `CLAUDE.md` is
the authoritative developer guide; `FIGURE_PROVENANCE.md` maps every paper figure to its
data and exact regeneration command.

```text
QNS-2Q/
├── README.md
├── CLAUDE.md                          # authoritative developer / agent guide
├── FIGURE_PROVENANCE.md              # per-figure -> (run folder, data, script, command) map
├── NOISE_MODEL_SPEC.md              # provenance of the hardcoded noise-model constants
├── DEPENDENCY_MAP.md                 # Python import graph: which module depends on which
├── pyproject.toml, requirements.txt  # pinned dependencies (same versions, either installs)
├── src/qns2q/                        # the package (import path: qns2q.*; run from repo root)
│   ├── paths.py                      # regime selection (QNS2Q_REGIME) + canonical run-folder paths
│   ├── noise/spectra.py             # noise PSD model; regimes: bland | featured | showcase
│   ├── model/
│   │   ├── trajectories.py          # pulse sequences, noise trajectories, Hamiltonian, propagators
│   │   └── observables.py           # POVM operators, expectation values, correlation functions
│   ├── characterize/                # QNS arm: experiments -> reconstruction
│   │   ├── experiments.py           # Stage 1: QNS experiment simulation
│   │   ├── single_qubit.py          # single-qubit QNS variant (the C_1_0_MT_vs_M figure)
│   │   ├── inversion.py             # least-squares spectral inversion + filter functions
│   │   ├── reconstruct.py           # Stage 2: spectral reconstruction + reconstruction figures
│   │   ├── spam.py                  # SPAM calibration / estimation / robust estimators
│   │   └── systematics.py           # forward-model comb-bias systematics
│   ├── control/                     # gate-design arm
│   │   ├── cz.py                    # Stage 3a: CZ gate optimization
│   │   ├── idle.py                  # Stage 3b: idle / dynamical-decoupling optimization
│   │   └── tails.py, padding.py     # spectral-tail model + pulse-sequence padding
│   └── viz/cz_pulse_plot.py        # CZ pulse-sequence figure (showcase_pulse_sequences.pdf)
├── scripts/                         # per-stage entry points, all run from the repo root
│   ├── generate_paper_figures.py    # ONE command for all 8 paper figures (wraps the rest)
│   ├── run_capture_arm.py  run_spam_experiments.py  run_spam_reconstruct.py  run_reconstruct.py
│   ├── run_margin_band.py  run_margin_band_idle.py  harvest_design_numbers.py
│   ├── run_cz_pulse_plot.py  run_single_qubit.py  calibrate_showcase.py
│   └── report_showcase_figs.py  showcase_storage_panel.py   # the paper's showcase figures
│       # CZ/idle optimizers run as: PYTHONPATH=src python -m qns2q.control.{cz,idle}
├── tests/                           # pytest suite (run: pytest tests/)
├── reports/showcase_0613/          # the paper's showcase report + its figure PDFs
└── DraftRun_{NoSPAM,SPAM}_showcase*/   # showcase run data (summary .npz) feeding the figures
```

---

## Pipeline Overview

The simulation pipeline has four stages; each reads the previous stage's outputs.

```text
Stage 1: QNS experiments    scripts/run_capture_arm.py        ->  results.npz, params.npz
         (SPAM arms)         scripts/run_spam_experiments.py
                                      |
Stage 2: Reconstruct        scripts/run_reconstruct.py        ->  specs.npz (+ recon figures)
                                      |
         +----------------------------+----------------------------+
         |                                                         |
Stage 3a: CZ gate           python -m qns2q.control.cz            Stage 3b: idle / DD
          ->  plotting_data_cz_v2.npz                             python -m qns2q.control.idle
                                                                  ->  optimization_data_all_M.npz

         (run_margin_band[_idle], harvest_design_numbers, showcase_storage_panel
          build the margin bands / design ladder / storage panel)
                                      |
Stage 4: Figures            scripts/report_showcase_figs.py   ->  the paper's 6 showcase panels
                            scripts/run_cz_pulse_plot.py      ->  pulse-sequence figure
                            scripts/run_single_qubit.py       ->  C_1_0_MT_vs_M
```

---

## Usage

All stages are run from the **repo root**: Stages 1/2/4 via the `scripts/*.py` entry
points, and the Stage-3 optimizers as modules (`PYTHONPATH=src python -m qns2q.control.cz`).
Paths resolve from `qns2q.paths.project_root()`, so there is no `cd src/` step.
Configuration is done by editing the config dataclass in each stage's `main()`.

The active noise model *and* output folder are selected by the `QNS2Q_REGIME` environment
variable (`bland` | `featured` | `showcase`, default `featured`), resolved centrally in
`qns2q/paths.py` — switch the whole pipeline with one variable, no source edits. **The
paper uses the `showcase` regime**; `FIGURE_PROVENANCE.md` gives the exact per-figure
commands.

```bash
source venv/bin/activate
export QNS2Q_REGIME=showcase    # or: featured | bland   (run from the repo root)
```

### Stage 1: Generate QNS Experiment Data

Simulates QNS experiments using configurable pulse sequences (CPMG, CDD1, CDD3) and computes correlation function observables ($C_{12,0}$, $C_{12,12}$, $C_{a,0}$, $C_{a,b}$) with error bars.

```bash
python scripts/run_capture_arm.py    # NoSPAM arm (also runs Stage 2); run_spam_experiments.py for SPAM arms
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
python scripts/run_reconstruct.py                       # active regime's NoSPAM folder
python scripts/run_reconstruct.py --folder DraftRun_NoSPAM_showcase_cap   # an explicit folder
```

**Configuration:** defaults to the active regime's folder; pass `--folder` to target another.

**Output:** `specs.npz` containing reconstructed spectra + PDF comparison plots

### Stage 3: Optimize Gate Sequences

Loads the noise spectra (the reconstructed `specs.npz`, or the ground-truth `simulated_spectra.npz`) and optimizes pulse timings to minimize gate infidelity. Two independent optimizations are available:

**CZ Gate Optimization** -- Builds pulse-sequence libraries, computes overlap integrals, and optimizes the coupling strength $J$ via `scipy.optimize`. Both optimizers default to the reconstructed `specs.npz` from Stage 2; flags select alternatives.

```bash
PYTHONPATH=src python -m qns2q.control.cz                       # reconstructed specs of the active regime
PYTHONPATH=src python -m qns2q.control.cz --simulated           # ground-truth simulated_spectra.npz
PYTHONPATH=src python -m qns2q.control.cz --protocol mitigated  # a SPAM arm's reconstruction
# regenerate ground truth if needed:  PYTHONPATH=src python -m qns2q.noise.spectra
```

**Configuration:** Edit `CZOptConfig` -- set `fname`/`max_pulses`/gate-time factors; flags `--folder`, `--protocol`, `--simulated`, `--no-cross`, `--spectral-model`.

**Identity Gate (DD) Optimization** -- Time-domain infidelity minimization with parametric pulse timing across M-repetition values.

```bash
PYTHONPATH=src python -m qns2q.control.idle
```

**Configuration:** Edit `Config` in `qns2q/control/idle.py`; same `--folder`/`--protocol`/`--simulated` flags.

**Output:** `infs_{known,opt}_*.npz` + infidelity vs. gate time PDF plots

### Stage 4: Generate Publication Figures

**The paper's figures (showcase regime)** are produced by `report_showcase_figs.py` (the
six showcase panels) plus two standalone figures; `FIGURE_PROVENANCE.md` is the
authoritative figure->data->command map. The one-command wrapper:

```bash
python scripts/generate_paper_figures.py               # assemble all 8 from committed data (minutes)
python scripts/generate_paper_figures.py --full-regen   # also re-derive every run folder first (hours, GPU)
python scripts/generate_paper_figures.py --dry-run       # preview the commands without running them
```

is equivalent to running the individual commands below by hand (do this instead if you
only want to regenerate one figure, or if you want to see/copy the exact underlying
command):

```bash
export QNS2Q_REGIME=showcase
SHOWCASE_FIGS_DIR=reports/showcase_0613/figs PYTHONPATH=src python scripts/report_showcase_figs.py
PYTHONPATH=src python scripts/run_single_qubit.py                                     # C_1_0_MT_vs_M
PYTHONPATH=src python scripts/run_cz_pulse_plot.py --folder DraftRun_NoSPAM_showcase_cap --tag _cap # pulse sequences
```

---

## Output Directory Structure

Each pipeline run generates an output directory named for the active regime (the paper uses `showcase`, in `DraftRun_NoSPAM_showcase_cap/`) with the following structure:

```text
DraftRun_NoSPAM_showcase_cap/         # the paper's NoSPAM run (showcase regime)
├── params.npz                        # experiment configuration parameters
├── results.npz                       # correlation-function observables (Stage 1)
├── specs.npz                         # reconstructed spectra (Stage 2)
├── simulated_spectra.npz             # ground-truth analytic spectra
├── optimization_data_all_M_cap.npz   # idle infidelities over M (Stage 3b)
├── plotting_data/
│   └── plotting_data_cz_v2_cap.npz   # CZ best-sequence plotting data (Stage 3a)
├── margin_band_{cz,id}_cap.npz       # NT-over-CDD margin bands
├── design_numbers.npz                # knowledge-ladder + SPAM design numbers (harvest)
└── storage_panel.npz                 # Bell-pair storage panel data
```

Only the small summary `.npz` above are tracked in git (see the `.gitignore` whitelist);
the multi-GB raw-trajectory cache `phases.npz` and the regenerable `figures/` tree are not
committed, and were trimmed from the showcase dirs. The `_cap` suffix is a vestigial
filename tag on the showcase run. The three SPAM arms live in
`DraftRun_SPAM_showcase_{reference,raw,mitigated}/` (each tracked down to its `specs.npz`
and design-ladder summaries).

---

## Running Tests

```bash
pytest tests/ -v
```

The suite covers the noise model, simulation, reconstruction, and gate machinery:

- **`test_spectra_input.py`, `test_noise_model.py`** -- noise-spectrum properties (symmetry, non-negativity, Cauchy-Schwarz) and the anchored model
- **`test_trajectories.py`** -- pulse sequences, control matrices, density-matrix validity, propagator unitarity
- **`test_observables.py`** -- operator unitarity, POVM structure, concurrence, bootstrap errors
- **`test_spectral_inversion.py`, `test_systematics.py`** -- filter functions, error propagation, comb-bias systematics
- **`test_spam.py`** -- SPAM calibration / mitigation / robust protocols
- **`test_control_gates.py`** -- CZ and idle gate optimization machinery
- **`test_tau_invariance.py`** -- invariance of the physics under time-unit re-anchoring

Run under the default `featured` regime the suite is green. Under `QNS2Q_REGIME=showcase`
two cross-spectra tests are known-stale (they assume the featured model's Im-dominant
cross-spectra, whereas the showcase model deliberately puts the common-mode carrier and
shared TLF into `Re S_{1,2}`).

---

## Core Modules

### `qns2q/noise/spectra.py`
Defines the analytic noise power spectral densities (Lorentzian/Gaussian components): self-spectra ($S_{11}$, $S_{22}$, $S_{12,12}$) and complex cross-spectra ($S_{1,2}$, $S_{1,12}$, $S_{2,12}$). The active regime (`bland | featured | showcase`) is selected at import by `QNS2Q_REGIME`.

### `qns2q/model/trajectories.py`
Generates temporally-correlated noise trajectories and simulates quantum evolution: pulse-sequence generators (CPMG, CDD, mqCDD), dephasing-noise Hamiltonian construction, and ensemble-averaged propagation (GPU-batched).

### `qns2q/model/observables.py`
Computes single- and two-qubit expectation values with SPAM-error modeling, and the correlation functions ($C_{12,0}$, $C_{12,12}$, $C_{a,b}$, $C_{a,0}$) consumed by reconstruction.

### `qns2q/characterize/inversion.py` and `reconstruct.py`
`inversion.py` implements the least-squares spectral inversion (filter functions, measurement-matrix construction, error propagation); `reconstruct.py` drives Stage 2 and writes `specs.npz` plus the reconstruction figures.

### `qns2q/paths.py`
Single source of truth for regime selection (`current_regime`) and canonical run-folder paths (`run_folder`, `run_path`, `project_root`).

---

## Citation

If you use this code in your research, please cite the associated paper and this repository.

- **Author:** Muhammad Qasim Khan
- **Affiliation:** Dartmouth College
- **Paper:** *Noise-tailored two-qubit gates*
