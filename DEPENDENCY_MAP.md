# Module dependency map

Who-imports-whom inside `src/qns2q/`, and how the `scripts/` entry points sit on
top of it. This is the *Python import graph* -- a different (and more granular)
view than the `data flow` pipeline diagram in `CLAUDE.md`/`README.md`, which is
about which `.npz` file feeds which stage. Use this map when you need to know
"if I change this function's signature, what else breaks", or "where do I even
start reading this codebase."

Regenerate/verify this map yourself at any time with:

```bash
grep -rn "^from qns2q\|^import qns2q" --include="*.py" src/ scripts/
```

## The graph is a strict DAG, in six layers

Each layer only imports from layers above it. There are no cycles.

```
Layer 0  paths.py                         (imports nothing in-repo)
             |
Layer 1  noise/spectra.py --------------- control/tails.py   control/padding.py
             |                             (leaf: no in-repo       (leaf: no in-repo
             |                              imports)                imports)
Layer 2  model/trajectories.py
             |
             +-------------------+-------------------+-------------------+
             |                   |                   |                   |
Layer 3  model/           characterize/       characterize/        control/cz.py
         observables.py   inversion.py        systematics.py       control/idle.py
                                                                    (also import
                                                                     noise/spectra,
                                                                     control/tails,
                                                                     control/padding,
                                                                     paths directly)
             |                                        |
             +--------------------+                   |
             |                    |                   |
Layer 4  characterize/    characterize/                |
         spam.py          single_qubit.py               |
             |                    |                     |
Layer 5  characterize/    (used only by                |
         experiments.py    scripts/run_single_qubit.py) |
         characterize/                                  |
         reconstruct.py                                  |
             |                                            |
             +--------------------+-----------------------+
                                   |
Layer 6  scripts/*.py  (the entry points -- see the table below)
                        viz/cz_pulse_plot.py also lives here (imports only paths.py)
```

## Two independent arms meet only at the scripts layer

`characterize/` (the QNS -> reconstruction arm) and `control/` (the gate-design
arm) **never import from each other**. Both are built on the same foundation
(`model/`, `noise/spectra.py`, `paths.py`), but nothing in `characterize/`
imports `control/cz.py`/`control/idle.py`, or vice versa. The two arms are
stitched together only by scripts that explicitly import both:

- `scripts/harvest_design_numbers.py` imports `control.cz` and `control.idle`
  (as `czmod`/`idmod`) to read their saved gate-optimization outputs, plus
  `paths` to find the reconstruction folder those outputs were built from.
- `scripts/showcase_storage_panel.py` imports `control.idle` only.

This mirrors the "two-arm pipeline" architecture described in `CLAUDE.md` --
you can read/modify one arm in isolation without touching the other, as long
as you don't change the `.npz` file formats they hand off through disk.

## Per-file import table

| File | Imports (within `qns2q`) | Imported by |
|---|---|---|
| `paths.py` | *(none)* | almost everything |
| `noise/spectra.py` | `paths` | `model/trajectories.py`, `control/cz.py`, `control/idle.py`, `characterize/experiments.py`, `characterize/reconstruct.py`, `characterize/single_qubit.py`, `scripts/report_showcase_figs.py` |
| `control/tails.py` | *(none)* | `control/cz.py`, `control/idle.py` |
| `control/padding.py` | *(none)* | `control/cz.py`, `control/idle.py` |
| `model/trajectories.py` | `noise/spectra` | `model/observables.py`, `characterize/inversion.py`, `characterize/systematics.py`, `characterize/experiments.py`, `characterize/single_qubit.py`, `characterize/spam.py` |
| `model/observables.py` | `model/trajectories` | `characterize/experiments.py`, `characterize/single_qubit.py`, `characterize/spam.py` |
| `characterize/inversion.py` | `model/trajectories` | `characterize/reconstruct.py` |
| `characterize/systematics.py` | `model/trajectories` | `characterize/reconstruct.py` |
| `control/cz.py` | `control/padding`, `control/tails`, `noise/spectra`, `paths` | `scripts/harvest_design_numbers.py` (as `czmod`) |
| `control/idle.py` | `control/padding`, `control/tails`, `noise/spectra`, `paths` | `scripts/harvest_design_numbers.py`, `scripts/showcase_storage_panel.py` (as `idmod`) |
| `viz/cz_pulse_plot.py` | `paths` | `scripts/run_cz_pulse_plot.py` (via `runpy`) |
| `characterize/spam.py` | `model/observables`, `model/trajectories` | `characterize/experiments.py` |
| `characterize/single_qubit.py` | `model/observables`, `model/trajectories`, `noise/spectra`, `paths` | `scripts/run_single_qubit.py` (via `runpy`) |
| `characterize/experiments.py` | `characterize/spam`, `model/observables`, `model/trajectories`, `noise/spectra`, `paths` | `scripts/run_capture_arm.py`, `scripts/run_spam_experiments.py` |
| `characterize/reconstruct.py` | `characterize/inversion`, `characterize/systematics`, `noise/spectra`, `paths` | `scripts/run_capture_arm.py`, `scripts/run_reconstruct.py` (via `runpy`), `scripts/run_spam_reconstruct.py`, `scripts/report_showcase_figs.py` |

`scripts/calibrate_showcase.py` and `scripts/compare_reopt_caches.py` import **no**
`qns2q` modules at all -- they are standalone (the first derives constants offline,
without importing the module they feed; the second is a numerical audit script
against saved `.npz` caches). `scripts/verify_embedding_reduction.py` is *almost*
standalone: its pure-NumPy identity checks need no `qns2q` import, but one optional
check (`extract_featured_pairblock`) does a LOCAL, function-scoped
`from qns2q.control.idle import (Config, prepare_time_domain_overlap,
evaluate_overlap_folded)` so the rest of the file stays importable without JAX/the
repo environment installed.

## Cross-file "attribute contracts" (not plain imports)

Two scripts reach into `control/cz.py`/`control/idle.py` via `module.attr`
rather than `from ... import name`, so a plain "who imports this name" grep
misses them. If you rename anything in `control/cz.py`/`control/idle.py`, also
check these usages:

- `scripts/harvest_design_numbers.py`: `czmod.CZOptConfig`,
  `czmod.calculate_infidelity`, `idmod.Config`, `idmod.calculate_infidelity`
- `scripts/showcase_storage_panel.py`: `idmod.Config`,
  `idmod.calculate_idling_fidelity`, `idmod.prepare_time_domain_overlap`,
  `idmod.evaluate_overlap_folded`, `idmod.make_tk12`

## External (third-party) dependencies

See `pyproject.toml` (canonical, pinned) and `requirements.txt` (same pins).
In brief: `numpy`/`scipy` (numerics), `jax`/`jaxlib` (JIT + GPU-batched
simulation, `jax_enable_x64` on everywhere), `qutip`/`qutip-qip` (operator
algebra), `matplotlib` (figures), `joblib` (parallel correlation-function
estimation), `pytest` (the test suite).
