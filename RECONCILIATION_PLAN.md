# Repository reconciliation plan — characterization & control pipeline

This plan reorganizes `QNS-2Q` to mirror the companion manuscript's end-to-end
pipeline (`Noise_Tailored_2Q_Gates/main_v9.tex`, Fig. 1): a **characterization**
arm (noise environment → frequency-comb QNS → unbiased spectra) feeding a
**control** arm (PTM gate model → constrained optimization → noise-tailored
pulses).

It is staged so the high-value, low-risk cleanup (already done) is separated from
the larger package move (deferred until you want it).

---

## Stage 0 — DONE (this pass)

- **Removed 5 dead source files** (no live importers; confirmed by grep):
  `cz_optimize_legacy.py`, `id_optimize_v2_legacy.py`, `id_optimize_v3_legacy.py`,
  `opt_plot.py`, `opt_window.py`. `src/` went 21 → 16 modules (+ the DC prototype).
- **Reconciled `README.md`**: dropped the dead-file listing; fixed the Stage-1
  parameter defaults (`M` 1→10, `t_grain` 10000→3000, `w_grain` 10000→500) to match
  `qns_experiments.py`. Fixed the stale `cz_optimize.py` module docstring.
- `GEMINI.md` does not exist (earlier inventory was mistaken); `CLAUDE.md` remains
  the authoritative run guide.

---

## Stage 1 — Target package layout (DONE — validated by 97 passing tests)

> Executed via `migrate_to_package.py` (now removed). All modules moved into the
> layout below, imports rewritten to absolute `qns2q.*` paths, path resolution
> centralized through `qns2q.paths.project_root()` (CWD-independent), `scripts/`
> entry points added, and `tests/` migrated. `pytest tests/` → 97 passed.


```
src/qns2q/
  __init__.py
  paths.py            # <- run_paths.py        (regime + canonical run folders)

  noise/              # the physical environment  (paper stage 1)
    __init__.py
    spectra.py        # <- spectra_input.py     (S_ab definitions, bland|featured)

  model/              # exact noisy-gate model    (paper stage 4: PTM / cumulant)
    __init__.py
    trajectories.py   # <- trajectories.py      (Hamiltonian, propagators, filter fns)
    observables.py    # <- observables.py       (POVMs, correlation functions)

  characterize/       # QNS arm                   (paper stages 2-3)
    __init__.py
    experiments.py    # <- qns_experiments.py
    single_qubit.py   # <- single_qubit_qns.py
    dc_ramsey.py      # <- dc_ramsey_prototype.py  (promote once accepted)
    inversion.py      # <- spectral_inversion.py
    reconstruct.py    # <- reconstruct_spectra.py

  control/            # gate-design arm           (paper stages 5-6)
    __init__.py
    cz.py             # <- cz_optimize.py
    idle.py           # <- id_optimize.py

  viz/
    __init__.py
    plot_utils.py · cz_plots.py · cz_pulse_plot.py · id_plots.py · qns_plots.py
    optimization.py   # <- plot_optimization.py

scripts/              # thin CLI entry points, one per pipeline stage
  run_experiments.py · run_reconstruct.py · run_dc_ramsey.py
  run_cz.py · run_idle.py · run_figures.py

archive/              # anything retired but worth keeping out of the tree
```

### Why this shape
- The two top-level arms (`characterize/`, `control/`) are exactly the paper's two
  halves, so a reader maps code ↔ manuscript at a glance.
- `noise/` + `model/` are the shared physics both arms import (today's
  `spectra_input` / `trajectories` / `observables`), so the dependency direction is
  acyclic: `noise, model → characterize, control → viz`.
- `scripts/` removes the brittle "must `cd src/` because of `os.pardir`" rule
  (CLAUDE.md): entry points resolve the repo root from `__file__` and pass an
  explicit run folder, so stages become CWD-independent.

## Stage 1 mechanics (when executed)
1. Add `__init__.py` at each level; move files (use `git mv` to preserve history).
2. Rewrite flat imports → package-relative (`from trajectories import …` →
   `from ..model.trajectories import …`). ~16 modules, mechanical.
3. Replace the `os.pardir` run-folder lookup in `qns_experiments`, `reconstruct`,
   `single_qubit` and `spectra_input.__main__` with a single `paths.run_path()` that
   resolves from the repo root (already the pattern in `cz_optimize`/`id_optimize`).
4. Update `tests/conftest.py` to put `src/` on `sys.path` as a package root; update
   the 4 test modules' imports.
5. Update `CLAUDE.md` + `README.md` run commands (`python scripts/run_*.py`).

**Risk / cost:** touches every import + the path logic + tests. ~1 focused pass,
fully mechanical, but it WILL churn every file — hence deferred behind explicit
approval rather than bundled with the science changes.

---

## Stage 2 — Method improvements already landed (selectable, default-off)

These are wired now and do not change default behavior:

- `trajectories.make_noise_mat[_arr](..., midpoint=True)` — bin-midpoint noise
  synthesis grid that removes the spurious `w=0` static tone (fixes a ~10-20% DC
  bias; see `dc_ramsey_prototype.py`).
- `reconstruct_spectra.SpectraReconConfig(inversion_method=…, reg_lambda=…,
  enforce_nonneg=…, diagnostics=…)` — robust inversion backends (`lstsq`,
  `tikhonov`), NNLS non-negativity for self-spectra, and `cond(U)` / truncation-bias
  diagnostics (`spectral_inversion.solve_inverse`, `truncation_bias_estimate`).
- `spectral_inversion.regress_observables_over_M(...)` — SPAM-free M-scaling
  regression core (consumes multi-M observables; the experiment stage must emit
  them — see that function's docstring).

---

## DC method — DONE
The comb-subtraction self-spectrum DC (`recon_S_11_dc`/`_22`/`_1212`) is replaced by
the partner-decoupled FID/CDD3 Ramsey slope `S_aa(0) = 2<C_{a,0}>/(MT)`
(`characterize/inversion.py`), fed by light `['FID','CDD3']`/`['CDD3','FID']`
experiments over a few fast control times (`characterize/experiments.py`). The
midpoint noise grid is the experiment-stage default. End-to-end smoke: S_11/S_22/
S_1212 DC ratios ~1.06 vs analytic truth (was ~5x off via the comb).

## Open items (tracked, not yet done)
- End-to-end M-regression: the reusable core (`inversion.regress_observables_over_M`)
  is in place; add an M-sweep emitter to the experiment stage and a consumer in
  `reconstruct` to use it.
- Re-run the downstream optimizers/figures on the midpoint-regenerated `specs.npz`
  for full consistency (DC barely affects balanced-sequence control, so second-order).
- `PAPER_CODE_AUDIT.md` item `RECON-PREFACTOR-2X` (one-sided PSD 2× convention at DC)
  intersects the DC rework — resolve together.
