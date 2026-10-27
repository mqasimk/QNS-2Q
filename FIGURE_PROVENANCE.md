# Figure provenance map

Maps every figure the paper uses (`~/Noise_Tailored_2Q_Gates/figures/*.pdf`, referenced
from the active `main_v12.tex`; older versions retain their historical sources) to the exact **(run folder, data file, script, command)** that
produces it. Rewritten 2026-06-16 after the repo was leaned down to the machinery that
regenerates the paper results (`CLEANUP-0616`); the pre-cleanup history of this file is in
git.

**TL;DR one-command version:** `python scripts/generate_paper_figures.py` runs every
command below in order (add `--full-regen` to also re-derive the run folders themselves
first, or `--dry-run` to preview). This file remains the source of truth for what each
individual command does and why -- read on if you want to regenerate just one figure, or
if the wrapper script and this file ever disagree (trust this file).

## One regime, five run folders

Everything the paper shows is the **showcase** regime (`QNS2Q_REGIME=showcase`), the
June-13 featured model with the shared TLF in `Re S_{1,2}`. All paper data
lives in five run folders (all other `DraftRun_*`/`GateRun_*`/`backup_*` dirs were
holdover and were deleted):

| Folder | Holds |
|---|---|
| `DraftRun_NoSPAM_showcase_cap` | the 256k characterization + both gates' optimization + margins + design ladder + storage panel (the bulk of the figures) |
| `DraftRun_SPAM_showcase_reference` | SPAM arm: no-SPAM reference (256k) |
| `DraftRun_SPAM_showcase_raw` | SPAM arm: unmitigated |
| `DraftRun_SPAM_showcase_mitigated` | SPAM arm: mitigated |
| `DraftRun_SPAM_showcase_robust` | SPAM arm: SPAM-robust (4 spectra, M-regression) |
| `DraftRun_NoSPAM_showcase_cap_diag3`, `…_robust4` | knowledge-ladder rungs feeding `design_numbers.npz` only |

These folders are tracked as **small summary npz only**; the multi-GB raw-trajectory
caches (`phases.npz`) and regenerable `figures/` trees were trimmed. The `.npz` files
needed for every figure (incl. the formerly-untracked `design_numbers.npz`,
`storage_panel.npz`, and the `*_short`/`_diag3`/`_robust4`/`_rung_d` summaries) are
whitelisted in `.gitignore` and committed, so a fresh clone regenerates all figures.

## The 8 figures

Run from the repo root with the venv active and `QNS2Q_REGIME=showcase`.

### Six showcase panels — `scripts/report_showcase_figs.py`

```bash
export QNS2Q_REGIME=showcase
SHOWCASE_FIGS_DIR=reports/showcase_0613/figs python scripts/report_showcase_figs.py
```

Reads `DraftRun_NoSPAM_showcase_cap/{specs,optimization_data_all_M_cap,margin_band_*_cap,
design_numbers,storage_panel}.npz` + `plotting_data/plotting_data_cz_v2_cap*.npz` and the
four `DraftRun_SPAM_showcase_*/specs.npz`. Emits to `reports/showcase_0613/figs/`; the
paper copies are byte-identical renames:

| Paper file | report fig | LaTeX |
|---|---|---|
| `showcase_model.pdf` | `fig_model_spectra.pdf` | the six-spectrum featured model |
| `showcase_recon.pdf` | `fig_recon_capture.pdf` | blind 256k reconstruction vs truth |
| `showcase_spam_arms.pdf` | `fig_spam_comparison.pdf` | four SPAM arms (reference, raw, mitigated, robust) |
| `showcase_design.pdf` | `fig_design_experiments.pdf` | knowledge-ladder + SPAM design (reads `design_numbers.npz`) |
| `showcase_storage.pdf` | `fig_storage.pdf` | Bell-pair storage split (reads `storage_panel.npz`) |
| `showcase_gates.pdf` | `fig_gates.pdf` | both gates' infidelity + margin bands |

### Two standalone figures

| Paper file | Command | Notes |
|---|---|---|
| `C_1_0_MT_vs_M.pdf` | see the 2026-10-26 command below | 256k realizations; corrected N=1 raw twisted estimator, CPMG controls, source data and provenance retained |
| `showcase_pulse_sequences.pdf` | `python scripts/run_cz_pulse_plot.py --folder DraftRun_NoSPAM_showcase_cap --tag _cap` | best-known vs best-NT CZ pulse comparison; faithful regeneration |

### Single-qubit SPAM-robust illustration: corrected 256k run (2026-10-26)

The current paper figure uses the N=1 protocol from Sec. IV.D: one +X
fiducial preparation, raw X/Y probabilities and an X_1 flip after the basis
rotation, followed by C_hat = -1/2 log[(E_X^-)^2 + (E_Y^-)^2]. There is no
confusion-matrix inversion or preparation/visibility division. Each X/Y
observable has 256,000 independent noise realizations; raw and flipped
probabilities are paired on those realizations, following the pinned robust
simulation convention. Projection sampling remains off (`n_meas=0`).

Only the zeta_1 Z_1 term couples. Its marginal spectrum is S_11 of the pinned
current `showcase` model, as Qasim requested. Spectator and auxiliary factors
in the shared solver are uncoupled identities. CPMG/CDD2 controls use five
cycle counts n=1,...,5, with M=5,...,19, T=160 tau, t_grain=1500,
w_grain=1000 and the existing SPAM parameters. The plot shows ensemble
estimates/errors, free linear fits and the known-model SPAM-offset reference.
Known SPAM values and exact Gaussian ensemble coefficients are diagnostics;
they are not supplied to the estimator or used to correct the fitted slope.
Finite-M filter effects can shift an extrapolated intercept relative to the
pure SPAM contribution; that distinction is retained in the caption.

Exact regeneration from this checkout:

```bash
QNS2Q_REGIME=showcase XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  venv/bin/python -u reports/single_qubit_robust_256k_2026-10-26/_runtime/scripts/run_single_qubit.py \
  --n-realizations 256000 \
  --output-dir /home/mqasimk/IdeaProjects/QNS-2Q/reports/single_qubit_robust_256k_2026-10-26
```

The user subsequently requested integer M tick labels. The simulations use
only M=5,6,...,19; the continuous curves are regression fits. Regenerate the
final plot from the saved numerical data, without another simulation, with:

```bash
QNS2Q_REGIME=showcase MPLBACKEND=Agg \
  PYTHONPATH=reports/single_qubit_robust_256k_2026-10-26/_runtime/src \
  venv/bin/python reports/single_qubit_robust_256k_2026-10-26/plot_source.py \
  --plot-only --output-dir reports/single_qubit_robust_256k_2026-10-26
```

The current stock entry point also supports `--plot-only` and integer ticks.
`plot_source.py` preserves the exact final rendering source; its hash is in
the manifest. Numerical NPZ hashes were unchanged by this axis correction.

The runtime is exported from 4241fbf plus the standalone protocol/streaming
patch. `provenance.json` records exact hashes; `implementation.patch` and
`_runtime/` retain the executed source. `M_XX.npz` records configuration,
twisted means/errors and the independently calculated model coefficient;
`C_1_0_MT_vs_M.npz` contains plotted data and free-fit results. The existing
parametric bootstrap remains unseeded, so saved errors/fit weights have
bootstrap-level variability; noise draws and point estimates use seed
20260608. Float32 arrays use full matrix-multiply precision.

Tests cover the nonzero-noise streamed/dense equivalence, the N=1 reduction,
exact Gaussian/SPAM-offset normalization, asymmetry cancellation, retention of
readout visibility, spectator independence and free-intercept slope recovery.

**Superseded first attempt.** `reports/single_qubit_256k_2026-10-26/` preserved
the legacy readout-corrected two-qubit local estimator. Retitling that result
as a local-coefficient illustration did not fulfill the intended paper
protocol; Qasim identified this mismatch and the corrected run above replaces
it. Keep the earlier outputs for audit, not as the paper's single-qubit result.
The original November-2025 PDF also predates the pinned current noise model;
its exact numerical source was absent. The two-qubit reconstruction and gate
results remain unchanged by either standalone rerun.

The `--folder`/`--tag` arguments were added in `CLEANUP-0616` so these stock entry points
target the `_cap` folder/filenames directly (the showcase data carries a vestigial `_cap`
filename tag).

`run_reconstruct.py` no longer produces a paper figure — the appendix cross-spectra
validation panel (`spectral_reconstruction_cross_pub.pdf`) was dropped from `main_v10.tex`.
It is kept as a load-bearing utility because it re-saves `specs.npz` into the target folder
(point it at a scratch copy if you want to preserve the committed `specs.npz`) and remains a
useful cross-spectra sanity check.

## Re-deriving the design ladder from scratch

`showcase_design.pdf` reads the committed `design_numbers.npz`. To rebuild it from the
underlying optimization runs:

```bash
python scripts/harvest_design_numbers.py    # -> DraftRun_NoSPAM_showcase_cap/design_numbers.npz
```

This consumes the `_rung_c`, `_diag3`, `_robust4`, and per-SPAM-arm `_rung_d` summaries
(all whitelisted/committed). `storage_panel.npz` is rebuilt with
`python scripts/showcase_storage_panel.py`.

**Robust SPAM arm — intentional design-data omission.** The `reference`, `raw`, and
`mitigated` arms carry `_rung_d`/`_rung_d_idle` gate-design summaries; the `robust` arm
deliberately does **not**. Its reconstruction (`specs.npz`) still feeds the robust curve in
`showcase_spam_arms.pdf`, but it is by design not carried through to gate design, so
`harvest_design_numbers.py` skips it (the `try/except FileNotFoundError` is expected, not an
error) and the robust bar is intentionally absent from `showcase_design.pdf` panels (b)/(d).

## Re-running characterization

The raw per-shot `phases.npz` caches were dropped. To re-derive `results.npz` (and hence
re-reconstruct from scratch) re-run Stage 1: `python scripts/run_capture_arm.py` (NoSPAM)
or `python scripts/run_spam_experiments.py {reference|raw|mitigated}` (SPAM arms). The
small summary npz already on disk are sufficient for every figure without this.
