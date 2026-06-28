# Figure provenance map

Maps every figure the paper uses (`~/Noise_Tailored_2Q_Gates/figures/*.pdf`, referenced
from `main_v10.tex`) to the exact **(run folder, data file, script, command)** that
produces it. Rewritten 2026-06-16 after the repo was leaned down to the machinery that
regenerates the paper results (`CLEANUP-0616`); the pre-cleanup history of this file is in
git.

## One regime, five run folders

Everything the paper shows is the **showcase** regime (`QNS2Q_REGIME=showcase`), the
June-13 featured model with the shared TLF in `Re S_{1,2}`. All paper data
lives in five run folders (all other `DraftRun_*`/`GateRun_*`/`backup_*` dirs were
holdover and were deleted):

| Folder | Holds |
|---|---|
| `DraftRun_NoSPAM_showcase_cap` | the 256k characterization + both gates' optimization + margins + design ladder + storage panel (the bulk of the figures) |
| `DraftRun_SPAM_showcase_reference` | SPAM arm: no-SPAM reference (64k) |
| `DraftRun_SPAM_showcase_raw` | SPAM arm: unmitigated |
| `DraftRun_SPAM_showcase_mitigated` | SPAM arm: mitigated |
| `DraftRun_SPAM_showcase_robust` | SPAM arm: SPAM-robust (4 spectra, M-regression) |
| `DraftRun_NoSPAM_showcase_cap_diag3`, `…_robust4` | knowledge-ladder rungs feeding `design_numbers.npz` only |

These folders are tracked as **small summary npz only**; the multi-GB raw-trajectory
caches (`phases.npz`) and regenerable `figures/` trees were trimmed. The `.npz` files
needed for every figure (incl. the formerly-untracked `design_numbers.npz`,
`storage_panel.npz`, and the `*_short`/`_diag3`/`_robust4`/`_rung_d` summaries) are
whitelisted in `.gitignore` and committed, so a fresh clone regenerates all figures.

## The 9 figures

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

### Three standalone figures

| Paper file | Command | Notes |
|---|---|---|
| `C_1_0_MT_vs_M.pdf` | `python scripts/run_single_qubit.py` | self-contained seeded single-qubit SPAM-robustness sim; no run folder needed (writes `C_1_0_MT_1_vs_M_formal.pdf`, manually renamed) |
| `spectral_reconstruction_cross_pub.pdf` | `python scripts/run_reconstruct.py --folder DraftRun_NoSPAM_showcase_cap` | appendix cross-spectra validation; faithful regeneration (matplotlib stamps a fresh timestamp, so not byte-identical to the shipped PDF) |
| `showcase_pulse_sequences.pdf` | `python scripts/run_cz_pulse_plot.py --folder DraftRun_NoSPAM_showcase_cap --tag _cap` | best-known vs best-NT CZ pulse comparison; faithful regeneration |

The `--folder`/`--tag` arguments were added in `CLEANUP-0616` so these two stock entry
points target the `_cap` folder/filenames directly (the showcase data carries a vestigial
`_cap` filename tag). `run_reconstruct.py` re-saves `specs.npz` into the target folder, so
point it at a scratch copy if you want to preserve the committed `specs.npz`.

## Re-deriving the design ladder from scratch

`showcase_design.pdf` reads the committed `design_numbers.npz`. To rebuild it from the
underlying optimization runs:

```bash
python scripts/harvest_design_numbers.py    # -> DraftRun_NoSPAM_showcase_cap/design_numbers.npz
```

This consumes the `_rung_c`, `_diag3`, `_robust4`, and per-SPAM-arm `_rung_d` summaries
(all whitelisted/committed). `storage_panel.npz` is rebuilt with
`python scripts/showcase_storage_panel.py`.

## Re-running characterization

The raw per-shot `phases.npz` caches were dropped. To re-derive `results.npz` (and hence
re-reconstruct from scratch) re-run Stage 1: `python scripts/run_capture_arm.py` (NoSPAM)
or `python scripts/run_spam_experiments.py {reference|raw|mitigated}` (SPAM arms). The
small summary npz already on disk are sufficient for every figure without this.
