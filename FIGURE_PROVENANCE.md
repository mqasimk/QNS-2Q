# Figure provenance map

Maps each publication figure used by the paper
(`~/Noise_Tailored_2Q_Gates/figures/*.pdf`, referenced from `main_v9.tex`) to the
exact **(noise model, run data, config, script)** that produced it.

**Status of this file (updated 2026-06-08, evening):** ⚠️ **Largely superseded.** The
regime/`run_paths.py` restructure described below as "planned" has **landed**, and the
featured-idling + CZ runs described below as "lost" have been **regenerated**. Read the
superseding section immediately below; the older sections are kept for historical context.

## ✅ UNCAP update (2026-06-12) — separation-limited pulse budgets

The 6/11 Lorenza report's gate figure (`reports/lorenza_0611/figs/fig_gates.pdf`)
and §gates numbers now come from **uncapped** (separation-limited, `--max-pulses 0`)
reruns on the 64k reference arm: `DraftRun_SPAM_featured_reference/plotting_data/
plotting_data_cz_v2_uncapped.npz`, `optimization_data_all_M_uncapped.npz`,
`margin_band_{cz,id}_uncapped.npz` (`run_cz.py`/`run_idle.py`/`run_margin_band*.py`
with `--tag uncapped`; idle additionally `--max-dim 2600`, clips logged). Published-cap
files remain untagged alongside. Every npz records its `max_pulses` (10^9 =
separation-limited). NoSPAM-arm uncapped twins live in `DraftRun_NoSPAM_featured/`.

## ✅ Current state (2026-06-08) — restructure landed + data regenerated

- **Regime selection** is now the `QNS2Q_REGIME` env var (`bland`|`featured`), resolved
  centrally in `src/run_paths.py`. The `git show 77e516a^:…` model-swap dance and the
  `Featureless`/`FeatureFull`/`Feature`/`Boring` folder names below are **obsolete**.
  Canonical run folders are now `DraftRun_NoSPAM_bland` and `DraftRun_NoSPAM_featured`
  (both tracked in git).
- **Data is no longer lost.** Seeded regenerations (`RANDOM_SEED = 20260608`) on
  2026-06-08:
  - **CZ-featured** — `cz_optimize.py` with `use_simulated=False` (reconstructed spectra)
    and the CZ-CUMULANT-4X fix → `plotting_data/plotting_data_cz_v2.npz`, `infs_*_cz_v2.npz`.
  - **idling-featured & idling-bland** — `id_optimize.py` → `optimization_data_all_M.npz`,
    `plotting_data/plotting_data_id_v4.npz`, `infs_*_id_M*.npz`.
- **Reproduce now** (no model-swap needed): `cd src/ && export QNS2Q_REGIME=featured`
  (or `bland`), then run the relevant stage; outputs land in `DraftRun_NoSPAM_<regime>/`.
- ⚠️ The regenerated CZ and featured-idling **numbers differ from `main_v9.tex`** — the
  published runs used a different, now-superseded noise model (the *deterministic* FID
  differs ~4.8×). The manuscript numbers need updating; see REPRO-CZ / REPRO-IDLE-FEAT
  in `PAPER_CODE_AUDIT.md`.
- The line 169–170 note below (`use_simulated=True` → reconstructed) is **obsolete on
  both counts**: `main()` now sets `use_simulated=False`, and `True` would load the
  *simulated* spectra, not reconstructed.

---

_(Everything below predates the restructure and is retained for history only.)_

**Status of this file (2026-06-08, original):** all nine referenced figures are now traced
and confirmed (previously four idling rows were "best guess" and the CvsM row was a TODO).

**Why a figure generally can't be reproduced by just running a script at `HEAD`:**
1. `DraftRun_*/` output dirs are **gitignored** (`.gitignore: DraftRun_*/`) — run data
   and figures were never tracked, so several runs are gone.
2. The noise model in `spectra_input.py` was **changed mid-project** — commit `77e516a`
   (2026-04-05), *"Replace bland noise spectra with featured multi-peak PSDs."* Switching
   back to bland currently requires `git show 77e516a^:src/spectra_input.py`.
3. The plot/optimize scripts carry **stale data-folder defaults** (see the "Stale
   defaults" table) — running them as-committed reads the wrong (or an empty/nonexistent)
   folder.
4. The runtime config **differs from the committed dataclass defaults** (see "Shared
   config") — so "just run `main()`" does not reproduce the paper numbers.

A repo restructure to make regime-switching painless is planned; paths/names below
describe the **current** state and will be updated when that lands.

---

## Noise models

| Tag | Where | Self-spectra (S_11, S_22, S_1212) |
|---|---|---|
| **BLAND** (monotonic) | `git show 77e516a^:src/spectra_input.py` | `S_11=2.5e5·L(w,0,1e-6)`, `S_22=1e5·L(w,0,1.5e-6)`, `S_1212=1e6/(1+2·1e-5·|w|)` — monotonic, no peaks |
| **FEATURED** (multi-peak) | `src/spectra_input.py` at `HEAD` (since `77e516a`) | `S_11`=peak@5MHz+plateau@16MHz+peak@27MHz+DC; `S_22`=plateau@8MHz+peak@20MHz+bump@28MHz+DC; `S_1212`=peak@12MHz+hump@23MHz+DC |

Empirically verified from the stored spectra: `Featureless/specs.npz` S_11 has **0 interior
peaks** (bland); `FeatureFull/specs.npz` S_11 has **5 peaks** (featured).

Cross-spectra (both models, unchanged across the switch):
`S_a,b = sqrt(S_aa·S_bb)·exp(-i·w·γ)`, helpers `L` (Lorentzian), `Gauss` (Gaussian).

## Run folders (all gitignored, under repo root)

⚠️ The folder names are **misleading**: `Featureless` holds the **bland** run,
`FeatureFull` holds the **featured** run.

| Folder | Model | Holds | Feeds paper figures |
|---|---|---|---|
| `DraftRun_NoSPAM_Featureless` | **BLAND** | QNS `results/specs.npz` + idling-opt (`optimization_data_all_M.npz`, `plotting_data/plotting_data_id_v4.npz`, `infs_*_id_M{1..128}.npz`, `params.npz`) | both reconstruction figs + both *boring* idling figs |
| `DraftRun_NoSPAM_FeatureFull` | **FEATURED** | QNS `results/specs.npz` only (no idling/CZ opt) | none directly (featured QNS recon exists but isn't the in-paper version) |
| `DraftRun_NoSPAM_Boring` | — | empty (only a stray `figures/`) | none — delete |
| `DraftRun_NoSPAM_Feature` | — | **does not exist** | the CZ scripts default here → they fail |

## Shared QNS / reconstruction config

These are the **runtime values stored in `params.npz`** (identical in both `Featureless`
and `FeatureFull`). ⚠️ They are **NOT** the committed `QNSExperimentConfig` dataclass
defaults — the committed defaults are `M=18, t_grain=1500, truncate=5, w_grain=1000,
n_shots=2000, SPAM ON (a_sp=[0.99,0.98], c=[0.01j,-0.02j], a1=0.99…)`, `fname=DraftRun_MScaling`.
Reproduction must use the `params.npz` values (or apply these as explicit overrides):

```
tau=2.5e-8 s   M=10   t_grain=3000   truncate=20   w_grain=500
T=160*tau=4.0e-6 s    gamma=T/14     gamma_12=T/28   n_shots=10000
SPAM OFF: a_sp=[1,1], c=[0,0], a1=b1=a2=b2=1, spMit=False
=> wmax = 2*pi*truncate/T = 3.14e7 rad/s  (= 5 MHz on the omega/2pi axis)
   comb harmonics wk = 2*pi*(k+1)/T,  k=0..truncate-1
```

## Stale defaults to override before reproducing

| Script | Hardcoded default | Problem | Fix when reproducing |
|---|---|---|---|
| `id_plots.py` `get_data_paths()` | `base_dir = DraftRun_NoSPAM_Boring` | empty folder → no data | repoint to `Featureless` (bland) or the featured idling run |
| `id_optimize.py` `Config(fname=...)` | `DraftRun_NoSPAM_Featureless` | always the bland folder | set a featured folder + switch model to make featured idling data |
| `cz_optimize.py` / `cz_plots.py` / `cz_pulse_plot.py` | `DraftRun_NoSPAM_Feature` | folder doesn't exist | point at a real CZ run folder |

## When the figures were generated

Git last-commit date of every paper figure is **2026-05-21** (commit `544b43f`, the v8
freeze) — except `spectral_reconstruction_cross_pub.pdf` (`7676f45`, 2026-06-08, the
LV0606-XFIG cleanup). The uniform `2026-06-08 07:28` filesystem mtimes are a bulk
`git checkout` artifact, **not** a regeneration — use the commit dates as the anchor.

---

## Figure map (all confirmed)

### QNS reconstruction (validation) — ✅ reproducible from disk

| Paper file | LaTeX label | Script :: function | Model | Run |
|---|---|---|---|---|
| `spectral_reconstruction_all_pub.pdf` | `fig::QNSboringself` (§IV) | `reconstruct_spectra.py :: plot_all_spectra` | **BLAND** | `Featureless` |
| `spectral_reconstruction_cross_pub.pdf` | `fig::QNScrossboring` (App.) | `reconstruct_spectra.py :: plot_cross_spectra` | **BLAND** | `Featureless` |

Reproduce (regenerates both into `Featureless/figures/reconstruction/`):
```bash
cd src/
cp spectra_input.py /tmp/featured_backup.py            # safety
git show 77e516a^:src/spectra_input.py > spectra_input.py   # restore BLAND model
../venv/bin/python -c "import reconstruct_spectra as r; \
  r.SpectraReconstructor(r.SpectraReconConfig('DraftRun_NoSPAM_Featureless')).run()"
git checkout -- spectra_input.py                       # restore FEATURED model
```
- `spectral_reconstruction_self_pub.pdf` exists in the paper repo but is an **orphan**
  (not referenced by `main_v9.tex`).

### SPAM-robust illustration — ⚠️ code-only (no saved data)

| Paper file | LaTeX label | Generator | Model | Status |
|---|---|---|---|---|
| `C_1_0_MT_vs_M.pdf` | `fig::CvsM` | `single_qubit_qns.py :: main()` | unrecorded | re-run from code |

- `main()` sweeps `m_values = range(5, 20)` (M=5..19), single qubit `l=1`, experiment
  `C_1_0_MT_1` with sequences `['CDD1', 'CDD1-1/2']`, `exp_type='C_a_0'`.
- Uses the **committed `QNSExperimentConfig` dataclass DEFAULTS** (only `M` is overridden):
  so `truncate=5`, `t_grain=1500`, `w_grain=1000`, `n_shots=2000`, **SPAM ON**
  (`a_sp=[0.99,0.98]`, `c=[0.01j,-0.02j]`). This is the **only** paper figure run with SPAM
  on — plausibly intentional (it illustrates the SPAM-robust estimator), but **confirm
  before regenerating**.
- `save_results()` is commented out (`single_qubit_qns.py:227`) → **no `params.npz`, no run
  folder** was ever saved. The only record is the code.
- Output filename is `C_1_0_MT_1_vs_M_formal.pdf` (`single_qubit_qns.py:253`); the paper file
  `C_1_0_MT_vs_M.pdf` is a **manual rename**.
- Model (bland vs featured) is unrecorded; committed 2026-05-21 (post featured-switch) so
  the live `spectra_input.py` would have been **featured** at run time, but unconfirmed.

### Idling gate — split: bland reproducible, featured data lost

| Paper file | LaTeX label | Generator | Model | Run / status |
|---|---|---|---|---|
| `infidelity_vs_gatetime_id_best_M_boring_pub.pdf` | `fig:idling_fidelity_boring` | `id_plots.py` best-M | **BLAND** | `Featureless` — ✅ on disk |
| `spectra_overlay_S1212_pub.pdf` | `fig:S1212overlapboring` | `id_plots.py :: generate_spectra_overlay_plot` (`suffix='S1212'`) | **BLAND** | `Featureless` seqs + bland `spectra_input` — ✅ on disk |
| `infidelity_vs_gatetime_id_best_M_pub.pdf` | `fig:idling_infidelity` | `id_plots.py` best-M | **FEATURED** | ❌ **data gone** — re-run `id_optimize.py` (featured) |
| `spectra_overlay_S22_pub.pdf` | `fig:S22overlap` | `id_plots.py :: generate_spectra_overlay_plot` (`suffix='S22'`) | **FEATURED** | ❌ **data gone** — needs the featured idling run |

Confirmation (visual + data): the `_boring` figure shows NT (optimized) lying exactly on
top of CDD (no optimization advantage; plain `CDD` labels), matching `Featureless`
(`plotting_data_id_v4.npz`: median NT/CDD ≈ 1.26, NT beats CDD at only 50% of gate times).
The non-`boring` figure shows NT clearly beating CDD and contains `mqCDD` sequences in its
labels — a different (featured) run with no surviving data.

Idling-run config (from `Featureless` data + `id_optimize.py`):
- `M_values = [2**i for i in range(8)]` = `{1,2,4,8,16,32,64,128}`
- gate-time grid = `320 * 2^k * tau` for `k=0..5` → `{320, 640, 1280, 2560, 5120, 10240} τ`
  (the `best-M` plot picks the best sequence at each gate time across all M)
- `min_gate_time = π/(4·Jmax) = 15.708 τ` with hardcoded `Jmax=2e6`
  (`id_optimize.py:1131`) — this is the paper caption's `T_{G,min}≈15.71τ`; sweet spot
  `T_G=320τ` is the grid start. **(Resolves tracker item LV0606-NUM-ANCHOR.)**
- `id_optimize.py` random-restart knobs: `reps_known=range(100,401,10)`,
  `reps_opt=range(100,401,20)`, SLSQP `maxiter=1000`.

Reproduce the **bland** idling figures (after repointing `id_plots.py` `base_dir` →
`Featureless`):
```bash
cd src/
git show 77e516a^:src/spectra_input.py > spectra_input.py   # bland (for the overlay spectrum)
../venv/bin/python id_plots.py
git checkout -- spectra_input.py
```

### Entangling gate (CZ) — ❌ data missing (re-run required)

| Paper file | LaTeX label | Generator | Model | Status |
|---|---|---|---|---|
| `infidelity_vs_gatetime_pub.pdf` | `fig:infidelity_vs_time` | `cz_plots.py` | featured¹ | no CZ run on disk; folder `DraftRun_NoSPAM_Feature` absent |
| `pulse_sequence_comparison_pub.pdf` | `fig:seq_comparison` | `cz_pulse_plot.py` | featured¹ | same — re-run `cz_optimize.py` |

¹ Inferred from the `…_Feature` folder name; unconfirmed. `cz_optimize.py` `main()` uses
`CZOptConfig(use_simulated=True)` → it consumes a reconstructed `specs.npz` (a featured one
exists in `FeatureFull`); `gate_time_factors=[-3..3]` (7 gate times), SLSQP `maxiter=2000`,
`min_gate_time = π/(4·Jmax)`.

### Pending (no figure file yet — `[FIGURE:]` stubs in `main_v9.tex`)
- SPAM-mitigated QNS validation — needs a SPAM-injected run (none on disk).
- CZ entangling spectral-overlap panel — needs the CZ run data.

---

## Reproducibility summary

| Figures | Count | Status |
|---|---|---|
| recon (×2) + boring idling (×2) | 4 | ✅ reproducible from on-disk `Featureless` (bland) |
| featured idling (×2) | 2 | ❌ re-run `id_optimize.py` with featured model |
| CZ (×2) | 2 | ❌ re-run `cz_optimize.py` |
| CvsM (×1) | 1 | ⚠️ re-run `single_qubit_qns.py` (confirm SPAM/model first) |

Figure-feeding data is **small** (`specs.npz` 8K, `optimization_data_all_M.npz` 220K,
`plotting_data_id_v4.npz` 4K, `params.npz` 268K) — committing per-regime summaries to git
would cost <0.5 MB/regime and prevent future loss.

---

_Last updated 2026-06-08. All nine referenced figures traced and confirmed._
