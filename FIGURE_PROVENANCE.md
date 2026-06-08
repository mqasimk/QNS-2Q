# Figure provenance map

Maps each publication figure used by the paper
(`~/Noise_Tailored_2Q_Gates/figures/*.pdf`, referenced from `main_v9.tex`) to the
exact **(noise model, run data, config, script)** used to generate it.

**Why this file exists.** A figure here generally *cannot* be reproduced by just
running a script at `HEAD`, because:
1. `DraftRun_*/` output dirs are **gitignored** — run data (`results.npz`, `specs.npz`)
   and figures were never tracked.
2. The noise model in `spectra_input.py` was **changed mid-project** — commit
   `77e516a` (2026-04-05), *"Replace bland noise spectra with featured multi-peak
   PSDs."* The reconstruction figures currently in the paper predate this and use the
   **bland** model, which now lives only in git history.
3. The `*_plots.py` / `reconstruct_spectra.py` mains carry **stale data-folder
   defaults** (e.g. `DraftRun_NoSPAM_Feature`, which does not exist).

So each figure needs its `(model, run, config)` triple recorded explicitly.

---

## Noise models

| Tag | Where | Self-spectra |
|---|---|---|
| **BLAND** (monotonic) | `git show 77e516a^:src/spectra_input.py` | `S_11=2.5e5·L(w,0,1e-6)`, `S_22=1e5·L(w,0,1.5e-6)`, `S_1212=1e6/(1+2·1e-5·|w|)` |
| **FEATURED** (multi-peak) | `spectra_input.py` at `HEAD` (since `77e516a`) | `S_11`=3 peaks; `S_22`=plateau+peak+bump; `S_1212`=resonance+hump; small DC backgrounds |

Cross-spectra (both models): `S_a,b = sqrt(S_aa·S_bb)·exp(-i·w·γ)`, helpers `L` (Lorentzian),
`Gauss` (Gaussian) — unchanged across the switch.

## Run folders (all gitignored, under repo root)

| Folder | Model | Holds | Used for |
|---|---|---|---|
| `DraftRun_NoSPAM_Featureless` | **BLAND** | QNS `results/specs.npz` + idling-opt `infs_*_id_M*.npz` | reconstruction figs in paper; bland/idling results |
| `DraftRun_NoSPAM_FeatureFull` | **FEATURED** | QNS `results/specs.npz` | featured QNS reconstruction (new model) |
| `DraftRun_NoSPAM_Boring` | — | empty (only `figures/`) | ignore |

## Shared QNS experiment config

Committed `QNSExperimentConfig` defaults (unchanged since the bland era; verified that
the only uncommitted edit is the `fname` output-folder line):

```
tau=2.5e-8 s   M=10   t_grain=3000   truncate=20   w_grain=500
T=160*tau=4.0e-6 s    gamma=T/14     gamma_12=T/28  n_shots=10000
SPAM off: a_sp=[1,1], c=[0,0], spMit=False
=> wmax = 2*pi*truncate/T  (≈ 31.4 MHz on the ω/2π axis)
   comb harmonics wk = 2*pi*(k+1)/T,  k=0..truncate-1
```

---

## Figure map

### QNS reconstruction (validation) — ✅ CONFIRMED reproduced 2026-06-08

| Paper file | LaTeX label | Script :: function | Model | Run |
|---|---|---|---|---|
| `spectral_reconstruction_all_pub.pdf` | `fig::QNSboringself` (§IV) | `reconstruct_spectra.py :: plot_all_spectra` | **BLAND** | `Featureless` |
| `spectral_reconstruction_cross_pub.pdf` | `fig::QNScrossboring` (App.) | `reconstruct_spectra.py :: plot_cross_spectra` | **BLAND** | `Featureless` |

Reproduce (regenerates both files into `Featureless/figures/reconstruction/`):
```bash
cd src/
cp spectra_input.py /tmp/featured_backup.py            # safety
git show 77e516a^:src/spectra_input.py > spectra_input.py   # restore BLAND model
../venv/bin/python -c "import reconstruct_spectra as r; \
  r.SpectraReconstructor(r.SpectraReconConfig('DraftRun_NoSPAM_Featureless')).run()"
git checkout -- spectra_input.py                       # restore FEATURED model
```
- `plot_cross_spectra` (added 2026-06-08) resolves L. Viola's note: in-panel legend
  (top panel) + asinh `SymmetricalLogLocator` y-ticks, replacing the legend-underneath /
  overlapping-ticks of the prior version.
- The FEATURED counterparts come from the same commands with the BLAND-restore lines
  omitted and `'DraftRun_NoSPAM_FeatureFull'` as the folder.
- `spectral_reconstruction_self_pub.pdf` exists in the paper repo but is an **orphan**
  (not referenced by `main_v9.tex`).

### SPAM-robust illustration

| Paper file | LaTeX label | Script | Status |
|---|---|---|---|
| `C_1_0_MT_vs_M.pdf` | `fig::CvsM` | (unverified) | ⚠️ TODO — trace generator + run |

### Idling gate — ⚠️ UNVERIFIED (best guess; trace before relying)

| Paper file | LaTeX label | Likely script | Likely run |
|---|---|---|---|
| `infidelity_vs_gatetime_id_best_M_boring_pub.pdf` | `fig:idling_fidelity_boring` | `id_plots.py` | `Featureless` (bland) |
| `infidelity_vs_gatetime_id_best_M_pub.pdf` | `fig:idling_infidelity` | `id_plots.py` | featured run (location unconfirmed) |
| `spectra_overlay_S1212_pub.pdf` | `fig:S1212overlapboring` | `id_plots.py`/`plot_utils.py` | `Featureless` (bland) |
| `spectra_overlay_S22_pub.pdf` | `fig:S22overlap` | `id_plots.py`/`plot_utils.py` | featured run (location unconfirmed) |

### Entangling gate — ⚠️ DATA MISSING (re-run required)

| Paper file | LaTeX label | Script | Note |
|---|---|---|---|
| `infidelity_vs_gatetime_pub.pdf` | `fig:infidelity_vs_time` | `cz_plots.py` | no CZ run data on disk; `cz_optimize.py` re-run needed |
| `pulse_sequence_comparison_pub.pdf` | `fig:seq_comparison` | `cz_pulse_plot.py` | same — CZ run data missing |

### Pending (no figure file yet — `[FIGURE:]` stubs in `main_v9.tex`)
- `main_v9.tex:649` — SPAM-mitigated QNS validation (needs a SPAM-injected run, none on disk).
- `main_v9.tex:855` — CZ entangling spectral-overlap panel (needs CZ run data).

---

_Last updated 2026-06-08. Extend the UNVERIFIED/TODO rows as each figure is traced._
