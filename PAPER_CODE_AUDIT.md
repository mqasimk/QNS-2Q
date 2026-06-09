# Paper ↔ Code agreement audit

**Audited:** `~/Noise_Tailored_2Q_Gates/main_v9.tex` (active manuscript) vs `~/IdeaProjects/QNS-2Q/src/*.py` + committed `DraftRun_NoSPAM_{bland,featured}/` data.
**Date:** 2026-06-08. **Scope:** static check — equations, parameters, methodology, captions vs implementation and committed data. The GPU pipeline was **not** re-run, so post-fix numbers still need a controlled (seeded) regeneration to confirm.
**Method:** 11 comparison dimensions, 121 raw findings, the 29 non-trivial ones adversarially re-verified by independent agents; the two highest-stakes items (CZ-CUMULANT-4X, RECON-PREFACTOR-2X) were additionally re-derived by hand from source.

## How to use this file across sessions
- Each item has a **stable ID** (e.g. `CZ-CUMULANT-4X`). Reference it in commit messages.
- **Status legend:** `OPEN` (not started) · `WIP` (in progress) · `DONE` (fixed + verified) · `WONTFIX` (decided not to act) · `REFUTED` (false alarm, no action).
- When you finish an item, set its status, add a one-line resolution note + commit hash under it, and update the table.
- Paper-side fixes can also be cross-filed in the manuscript's `REVIEW_TRACKER.md`; code-side fixes live here.
- ⚠️ The shipped spectra reconstructions are numerically correct in both regimes — most items below are about **reproducibility, conventions, and presentation**, not wrong physics. The two exceptions are CZ-CUMULANT-4X (real code bug) and the REPRO-* cluster (committed data ≠ published figures).

## Status table

| ID | Sev | Status | One-liner | Fix side |
|---|---|---|---|---|
| CZ-CUMULANT-4X | 🔴 critical | DONE | `calculate_cz_fidelity` over-counts 2nd cumulant 4× | code |
| REPRO-CZ | 🟠 major | DONE | CZ fig/table numbers not reproducible from on-disk npz | data/code |
| REPRO-IDLE-FEAT | 🟠 major | DONE | Featured-idling fig/table numbers not reproducible (diff. noise model) | data/code |
| CZ-USE-SIMULATED | 🟠 major | DONE | CZ optimized on ground-truth spectra, text says reconstructed | code/paper |
| RECON-AXIS-MHZ | 🟠 major | DONE | Reconstruction figure axis mislabeled "MHz" (off by 2π) | code |
| SEED-OPT | 🟠 major | DONE | Optimizers unseeded → headline numbers not pinned | code |
| RECON-PREFACTOR-2X | 🟡 moderate | OPEN | Paper U prefactor 2× code's (self-consistent convention) | paper (doc) |
| HAM-HALF-CONV | ⚪ minor | OPEN | Code H has ½ on noise terms; paper unit-coeff; LV margin note open | paper (doc) |
| CVSM-LCURVES | ⚪ minor | DONE | `single_qubit_qns.py` computes only l=1 but hardcodes 5-curve legend | code |
| OVERLAY-LABELS | ⚪ minor | DONE | Overlay best-seq caption labels don't match current data | paper/data |
| OVERLAY-G-NORM | ⚪ minor | DONE | Overlay right-axis G has extra 1/T_G vs paper def | code |
| MINGATE-METADATA | ⚪ minor | DONE | CZ `min_gate_time` carried into idling data as dead metadata | code |
| CAPTION-DECADES | ⚪ minor | OPEN | "roughly two decades" — bland self-spectra span ~3 | paper |
| FILENAME-CVSM | ⚪ minor | DONE | CvsM output filename manually renamed | code |
| CPMG-DOCSTRING | ⚪ minor | DONE | `cpmg()` docstring says "n π-pulses", produces 2n | code |
| DC-EXP-UNDOC | ⚪ minor | OPEN | FID-based DC (k=0) experiments not in `tab:exps` | paper |
| MULT-NOISE-WORDING | ⚪ minor | OPEN | "actively mitigates multiplicative noise" overstates | paper |
| PROVENANCE-STALE | ⚪ minor | DONE | `FIGURE_PROVENANCE.md` predates regime restructure | doc |
| REFUTED-DC-CONV | — | REFUTED | DC ×2 convention — paper does state it; code correct | — |
| REFUTED-S212-PHASE | — | REFUTED | S_2_12 overlay phase is the physically correct one | — |
| REFUTED-BLAND-RATIO | — | REFUTED | Bland NT/CDD caption matches the data it plots | — |

> **Session 2026-06-08 progress (uncommitted):** all 9 GPU-independent code findings
> are fixed in `src/`, and the seeded GPU regeneration has now run end-to-end.
> CZ-CUMULANT-4X was verified to machine precision (the fixed CZ decay reduces
> *exactly* to the verified-correct `calculate_idling_fidelity` with the rotation off;
> a bit-identical reconstruction confirms the change is live; old/new infidelity ratio
> ≈ 4.01). **Regeneration complete** (seed 20260608): CZ-featured + idling-{featured,
> bland} re-ran cleanly, so REPRO-CZ, REPRO-IDLE-FEAT and OVERLAY-LABELS are now
> resolved on the *data* side — the on-disk npz are correct and reproducible.
> Confirmed: the regenerated CZ and featured-idling numbers do **not** match the
> manuscript (the published runs used a different, now-superseded noise model — the
> *deterministic* FID differs ~4.8×), so the paper numbers will need updating
> (paper-side, deferred by the user). PROVENANCE-STALE refreshed. Remaining: the 5
> `OPEN` paper items in `~/Noise_Tailored_2Q_Gates/`, and a git commit of the
> regenerated data (not yet committed — awaiting go-ahead).

---

## 🔴 CRITICAL

### CZ-CUMULANT-4X — `calculate_cz_fidelity` over-counts the second cumulant by 4×
**Status:** DONE (code; uncommitted) · **Files:** `src/cz_optimize.py:539–556` (objective at `:614`) · **Paper:** eq `c2_spectra` (main_v9.tex:259–260), PTM `:356`, App E.

**Resolution (2026-06-08):** Applied the prescribed fix in `lambda_element`: the decay term is now `-0.5*(sgn(O,a,0)-1) * I_matrix[a,b] * (z2q[a]@z2q[b])` summed over the full `a,b ∈ {1,2,12}` (no `sgn(O,a,b)` gating, no spurious `2.0`); `-1j*rot_op` kept, misleading comment removed. Verified numerically in the `src/` venv: (i) a bit-identical reconstruction of the decay reproduces the **live** `calculate_cz_fidelity` to `0.0` (binds the check to source, J=0 and J≠0); (ii) with rotation off and a *physical* overlap matrix, the fixed CZ fidelity equals `calculate_idling_fidelity` to `0.0` (machine precision) — same verified normalization; (iii) old/new full-PTM infidelity ratio ≈ **4.01** (consistent with the audit's ≈3.92). Headline-number regeneration is tracked under REPRO-CZ.

**Verified by hand from source.** Each decay term is built as:
```python
pre_factor   = -0.5 * (sgn(Oi, idx_i, idx_j) + 1.0)
sgn_term     =  sgn(Oi, idx_i, 0) - 1.0
overlap_term =  2.0 * I_matrix[idx_i, idx_j]   # comment: "Factor of 2 accounts for 1/pi vs 1/2pi"
val_CO      += pre_factor * sgn_term * overlap_term * (z2q[i] @ z2q[j])
# G = expm(-1j*rot_op - val_CO)
```
On the diagonal (`a=b`, `sgn(O,a,a)=+1`) this gives `-2·(sgn(O,a,0)-1)·I`, whereas the paper requires `-(sgn(O,a,0)-1)·(1/2)·I` → **4× too large** (2× from `pre_factor=-1` vs −½, 2× from the spurious `2.0*I`). Off-diagonal terms with `sgn(O,a,b)=-1` are also wrongly zeroed (paper keeps them).

**Proof it's a bug, not a convention:** `calculate_idling_fidelity` (`src/id_optimize.py:644–714`) consumes an `I_matrix` from a **byte-identical** `evaluate_overlap_comb` (both `cz_optimize.py:389–416` and `id_optimize.py:525–580` return `(term_dc + 2*sum_ac)*M/T_seq`), uses `I_matrix` **directly** with the ½ correctly in `E1..E4`, and reproduces eq `c2_spectra` to machine precision. The two paths share the same overlap normalization; only CZ inflates it. Measured full-PTM infidelity ratio (code vs paper-faithful) ≈ **3.92**. The "1/π vs 1/2π" comment is wrong — `evaluate_overlap_comb` already returns the 1/2π-normalized ∫dω/2π S·G.

**Impact:** live optimization objective → every reported CZ infidelity inflated ~4× and the optimum can shift.

**Fix:** mirror `calculate_idling_fidelity`. Remove `2.0*` (use `I_matrix` directly), replace the `pre_factor` with `-0.5*(sgn(O,a,0)-1)` summed over the full `a,b ∈ {1,2,12}` double sum (do **not** gate by `sgn(O,a,b)`), keep `-1j*rot_op`, delete the misleading comment. Then regenerate `infs_*_cz_v2.npz` and re-verify `tab:fidelity_summary`.

---

## 🟠 MAJOR

### REPRO-CZ — CZ figure/table numbers not reproducible from committed data
**Status:** DONE (data regenerated; uncommitted — paper reconciliation deferred) · **Files:** `DraftRun_NoSPAM_featured/plotting_data/plotting_data_cz_v2.npz`, `cz_plots.py` · **Paper:** `fig:infidelity_vs_time` (caption :895), `tab:fidelity_summary` (:880). **Resolution (2026-06-08):** re-ran `cz_optimize.py` (featured) — seeded (20260608), CZ-CUMULANT-4X fixed, reconstructed spectra (`use_simulated=False`). New on-disk values @320τ: **NT 2.06e-2 / CDD 7.88e-2 / FID 1.51e-1** (paper: 4.40e-4 / 1.43e-1 / 4.75e-1). The data is now reproducible from the repo, but the paper's headline CZ sweet-spot (NT 4.4e-4) does **not** reproduce — best NT across all gate times is only ~2.6e-3 (@40τ), with no sharp dip. Remaining (paper-side, deferred): commit the npz + revisit the manuscript CZ numbers.

Paper at `T_G=320τ`: NT `4.40e-4` / CDD `1.43e-1` / FID `4.75e-1`. On-disk: `6.66e-2 / 2.68e-1 / 3.52e-1` (NT ~150× off; global NT min only `7.5e-3`, no sweet-spot dip). The committed PDF (mtime 07:28) **does** show `4.40e-4`; the on-disk npz (mtime 14:20) was regenerated ~7h later by an unseeded re-run using the buggy code (CZ-CUMULANT-4X). Paper is internally consistent (caption = table = PDF); it just can't be regenerated from the repo. **Fix:** after fixing CZ-CUMULANT-4X and SEED-OPT, re-run, confirm/repair the table numbers, commit the exact figure-source npz.

### REPRO-IDLE-FEAT — featured idling figure/table not reproducible (different noise model)
**Status:** DONE (data regenerated; uncommitted — paper reconciliation deferred) · **Files:** `DraftRun_NoSPAM_featured/optimization_data_all_M.npz`, `id_plots.py` · **Paper:** `fig:idling_infidelity` (:826), `tab:fidelity_summary` (:874–877). **Resolution (2026-06-08):** re-ran `id_optimize.py` (featured), seeded. New best-over-M @640τ: **NT 2.93e-3 / CDD 5.76e-3 / FID 2.47e-1** (paper: 7.06e-5 / 3.67e-5 / 5.13e-2). This confirms the diagnosis: values are 1–2 orders off the paper, the **deterministic FID is ~4.8× off** (proves the published figure used a different noise model — not just a different seed), and the 640τ ordering is **reversed** (regenerated NT<CDD at every gate time). Remaining (paper-side, deferred): commit npz + update the six manuscript numbers and the 640τ NT-vs-CDD wording, or locate/restore the original noise model.

The six caption/table numbers (e.g. `640τ`: CDD `3.67e-5`, NT `7.06e-5`, FID `5.13e-2`) are off from on-disk by 2–3 orders of magnitude, and the **`640τ` CDD-beats-NT ordering is reversed** (on disk NT < CDD everywhere). Critically, even the *deterministic* no-pulse FID curve differs ~5× — so the published figure used a **different noise model/parameterization**, not just a different seed. That data is effectively lost. **Fix:** locate/regenerate the source run, re-pin params in provenance; if regenerating, update all six numbers + the 640τ NT-vs-CDD sweet-spot wording.

### CZ-USE-SIMULATED — CZ optimized on ground-truth, not reconstructed, spectra
**Status:** DONE (code; uncommitted) · **File:** `src/cz_optimize.py:1061` (`main()` now sets `use_simulated=False` → loads the reconstructed `specs.npz`, matching the text; was `use_simulated=True`→`simulated_spectra.npz`). **Resolution (2026-06-08):** flag flipped (preferred fix); CZ regen tracked under REPRO-CZ, and the `FIGURE_PROVENANCE.md:170` correction folded into PROVENANCE-STALE. The two files are materially different objects. · **Paper:** lines 695, 708, 928 say the optimization uses the SPAM-free **reconstructed** spectra. `id_optimize.py:1223` correctly uses `use_simulated=False`.

**Fix (preferred):** set CZ `main()` to `use_simulated=False` to match the text, regenerate. **Alt:** add a sentence in §V.C and the `fig:infidelity_vs_time`/`tab:fidelity_summary` captions disclosing CZ used ground-truth spectra while idling used reconstructed. (Also: `FIGURE_PROVENANCE.md:170` wrongly says `True` loads reconstructed — it loads simulated.)

### RECON-AXIS-MHZ — reconstruction figure x-axis mislabeled "MHz"
**Status:** DONE (code; uncommitted) · **File:** `src/reconstruct_spectra.py:247,354,382,445` (plots `ω/1e6` = Mrad/s, labels `ω (MHz)`). **Resolution (2026-06-08):** set `xunits = 2*np.pi*1e6` in both plot methods (matches `plot_utils.py`); the `ω (MHz)` labels are kept. PDFs `spectral_reconstruction_{all,cross}_pub.pdf` still need a plot regen. `plot_utils.py:306,376,482,547` correctly uses `ω/(2π·1e6)`. Off by 2π — the published "5–30 MHz" axis is physically ~0.8–5 MHz. Captions quote no numbers, so no sentence is false. **Fix:** set `xunits = 2*np.pi*1e6` where labeled MHz (or relabel "Mrad/s"); regenerate `spectral_reconstruction_{all,cross}_pub.pdf`.

### SEED-OPT — optimizers unseeded; figure-source data not pinned
**Status:** DONE (code; uncommitted) · **Files:** `cz_optimize.py`, `id_optimize.py` (np.random restarts, unseeded per CLAUDE.md). Root cause of REPRO-CZ / REPRO-IDLE-FEAT recurring. **Resolution (2026-06-08):** added module-level `RANDOM_SEED = 20260608`, seed `np.random` at each optimizer entry point, and record `seed` in the saved npz. Also seeded `single_qubit_qns.py` (its per-shot noise keys come from `np.random.randint`). Committing the exact per-regime figure-source npz is folded into REPRO-CZ / REPRO-IDLE-FEAT. **Fix:** add a fixed RNG seed (and record it), commit the exact per-regime npz backing each published figure (data is small, <0.5 MB/regime).

---

## 🟡 MODERATE

### RECON-PREFACTOR-2X — reconstruction-matrix prefactor convention (no results impact)
**Status:** OPEN · **Files:** `src/spectral_inversion.py` (self `m/T`, cross `2m/T`) · **Paper:** `gencoeff` `:472` (`2M/T` self, `4M/T` cross), App B.

The paper's displayed prefactors are exactly 2× the code's. **Adjudicated by hand:** the code's Hamiltonian carries an explicit `0.5` on every noise term (`trajectories.py:248–250`), halving the measured coefficient (`a()=-0.25*log`, `observables.py:298`); the `M/T` matrix is correspondingly halved — they cancel, and the shipped spectra match ground truth ~1× in both regimes. The paper is internally self-consistent in a two-sided/unit-coefficient convention (`E[X]=e^{-C}`, :1154); the code in a one-sided/half-coefficient convention. **No published number is affected.** Two review agents read it as "paper over-counts, fix paper"; two (with the more careful internal-consistency trace) and my own check say "self-consistent convention." **Fix:** add one sentence near `gencoeff` pinning the `(2−δ_{k,0})` folding convention; re-derive App B once to be certain. No code change.

---

## ⚪ MINOR

### HAM-HALF-CONV — ½ on stochastic Hamiltonian terms vs paper's unit-coefficient H_R
**Status:** OPEN · `trajectories.py:248–250` uses `0.5*b*Z`; paper `H_R`/`H_I` (eqs :151,:181) carry unit coefficient. Zero numerical impact (absorbed by Z's eigenvalue-2 spread; spectra round-trip ~1×). Resolves the authors' open margin note `%% LV: Do you have the 1/2 factor…?` (main_v9.tex:152). **Fix:** add one clarifying sentence in §II.A (state the convention) and delete the LV note.

### CVSM-LCURVES — CvsM script computes only l=1 but hardcodes a 5-curve legend
**Status:** DONE (code; uncommitted) · `single_qubit_qns.py:213` fixes `l_index=1`; runs one experiment; `:247` hardcodes `['l=1'…'l=5']`; `save_results()` commented out `:227`. Per-curve params confirmed correct (T=160τ, τ=25ns, n_shots=2000, α_M=0.97, α_SP=0.99). **Correction to the premise:** the 5 curves are *not* 5 values of the experiment's `l` — there `l` is the **qubit** index, only ever 1 or 2 (`make_c_a_0_mt:550–552` sets `state='p0' if l==1 else '0p'`). They are the `truncate=5` **control-time harmonics** `c_times=T/k` (k=1..5) that `make_c_a_0_mt` already returns per M; `ax.plot(m_values, results)` plots those 5 columns. So no multi-`l` sweep was missing — the hardcoded legend merely mislabeled the harmonic index, and adding a qubit-`l` loop would be wrong. **Resolution (2026-06-08):** legend now built from the actual curve count (`results.shape[1]`); output renamed to `C_1_0_MT_vs_M.pdf`; figure-source data persisted to `C_1_0_MT_vs_M.npz`; RNG seeded; docstring/comments document the `l`(qubit)-vs-harmonic-index collision. The caption clarification remains paper-side.

### OVERLAY-LABELS — overlay caption best-sequence labels don't match current data
**Status:** DONE (data pinned; uncommitted — caption re-transcription deferred, paper-side) · `fig:S1212overlapboring` (:813), `fig:S22overlap` (:835). Captions hand-transcribe `CDD_{3,2}^128`/`NT_{7,13}^64`/`NT_{275,391}^2`; current on-disk winners differ. Overlay code only emits "Known"/"Optimized". **Resolution (2026-06-08):** with SEED-OPT the winners are now pinned/reproducible. New best-over-M labels (both regimes): **CDD(1,2)^64** and **NT(79,158)^2** @320τ; the overlay figures themselves use the longest gate time, so the exact caption labels come from re-running `python id_plots.py`. Remaining (paper-side, deferred): regenerate the overlay PDFs and re-transcribe the three caption labels.

### OVERLAY-G-NORM — overlay right-axis G has an extra 1/T_G
**Status:** DONE (code; uncommitted) · `id_plots.py:868` computes `G = Za·conj(Zb)/(w²·longest_gt)`; paper defines `G = A_aA_b*/ω²` (App E, :1202). Display-only rescaling; optimization unaffected. **Resolution (2026-06-08):** dropped `* longest_gt` so `G = A_a A_b*/ω²` per the paper (needs a plot regen to take visual effect).

### MINGATE-METADATA — CZ min_gate_time carried into idling data
**Status:** DONE (code; uncommitted) · `id_optimize.py:1131` hardcodes `min_gate_time = π/(4·2e6)` (a CZ bound) and saves it into idling `plotting_data`; `id_plots.py` only prints it (axvline commented out). Value 15.708τ matches the paper's `T_{G,min}≈15.71τ` (correctly scoped to the entangling gate at :853). Dead metadata. **Resolution (2026-06-08):** removed the computation and the `plotting_data_id_v4.npz` dict entry; `id_plots.py:140` already reads it guarded (`if 'min_gate_time' in data`), so no plot-side change was needed.

### CAPTION-DECADES — "roughly two decades" understates the span
**Status:** OPEN · `fig::QNSboringself` caption (:787). Bland S11 spans ~3.0 decades, S22 ~3.6 (S1212 ~1.84). **Fix:** "two to three decades" (cosmetic).

### FILENAME-CVSM — CvsM output filename manually renamed
**Status:** DONE (code; uncommitted) · `single_qubit_qns.py:253` saves `C_1_0_MT_1_vs_M_formal.pdf`; paper includes `C_1_0_MT_vs_M.pdf`. **Resolution (2026-06-08):** now saves directly as `C_1_0_MT_vs_M.pdf` (plus a companion figure-source `C_1_0_MT_vs_M.npz`).

### CPMG-DOCSTRING — docstring count off by 2
**Status:** DONE (code; uncommitted) · `trajectories.py:283–284` calls `n` "Number of π-pulses" but `cpmg()` produces 2n (matches the paper's `[CDD₂]_n` — code is correct). **Resolution (2026-06-08):** docstring now reads "n base-cycle (CDD2) repetitions; the sequence has 2n pi-pulses".

### DC-EXP-UNDOC — FID-based DC (k=0) experiments not tabulated
**Status:** OPEN · `qns_experiments.py:233–239` defines 6 FID DC experiments (consumed by `recon_*_dc`, prepended ×2 for one-sided PSD) that materially set the ω=0 points in the reconstruction figures, but they aren't in `tab:exps`. Numerically consistent with the `(2−δ_{k,0})` factor. **Fix:** add a sentence/footnote (or rows) in §IV.A/App B describing the DC protocol.

### MULT-NOISE-WORDING — "actively mitigates multiplicative noise" overstates
**Status:** OPEN · Conclusion :930 says the framework "actively mitigates" multiplicative control noise, but the code (`cz_optimize.py`) only minimizes the additive-channel cost L_G; multiplicative suppression is a real *corollary* (same `F^(1)_12` filter function). Line 940 is adequately hedged. **Fix:** soften :930 to "incidentally suppressing … as a corollary of the y₁₂ toggling (App C)"; optionally add the `|F̄^(1)_12|² = J²|F^(1)_12|²` identity to App C.2.

### PROVENANCE-STALE — FIGURE_PROVENANCE.md predates the regime restructure
**Status:** DONE (doc; uncommitted) · It used old folder names (`Featureless`/`FeatureFull`), claimed CZ/featured data is "lost" (now present under `DraftRun_NoSPAM_{bland,featured}`), and misstated `use_simulated=True`. **Resolution (2026-06-08):** prepended a "✅ Current state" superseding section to `FIGURE_PROVENANCE.md` documenting the landed `QNS2Q_REGIME`/`run_paths.py` restructure, the canonical `DraftRun_NoSPAM_{bland,featured}` folders, the seeded regeneration of the previously-"lost" CZ + featured-idling data, the `use_simulated=False` correction, and the noise-model caveat vs the manuscript; the pre-restructure sections are retained for history.

---

## ✅ REFUTED (false alarms — no action; do not re-investigate)

- **REFUTED-DC-CONV** — claim that the paper omits the k=0 / one-sided ×2 convention. False: `gencoeff` sums from k=0 with `(2−δ_{k,0})` (:468,:472) and App E gives `I^DC` (:1233–1236). Code's `2·S_dc` matches theory to ~3%.
- **REFUTED-S212-PHASE** — claim that the `S_2_12` overlay phase (`γ₁₂−γ`) is wrong. False: it's the *physically realized* relative phase (per-channel time shifts in `trajectories.py`) and matches the reconstruction and every downstream consumer; the bare-`γ₁₂` outlier is only `spectra_input.__main__`, which never feeds the figure.
- **REFUTED-BLAND-RATIO** — claim that the bland "NT doesn't beat CDD" caption is inverted. False: the caption matches the best-over-M data the figure actually plots (median NT/CDD 0.91, small offset on a 2.5-decade log axis); the "1.26" is from a different single-M data product in the (non-authoritative) provenance doc.

---

## Recommended order of work
1. **CZ-CUMULANT-4X** (correctness) → then **SEED-OPT** → regenerate → **REPRO-CZ**, **REPRO-IDLE-FEAT**.
2. **CZ-USE-SIMULATED** (flip flag or disclose).
3. **RECON-AXIS-MHZ**; **RECON-PREFACTOR-2X** + **HAM-HALF-CONV** convention notes.
4. Minor caption/label sweep; **PROVENANCE-STALE** last (after data settles).
