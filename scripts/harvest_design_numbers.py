"""SHOWCASE-0612 design-section harvest (conformed to the 06/11 template).

**What "SHOWCASE-0612"/"06/11 template" mean, in plain language.** These are
dated internal shorthand, not physics: "SHOWCASE-0612" tags every code path
(here and in the sibling `control/cz.py`/`control/idle.py`) that belongs to
the showcase-regime control-bandwidth gate-design battery the manuscript's
numbers are drawn from -- kept as a grep-able cross-reference across those
files, the tag itself carries no numeric meaning. "The 06/11 template" refers
to the structure of `report_lorenza_figs.py`, an earlier report-generation
script (deleted in the `CLEANUP-0616` repo cleanup, see CLAUDE.md) that first
established which numbers the design section needed and how to lay out this
harvest; this file is the showcase-specific rewrite that still follows that
layout.

**Physics role / pipeline position.** This script is Stage 4 bookkeeping, not
a simulation stage: it runs strictly AFTER Stage 3a/3b's gate-optimization
batteries (`control/cz.py`, `control/idle.py`, invoked in bulk by
`scripts/run_carrier_battery_0616.sh`) have already been executed and their
results saved to disk. All it does is re-load those already-computed `.npz`
outputs, re-evaluate a couple of saved winning pulse sequences against
different noise-model assumptions (predicted-vs-true, with/without two-qubit
channels), and re-package the resulting handful of numbers into one small
summary file, `design_numbers.npz`, plus a human-readable printout, for the
manuscript's "What the gate design needs from the spectra" section.

**Inputs.** Per-gate-time-point CZ optimizer output
(`<folder>/plotting_data/plotting_data_cz_v2_<tag>.npz`) and per-M idle
optimizer output (`<folder>/optimization_data_all_M_<tag>.npz`) from several
run folders/tags (the knowledge-ladder rungs `DraftRun_NoSPAM_showcase_cap[_diag3|_robust4]`
and the four SPAM arms `DraftRun_SPAM_showcase_<reference|raw|mitigated|robust>`,
resolved via `qns2q.paths.run_folder`), plus the `control.cz`/`control.idle`
modules themselves (imported as `czmod`/`idmod`) to re-evaluate saved winner
sequences with `CZOptConfig`/`Config` + `calculate_infidelity` under
counterfactual spectral models (see `_zero_2q` and the "error budget" section
of `main`).

**Outputs.** `<CAP>/design_numbers.npz`, where `CAP =
"DraftRun_NoSPAM_showcase_cap"`. Its key names (`cz_ladder_*`, `id_ladder_*`,
`cz_arm_*`, `id_arm_*`, `cz_budget_share_*`, `id_budget_share_nt`) are a
NAMING CONTRACT with the only downstream reader, `scripts/report_showcase_figs.py`
(panel `fig_design`, which draws `showcase_design.pdf` -- see
FIGURE_PROVENANCE.md) -- do not rename them without updating that script too.

Collects, from the shared-carrier battery outputs, everything Sec. "What the
gate design needs from the spectra" quotes:

  * the knowledge ladder, both gates: true infidelity of the blind NT
    ("noise-tailored" -- a free-pulse-timing optimization, as opposed to a
    textbook library sequence) winner, where the optimizer was restricted to
    see only a subset of the reconstructed 4x4 spectral matrix: 1Q-2 = only
    the two single-qubit self-spectra S_11/S_22 ("rung_c" below, built with
    `--self-only`); diag-3 = + the S_1212 (two-qubit Ising/"ZZ") self-
    spectrum; robust-4 = + the S_1_2 cross-spectrum (the most any SPAM-robust
    reconstruction can ever recover, since it cannot access S_1_12/S_2_12);
    full-6 = every channel. This ladder is what quantifies, in the paper,
    "how much does actually characterizing more of the noise environment buy
    the gate design" -- plus the best textbook (CDD-family) sequence as a
    baseline for comparison. Evaluated at CZ Tg = 320 tau; idle best-over-M
    at Tg = 640 and 10240 tau (`ID_TGS` below -- a DIFFERENT idle gate time,
    2560 tau, is used later for the error-budget section, computed
    independently of this ladder, so don't conflate the two 2560-vs-10240
    numbers if re-deriving this file from the paper text);
  * the SPAM-arm design blocks, both gates: best CDD (true infidelity), best
    NT (true infidelity), and best NT (infidelity PREDICTED using only that
    SPAM arm's own -- possibly SPAM-biased -- reconstruction, i.e. what an
    experimentalist running that arm would have believed the gate could do
    before checking it against the true noise model). Comparing true vs.
    predicted for the SPAM-free `reference` arm specifically is what the
    paper calls "prediction accuracy": with no SPAM bias present, any gap
    there is pure reconstruction noise/model error, which calibrates how
    much of the larger gaps seen for `raw`/`mitigated`/`robust` is instead
    attributable to SPAM;
  * the error budget: share of the FID (free-induction decay, i.e. the
    "do-nothing" identity sequence) and of the blind winner's residual error
    that is carried by the two-qubit channels (S_1212 + all three
    cross-spectra), obtained by zeroing exactly those channels (`_zero_2q`)
    in the TRUE-model evaluation of the fixed, already-chosen sequences --
    i.e. "how much of the error would disappear if the qubits only ever felt
    independent single-qubit dephasing, with the sequence held fixed";
  * prediction accuracy of the reference arm (predicted vs true, both
    gates) -- not separately computed; read directly off the `reference`
    entry written by the SPAM-arm design blocks above.

Writes <cap folder>/design_numbers.npz and prints a readable summary.

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/harvest_design_numbers.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root, run_folder
from qns2q.control import cz as czmod
from qns2q.control import idle as idmod

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"
# ***IMPORTANT / NOT A BUG***: the robust SPAM arm intentionally carries NO
# "rung_d" gate-design data. Only its noise-spectrum RECONSTRUCTION
# (specs.npz) feeds the robust curve in showcase_spam_arms.pdf; nobody ever
# ran control/cz.py or control/idle.py against that reconstruction to design
# a gate from it (see FIGURE_PROVENANCE.md, "Robust SPAM arm — intentional
# design-data omission"). Every "rung_d" loop in `main()` below therefore
# wraps its file load in `try/except (FileNotFoundError, KeyError)` and just
# SKIPS the robust arm when it hits the missing files -- that exception IS
# the expected, correct behavior here, not a sign that a battery run failed.
# The practical upshot: the robust arm ends up with no bar at all in the
# showcase_design.pdf SPAM-arm panels (b)/(d).
ARMS = ("reference", "raw", "mitigated", "robust")
CZ_TG = 320.0
ID_TGS = (640.0, 10240.0)


def _zero_2q(SMat):
    """Zero out every two-qubit-noise entry of a 4x4 spectral matrix in place
    (functionally -- JAX arrays are immutable, so each `.at[...].set(...)`
    returns a new array), leaving only the two single-qubit self-spectra
    S_11 (index 1) and S_22 (index 2) untouched.

    `SMat`/`SMat_ideal` (built by `CZOptConfig`/`idle.Config`) index the
    three physical noise channels as 1 = qubit-1 Z dephasing, 2 = qubit-2 Z
    dephasing, 3 = the two-qubit Ising ("ZZ") coupling channel S_1212; index
    0 is an unused placeholder row/column carried along for indexing
    convenience. This helper clears row/col 3 (S_1212) and all three
    off-diagonal cross-spectra pairs ((1,2), (1,3), (2,3) + their Hermitian-
    conjugate transposes), which is exactly "every channel a single isolated
    qubit could not feel by itself". Passing the result back into
    `calculate_infidelity(..., use_ideal=True)` therefore answers a
    counterfactual: what would this SAME fixed pulse sequence's true
    infidelity be if the environment only ever produced independent
    single-qubit dephasing noise? The difference from the un-zeroed
    infidelity is the "two-qubit-channel share" of the error budget quoted
    in `main()`.
    """
    for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)):
        SMat = SMat.at[r, c].set(0.0)
    return SMat


def cz_true_at(folder, tag, tg=CZ_TG, kind='opt'):
    """Look up one CZ-gate infidelity number from an already-finished
    `control/cz.py` optimization run.

    Loads `<folder>/plotting_data/plotting_data_cz_v2_<tag>.npz` (written by
    `control/cz.py`'s `run_optimization()`; `allow_pickle=True` is required
    because that file also stores ragged per-gate-time pulse-sequence arrays,
    not just plain numeric columns -- see `cz_winner_seq` below), finds the
    saved gate-time grid point (`taxis`) closest to the requested `tg` (a
    nearest-index match via `argmin`, since the optimizer only swept a
    handful of discrete gate times), and returns the corresponding
    infidelity. `kind='known'` selects the best textbook/library (e.g. CDD)
    sequence found for that gate time; `kind='opt'` (default) selects the
    best noise-tailored (NT), free-pulse-timing-optimized sequence -- these
    match the `infs_known`/`infs_opt` array names `control/cz.py` writes.
    """
    d = np.load(os.path.join(ROOT, folder, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    tgs = np.asarray(d['taxis'], dtype=float)
    i = int(np.argmin(np.abs(tgs - tg)))
    return float(np.asarray(d[f'infs_{kind}'], dtype=float)[i])


def cz_winner_seq(folder, tag, tg=CZ_TG, kind='opt'):
    """Same lookup as `cz_true_at`, but returns the winning PULSE SEQUENCE
    itself instead of its infidelity number: a `(pt1, pt2)` pair of pulse-
    timing arrays for qubit 1 and qubit 2 respectively. Converting the
    loaded plain-NumPy arrays to `jnp.asarray` here (rather than leaving them
    as NumPy) is needed because this sequence is later fed back into
    `czmod.calculate_infidelity`, a JAX-based function that expects
    `jax.numpy` arrays. This is used when `main()` needs to re-evaluate an
    already-chosen winner against a DIFFERENT spectral model than the one it
    was designed on (e.g. predicted-vs-true, or with two-qubit channels
    zeroed by `_zero_2q`), rather than just reading off its originally
    recorded infidelity.
    """
    d = np.load(os.path.join(ROOT, folder, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    tgs = np.asarray(d['taxis'], dtype=float)
    i = int(np.argmin(np.abs(tgs - tg)))
    s = d[f'sequences_{kind}'][i]
    return (jnp.asarray(s[0]), jnp.asarray(s[1]))


def idle_best_over_M(folder, tag, tg, kind='opt', want_seq=False):
    """Look up the idle-gate infidelity at total gate time `tg`, taking the
    BEST result over every repetition count `M` that was swept for this run.

    `control/idle.py` decomposes a total idle duration `Tg` into `M`
    back-to-back repeats of a shorter base pulse block (`Tg = M * (base
    block duration)`); for a fixed `Tg`, different choices of `M` trade off
    differently against the noise spectrum, so `optimization_data_all_M_<tag>.npz`
    stores results for every `M` in `d['M_values']` under keys
    `M<m>_gate_times`, `M<m>_infs_known`/`M<m>_infs_opt`, and
    `M<m>_sequences_known`/`M<m>_sequences_opt`. This helper scans all of
    those `M` values, keeps only the ones whose swept gate-time grid actually
    contains `tg` (`np.isclose` rather than exact `==` because the grid is
    floating point), and returns whichever `M` gave the lowest infidelity --
    matching how the paper always reports "best over M" rather than a single
    fixed M. `kind='known'`/`'opt'` select the best library vs. NT sequence,
    exactly as in `cz_true_at` above. If `tg` isn't present for ANY swept M
    in this run, returns NaN (and, if `want_seq`, `m=0` and `seq=None`)
    rather than raising, so a caller can detect "this design point wasn't
    run" without a crash. Pass `want_seq=True` to also get back the winning
    `M` and its pulse sequence (needed the same way as `cz_winner_seq`
    above, to re-evaluate that sequence under a different spectral model)
    instead of just the bare infidelity float.
    """
    d = np.load(os.path.join(ROOT, folder, f"optimization_data_all_M_{tag}.npz"),
                allow_pickle=True)
    best = None
    for m in (int(x) for x in d['M_values']):
        gts = np.asarray(d[f'M{m}_gate_times'], dtype=float)
        ix = np.where(np.isclose(gts, tg))[0]
        if not ix.size:
            continue
        k = int(ix[0])
        inf = float(d[f'M{m}_infs_{kind}'][k])
        seq = d[f'M{m}_sequences_{kind}'][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, (jnp.asarray(seq[0]), jnp.asarray(seq[1])))
    if best is None:
        return (np.nan, 0, None) if want_seq else np.nan
    return best if want_seq else best[0]


def main():
    """Run every harvest step described in the module docstring, in order
    (knowledge ladders -> SPAM-arm design blocks -> error budget), printing a
    human-readable summary as it goes and accumulating every number into the
    plain dict `out`. At the very end `out` is unpacked with `np.savez(path,
    **out)` -- the `**out` here is Python's "unpack a dict into keyword
    arguments" syntax, which is how one dict holding an arbitrary, growing
    set of `name: value` pairs turns into `np.savez`'s `name=value, ...`
    call without having to list every field twice; every array later comes
    back out of the saved `.npz` file under that same key name.
    """
    out = {}

    # ---- knowledge ladders --------------------------------------------------
    # rung_c/diag3/robust4/full are the four "how much of the reconstructed
    # spectrum did the optimizer get to see" rungs described in the module
    # docstring above; the run-folder suffixes ("_diag3", "_robust4") and
    # filename tags ("rung_c_cap", "diag3_cap", ...) are a naming CONTRACT
    # with scripts/run_carrier_battery_0616.sh (which produces these folders)
    # and scripts/report_showcase_figs.py (which reads the `out` keys below
    # via its own hard-coded rung names) -- do not rename them here without
    # updating both of those scripts.
    print("== CZ knowledge ladder (true 1-F at Tg=320) ==")
    cz_ladder = dict(
        cdd=cz_true_at(CAP, 'cap', kind='known'),
        rung_c=cz_true_at(CAP, 'rung_c_cap'),
        diag3=cz_true_at(f"{CAP}_diag3", 'diag3_cap'),
        robust4=cz_true_at(f"{CAP}_robust4", 'robust4_cap'),
        full=cz_true_at(CAP, 'cap'),
    )
    for k, v in cz_ladder.items():
        print(f"  {k:8s} {v:.4e}")
        out[f'cz_ladder_{k}'] = v

    print("== idle knowledge ladder (true 1-F, best over M) ==")
    # Same four rungs as the CZ ladder above, but each entry here is a
    # (run-folder suffix, filename tag) pair rather than a bare tag, because
    # the idle battery writes one folder per rung (matching the CZ rungs)
    # under a DIFFERENT tag convention (`*_idle_cap`); `rung_c`/`full` reuse
    # the base `CAP` folder (no extra suffix) since only diag3/robust4 got
    # their own copied-and-redacted `specs.npz` (see
    # scripts/run_carrier_battery_0616.sh step "[2/8]").
    id_tags = dict(rung_c=('', 'rung_c_idle_cap'),
                   diag3=('_diag3', 'diag3_idle_cap'),
                   robust4=('_robust4', 'robust4_idle_cap'),
                   full=('', 'cap'))
    for tg in ID_TGS:
        cdd = idle_best_over_M(CAP, 'cap', tg, kind='known')
        out[f'id_ladder_cdd_{int(tg)}'] = cdd
        print(f"  Tg={tg:6.0f}  cdd      {cdd:.4e}")
        for k, (sub, tag) in id_tags.items():
            v = idle_best_over_M(f"{CAP}{sub}", tag, tg)
            out[f'id_ladder_{k}_{int(tg)}'] = v
            print(f"  Tg={tg:6.0f}  {k:8s} {v:.4e}")

    # ---- SPAM-arm design blocks --------------------------------------------
    # For each SPAM protocol arm: kt = best known/CDD-family sequence's TRUE
    # infidelity, nt = best noise-tailored (NT) sequence's TRUE infidelity,
    # npred = that same NT winner's infidelity as PREDICTED from the arm's
    # own reconstruction alone (`use_ideal=False` below -- see the note next
    # to `calculate_infidelity`'s two branches in control/cz.py /
    # control/idle.py: `use_ideal=True` always evaluates against the current
    # analytic ground-truth model regardless of what was reconstructed, while
    # `use_ideal=False` evaluates against `config.SMat`, i.e. whatever
    # spectra this specific config/run actually characterized). Comparing nt
    # (true) vs npred (predicted) for the `reference` (no-SPAM) arm is the
    # "prediction accuracy" number described in the module docstring.
    # `ARMS` is iterated in the FIXED order (reference, raw, mitigated,
    # robust); see the big comment at the `ARMS` definition above for why the
    # robust arm's `try` block is EXPECTED to always raise FileNotFoundError
    # here and get skipped -- that arm has no rung_d gate-design data by
    # design, not by omission/bug.
    print("== CZ SPAM-arm designs (Tg=320): known_true / nt_true / nt_pred ==")
    for arm in ARMS:
        folder = run_folder(spam=True, protocol=arm)
        tag = f"rung_d_{arm}"
        try:
            kt = cz_true_at(folder, tag, kind='known')
            nt = cz_true_at(folder, tag, kind='opt')
            cfg = czmod.CZOptConfig(fname=folder, min_sep_factor=8.0,
                                    max_pulses=10**9, gate_time_factors=[])
            seq = cz_winner_seq(folder, tag)
            npred = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG,
                                                     use_ideal=False))
        except (FileNotFoundError, KeyError) as e:
            # Expected for the robust arm (see comment at ARMS above) --
            # print-and-continue rather than letting the battery abort.
            print(f"  {arm:10s} SKIPPED (no rung-d gate data: {e})")
            continue
        out[f'cz_arm_{arm}'] = np.array([kt, nt, npred])
        # seq[i] is the (n_i + 2)-length array of absolute pulse times
        # [0, t1, ..., T] for qubit i (see control/cz.py's
        # `delays_to_pulse_times`); subtracting the 2 boundary sentinels (the
        # start-of-gate and end-of-gate times, which are not real pi-pulses)
        # gives n_i, the actual pulse COUNT reported in the "NT(n1,n2)"
        # sequence label.
        print(f"  {arm:10s} {kt:.4e} / {nt:.4e} / {npred:.4e}  "
              f"(label NT({len(seq[0])-2},{len(seq[1])-2}))")

    print("== idle SPAM-arm designs (Tg=640, best over M) ==")
    for arm in ARMS:
        folder = run_folder(spam=True, protocol=arm)
        tag = f"rung_d_idle_{arm}"
        try:
            kt = idle_best_over_M(folder, tag, 640.0, kind='known')
            nt, m, seq = idle_best_over_M(folder, tag, 640.0, want_seq=True)
            cfg = idmod.Config(fname=folder, M=m, max_pulses=10**9,
                               min_sep_factor=8.0)
            npred = float(idmod.calculate_infidelity(seq, cfg, m, 640.0 / m,
                                                     use_ideal=False))
        except (FileNotFoundError, KeyError) as e:
            # Same expected robust-arm skip as the CZ block above.
            print(f"  {arm:10s} SKIPPED (no rung-d gate data: {e})")
            continue
        out[f'id_arm_{arm}'] = np.array([kt, nt, npred])
        print(f"  {arm:10s} {kt:.4e} / {nt:.4e} / {npred:.4e}  (M={m})")

    # ---- error budget: 2Q-channel share -------------------------------------
    # "Share of the error carried by the two-qubit channels" = 1 - (infidelity
    # with only single-qubit noise) / (infidelity with the full noise model),
    # for a FIXED, already-chosen pulse sequence. Both infidelities are
    # evaluated with `use_ideal=True` (the true analytic ground-truth model,
    # never the reconstruction) since this is a property of the physical
    # noise model and the sequence, not of what any QNS experiment measured.
    # `_zero_2q` (see its docstring above) is what removes the two-qubit
    # channels between the "full" and "only1q" evaluations.
    print("== error budget: two-qubit-channel share (truth, fixed sequences) ==")
    cfg = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                            gate_time_factors=[])
    seq = cz_winner_seq(CAP, 'cap')  # the blind NT winner at Tg=320 (full-6 rung)
    full = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    cfg.SMat_ideal = _zero_2q(cfg.SMat_ideal)
    only1q = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    out['cz_budget_share_nt'] = (full - only1q) / full
    # The FID (free-induction decay) comparison sequence is not a real
    # optimized sequence at all -- it is just "no pulses in between the start
    # and end of the gate", i.e. the qubits simply idle and dephase freely
    # for the whole Tg. Building it directly as a 2-point [0, Tg] time array
    # (rather than calling into a sequence-library constructor) is enough
    # because `calculate_infidelity` only needs the pulse TIMES, and "no
    # pulses" is exactly the sequence with just the start/end boundary times.
    fid_seq = (jnp.array([0., CZ_TG]), jnp.array([0., CZ_TG]))
    cfg2 = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                             gate_time_factors=[])
    fid_full = float(czmod.calculate_infidelity(fid_seq, cfg2, 1, CZ_TG,
                                                use_ideal=True))
    cfg2.SMat_ideal = _zero_2q(cfg2.SMat_ideal)
    fid_1q = float(czmod.calculate_infidelity(fid_seq, cfg2, 1, CZ_TG,
                                              use_ideal=True))
    out['cz_budget_share_fid'] = (fid_full - fid_1q) / fid_full
    print(f"  CZ 320: FID share {out['cz_budget_share_fid']:.1%}, "
          f"NT-winner share {out['cz_budget_share_nt']:.1%}")

    # Idle-gate error budget at Tg=2560 tau specifically (a THIRD gate time,
    # independent of the CZ_TG=320 used above and the ID_TGS=(640,10240) used
    # in the knowledge ladder -- see the module docstring's note on not
    # conflating these). `idle_best_over_M` re-finds whichever M gave the
    # best NT winner at this Tg, and `idmod.Config(..., M=m, ...)` rebuilds a
    # matching config so the sequence can be re-evaluated.
    nt, m, seq = idle_best_over_M(CAP, 'cap', 2560.0, want_seq=True)
    icfg = idmod.Config(fname=CAP, M=m, max_pulses=10**9, min_sep_factor=8.0)
    full = float(idmod.calculate_infidelity(seq, icfg, m, 2560.0 / m,
                                            use_ideal=True))
    icfg.SMat_ideal = _zero_2q(icfg.SMat_ideal)
    only1q = float(idmod.calculate_infidelity(seq, icfg, m, 2560.0 / m,
                                              use_ideal=True))
    out['id_budget_share_nt'] = (full - only1q) / full
    print(f"  idle 2560 (M={m}): NT-winner share {out['id_budget_share_nt']:.1%}")

    path = os.path.join(ROOT, CAP, "design_numbers.npz")
    np.savez(path, **out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
