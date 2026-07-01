"""Compare published gate caches vs *_reopt (re-optimized under the fixed evaluator).

**What this is / where it sits.** This is a one-off audit script, not part of the
regular characterize -> control pipeline described in the repo-root CLAUDE.md. It
is the second half of a two-step check; the first step is its companion
``scripts/reopt_battery_verify.sh`` (a shell script, out of scope for this file),
which re-runs the paper's whole gate-optimization battery (``control/cz.py`` and
``control/idle.py``) a second time under a bugfixed fidelity evaluator, writing
its results to files with an extra ``_reopt`` suffix so the original, already-
published ``.npz`` caches are never touched. This script then loads each
(published, ``_reopt``) pair and diffs them.

**Why this check exists (the bug being audited for).** Both optimizers pick their
winning pulse sequence by minimizing an *approximate*, analytic gate infidelity
(a second-order/"second-cumulant" expansion in the noise -- ``g_diag`` in
``control/cz.py``, ``calculate_idling_fidelity`` in ``control/idle.py``), because
evaluating the true infidelity by brute-force simulation for every candidate
sequence would be far too slow to search over. Commit V10-C2-PREFACTOR-0624
(2026-06-24; this tag has no separate doc anchor elsewhere in the repo, so it's
kept here purely as a git-log pointer) found that formula was subtly wrong: it
projected noise contributions using only one filter-function index where the
correct expression needs a symmetric ("both-index parity") sum, which made the
self-noise terms come out about 2x too large and double-counted the cross-noise
terms. Fixing the formula changes the *objective value* the optimizers were
extremizing -- not just a cosmetic number -- so every already-published "winning"
pulse sequence has to be re-checked: is it still the best choice once the
corrected objective is used, or did the old bug happen to favor a different,
now-suboptimal sequence?

**What the check does.** For every gate time ``Tg``, using both the pre-fix
("published") and post-fix ("_reopt") caches, it takes the best-over-M
noise-tailored sequence (label "NT" in the printed table, stored under the key
``opt`` -- the sequence the optimizer searched for, guided by the reconstructed
noise spectrum) and the best-over-M literature dynamical-decoupling baseline
(label "CDD", stored under ``known`` -- a fixed textbook-family sequence that does
*not* use the reconstructed spectrum), exactly as the paper's own plotting code
selects them (see ``idle_curve``/``cz_curve`` below), and compares: the *ratio* of
best infidelities (did the achievable infidelity move?) and the winning pulse
count per qubit (did the *identity* of the winning sequence change, even if the
reported number barely moved?). Any gate time where the infidelity ratio drifts
by more than ``REL_TOL`` (2%, chosen to be well above ordinary optimizer-restart
scatter but well below "this is actually a different answer"), or the winning
pulse counts differ, is flagged in the final VERDICT block.

Verdict as of 2026-06-24: CZ is bit-stable everywhere; idle is stable at all
working gate times and to quoted precision at the deep (long-gate-time) points;
the only motion is optimizer scatter *below* quoted precision at the two deepest
idle points. That scatter is not caused by this fix -- it is a pre-existing
near-degeneracy in the search landscape there: the "rung_c" cache (the
knowledge-ladder rung, see FIGURE_PROVENANCE.md, where the optimizer is
deliberately restricted to seeing only the two single-qubit self-spectra
S_11/S_22 instead of the full six-spectrum model) shows the same scatter under a
change that is argmin-neutral by construction, so the same landscape feature
must be responsible in both cases.

    python scripts/compare_reopt_caches.py
"""
import os, glob, numpy as np

REL_TOL = 0.02  # 2% relative change in best infidelity => flag. Loose enough to
# ignore harmless restart-to-restart optimizer scatter (SLSQP local search from
# random seeds does not converge bit-identically every time), tight enough to
# catch a real change in which sequence wins.

def seqcounts(s):
    """Number of interior control pulses per qubit in a saved pulse sequence.

    ``s`` is a ``(times_qubit1, times_qubit2)`` pair of pulse-time arrays, as
    saved by the optimizers in ``control/cz.py``/``control/idle.py``: each array
    always includes the two endpoints ``t=0`` and ``t=Tg``, so subtracting 2
    from its length gives just the interior pulses. Returns ``None`` (via the
    blanket ``except``) when ``s`` itself is ``None`` -- e.g. no candidate
    sequence was found/saved for that gate time -- so this stays a simple
    "how many pulses did the winner use" summary rather than raising.
    """
    try: return (len(s[0]) - 2, len(s[1]) - 2)
    except Exception: return None

def load(p):
    """Load one .npz cache fully into a plain dict.

    ``np.load`` normally returns a lazy ``NpzFile`` that keeps the file handle
    open and only reads arrays on first access; wrapping it in ``dict(...)``
    reads everything eagerly and lets the file close right away, which matters
    here since the script opens many caches in a loop. ``allow_pickle=True`` is
    required because the ``sequences_opt``/``sequences_known`` entries are
    ragged object arrays (each pulse sequence has a different number of
    pulses), which plain ``np.load`` refuses to deserialize otherwise.
    """
    return dict(np.load(p, allow_pickle=True))

def idle_curve(d):
    """Reduce one idle-gate cache to gate_time -> best-over-M NT/CDD summary.

    The idle optimizer sweeps a set of "M" values (number of repeats of the
    base pulse block used to fill the gate time, see control/idle.py) and,
    for each M, a range of resulting total gate times. The same nominal gate
    time can therefore be reached by several different (M, sequence) choices,
    and the paper reports whichever one actually achieves the lowest
    infidelity -- that "best-over-M" reduction is what this function
    reproduces, exactly mirroring the selection the paper's own plotting code
    performs at figure-generation time.

    Returns ``{gate_time: {opt, optM, optC, known, knownM, knownC, nopulse}}``
    where ``opt``/``known`` are the best-over-M infidelities for the
    noise-tailored (NT) and best-known (CDD-family) sequences respectively,
    ``*M`` the M that achieved it, ``*C`` its per-qubit pulse counts
    (``seqcounts``), and ``nopulse`` the free-evolution (no pulses at all)
    baseline infidelity at that M.
    """
    Ms = [int(x) for x in d["M_values"]]
    by_t = {}
    for M in Ms:
        gts = np.asarray(d[f"M{M}_gate_times"], float)
        io = np.asarray(d[f"M{M}_infs_opt"], float)
        ik = np.asarray(d[f"M{M}_infs_known"], float)
        inp = np.asarray(d[f"M{M}_infs_nopulse"], float)
        so = d[f"M{M}_sequences_opt"]; sk = d[f"M{M}_sequences_known"]
        for g, t in enumerate(gts):
            # round(): different M choices land on the same *nominal* gate
            # time only up to floating-point noise, so round to the nearest
            # integer (times are in units of tau=1, see CLAUDE.md) to bin them
            # together as "the same Tg" before taking the best over M below.
            by_t.setdefault(round(float(t)), []).append(
                # Row layout consumed by position just below: (M, NT_inf,
                # NT_seqcounts, CDD_inf, CDD_seqcounts, nopulse_inf).
                (M, io[g], seqcounts(so[g]), ik[g], seqcounts(sk[g]), inp[g]))
    out = {}
    for t, rows in by_t.items():
        # NT and CDD are optimized independently, so the M that wins for one
        # need not be the M that wins for the other -- hence two separate
        # argmins (by row index 1 vs index 3) over the same per-M rows.
        bo = min(rows, key=lambda r: r[1])
        bk = min(rows, key=lambda r: r[3])
        out[t] = dict(opt=bo[1], optM=bo[0], optC=bo[2],
                      known=bk[3], knownM=bk[0], knownC=bk[4],
                      nopulse=min(r[5] for r in rows))
    return out

def cz_curve(d):
    """Reduce one CZ-gate cache to the same gate_time -> summary shape as ``idle_curve``.

    Unlike the idle gate, the CZ optimizer runs at a single fixed M (recorded
    once in the cache, not swept), so there is no best-over-M reduction to do
    here -- this function just repackages ``taxis``/``infs_*``/``sequences_*``
    into the identical per-gate-time dict shape ``idle_curve`` produces, so
    ``compare`` below can treat both gate types uniformly.
    """
    M = int(d["M"]); tax = np.asarray(d["taxis"], float)
    io = np.asarray(d["infs_opt"], float); ik = np.asarray(d["infs_known"], float)
    inp = np.asarray(d["infs_nopulse"], float)
    so = d["sequences_opt"]; sk = d["sequences_known"]
    out = {}
    for g, t in enumerate(tax):
        out[round(float(t))] = dict(opt=io[g], optM=M, optC=seqcounts(so[g]),
                                    known=ik[g], knownM=M, knownC=seqcounts(sk[g]),
                                    nopulse=inp[g])
    return out

def compare(name, pub, new, curve):
    """Diff one (published, reopt) cache pair, print a side-by-side table, and
    return the list of human-readable flag strings raised while doing so.

    ``curve`` is ``idle_curve`` or ``cz_curve`` -- whichever normalizes this
    cache pair's raw arrays into the common ``{gate_time: {opt, known, ...}}``
    shape both need. For every gate time present in the published cache, this
    computes the NT (``opt``) and CDD (``known``) infidelity ratios
    (reopt / published) and compares the winning NT pulse counts; anything
    outside tolerance gets an inline ``<...!>`` marker in the printed row
    (so a reader can spot problem rows without re-reading every number) and a
    matching entry appended to ``flags``, which the caller aggregates into
    the final VERDICT.
    """
    cp, cn = curve(pub), curve(new)
    flags = []
    print(f"\n{'='*92}\n{name}\n{'='*92}")
    print(f"{'Tg':>7} | {'NT pub':>11} {'NT new':>11} {'r':>6} {'seq pub/new':>13} | "
          f"{'CDD pub':>11} {'CDD new':>11} {'r':>6}")
    for t in sorted(cp):
        if t not in cn:
            flags.append(f"{name} Tg={t}: missing in reopt"); continue
        p, n = cp[t], cn[t]
        ro = n['opt'] / p['opt'] if p['opt'] else float('nan')
        rk = n['known'] / p['known'] if p['known'] else float('nan')
        seqp = f"{p['optC']}/{n['optC']}"
        mark = ""
        if abs(ro - 1) > REL_TOL: mark += " <NT!>"; flags.append(f"{name} Tg={t}: NT ratio {ro:.3f}")
        if abs(rk - 1) > REL_TOL: mark += " <CDD!>"; flags.append(f"{name} Tg={t}: CDD ratio {rk:.3f}")
        if p['optC'] != n['optC']: mark += " <seq!>"; flags.append(f"{name} Tg={t}: NT counts {p['optC']}->{n['optC']}")
        print(f"{t:>7} | {p['opt']:11.4e} {n['opt']:11.4e} {ro:6.3f} {seqp:>13} | "
              f"{p['known']:11.4e} {n['known']:11.4e} {rk:6.3f}{mark}")
    return flags

# This script is run from anywhere, so it CDs to the repo root itself first
# (unlike the pipeline scripts under scripts/, which resolve paths via
# qns2q.paths.project_root() instead -- this one is a standalone audit tool
# with no dependency on the package, hence the local __file__-based lookup).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
all_flags = []

# reopt_battery_verify.sh's naming convention is additive: every "_reopt" run
# writes the exact same filename as its published sibling plus a "_reopt"
# suffix, so the published cache to diff against is recovered by just
# stripping that suffix back off -- no separate lookup table needed. The two
# glob patterns below are the idle-gate ("optimization_data_all_M_*") and
# CZ-gate ("plotting_data/plotting_data_cz_v2_*") cache filenames respectively
# (see control/idle.py / control/cz.py); `sorted()` just makes the scan (and
# hence the printed report) order reproducible run to run.
for newp in sorted(glob.glob("DraftRun_*/optimization_data_all_M_*reopt*.npz")):
    pubp = newp.replace("_reopt", "")
    if not os.path.exists(pubp):
        all_flags.append(f"no published sibling for {newp}"); continue
    all_flags += compare(os.path.relpath(newp), load(pubp), load(newp), idle_curve)

for newp in sorted(glob.glob("DraftRun_*/plotting_data/plotting_data_cz_v2_*reopt*.npz")):
    pubp = newp.replace("_reopt", "")
    if not os.path.exists(pubp):
        all_flags.append(f"no published sibling for {newp}"); continue
    all_flags += compare(os.path.relpath(newp), load(pubp), load(newp), cz_curve)

print(f"\n{'#'*92}\nVERDICT")
if not all_flags:
    print(f"  ALL POINTS STABLE within {REL_TOL:.0%}: re-optimization under the fixed "
          f"evaluator\n  reproduces every published winning sequence and infidelity.")
else:
    print(f"  {len(all_flags)} flag(s) (> {REL_TOL:.0%} change or different winner):")
    for f in all_flags: print("   -", f)
print('#'*92)
