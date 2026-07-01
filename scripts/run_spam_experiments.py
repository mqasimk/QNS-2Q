"""Stage 1 of the SPAM (State Preparation And Measurement) pipeline: this is the
entry point that actually runs the simulated QNS experiment suite with realistic
SPAM errors deliberately injected, so the paper can show that its error-mitigation
math (the "SPAM-Mitigated"/"SPAM-Robust" protocols) recovers the correct noise
spectra even when readout and state prep are imperfect.

Where this sits in the pipeline (see CLAUDE.md's "Data Flow" diagram and the
"SPAM pipeline" section): it is the SPAM-arm sibling of
`scripts/run_capture_arm.py` (which runs the idealized, SPAM-free "characterize"
arm). Both scripts are thin wrappers around the same underlying machinery in
`qns2q.characterize.experiments` (the actual simulator: pulse sequences, noise
synthesis, POVM readout) -- this script's only job is to pick a fixed set of
"true" SPAM error values, build the right `QNSExperimentConfig` for the
requested mitigation protocol, and hand both to that module's `main()`. It
takes no `.npz` inputs (Stage 1 always starts from a clean simulated state);
it writes `results.npz` (the measured correlation coefficients + standard
errors) and `params.npz` (the frozen config needed to interpret them) into
`DraftRun_SPAM_<regime>_<protocol>/`, and optionally `phases.npz` (a cached
per-shot noise-phase dataset, see `--record`/`--replay` below). Downstream,
`scripts/run_spam_reconstruct.py` (Stage 2) reads those two files to invert
the correlation coefficients into reconstructed power spectral densities, and
that in turn feeds the paper's SPAM-comparison figures.

Usage (from the repo root, regime via QNS2Q_REGIME):

    python scripts/run_spam_experiments.py [raw|mitigated|robust] [--reduced]

Writes to ``DraftRun_SPAM_<regime>_<protocol>/``. The protocols (see
`QNSExperimentConfig.spam_protocol`'s docstring in
`qns2q/characterize/experiments.py` for the full mechanics of each):

    raw       -- SPAM injected, NO mitigation (quantifies the corruption)
    mitigated -- estimated-parameter mitigation (paper Sec. SPAM-Mitigated QNS):
                 all six spectra
    robust    -- twisting + wringing + M-regression (paper Sec. SPAM-Robust QNS):
                 S_11, S_22, S_1212, S_1_2 (S_l_12 not accessible)

``--reduced`` uses a light config (fewer harmonics/shots/time points) for quick
validation runs; omit it for paper-quality statistics.

The injected SPAM parameters follow the companion paper's simulation values
(alpha_M ~ 0.95-0.97, alpha_SP^z ~ 0.98-0.99) plus nonzero readout asymmetries
delta_l and transverse SP components c_l, so every mitigation channel
(visibility, asymmetry, z- and transverse-SP) is exercised.
"""
import os
import sys

# This script lives in scripts/, one directory below the repo root, but the
# actual physics code is an installed-looking package under src/qns2q/ that
# isn't on Python's default import path when you just run this file directly.
# Inserting "../src" onto sys.path here (before the `import qns2q...` line
# below) lets `python scripts/run_spam_experiments.py` work from any working
# directory without requiring the caller to set PYTHONPATH by hand -- see
# CLAUDE.md's note that every pipeline stage is meant to be CWD-independent.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.characterize.experiments import QNSExperimentConfig, main
from qns2q.paths import run_folder, project_root

# Injected (true) SPAM parameters. alpha_M = a + b - 1, delta = a - b, so
# qubit 1: alpha_M = 0.97, delta = +0.03; qubit 2: alpha_M = 0.95, delta = -0.02.
# These match the companion paper's simulation values (mild, hardware-realistic).
# a1/b1 (qubit 1) and a2/b2 (qubit 2) are the raw per-outcome measurement-fidelity
# parameters from QNSExperimentConfig (see its docstring): they get combined
# downstream into the paper's readout visibility alpha_M = a+b-1 and asymmetry
# delta = a-b. a_sp is the true state-prep visibility alpha_SP^z per qubit
# (1.0 = no error); c is the true transverse (complex) state-prep error per
# qubit. Together these are the "ground truth" SPAM errors that get injected
# into the simulated experiment; the mitigation protocols below only ever see
# ESTIMATES of them (except spam_protocol='none', the oracle case).
SPAM_TRUE = dict(
    a1=1.00, b1=0.97,
    a2=0.965, b2=0.985,
    a_sp=jnp.array([0.99, 0.98]),
    c=np.array([jnp.array(0.02 + 0.04j), jnp.array(0.03 - 0.02j)]),
)

# Strong-SPAM variant (--strong): qubit 1 alpha_M = 0.88, delta = +0.06; qubit 2
# alpha_M = 0.85, delta = -0.05. Why a second, harsher parameter set exists at
# all: at the mild SPAM_TRUE values above, some of the legacy (un-mitigated,
# spam_protocol='raw') estimators' coefficient DIFFERENCES (e.g.
# C_12_0_MT_1 - C_12_0_MT_2) happen to cancel most of the SPAM-induced offset
# algebraically, so a plot of the 'raw' arm at reduced (few-shot) statistics
# can look deceptively close to the SPAM-free answer -- not because the
# mitigation is unnecessary, but because the residual bias is too small to see
# above the statistical noise at that sample size. Injecting these larger,
# still hardware-plausible SPAM values instead makes that residual bias large
# enough to see clearly, which is what a validation run of the 'raw' arm is
# actually trying to demonstrate.
SPAM_TRUE_STRONG = dict(
    a1=0.97, b1=0.91,
    a2=0.90, b2=0.95,
    a_sp=jnp.array([0.95, 0.92]),
    c=np.array([jnp.array(0.06 + 0.10j), jnp.array(0.08 - 0.06j)]),
)


def build_config(protocol: str, reduced: bool, strong: bool = False,
                 medium: bool = False, tuned: bool = False,
                 fine: bool = False) -> QNSExperimentConfig:
    """Assemble the `QNSExperimentConfig` for one SPAM arm.

    Picks which SPAM-error values get injected (mild `SPAM_TRUE`, or the
    harsher `SPAM_TRUE_STRONG` if `strong`), which mitigation `protocol` the
    estimators should use, and which grid/shot-count preset (`reduced` <
    `medium` < `tuned` < `fine`, in increasing cost/precision -- see the
    inline comments below for what each buys) to run at. Only one of
    `reduced`/`medium`/`tuned`/`fine` should be True at a time; if none are,
    `QNSExperimentConfig`'s own defaults are used. Returns the config object
    without running anything -- the caller passes it to
    `qns2q.characterize.experiments.main()` to actually execute the suite.
    """
    if protocol == 'reference':
        # SPAM-free reference arm at the same (reduced) statistics: isolates the
        # SPAM-specific reconstruction bias from the comb/truncation systematics.
        common = dict(spam_protocol='none',
                      fname=run_folder(spam=True, protocol='reference'))
    else:
        # `common` collects the keyword arguments that will eventually be
        # passed to `QNSExperimentConfig(**common)` below. The `**(...)` here
        # is Python's dict-unpacking syntax: it splices every key/value pair
        # from the chosen SPAM dict (SPAM_TRUE or SPAM_TRUE_STRONG) into
        # `common` as if they had been written out by hand (e.g. `a1=1.00,
        # b1=0.97, ...`) -- a compact way to build up a config from reusable
        # named presets instead of repeating every field inline.
        common = dict(
            spam_protocol=protocol,
            fname=run_folder(spam=True, protocol=protocol),
            **(SPAM_TRUE_STRONG if strong else SPAM_TRUE),
        )
    if reduced:
        # Light validation config: same physics, coarser grids and fewer shots.
        common.update(dict(M=8, t_grain=600, truncate=10, w_grain=200,
                           n_shots=2000))
    elif medium:
        # Medium config: full harmonic count at moderate statistics (~7x cheaper
        # than the full paper config per point).
        common.update(dict(t_grain=1000, truncate=20, n_shots=4000))
    elif tuned:
        # Tuned comparison config: an intermediate stepping-stone preset used
        # while dialing in the final paper-quality settings below (`fine`).
        # It doubles the shot count (more statistical precision) and pays for
        # the extra compute by coarsening the time/frequency grids slightly
        # (t_grain 1000->800, w_grain 500->350) -- the comment records that
        # this coarsening was checked beforehand to stay well within the
        # numerical-resolution requirements (grid spacing dt = 0.2 tau is
        # still much finer than the fastest timescale 1/wmax the comb needs to
        # resolve; grid spacing dw = 2.2e-3 is still much finer than the
        # narrowest noise-spectrum feature, of width 0.02, in the Class-F
        # noise model) so it doesn't sacrifice accuracy, only cost. Same
        # number of reconstructed harmonics (truncate=20) as medium/fine.
        # ~37 min/arm; the statistical error bars (which dominate the small
        # channels at the Class-F noise-model's line features) tighten by
        # sqrt(2) vs --medium.
        common.update(dict(t_grain=800, w_grain=350, truncate=20, n_shots=8000))
    elif fine:
        # Fine preset: the paper-quality setting, meant for use with
        # --record/--replay (see the __main__ block below) since replaying a
        # cached noise-phase dataset is what makes running this many shots at
        # full grid resolution affordable -- otherwise synthesizing the noise
        # from scratch at this shot count would be far too slow. It restores
        # the full (uncoarsened) time/frequency grids AND uses the same shot
        # count (256000) as the companion NoSPAM capture arm
        # (`scripts/run_capture_arm.py`), so that every arm quoted in the
        # paper -- SPAM and non-SPAM alike -- shares one shot budget and their
        # statistical error bars are directly comparable; this preset used to
        # default to 64k shots, so bars here are 2x tighter than that older
        # setting.
        common.update(dict(t_grain=1000, w_grain=500, truncate=20, n_shots=256000))
    return QNSExperimentConfig(**common)


def dataset_path():
    """Where the cached per-shot noise-phase dataset (`phases.npz`) lives.

    Always the (SPAM-free) reference arm's run folder, regardless of which
    protocol is actually being run: the synthesized noise itself does not
    depend on the SPAM protocol or SPAM strength (those only change how
    readout/state-prep is modeled and how the estimators post-process the
    results), so one recording -- made once with `--record` on the reference
    arm -- can be replayed (`--replay`) into every other arm's preps and
    estimators without re-synthesizing the noise each time.
    """
    return os.path.join(project_root(),
                        run_folder(spam=True, protocol='reference'),
                        'phases.npz')


if __name__ == "__main__":
    # This block only runs when the file is executed directly (e.g.
    # `python scripts/run_spam_experiments.py ...`), not when it is imported
    # by something else -- the standard Python idiom for "script entry point".
    # Command-line flags are parsed by hand below (checking `'--flag' in args`
    # and then stripping every `--...` token) rather than with the `argparse`
    # module; it is less featureful (no auto-generated --help, no type
    # checking) but keeps this small script self-contained. The one required
    # positional argument is the protocol name, read after the `--flags` are
    # stripped out.
    args = [a for a in sys.argv[1:]]
    reduced = '--reduced' in args
    medium = '--medium' in args
    tuned = '--tuned' in args
    fine = '--fine' in args
    strong = '--strong' in args
    # --record: run this arm with a PhaseRecorder and save the per-shot phase
    #   dataset alongside it (use with the reference arm: the noise is
    #   protocol- and SPAM-strength-independent, so ONE recording serves every
    #   subsequent arm and strength).
    # --replay: skip noise synthesis and replay the recorded dataset through
    #   this arm's preps + estimators (~minutes instead of ~40 at --tuned).
    record = '--record' in args
    replay = '--replay' in args
    # --fast: solve on the filter-vector PhasedState path (exact for this
    #   dephasing model) instead of the dense solver_prop. --dense forces the
    #   old path (validation only). In plain terms: this noise model only
    #   ever dephases (never flips) the qubits, so its time evolution can be
    #   tracked as a handful of accumulated phases per shot instead of
    #   propagating the full 8x8 density matrix -- mathematically identical,
    #   but dramatically cheaper, which is what makes the large `fine`/robust
    #   shot counts above tractable at all.
    fast = '--fast' in args
    dense = '--dense' in args
    args = [a for a in args if not a.startswith('--')]
    protocol = args[0] if args else 'mitigated'
    if protocol not in ('raw', 'mitigated', 'robust', 'reference'):
        raise SystemExit(f"Unknown protocol {protocol!r}; "
                         "expected raw|mitigated|robust|reference")
    if record and replay:
        raise SystemExit("--record and --replay are mutually exclusive")
    # The SPAM-robust suite cannot replay the non-robust dataset (its readout
    # estimators are structurally different from the other protocols') and is
    # intractable on the dense solver (~hours/experiment), so default it to the
    # exact fast path; --dense opts back into solver_prop for validation.
    if protocol == 'robust' and not dense:
        fast = True
    config = build_config(protocol, reduced, strong, medium, tuned, fine)
    print(f"[spam] protocol={protocol} reduced={reduced} medium={medium} "
          f"tuned={tuned} fine={fine} strong={strong} record={record} "
          f"replay={replay} -> {config.fname}")
    main(config,
         record_to=dataset_path() if record else None,
         replay_from=dataset_path() if replay else None,
         fast=fast)
