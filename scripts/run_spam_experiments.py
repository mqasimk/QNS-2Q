"""Stage 1 of the SPAM pipeline: QNS experiments with SPAM errors injected.

Usage (from the repo root, regime via QNS2Q_REGIME):

    python scripts/run_spam_experiments.py [raw|mitigated|robust] [--reduced]

Writes to ``DraftRun_SPAM_<regime>_<protocol>/``. The protocols:

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.characterize.experiments import QNSExperimentConfig, main
from qns2q.paths import run_folder, project_root

# Injected (true) SPAM parameters. alpha_M = a + b - 1, delta = a - b, so
# qubit 1: alpha_M = 0.97, delta = +0.03; qubit 2: alpha_M = 0.95, delta = -0.02.
# These match the companion paper's simulation values (mild, hardware-realistic).
SPAM_TRUE = dict(
    a1=1.00, b1=0.97,
    a2=0.965, b2=0.985,
    a_sp=jnp.array([0.99, 0.98]),
    c=np.array([jnp.array(0.02 + 0.04j), jnp.array(0.03 - 0.02j)]),
)

# Strong-SPAM variant (--strong): qubit 1 alpha_M = 0.88, delta = +0.06; qubit 2
# alpha_M = 0.85, delta = -0.05. At mild SPAM the legacy coefficient DIFFERENCES
# (e.g. C_12_0_MT_1 - C_12_0_MT_2) already cancel the log-additive SPAM offset,
# so the raw arm hides inside reduced-run statistics; the strong values make the
# residual (delta cross-term) bias clearly resolvable for validation.
SPAM_TRUE_STRONG = dict(
    a1=0.97, b1=0.91,
    a2=0.90, b2=0.95,
    a_sp=jnp.array([0.95, 0.92]),
    c=np.array([jnp.array(0.06 + 0.10j), jnp.array(0.08 - 0.06j)]),
)


def build_config(protocol: str, reduced: bool, strong: bool = False,
                 medium: bool = False, tuned: bool = False,
                 fine: bool = False) -> QNSExperimentConfig:
    if protocol == 'reference':
        # SPAM-free reference arm at the same (reduced) statistics: isolates the
        # SPAM-specific reconstruction bias from the comb/truncation systematics.
        common = dict(spam_protocol='none',
                      fname=run_folder(spam=True, protocol='reference'))
    else:
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
        # Tuned comparison config (2026-06-10): doubles the shots and funds it
        # by trimming grid overhead (t_grain 1000->800, w_grain 500->350; both
        # validated headroom: dt = 0.2 tau << 1/wmax, dw = 2.2e-3 << line sigma
        # 0.02). Same wk grid as medium/full (truncate 20). ~37 min/arm; the
        # statistical bars (which dominate the small channels at the Class-F
        # lines) tighten by sqrt(2) vs --medium.
        common.update(dict(t_grain=800, w_grain=350, truncate=20, n_shots=8000))
    elif fine:
        # Fine preset (2026-06-10): for use with --record/--replay. The
        # filter-vector phase solver makes shots nearly free, so this restores
        # the full grids AND takes 8x the tuned shots -- statistical bars
        # tighten ~2.8x vs --tuned, sqrt(16)=4x vs --medium.
        common.update(dict(t_grain=1000, w_grain=500, truncate=20, n_shots=64000))
    return QNSExperimentConfig(**common)


def dataset_path():
    """The phase dataset lives with the (SPAM-free) reference arm."""
    return os.path.join(project_root(),
                        run_folder(spam=True, protocol='reference'),
                        'phases.npz')


if __name__ == "__main__":
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
    #   old path (validation only).
    fast = '--fast' in args
    dense = '--dense' in args
    args = [a for a in args if not a.startswith('--')]
    protocol = args[0] if args else 'mitigated'
    if protocol not in ('raw', 'mitigated', 'robust', 'reference'):
        raise SystemExit(f"Unknown protocol {protocol!r}; "
                         "expected raw|mitigated|robust|reference")
    if record and replay:
        raise SystemExit("--record and --replay are mutually exclusive")
    # The SPAM-robust suite cannot replay the non-robust dataset and is
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
