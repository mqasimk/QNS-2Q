"""SHOWCASE-0612 capture-grade NoSPAM arm (stage 1 + stage 2).

Reproduces the 2026-06-12 capture battery exactly; the config is the one
recovered from the cap arm's params.npz: 128k shots per estimator, M=16 DC
sweeps with the harmonic sweeps extended to 160*16 = 2560 tau, t_grain=1600,
truncate=20, w_grain=350, midpoint synthesis grid. Records the per-shot phase
dataset alongside (phases.npz) and runs the reconstruction in the same
process, so one command refreshes the whole capture arm after a model change.

Usage:
    QNS2Q_REGIME=showcase ./venv/bin/python scripts/run_capture_arm.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qns2q.characterize.experiments import QNSExperimentConfig, main as run_experiments
from qns2q.characterize.reconstruct import main as run_reconstruct
from qns2q.paths import project_root

FNAME = "DraftRun_NoSPAM_showcase_cap"

if __name__ == "__main__":
    # VISUAL-PARITY-0616: 128k -> 256k shots (2x) tightens the shot-limited
    # cross-spectra bars (S112/S212/Im) ~29% toward self-spectra visual parity.
    # Combined with the DT=4 Im boost this substantially cleans the cross panels.
    # NB: record_to MUST stay set -- the PhaseRecorder routes the suite onto the
    # filter-vector fast solver (exact for this diagonal-propagator dephasing
    # model, 3 phase coeffs/shot), which makes 256k tractable. record_to=None
    # falls back to the dense solver_prop, whose ~5.7 GB/2000-shot batch OOMs the
    # 12 GB GPU at 256k. The phases.npz it leaves behind is just an unused
    # artifact here (no replay needed for figure regen).
    config = QNSExperimentConfig(M=16, t_grain=1600, truncate=20, w_grain=350,
                                 n_shots=256000, spam_protocol="none",
                                 fname=FNAME, midpoint=True)
    run_experiments(config,
                    record_to=os.path.join(project_root(), FNAME, "phases.npz"))
    run_reconstruct(FNAME)
    print("CAPTURE-ARM-DONE")
