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
    config = QNSExperimentConfig(M=16, t_grain=1600, truncate=20, w_grain=350,
                                 n_shots=128000, spam_protocol="none",
                                 fname=FNAME, midpoint=True)
    run_experiments(config,
                    record_to=os.path.join(project_root(), FNAME, "phases.npz"))
    run_reconstruct(FNAME)
    print("CAPTURE-ARM-DONE")
