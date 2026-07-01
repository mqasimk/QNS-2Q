"""Stage 2 entry point: reconstructs noise power spectral densities from a
Stage-1 NoSPAM run's results.npz, writing specs.npz plus reconstruction
comparison PDFs into the same run folder. Run from the repo root; regime via
QNS2Q_REGIME.

This file is a thin shim: it just hands argv to qns2q.characterize.reconstruct's
own __main__ block via runpy, so the real --folder argument parsing and
reconstruction logic live there, not here.

    PYTHONPATH=src python scripts/run_reconstruct.py                                     # active regime's NoSPAM folder
    PYTHONPATH=src python scripts/run_reconstruct.py --folder DraftRun_NoSPAM_showcase_cap  # an explicit folder
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.characterize.reconstruct", run_name="__main__", alter_sys=True)
