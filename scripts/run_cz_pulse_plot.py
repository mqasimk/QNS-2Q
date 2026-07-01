"""Draws the showcase_pulse_sequences.pdf paper figure: the best-known (CDD/library)
CZ pulse sequence vs. the best noise-tailored (NT) one it was optimized against,
for visual comparison. Run from the repo root; regime via QNS2Q_REGIME.

This file is a thin shim: it just hands argv to qns2q.viz.cz_pulse_plot's own
__main__ block via runpy, so all the real argument parsing (--folder/--tag) and
plotting logic lives there, not here.

    PYTHONPATH=src python scripts/run_cz_pulse_plot.py --folder DraftRun_NoSPAM_showcase_cap --tag _cap
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.viz.cz_pulse_plot", run_name="__main__", alter_sys=True)
