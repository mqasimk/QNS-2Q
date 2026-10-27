"""Produces the standalone C_1_0_MT_vs_M.pdf paper figure: scaling of the
C_1,0 correlation-function estimator vs. the number of dynamical-decoupling
blocks M, for a single qubit. Not part of the two-qubit Stage 1/2 pipeline -- a
self-contained simulation via qns2q.characterize.single_qubit. Run from the repo
root; regime via QNS2Q_REGIME. The default ensemble is 256,000 noise
realizations per observable estimate, with projection noise excluded.
Time-chunked filter contraction and batched noise draws bound GPU memory.
The estimator corrects readout using the exact confusion matrix and retains
preparation bias; this is a local-coefficient illustration, not the raw
SPAM-robust twisting/wringing estimator.

This file is a thin shim: it just hands argv to that module's own __main__ block
via runpy.

    QNS2Q_REGIME=showcase PYTHONPATH=src python scripts/run_single_qubit.py --output-dir reports/single_qubit_256k_2026-10-26
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.characterize.single_qubit", run_name="__main__", alter_sys=True)
