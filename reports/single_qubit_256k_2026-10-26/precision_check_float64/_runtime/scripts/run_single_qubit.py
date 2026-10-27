"""Produces the standalone C_1_0_MT_vs_M.pdf paper figure: SPAM-robustness of the
C_1,0 correlation-function estimator vs. the number of dynamical-decoupling
blocks M, for a single qubit. Not part of the two-qubit Stage 1/2 pipeline -- a
self-contained simulation via qns2q.characterize.single_qubit. Run from the repo
root; regime via QNS2Q_REGIME. GPU-memory-hungry at the higher M values: noise
matrices grow with M, and on a 12 GB GPU this can raise a JAX RESOURCE_EXHAUSTED
error partway through even with nothing else running and
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 set (verified: this is a real memory ceiling
on this hardware, not a bug -- it reproduces identically on an unmodified
checkout). If you hit it, you need a GPU with more headroom (16 GB+) for this
one figure; the other seven don't need nearly as much memory.

This file is a thin shim: it just hands argv to that module's own __main__ block
via runpy.

    PYTHONPATH=src python scripts/run_single_qubit.py
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.characterize.single_qubit", run_name="__main__", alter_sys=True)
