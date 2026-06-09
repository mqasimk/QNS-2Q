"""Entry point: run from the repo root. Regime via QNS2Q_REGIME.

Executes qns2q.characterize.single_qubit's top-level (__main__) routine.
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.characterize.single_qubit", run_name="__main__", alter_sys=True)
