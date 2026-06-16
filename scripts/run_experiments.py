"""Entry point: run from the repo root. Regime via QNS2Q_REGIME.

Executes qns2q.characterize.experiments's top-level (__main__) routine.

Optional flags (mirroring run_spam_experiments):
  --record   save the per-shot phase dataset to <run folder>/phases.npz
  --replay   skip noise synthesis; replay that dataset (~minutes). Replay
             reuses the SAME Monte Carlo -- re-analysis, not a fresh repeat.
"""
import os, sys, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("qns2q.characterize.experiments", run_name="__main__", alter_sys=True)
