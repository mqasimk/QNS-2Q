"""Verify the strong-ZZ reconstruction: every channel reconstructed (no all-
NaN), the ZZ knee captured, and the self-spectra NOT corrupted by the now-
dominant ZZ channel. Compares specs.npz teeth to the analytic truth.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from qns2q.noise.spectra import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
from qns2q.paths import project_root

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"
d = np.load(os.path.join(ROOT, CAP, "specs.npz"), allow_pickle=True)
print("specs keys:", [k for k in d.files])
wk = np.asarray(d['wkqns']) if 'wkqns' in d.files else np.asarray(d['wk'])
pos = wk > 1e-9
truth = {'S11': S_11, 'S22': S_22, 'S1212': S_1212,
         'S12': S_1_2, 'S112': S_1_12, 'S212': S_2_12}
print(f"\n{'chan':>6} | {'NaN':>7} | {'recon DC':>11} | {'truth DC':>11} | {'med |rel err| (teeth)':>20}")
print("-" * 72)
for k, fn in truth.items():
    rec = np.asarray(d[k])
    tru = np.asarray(fn(wk))
    nnan = int(np.isnan(rec).sum())
    idc = int(np.argmin(np.abs(wk)))
    rdc, tdc = float(np.real(rec[idc])), float(np.real(tru[idc]))
    m = pos & np.isfinite(np.real(rec)) & (np.abs(np.real(tru)) > 1e-12)
    rel = np.abs((np.real(rec)[m] - np.real(tru)[m]) / np.real(tru)[m])
    medrel = float(np.median(rel)) if rel.size else float('nan')
    print(f"{k:>6} | {nnan:>3d}/{rec.size:<3d} | {rdc:>11.3e} | {tdc:>11.3e} | {medrel:>20.1%}")
