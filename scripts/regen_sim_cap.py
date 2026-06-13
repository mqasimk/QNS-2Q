"""Regenerate simulated_spectra.npz (ground truth, comb-sampled) for the CAP
folder from the CURRENT analytic model (strong-ZZ knee). Mirrors the sampling
in qns2q.noise.spectra.__main__ but targets the CAP folder directly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 MODEL_VERSION, DT_SHIFT)
from qns2q.paths import project_root

T = 160.0
truncate = 20
wk = jnp.linspace(-2 * np.pi * truncate / T, 2 * np.pi * truncate / T, 2 * truncate + 1)
km = {"S_11": "S11", "S_22": "S22", "S_1212": "S1212",
      "S_1_2": "S12", "S_1_12": "S112", "S_2_12": "S212"}
sk = {"S_11": S_11(wk), "S_22": S_22(wk), "S_1212": S_1212(wk),
      "S_1_2": S_1_2(wk), "S_1_12": S_1_12(wk), "S_2_12": S_2_12(wk)}
sd = {km[k]: np.array(v) for k, v in sk.items()}
sd['wk'] = np.array(wk)
sd['T'] = T
sd['truncate'] = truncate
sd['dt_shift'] = DT_SHIFT
sd['model_version'] = MODEL_VERSION
p = os.path.join(project_root(), "DraftRun_NoSPAM_showcase_cap", "simulated_spectra.npz")
np.savez(p, **sd)
print(f"wrote {p}")
print(f"S1212 at ZZ knee DC: {float(np.real(sd['S1212'][truncate])):.3e}")
