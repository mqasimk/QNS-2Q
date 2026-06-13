"""CZ NT-vs-CDD margin at the strong ZZ knee (model already edited to 1.4e-5).
The CZ design is forced/unchanged, so evaluate the stored NT and best-CDD CZ
winners on the new analytic truth. Tells us if the 44x headline survives.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.control import cz as czmod
from qns2q.paths import project_root

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"
CZ_TG = 320.0


def cz_seq(tag, kind):
    d = np.load(os.path.join(ROOT, CAP, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    i = int(np.argmin(np.abs(np.asarray(d['taxis'], float) - CZ_TG)))
    s = d[f'sequences_{kind}'][i]
    return (jnp.asarray(s[0]), jnp.asarray(s[1]))


cfg = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                        gate_time_factors=[])
nt = float(czmod.calculate_infidelity(cz_seq('cap', 'opt'), cfg, 1, CZ_TG, use_ideal=True))
cdd = float(czmod.calculate_infidelity(cz_seq('cap', 'known'), cfg, 1, CZ_TG, use_ideal=True))
fid = float(czmod.calculate_infidelity((jnp.array([0., CZ_TG]), jnp.array([0., CZ_TG])),
                                       cfg, 1, CZ_TG, use_ideal=True))
print(f"CZ @ knee 1.4e-5:  NT {nt:.3e} | best CDD {cdd:.3e} | FID {fid:.3e}")
print(f"  NT-over-CDD margin: {cdd/nt:.1f}x   (published at old knee: ~44x)")
