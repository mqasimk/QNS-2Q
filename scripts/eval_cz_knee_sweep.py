"""CZ ZZ-budget sweep (SHOWCASE-strong-ZZ certification case): how the 2Q-
channel share of the CZ residual grows with the FORCED low-frequency ZZ knee
H_ZZ_KNEE, and what the CZ residual costs. Fixed NT-winner (the knee is
undodgeable for the CZ, so the design does not change). No synthesis.

Reports, per candidate knee: CZ true residual (full truth) and the share a
single-qubit campaign would MISS = (full - self_only)/full. We want the
largest share that keeps the CZ in the 1e-4..1e-3 class.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.control import cz as czmod
from qns2q.paths import project_root
from qns2q.noise.spectra import _SC_W_TLF, _SC_H_ZZ_KNEE

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"
CZ_TG = 320.0


def cz_winner_seq(tag, tg=CZ_TG):
    d = np.load(os.path.join(ROOT, CAP, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    tgs = np.asarray(d['taxis'], dtype=float)
    i = int(np.argmin(np.abs(tgs - tg)))
    s = d['sequences_opt'][i]
    return (jnp.asarray(s[0]), jnp.asarray(s[1]))


def zero_2q_np(S):
    S = S.copy()
    for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)):
        S[r, c, :] = 0.0
    return S


def knee_np(w, h, wc):
    return h / (1 + (np.asarray(w) / wc) ** 2)


cfg = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                        gate_time_factors=[])
seq = cz_winner_seq('cap')
w = np.asarray(cfg.w_ideal)
base = np.asarray(cfg.SMat_ideal)   # current model (knee = _SC_H_ZZ_KNEE)

print(f"current H_ZZ_KNEE={_SC_H_ZZ_KNEE:.2e}, W_TLF={_SC_W_TLF}\n")
print(f"{'H_ZZ_KNEE':>10} | {'CZ residual':>11} | {'2Q share':>8} | {'self-only':>10}")
print("-" * 50)
for new_knee in [0.5e-6, 2e-6, 5e-6, 9e-6, 1.4e-5, 2.0e-5, 3.0e-5, 5.0e-5]:
    Sfull = base.copy()
    Sfull[3, 3] = base[3, 3] + knee_np(w, new_knee - float(_SC_H_ZZ_KNEE), _SC_W_TLF)
    cfg.SMat_ideal = jnp.asarray(Sfull)
    full = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    cfg.SMat_ideal = jnp.asarray(zero_2q_np(Sfull))
    only1q = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    share = (full - only1q) / full
    flag = "" if full < 1.1e-3 else "  <-- CZ off 1e-3-class"
    print(f"{new_knee:.2e} | {full:.3e}   | {share:7.1%} | {only1q:.3e}{flag}")
