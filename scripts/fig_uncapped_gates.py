"""UNCAP-0611 follow-up figure (2026-06-12, pre-meeting): both gates, published
pulse caps vs separation-limited pulse budgets, on the anchored model.

Layout mirrors the 0611 report's fig_gates: (a,c) true process infidelity vs
gate time, (b,d) NT-over-best-CDD margin; top row entangling (CZ), bottom row
idle. Each run's margin uses its OWN best CDD (uncapping admits higher CDD
orders, so the baseline strengthens too).
"""
import os, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from qns2q.characterize.reconstruct import setup_pub_rcparams
from qns2q.paths import project_root
from compare_uncapped import load_cz, merge, load_idle

C_NT = '#0072B2'     # blue: NT, separation-limited
C_NTCAP = '#56B4E9'  # light blue: NT, published cap
C_CDD = '#D55E00'    # vermillion: best CDD (uncapped library)
C_FID = '#999999'    # grey: free induction

T2_FID = 771.0

setup_pub_rcparams()
OUT = os.path.join(project_root(), "reports", "uncapped_0612")
os.makedirs(OUT, exist_ok=True)

# ---- data -------------------------------------------------------------
cz_cap = load_cz("plotting_data_cz_v2.npz")
cz_unc = merge(load_cz("plotting_data_cz_v2_uncapped.npz"),
               load_cz("plotting_data_cz_v2_uncapped2560.npz"))

id_cap = load_idle("optimization_data_all_M.npz")
id_unc = load_idle("optimization_data_all_M_uncapped.npz")
id_Tg = np.array(sorted(set(id_cap) & set(id_unc)))

# idle FID curve (M-independent): take it from the capped all-M file, M=1
_d = np.load(os.path.join(project_root(), "DraftRun_NoSPAM_featured",
                          "optimization_data_all_M.npz"), allow_pickle=True)
id_fid = dict(zip(np.asarray(_d['M1_gate_times'], dtype=float),
                  np.asarray(_d['M1_infs_nopulse'], dtype=float)))

# ---- figure -----------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.2))

# (a) CZ infidelity
ax = axs[0, 0]
ax.loglog(cz_cap['Tg'], cz_cap['nopulse'], 'o-', color=C_FID, ms=3.5,
          label='free induction')
ax.loglog(cz_unc['Tg'], cz_unc['known'], 's-', color=C_CDD, ms=3.5,
          label='best CDD (separation-limited library)')
ax.loglog(cz_cap['Tg'], cz_cap['opt'], 'd--', color=C_NTCAP, ms=3.5,
          label='best NT (published cap)')
ax.loglog(cz_unc['Tg'], cz_unc['opt'], 'd-', color=C_NT, ms=4,
          label='best NT (separation-limited)')
ax.set_ylabel(r'true process infidelity $1-F$')
ax.set_title('(a) entangling (CZ)', loc='left', fontsize=9)

# (b) CZ margin
ax = axs[0, 1]
m_cap = cz_cap['known'] / cz_cap['opt']
cdd_best = np.array([min(ku, kc) for ku, kc in zip(cz_unc['known'],
                                                   cz_cap['known'])])
m_unc = cdd_best / cz_unc['opt']
ax.semilogx(cz_cap['Tg'], m_cap, 'd--', color=C_NTCAP, ms=3.5,
            label='published cap (150/qubit)')
ax.semilogx(cz_unc['Tg'], m_unc, 'd-', color=C_NT, ms=4,
            label='separation-limited')
ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
ax.set_ylabel(r'NT-over-CDD margin')
ax.set_title('(b) entangling margin', loc='left', fontsize=9)

# (c) idle infidelity
ax = axs[1, 0]
ax.loglog(sorted(id_fid), [id_fid[t] for t in sorted(id_fid)], 'o-',
          color=C_FID, ms=3.5)
ax.loglog(id_Tg, [id_unc[t]['cdd'] for t in id_Tg], 's-', color=C_CDD, ms=3.5)
ax.loglog(id_Tg, [id_cap[t]['nt'] for t in id_Tg], 'd--', color=C_NTCAP, ms=3.5)
ax.loglog(id_Tg, [id_unc[t]['nt'] for t in id_Tg], 'd-', color=C_NT, ms=4)
ax.set_xlabel(r'gate time $T_g/\tau$')
ax.set_ylabel(r'true process infidelity $1-F$')
ax.set_title('(c) idle (winner over $M$)', loc='left', fontsize=9)

# (d) idle margin
ax = axs[1, 1]
m_cap_id = [id_cap[t]['cdd'] / id_cap[t]['nt'] for t in id_Tg]
m_unc_id = [min(id_cap[t]['cdd'], id_unc[t]['cdd']) / id_unc[t]['nt']
            for t in id_Tg]
ax.semilogx(id_Tg, m_cap_id, 'd--', color=C_NTCAP, ms=3.5,
            label='published cap (1000 total)')
ax.semilogx(id_Tg, m_unc_id, 'd-', color=C_NT, ms=4,
            label='separation-limited')
ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel(r'gate time $T_g/\tau$')
ax.set_ylabel(r'NT-over-CDD margin')
ax.set_title('(d) idle margin', loc='left', fontsize=9)

for ax in axs.flat:
    ax.grid(True, alpha=0.25)
    ax.axvline(T2_FID, color='k', lw=0.8, ls=':', alpha=0.6)
for ax in axs[:, 1]:
    ax.set_ylim(0.9, 2.0)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
           fontsize=8)
fig.suptitle('Noise-tailored gates with the pulse budget at the minimum-'
             'separation limit (anchored model, blind design)', fontsize=10)
fig.tight_layout(rect=[0, 0.05, 1, 0.97])

for ext in ('pdf', 'png'):
    p = os.path.join(OUT, f"fig_uncapped_gates.{ext}")
    fig.savefig(p, dpi=220)
    print(f"saved {p}")
