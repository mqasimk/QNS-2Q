"""'Don't lose anything else' check: at the candidate CZ-budget knees, does the
IDLE 2Q (full) design stay 1e-4-class (it can dodge low-freq ZZ by interleaving,
unlike the forced CZ), and does the 1Q-blind idle balloon (bonus gap)?
Old sequences => full is CONSERVATIVE (re-opt only lowers it); rung_c is exact
(self-only design, unchanged). No synthesis.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.control import idle as idmod
from qns2q.paths import project_root
from qns2q.noise.spectra import _SC_W_TLF, _SC_H_ZZ_KNEE

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"


def knee_np(w, h, wc):
    return h / (1 + (np.asarray(w) / wc) ** 2)


def fpro(SMat, w_ideal, tau, pt1, pt2, T_seq, M):
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_ideal, tau, T_seq, M)
    pt12 = idmod.make_tk12(pt1, pt2)
    pts = [jnp.array([0., T_seq]), pt1, pt2, pt12]
    I = np.zeros((4, 4), complex)
    for r in range(4):
        for c in range(4):
            I[r, c] = complex(idmod.evaluate_overlap_folded(pts[r], pts[c], RMat[r, c], dt, nbs))
    return 1.0 - float(idmod.calculate_idling_fidelity(jnp.asarray(I))) / 16.0


def best(path, Tg):
    d = np.load(path, allow_pickle=True)
    b = None
    for m in (int(x) for x in d['M_values']):
        gts = np.asarray(d[f'M{m}_gate_times'], float)
        ix = np.where(np.isclose(gts, Tg))[0]
        if not ix.size:
            continue
        k = int(ix[0])
        inf = float(d[f'M{m}_infs_opt'][k])
        s = d[f'M{m}_sequences_opt'][k]
        if s is not None and (b is None or inf < b[0]):
            b = (inf, m, (jnp.asarray(s[0]), jnp.asarray(s[1])))
    return b


full_p = os.path.join(ROOT, CAP, "optimization_data_all_M_cap.npz")
rungc_p = os.path.join(ROOT, CAP, "optimization_data_all_M_rung_c_idle_cap.npz")
refs = []  # keep SMat refs alive to avoid id()-cache collisions
for knee in [1.4e-5, 2.0e-5]:
    print(f"\n=== knee {knee:.1e} ===")
    for Tg in [640., 2560.]:
        def evl(win):
            _, m, (p1, p2) = win
            cfg = idmod.Config(fname=CAP, M=m, max_pulses=10**9, min_sep_factor=8.0)
            base = np.asarray(cfg.SMat_ideal)
            S = base.copy()
            S[3, 3] = base[3, 3] + knee_np(np.asarray(cfg.w_ideal),
                                           knee - float(_SC_H_ZZ_KNEE), _SC_W_TLF)
            Sj = jnp.asarray(S)
            refs.append(Sj)
            return fpro(Sj, cfg.w_ideal, cfg.tau, p1, p2, Tg / m, m), m
        ff, mf = evl(best(full_p, Tg))
        fr, mr = evl(best(rungc_p, Tg))
        print(f"  Tg={Tg:5.0f}: full(2Q,M={mf:<3d}) {ff:.3e} | "
              f"rung_c(1Q,M={mr:<3d}) {fr:.3e} | idle gap {fr/ff:4.1f}x")
