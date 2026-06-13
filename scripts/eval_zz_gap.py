"""Stage-1 ZZ-gap diagnostic (SHOWCASE-strong-ZZ, line repositioned to 0.285).

Best-case gap WITHOUT running the optimizer: the 2Q design's floor is the
line-NULLED cost. The rung_c (1Q-blind) sequence parks self in the window
(unchanged by the ZZ edits), so:
    1Q   = rung_c on the FULL truth (it parks ON the line, can't dodge)
    floor= rung_c on truth with the ZZ LINE removed (== best a perfect 2Q
           null reaches, since the 2Q design parks self the same way)
    gap_max = 1Q / floor   (the optimizer lands at or below this)
If gap_max < 10x at a ~1e-4 floor, repositioning isn't enough. No synthesis.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.control import idle as idmod
from qns2q.paths import project_root
from qns2q.noise.spectra import _SC_H_ZZ_LINE, _SC_ZZ_W0, _SC_ZZ_SIG

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"


def gauss_pair(w, w0, sig):
    w = np.asarray(w)
    return 0.5 * (np.exp(-(w - w0) ** 2 / (2 * sig ** 2))
                  + np.exp(-(w + w0) ** 2 / (2 * sig ** 2)))


def fpro(SMat, w_ideal, tau, pt1, pt2, T_seq, M):
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_ideal, tau, T_seq, M)
    pt12 = idmod.make_tk12(pt1, pt2)
    pts = [jnp.array([0., T_seq]), pt1, pt2, pt12]
    I = np.zeros((4, 4), dtype=complex)
    for r in range(4):
        for c in range(4):
            I[r, c] = complex(idmod.evaluate_overlap_folded(
                pts[r], pts[c], RMat[r, c], dt, nbs))
    return 1.0 - float(idmod.calculate_idling_fidelity(jnp.asarray(I))) / 16.0


def best_over_M(path, Tg):
    d = np.load(path, allow_pickle=True)
    best = None
    for m in (int(x) for x in d['M_values']):
        gts = np.asarray(d[f'M{m}_gate_times'], dtype=float)
        ix = np.where(np.isclose(gts, Tg))[0]
        if not ix.size:
            continue
        k = int(ix[0])
        inf = float(d[f'M{m}_infs_opt'][k])
        seq = d[f'M{m}_sequences_opt'][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, (jnp.asarray(seq[0]), jnp.asarray(seq[1])))
    return best


cfg0 = idmod.Config(fname=CAP, M=1, max_pulses=10**9, min_sep_factor=8.0)
wi = np.asarray(cfg0.w_ideal)
SMat_full = np.asarray(cfg0.SMat_ideal)
# truth with the ZZ LINE subtracted from S_1212 (keep knee, self, cross)
SMat_noline = SMat_full.copy()
SMat_noline[3, 3] = SMat_full[3, 3] - _SC_H_ZZ_LINE * gauss_pair(wi, _SC_ZZ_W0, _SC_ZZ_SIG)

mineig = min(np.linalg.eigvalsh(SMat_full[1:, 1:, i])[0] for i in range(SMat_full.shape[-1]))
iz = int(np.argmin(np.abs(wi - _SC_ZZ_W0)))
ratio = float(np.real(SMat_full[3, 3, iz]) / np.sqrt(np.real(SMat_full[1, 1, iz]) * np.real(SMat_full[2, 2, iz])))
print(f"line at w0={_SC_ZZ_W0}, PSD min eig {mineig:.2e}, S1212/sqrt(S11 S22) at line = {ratio:.1f}\n")

rungc_path = os.path.join(ROOT, CAP, "optimization_data_all_M_rung_c_idle_cap.npz")
print(f"{'Tg':>6} | {'1Q (parks on line)':>18} | {'2Q floor (nulled)':>18} | gap_max")
print("-" * 64)
for Tg in [640., 1280., 2560.]:
    win = best_over_M(rungc_path, Tg)
    if win is None:
        print(f"{Tg:6.0f} | (missing)")
        continue
    _, m, (pt1, pt2) = win
    cfgm = cfg0 if m == 1 else idmod.Config(fname=CAP, M=m, max_pulses=10**9, min_sep_factor=8.0)
    wim, tau = np.asarray(cfgm.w_ideal), cfgm.tau
    Sf = np.asarray(cfgm.SMat_ideal)
    Sn = Sf.copy()
    Sn[3, 3] = Sf[3, 3] - _SC_H_ZZ_LINE * gauss_pair(wim, _SC_ZZ_W0, _SC_ZZ_SIG)
    one_q = fpro(jnp.asarray(Sf), cfgm.w_ideal, tau, pt1, pt2, Tg / m, m)
    floor = fpro(jnp.asarray(Sn), cfgm.w_ideal, tau, pt1, pt2, Tg / m, m)
    print(f"{Tg:6.0f} | {one_q:.3e} (M={m:<3d}) | {floor:.3e}        | {one_q/floor:5.1f}x")
