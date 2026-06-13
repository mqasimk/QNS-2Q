"""Fast scan: which shared common-mode line maximizes the decoupled Bell-storage
split? Perturbs ONLY the analytic truth's cross entry Re S_1_2 (move-mode: an
existing local defect line is declared shared, self-spectra unchanged) or also
the self entries (add-mode: a fresh shared line). Reuses the exact idle overlap
evaluator so the numbers match the real storage panel. No reconstruction run.

    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python scripts/scan_shared_line.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import jax.numpy as jnp
from qns2q.control import idle as idmod
from qns2q.noise import spectra as sp

FOLDER = "DraftRun_NoSPAM_showcase_cap"
GATE_TIMES = [320., 640., 1280., 2560., 5120., 10240.]
C = float(sp._SC_C2_QS)


def cpmg_pt(T, n):
    tk = [(k + 0.5) * T / (2 * n) for k in range(2 * n)]
    return jnp.array([0.] + tk + [T])


def overlaps(SMat, w, tau, pt1, pt2, T, M):
    R, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w, tau, T, M)
    i11 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt1, R[1, 1], dt, nbs)))
    i22 = float(np.real(idmod.evaluate_overlap_folded(pt2, pt2, R[2, 2], dt, nbs)))
    i12 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt2, R[1, 2], dt, nbs)))
    return i11, i22, i12


def bell(i11, i22, i12):
    return (0.5 * (1 - np.exp(-0.5 * (i11 + i22 + 2 * i12))),
            0.5 * (1 - np.exp(-0.5 * (i11 + i22 - 2 * i12))))


def perturbed_SMat(base, w, w_sh, sig, h_cross, h_self):
    SMat = np.array(base)
    g = np.asarray(sp.Gauss(w, w_sh, sig))
    SMat[1, 1] = SMat[1, 1] + h_self * g
    SMat[2, 2] = SMat[2, 2] + h_self * g
    SMat[1, 2] = SMat[1, 2] + h_cross * g
    SMat[2, 1] = SMat[2, 1] + h_cross * g
    return jnp.asarray(SMat)


def split_table(SMat, w, tau, min_sep):
    rows = []
    for Tg in GATE_TIMES:
        # blind-ish: pick n minimizing anti-phase on this same SMat
        best = None
        n = 4
        while (Tg / (2 * n)) >= min_sep:
            pt = cpmg_pt(Tg, n)
            i = overlaps(SMat, w, tau, pt, pt, Tg, 1)
            anti = bell(i[0], i[1], -abs(i[2]))[0]
            if best is None or anti < best[1]:
                best = (n, anti, i)
            n *= 2
        n, _, i = best
        sync_phi, _ = bell(*i)
        anti_phi, _ = bell(i[0], i[1], -i[2])
        rows.append((Tg, n, sync_phi, anti_phi, sync_phi / anti_phi))
    return rows


def main():
    cfg = idmod.Config(fname=FOLDER, M=1, max_pulses=10**9, min_sep_factor=8.0)
    base = np.array(cfg.SMat_ideal)
    w = cfg.w_ideal
    tau, min_sep = cfg.tau, cfg.min_sep

    print(f"c_QS = {C}, (1+c)/(1-c) = {(1+C)/(1-C):.1f}x cap\n")
    print("=== BASELINE (no shared line) ===")
    for Tg, n, s, a, r in split_table(jnp.asarray(base), w, tau, min_sep):
        print(f"  Tg={Tg:7.0f} CPMG-{n:<3d} sync {s:.3e} anti {a:.3e} split {r:5.1f}x")

    # move-mode: each existing defect line declared shared (self unchanged)
    cen = np.asarray(sp._SC_LINE_CENTERS); sig = np.asarray(sp._SC_LINE_SIGMAS)
    aq1 = np.asarray(sp._SC_LINE_AMP_Q1); aq2 = np.asarray(sp._SC_LINE_AMP_Q2)
    for k in range(len(cen)):
        h_cross = C * np.sqrt(aq1[k] * aq2[k])
        print(f"\n=== MOVE line k={k}: w_sh={cen[k]:.3f} sig={sig[k]:.3f} "
              f"h_cross={h_cross:.2e} (self unchanged) ===")
        SMat = perturbed_SMat(base, w, cen[k], sig[k], h_cross, 0.0)
        for Tg, n, s, a, r in split_table(SMat, w, tau, min_sep):
            print(f"  Tg={Tg:7.0f} CPMG-{n:<3d} sync {s:.3e} anti {a:.3e} split {r:5.1f}x")

    # add-mode: fresh shared line at candidate freqs, symmetric height H
    for w_sh in (0.27, 0.30, 0.34, 0.39):
        H = 1.2e-4; s_sig = 0.015
        print(f"\n=== ADD fresh shared line w_sh={w_sh:.3f} H={H:.1e} "
              f"(self += line) ===")
        SMat = perturbed_SMat(base, w, w_sh, s_sig, C * H, H)
        for Tg, n, s, a, r in split_table(SMat, w, tau, min_sep):
            print(f"  Tg={Tg:7.0f} CPMG-{n:<3d} sync {s:.3e} anti {a:.3e} split {r:5.1f}x")


if __name__ == "__main__":
    main()
