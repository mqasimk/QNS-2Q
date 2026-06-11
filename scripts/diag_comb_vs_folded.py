"""OPT-COMB-M16: validate the M > 10 frequency-comb overlap approximation
against the time-domain folded evaluator under the featured (line) model.

The idle pipeline switches from the folded time-domain overlap to the comb
approximation at M > 10 for speed. The comb samples S only at the gate
harmonics k*2pi/T_seq; the finite-M filter has tooth width ~ spacing/M, and at
M = 16-32 that width is comparable to the nuclear-line width sigma = 0.02, so
the comb could mis-weight the lines. The folded evaluator carries the full
spectrum through an IFFT and the exact (M - |p|) triangular fold, so it is the
reference at any M (it is the M <= 10 path that the gate-probe truth check
validated at M = 1).

For each (Tg, M) of the published idle sweep this script evaluates every
library CDD/mqCDD pair both ways on the IDEAL featured SMat (exact lines --
the benchmark curve quoted in the figures) and reports the worst and median
relative deviation of the idling infidelity, plus the deviation for the
best-by-folded sequence (the one the figures actually quote).

Usage:  python scripts/diag_comb_vs_folded.py  [--tg 320 10240] [--m 16 32 64 128]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tg', type=float, nargs='+', default=[320.0, 2560.0, 10240.0],
                    help="gate times (tau) -- defaults span the published sweep")
    ap.add_argument('--m', type=int, nargs='+', default=[16, 32, 64, 128],
                    help="repetition counts (the comb-path regime)")
    a = ap.parse_args()

    import jax.numpy as jnp
    from qns2q.control import idle

    cfg = idle.Config(use_simulated=True)   # SMat_ideal is all we use
    SMat, w_grid = cfg.SMat_ideal, cfg.w_ideal

    print(f"[diag] comb-vs-folded on the IDEAL featured SMat "
          f"(grid to {float(w_grid[-1]):.2f}/tau), tau={float(cfg.tau):g}")
    print(f"{'Tg':>8} {'M':>4} {'T_seq':>8} {'#seq':>5} | "
          f"{'median |dev|':>12} {'max |dev|':>12} | {'best-seq dev':>12}")

    for Tg in a.tg:
        for M in a.m:
            T_seq = Tg / M
            if T_seq < 2 * cfg.tau:
                continue
            pLib, pDesc = idle.construct_pulse_library(T_seq, cfg.tau, 1000)
            if not pLib:
                continue

            idle._OVERLAP_SETUP_CACHE.clear()
            R_fold, dt, nbs = idle.prepare_time_domain_overlap(
                SMat, w_grid, cfg.tau, T_seq, M)

            w0 = 2 * jnp.pi / T_seq
            max_k = int(float(w_grid[-1]) / float(w0))
            omega_k = jnp.arange(1, max_k + 1) * w0
            S_flat = SMat.reshape(-1, SMat.shape[-1])
            S_h = np.stack([
                np.interp(np.asarray(omega_k), np.asarray(w_grid),
                          np.real(np.asarray(fp)), right=0.)
                + 1j * np.interp(np.asarray(omega_k), np.asarray(w_grid),
                                 np.imag(np.asarray(fp)), right=0.)
                for fp in np.asarray(S_flat)])
            S_packed = jnp.asarray(
                np.concatenate([np.asarray(S_flat[:, :1]), S_h], axis=1)
            ).reshape(4, 4, -1)

            def infidelity(d1, d2, comb):
                pt1 = idle.delays_to_pulse_times(d1, T_seq)
                pt2 = idle.delays_to_pulse_times(d2, T_seq)
                pt12 = idle.make_tk12(pt1, pt2)
                pt0 = jnp.array([0., T_seq])
                pts = [pt0, pt1, pt2, pt12]
                I = jnp.array([[
                    (idle.evaluate_overlap_comb(pts[r], pts[c], S_packed[r, c],
                                                omega_k, T_seq, M) if comb else
                     idle.evaluate_overlap_folded(pts[r], pts[c], R_fold[r, c],
                                                  dt, nbs))
                    for c in range(4)] for r in range(4)])
                return 1.0 - idle.calculate_idling_fidelity(I) / 16.0

            devs, infs_f = [], []
            for d1, d2 in pLib:
                f = float(infidelity(d1, d2, comb=False))
                c = float(infidelity(d1, d2, comb=True))
                infs_f.append(f)
                devs.append(abs(c - f) / max(abs(f), 1e-30))
            devs = np.array(devs)
            best = int(np.argmin(infs_f))
            print(f"{Tg:8.0f} {M:4d} {T_seq:8.2f} {len(pLib):5d} | "
                  f"{np.median(devs):12.2%} {devs.max():12.2%} | "
                  f"{devs[best]:12.2%}  (best: {pDesc[best]}, "
                  f"inf={infs_f[best]:.3e})")


if __name__ == '__main__':
    main()
