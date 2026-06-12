"""SHOWCASE-0612 entanglement-storage panel: what the cross-spectra are FOR.

Task: the idle holds half of a Bell pair (Phi+ = (|00> + |11>)/sqrt(2)) for a
time Tg. The Bell coherences are exactly ZZ-immune (|00> and |11> share the
Z1Z2 eigenvalue, |01> and |10> likewise), so only S_11, S_22 and the
inter-qubit cross-spectrum S_1_2 enter:

    1 - F_{Phi+} = (1 - exp(-v_DQ/2))/2,   v_DQ = I_11 + I_22 + 2 Re I_12
    1 - F_{Psi+} = (1 - exp(-v_ZQ/2))/2,   v_ZQ = I_11 + I_22 - 2 Re I_12

with I_ab the same overlap integrals the gate optimizer uses (control.idle).
For IDENTICAL pulse trains on the two qubits the only remaining design choice
is the relative toggling-frame parity (pulse the qubits simultaneously, or
bracket one qubit's idle with X frame flips): in-phase gives y_2 = +y_1 and
pays the (S_11 + S_22 + 2 Re S_12) combination on Phi+, anti-phase gives
y_2 = -y_1 and pays (S_11 + S_22 - 2 Re S_12). With the slow carrier
common-mode at c = 0.85 these differ by up to (1+c)/(1-c) ~ 12x -- and the
two implementations have identical single-qubit marginals and (to second
order) identical AVERAGE fidelity, so nothing short of the measured two-qubit
cross-spectrum can tell you which one protects the pair.

The panel evaluates, per idle gate time:
  FID         -- no pulses, Phi+ / Psi+ (the bare carrier hit and DFS split);
  sync        -- best symmetric CPMG (predicted-optimal n), simultaneous
                 pulses (the hardware-default implementation);
  anti        -- the SAME train, frame-flipped on qubit 2 (the choice the
                 measured Re S_1_2 > 0 dictates);
  NT winner   -- the blind average-fidelity-optimal idle (best over M), for
                 reference: best F_pro, but its Phi+ storage sits between.
Each on the analytic truth, plus predicted values from the blind
reconstruction for the sync/anti pair. A Monte-Carlo cross-check (the same
record/replay phase solver the experiments use) validates the formula AND the
shared-carrier synthesis end to end at one gate time.

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/showcase_storage_panel.py [--mc-check] [--out FILE]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root
from qns2q.control import idle as idmod

FOLDER = "DraftRun_NoSPAM_showcase_cap"
GATE_TIMES = [320., 640., 1280., 2560., 5120., 10240.]


def cpmg_pt(T, n):
    """CPMG pulse-time array [0, t_1, ..., t_{2n}, T] (2n pi pulses)."""
    tk = [(k + 0.5) * T / (2 * n) for k in range(2 * n)]
    return jnp.array([0.] + tk + [T])


def overlap_elements(cfg, pt1, pt2, T_seq, M, use_ideal):
    """(I_11, I_22, Re I_12) for one train pair via the exact folded evaluator."""
    SMat = cfg.SMat_ideal if use_ideal else cfg.SMat
    w_grid = cfg.w_ideal if use_ideal else cfg.w
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_grid, cfg.tau,
                                                      T_seq, M)
    i11 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt1, RMat[1, 1], dt, nbs)))
    i22 = float(np.real(idmod.evaluate_overlap_folded(pt2, pt2, RMat[2, 2], dt, nbs)))
    i12 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt2, RMat[1, 2], dt, nbs)))
    return i11, i22, i12


def bell_infs(i11, i22, i12):
    """(1 - F_Phi+, 1 - F_Psi+) from the overlap elements."""
    v_dq = i11 + i22 + 2 * i12
    v_zq = i11 + i22 - 2 * i12
    return (0.5 * (1 - np.exp(-0.5 * v_dq)),
            0.5 * (1 - np.exp(-0.5 * v_zq)))


def fpro_for_parity(cfg, pt1, pt2, T_seq, M, flip):
    """Average (process) infidelity of the pair, optionally frame-flipped on
    qubit 2. The flip negates y_2 and y_12, i.e. the (1,2)/(2,1) and
    (1,3)/(3,1) entries of the I matrix; (2,3) is invariant (both flip)."""
    SMat = cfg.SMat_ideal
    w_grid = cfg.w_ideal
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_grid, cfg.tau,
                                                      T_seq, M)
    pt12 = idmod.make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    pts = [pt0, pt1, pt2, pt12]
    I = np.zeros((4, 4), dtype=complex)
    for r in range(4):
        for c in range(4):
            I[r, c] = complex(idmod.evaluate_overlap_folded(
                pts[r], pts[c], RMat[r, c], dt, nbs))
    if flip:
        for r, c in ((1, 2), (2, 1), (1, 3), (3, 1)):
            I[r, c] = -I[r, c]
    fid = float(idmod.calculate_idling_fidelity(jnp.asarray(I)))
    return 1.0 - fid / 16.0


def pick_n(cfg, Tg, min_sep):
    """Predicted-optimal symmetric CPMG order from the CHARACTERIZED spectra
    (blind choice), among powers of two respecting the pulse-spacing floor."""
    best = None
    n = 4
    while (Tg / (2 * n)) >= min_sep:
        pt = cpmg_pt(Tg, n)
        i11, i22, i12 = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=False)
        inf_anti = bell_infs(i11, i22, -abs(i12))[0]
        if best is None or inf_anti < best[1]:
            best = (n, inf_anti)
        n *= 2
    return best[0]


def nt_winner(data, Tg):
    """Best-over-M blind NT idle winner (sequence, M) at gate time Tg."""
    best = None
    for m in (int(x) for x in data["M_values"]):
        gts = np.asarray(data[f"M{m}_gate_times"], dtype=float)
        idx = np.where(np.isclose(gts, Tg))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        inf = float(data[f"M{m}_infs_opt"][k])
        seq = data[f"M{m}_sequences_opt"][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, seq)
    return best


def mc_check(cfg, Tg, n, n_shots=20000):
    """Monte-Carlo validation of the Phi+/Psi+ formulas through the actual
    shared-carrier noise synthesis (filter-vector phase solver). Shots run in
    2k-chunks so the per-shot Gaussian draws never exceed GPU memory."""
    import jax
    from qns2q.model.trajectories import make_noise_mat_arr, _filter_vectors, \
        _shot_coeffs_from_filters
    t_vec = jnp.linspace(0, Tg, 4096)
    wmax = float(cfg.w_max) / 8.   # synthesis band: the comb band is enough
    mats = make_noise_mat_arr('make', t_vec=t_vec, w_grain=600, wmax=wmax,
                              truncate=int(cfg.mc), midpoint=True)
    pt = np.asarray(cpmg_pt(Tg, n))
    y = np.zeros((3, 3, t_vec.size))
    idx = np.searchsorted(pt, np.asarray(t_vec), side='right')
    y_t = (-1.0) ** (idx - 1)
    y[0, 0] = y_t
    y[1, 1] = y_t
    y[2, 2] = y_t * y_t
    F = _filter_vectors(jnp.asarray(mats), t_vec, jnp.asarray(y))
    rng = np.random.default_rng(20260612)
    chunks = []
    for s in range(0, n_shots, 2000):
        keys = jnp.asarray(rng.integers(0, 10000, (min(2000, n_shots - s), 2)))
        chunks.append(jax.vmap(_shot_coeffs_from_filters,
                               in_axes=[None, 0])(F, keys))
    coeffs = jnp.concatenate(chunks, axis=0)
    c1, c2 = np.asarray(coeffs[:, 0]), np.asarray(coeffs[:, 1])
    f_phi = float(np.mean(0.5 * (1 + np.cos(2 * (c1 + c2)))))
    f_psi = float(np.mean(0.5 * (1 + np.cos(2 * (c1 - c2)))))
    return 1 - f_phi, 1 - f_psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc-check', action='store_true')
    ap.add_argument('--out', default=None,
                    help="npz output (default <folder>/storage_panel.npz)")
    args = ap.parse_args()

    cfg = idmod.Config(fname=FOLDER, M=1, max_pulses=10**9, min_sep_factor=8.0)
    opt_path = os.path.join(project_root(), FOLDER,
                            "optimization_data_all_M_cap.npz")
    opt = np.load(opt_path, allow_pickle=True)

    rows = []
    for Tg in GATE_TIMES:
        pt_fid = jnp.array([0., Tg])
        i = overlap_elements(cfg, pt_fid, pt_fid, Tg, 1, use_ideal=True)
        fid_phi, fid_psi = bell_infs(*i)

        n = pick_n(cfg, Tg, cfg.min_sep)
        pt = cpmg_pt(Tg, n)
        it = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=True)
        sync_phi, sync_psi = bell_infs(*it)
        anti_phi, anti_psi = bell_infs(it[0], it[1], -it[2])
        ip = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=False)
        sync_phi_pred = bell_infs(*ip)[0]
        anti_phi_pred = bell_infs(ip[0], ip[1], -ip[2])[0]

        fpro_sync = fpro_for_parity(cfg, pt, pt, Tg, 1, flip=False)
        fpro_anti = fpro_for_parity(cfg, pt, pt, Tg, 1, flip=True)

        win = nt_winner(opt, Tg)
        if win is not None:
            inf_nt, m_nt, seq_nt = win
            cfg_m = (cfg if m_nt == 1 else
                     idmod.Config(fname=FOLDER, M=m_nt, max_pulses=10**9,
                                  min_sep_factor=8.0))
            pt1, pt2 = jnp.asarray(seq_nt[0]), jnp.asarray(seq_nt[1])
            iw = overlap_elements(cfg_m, pt1, pt2, Tg / m_nt, m_nt,
                                  use_ideal=True)
            nt_phi, nt_psi = bell_infs(*iw)
        else:
            inf_nt = m_nt = nt_phi = nt_psi = np.nan

        rows.append(dict(Tg=Tg, n_cpmg=n,
                         fid_phi=fid_phi, fid_psi=fid_psi,
                         sync_phi=sync_phi, anti_phi=anti_phi,
                         sync_psi=sync_psi, anti_psi=anti_psi,
                         sync_phi_pred=sync_phi_pred,
                         anti_phi_pred=anti_phi_pred,
                         fpro_sync=fpro_sync, fpro_anti=fpro_anti,
                         nt_fpro=inf_nt, nt_M=m_nt, nt_phi=nt_phi,
                         nt_psi=nt_psi))
        r = rows[-1]
        print(f"Tg={Tg:7.0f}  CPMG-{n:<3d} | FID Phi+ {fid_phi:.3e}  "
              f"sync Phi+ {sync_phi:.3e} (pred {sync_phi_pred:.3e})  "
              f"anti Phi+ {anti_phi:.3e} (pred {anti_phi_pred:.3e})  "
              f"split {sync_phi/anti_phi:5.1f}x | F_pro sync/anti "
              f"{fpro_sync:.3e}/{fpro_anti:.3e} (ratio "
              f"{fpro_sync/fpro_anti:.3f}) | NT(M={m_nt}) F_pro "
              f"{inf_nt:.3e} Phi+ {nt_phi:.3e}")

    out = args.out or os.path.join(project_root(), FOLDER, "storage_panel.npz")
    keys = rows[0].keys()
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"\nSaved {out}")

    if args.mc_check:
        Tg, n = 2560., None
        for r in rows:
            if r['Tg'] == Tg:
                n = r['n_cpmg']
        mc_phi, mc_psi = mc_check(cfg, Tg, n)
        print(f"\n[MC check @ Tg={Tg:.0f}, CPMG-{n}] Phi+ {mc_phi:.3e}, "
              f"Psi+ {mc_psi:.3e} (sync parity; compare the sync row above)")


if __name__ == "__main__":
    main()
