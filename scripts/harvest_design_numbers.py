"""SHOWCASE-0612 design-section harvest (conformed to the 06/11 template).

Collects, from the shared-carrier battery outputs, everything Sec. "What the
gate design needs from the spectra" quotes:

  * the knowledge ladder, both gates: true infidelity of the blind NT winner
    designed on {1Q-2, diag-3, robust-4, full-6} channel sets (+ best CDD);
    CZ at Tg = 320 tau, idle best-over-M at 640 and 2560 tau;
  * the SPAM-arm design blocks, both gates: best CDD (true), best NT (true),
    best NT (predicted on that arm's own reconstruction);
  * the error budget: share of the FID and of the blind winner's residual
    error carried by the two-qubit channels (S_1212 + all cross-spectra),
    obtained by zeroing those channels in the TRUTH evaluation of the fixed
    sequences;
  * prediction accuracy of the reference arm (predicted vs true, both gates).

Writes <cap folder>/design_numbers.npz and prints a readable summary.

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/harvest_design_numbers.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root, run_folder
from qns2q.control import cz as czmod
from qns2q.control import idle as idmod

ROOT = project_root()
CAP = "DraftRun_NoSPAM_showcase_cap"
ARMS = ("reference", "raw", "mitigated", "robust")
CZ_TG = 320.0
ID_TGS = (640.0, 10240.0)


def _zero_2q(SMat):
    for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)):
        SMat = SMat.at[r, c].set(0.0)
    return SMat


def cz_true_at(folder, tag, tg=CZ_TG, kind='opt'):
    d = np.load(os.path.join(ROOT, folder, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    tgs = np.asarray(d['taxis'], dtype=float)
    i = int(np.argmin(np.abs(tgs - tg)))
    return float(np.asarray(d[f'infs_{kind}'], dtype=float)[i])


def cz_winner_seq(folder, tag, tg=CZ_TG, kind='opt'):
    d = np.load(os.path.join(ROOT, folder, "plotting_data",
                             f"plotting_data_cz_v2_{tag}.npz"), allow_pickle=True)
    tgs = np.asarray(d['taxis'], dtype=float)
    i = int(np.argmin(np.abs(tgs - tg)))
    s = d[f'sequences_{kind}'][i]
    return (jnp.asarray(s[0]), jnp.asarray(s[1]))


def idle_best_over_M(folder, tag, tg, kind='opt', want_seq=False):
    d = np.load(os.path.join(ROOT, folder, f"optimization_data_all_M_{tag}.npz"),
                allow_pickle=True)
    best = None
    for m in (int(x) for x in d['M_values']):
        gts = np.asarray(d[f'M{m}_gate_times'], dtype=float)
        ix = np.where(np.isclose(gts, tg))[0]
        if not ix.size:
            continue
        k = int(ix[0])
        inf = float(d[f'M{m}_infs_{kind}'][k])
        seq = d[f'M{m}_sequences_{kind}'][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, (jnp.asarray(seq[0]), jnp.asarray(seq[1])))
    if best is None:
        return (np.nan, 0, None) if want_seq else np.nan
    return best if want_seq else best[0]


def main():
    out = {}

    # ---- knowledge ladders --------------------------------------------------
    print("== CZ knowledge ladder (true 1-F at Tg=320) ==")
    cz_ladder = dict(
        cdd=cz_true_at(CAP, 'cap', kind='known'),
        rung_c=cz_true_at(CAP, 'rung_c_cap'),
        diag3=cz_true_at(f"{CAP}_diag3", 'diag3_cap'),
        robust4=cz_true_at(f"{CAP}_robust4", 'robust4_cap'),
        full=cz_true_at(CAP, 'cap'),
    )
    for k, v in cz_ladder.items():
        print(f"  {k:8s} {v:.4e}")
        out[f'cz_ladder_{k}'] = v

    print("== idle knowledge ladder (true 1-F, best over M) ==")
    id_tags = dict(rung_c=('', 'rung_c_idle_cap'),
                   diag3=('_diag3', 'diag3_idle_cap'),
                   robust4=('_robust4', 'robust4_idle_cap'),
                   full=('', 'cap'))
    for tg in ID_TGS:
        cdd = idle_best_over_M(CAP, 'cap', tg, kind='known')
        out[f'id_ladder_cdd_{int(tg)}'] = cdd
        print(f"  Tg={tg:6.0f}  cdd      {cdd:.4e}")
        for k, (sub, tag) in id_tags.items():
            v = idle_best_over_M(f"{CAP}{sub}", tag, tg)
            out[f'id_ladder_{k}_{int(tg)}'] = v
            print(f"  Tg={tg:6.0f}  {k:8s} {v:.4e}")

    # ---- SPAM-arm design blocks --------------------------------------------
    print("== CZ SPAM-arm designs (Tg=320): known_true / nt_true / nt_pred ==")
    for arm in ARMS:
        folder = run_folder(spam=True, protocol=arm)
        tag = f"rung_d_{arm}"
        try:
            kt = cz_true_at(folder, tag, kind='known')
            nt = cz_true_at(folder, tag, kind='opt')
            cfg = czmod.CZOptConfig(fname=folder, min_sep_factor=8.0,
                                    max_pulses=10**9, gate_time_factors=[])
            seq = cz_winner_seq(folder, tag)
            npred = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG,
                                                     use_ideal=False))
        except (FileNotFoundError, KeyError) as e:
            print(f"  {arm:10s} SKIPPED (no rung-d gate data: {e})")
            continue
        out[f'cz_arm_{arm}'] = np.array([kt, nt, npred])
        print(f"  {arm:10s} {kt:.4e} / {nt:.4e} / {npred:.4e}  "
              f"(label NT({len(seq[0])-2},{len(seq[1])-2}))")

    print("== idle SPAM-arm designs (Tg=640, best over M) ==")
    for arm in ARMS:
        folder = run_folder(spam=True, protocol=arm)
        tag = f"rung_d_idle_{arm}"
        try:
            kt = idle_best_over_M(folder, tag, 640.0, kind='known')
            nt, m, seq = idle_best_over_M(folder, tag, 640.0, want_seq=True)
            cfg = idmod.Config(fname=folder, M=m, max_pulses=10**9,
                               min_sep_factor=8.0)
            npred = float(idmod.calculate_infidelity(seq, cfg, m, 640.0 / m,
                                                     use_ideal=False))
        except (FileNotFoundError, KeyError) as e:
            print(f"  {arm:10s} SKIPPED (no rung-d gate data: {e})")
            continue
        out[f'id_arm_{arm}'] = np.array([kt, nt, npred])
        print(f"  {arm:10s} {kt:.4e} / {nt:.4e} / {npred:.4e}  (M={m})")

    # ---- error budget: 2Q-channel share -------------------------------------
    print("== error budget: two-qubit-channel share (truth, fixed sequences) ==")
    cfg = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                            gate_time_factors=[])
    seq = cz_winner_seq(CAP, 'cap')
    full = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    cfg.SMat_ideal = _zero_2q(cfg.SMat_ideal)
    only1q = float(czmod.calculate_infidelity(seq, cfg, 1, CZ_TG, use_ideal=True))
    out['cz_budget_share_nt'] = (full - only1q) / full
    fid_seq = (jnp.array([0., CZ_TG]), jnp.array([0., CZ_TG]))
    cfg2 = czmod.CZOptConfig(fname=CAP, min_sep_factor=8.0, max_pulses=10**9,
                             gate_time_factors=[])
    fid_full = float(czmod.calculate_infidelity(fid_seq, cfg2, 1, CZ_TG,
                                                use_ideal=True))
    cfg2.SMat_ideal = _zero_2q(cfg2.SMat_ideal)
    fid_1q = float(czmod.calculate_infidelity(fid_seq, cfg2, 1, CZ_TG,
                                              use_ideal=True))
    out['cz_budget_share_fid'] = (fid_full - fid_1q) / fid_full
    print(f"  CZ 320: FID share {out['cz_budget_share_fid']:.1%}, "
          f"NT-winner share {out['cz_budget_share_nt']:.1%}")

    nt, m, seq = idle_best_over_M(CAP, 'cap', 2560.0, want_seq=True)
    icfg = idmod.Config(fname=CAP, M=m, max_pulses=10**9, min_sep_factor=8.0)
    full = float(idmod.calculate_infidelity(seq, icfg, m, 2560.0 / m,
                                            use_ideal=True))
    icfg.SMat_ideal = _zero_2q(icfg.SMat_ideal)
    only1q = float(idmod.calculate_infidelity(seq, icfg, m, 2560.0 / m,
                                              use_ideal=True))
    out['id_budget_share_nt'] = (full - only1q) / full
    print(f"  idle 2560 (M={m}): NT-winner share {out['id_budget_share_nt']:.1%}")

    path = os.path.join(ROOT, CAP, "design_numbers.npz")
    np.savez(path, **out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
