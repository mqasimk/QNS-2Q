"""SHOWCASE-0612 shared-carrier invariance check.

The shared-carrier edit (noise/spectra.py: _SC_C2_QS) moves slow-carrier power
into the inter-qubit cross-spectrum S_1_2 without touching any self-spectrum
or the qubit-exchange crosses. Average process fidelity is first-order blind
to the off-diagonal cross-spectra (the +/- cross terms cancel pairwise over
the PTM elements), so the true infidelity of every stored winner should move
only at second order -- at the 1e-4-class NT points, invisibly; at the
deep-decoherence FID points, by at most the cosh(delta) convexity factor.

Re-scores the stored cap winners (CZ NT/CDD/FID per gate time; idle NT/CDD
per (M, Tg)) on the CURRENT model truth (SMat_ideal) and prints old vs new.

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/verify_carrier_invariance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root
from qns2q.control import cz as czmod
from qns2q.control import idle as idmod

FOLDER = "DraftRun_NoSPAM_showcase_cap"


def check_cz(tag):
    path = os.path.join(project_root(), FOLDER, "plotting_data",
                        f"plotting_data_cz_v2_{tag}.npz")
    if not os.path.exists(path):
        print(f"[cz:{tag}] no file, skipped")
        return
    d = np.load(path, allow_pickle=True)
    cfg = czmod.CZOptConfig(fname=FOLDER, min_sep_factor=8.0,
                            max_pulses=10**9)
    taxis = np.asarray(d["taxis"], dtype=float)
    print(f"\n[cz:{tag}]  Tg      kind   old           new           rel.diff")
    for k, tg in enumerate(taxis):
        rows = [("FID", (jnp.array([0., tg]), jnp.array([0., tg])),
                 float(d["infs_nopulse"][k])),
                ("CDD", d["sequences_known"][k], float(d["infs_known"][k])),
                ("NT", d["sequences_opt"][k], float(d["infs_opt"][k]))]
        for kind, seq, old in rows:
            if seq is None:
                continue
            seq = (jnp.asarray(seq[0]), jnp.asarray(seq[1]))
            new = float(czmod.calculate_infidelity(seq, cfg, 1, tg,
                                                   use_ideal=True))
            rel = (new - old) / old if old else float("nan")
            print(f"  {tg:7.0f}  {kind:4s}  {old:.6e}  {new:.6e}  {rel:+.2e}")


def check_idle():
    path = os.path.join(project_root(), FOLDER,
                        "optimization_data_all_M_cap.npz")
    d = np.load(path, allow_pickle=True)
    m_values = [int(m) for m in d["M_values"]]
    print(f"\n[idle]  M    Tg      kind   old           new           rel.diff")
    for m in m_values:
        cfg = idmod.Config(fname=FOLDER, M=m, max_pulses=10**9,
                           min_sep_factor=8.0)
        gts = np.asarray(d[f"M{m}_gate_times"], dtype=float)
        for k, tg in enumerate(gts):
            t_seq = tg / m
            for kind, skey, ikey in (("CDD", "sequences_known", "infs_known"),
                                     ("NT", "sequences_opt", "infs_opt")):
                seq = d[f"M{m}_{skey}"][k]
                if seq is None:
                    continue
                old = float(d[f"M{m}_{ikey}"][k])
                seq = (jnp.asarray(seq[0]), jnp.asarray(seq[1]))
                new = float(idmod.calculate_infidelity(seq, cfg, m, t_seq,
                                                       use_ideal=True))
                rel = (new - old) / old if old else float("nan")
                print(f"  {m:3d}  {tg:7.0f}  {kind:4s}  {old:.6e}  "
                      f"{new:.6e}  {rel:+.2e}")


if __name__ == "__main__":
    check_cz("cap")
    check_cz("cap_short")
    check_idle()
    print("\nExpectation: NT/CDD rel.diff at the 1e-16..1e-3 level "
          "(second order); FID may move at the convexity level deep past T2*.")
