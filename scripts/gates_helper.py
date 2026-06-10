"""Acceptance-gate helper for NOISE_MODEL_SPEC.md section 6 (driven by
run_acceptance_gates.sh; regime via QNS2Q_REGIME, one gate per process).

    python scripts/gates_helper.py qns   # gate iii: medium-stats QNS run +
                                         # reconstruction + truth-deviation table
    python scripts/gates_helper.py cz    # gates iv/v: NT-vs-CDD probe at
                                         # {80, 320, 640} tau + ZZ-relevance share
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import current_regime, project_root


def gate_qns():
    """Gate iii: medium-stats experiments + reconstruction in a scratch folder,
    then per-spectrum deviation from the analytic truth."""
    from qns2q.characterize.experiments import QNSExperimentConfig, main as exp_main
    from qns2q.characterize.reconstruct import main as rec_main
    from qns2q.characterize.systematics import analytic_spectra

    regime = current_regime()
    folder = f"GateRun_{regime}_qns"
    print(f"[gate iii / {regime}] medium-stats QNS run -> {folder}")
    cfg = QNSExperimentConfig(t_grain=1000, truncate=20, n_shots=4000, fname=folder)
    exp_main(cfg)
    rec_main(data_folder=folder)

    specs = np.load(os.path.join(project_root(), folder, "specs.npz"))
    print(f"[gate iii / {regime}] specs.npz keys: {sorted(specs.files)}")
    truth = analytic_spectra()
    wk = np.asarray(specs['wk']) if 'wk' in specs.files else None
    print(f"[gate iii / {regime}] reconstruction vs analytic truth:")
    for key in ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212'):
        if key not in specs.files or wk is None:
            print(f"  {key}: MISSING from specs.npz")
            continue
        rec = np.asarray(specs[key])
        tr = np.asarray(truth[key](wk))
        scale = np.median(np.abs(tr)) + 1e-30
        rel = np.abs(rec - tr) / (np.abs(tr) + 0.05 * scale)
        print(f"  {key}: median rel dev {np.median(rel):6.1%}   max {np.max(rel):6.1%}")
    print(f"[gate iii / {regime}] done")


def gate_cz():
    """Gates iv/v: CZ NT-vs-CDD probe on ground-truth spectra at three gate
    times; then the ZZ-channel share of the gate error (gate v)."""
    from qns2q.control.cz import (CZOptConfig, run_optimization, calculate_infidelity,
                                  construct_pulse_library, evaluate_known_sequences_with_T,
                                  _OVERLAP_SETUP_CACHE)

    regime = current_regime()
    print(f"[gate iv / {regime}] CZ probe at gate-time factors [2, 0, -1] "
          f"(Tg = 80, 320, 640 tau), ground-truth spectra")
    cfg = CZOptConfig(use_simulated=True, gate_time_factors=[2, 0, -1])
    run_optimization(cfg)

    # --- gate v: ZZ-relevance at the 320-tau point --------------------------------
    print(f"[gate v / {regime}] ZZ-channel share of the gate error at Tg = 320 tau")
    cfg2 = CZOptConfig(use_simulated=True, gate_time_factors=[0])
    T_seq = cfg2.Tqns / 2 ** (0 - 1)            # 320 tau, M = 1
    pt = jnp.array([0., T_seq])
    seq_nopulse = (pt, pt)
    pLib, pDesc = construct_pulse_library(T_seq, cfg2.tau, cfg2.max_pulses)
    best_seq, _, idx = evaluate_known_sequences_with_T(cfg2, 1, T_seq, pLib)

    def both(tag, seq):
        _OVERLAP_SETUP_CACHE.clear()
        full = float(calculate_infidelity(seq, cfg2, 1, T_seq, use_ideal=True))
        keep_i, keep_id = cfg2.SMat, cfg2.SMat_ideal
        cfg2.SMat = cfg2.SMat.at[3, :, :].set(0).at[:, 3, :].set(0)
        cfg2.SMat_ideal = cfg2.SMat_ideal.at[3, :, :].set(0).at[:, 3, :].set(0)
        _OVERLAP_SETUP_CACHE.clear()
        nozz = float(calculate_infidelity(seq, cfg2, 1, T_seq, use_ideal=True))
        cfg2.SMat, cfg2.SMat_ideal = keep_i, keep_id
        _OVERLAP_SETUP_CACHE.clear()
        share = 1 - nozz / full if full > 0 else float('nan')
        print(f"  {tag}: infidelity {full:.4e} -> {nozz:.4e} without ZZ channel "
              f"(ZZ+cross share {share:.1%})")

    both("no-pulse (FID)", seq_nopulse)
    both(f"best known ({pDesc[idx]})", best_seq)
    print(f"[gate v / {regime}] done")


if __name__ == "__main__":
    gate = sys.argv[1] if len(sys.argv) > 1 else 'qns'
    if gate == 'qns':
        gate_qns()
    elif gate == 'cz':
        gate_cz()
    else:
        raise SystemExit(f"Unknown gate {gate!r}; expected qns|cz")
