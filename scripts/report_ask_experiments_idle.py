"""Idle-gate twin of report_ask_experiments.py (knowledge subsets + SPAM arms).

Same two experiments as the CZ version, on the idle gate at the report's two
idle rows, with the repetition number M FIXED to the main run's blind
best-over-M winner at each gate time (Tg=640: M=2, Tg=2560: M=8) so the
within-block design comparison is controlled:

  subsets : all-6 / robust-4 / diag-3 / 1Q-2 channel knowledge given to the
            blind optimizer (ideal benchmark always keeps full truth)
  arms    : full-knowledge optimization on the reference vs raw (SPAM-biased)
            arm reconstructions, at Tg=640

The RNG is re-seeded before every variant so all variants within a block
share identical random pulse-count draws and delay seeds, mirroring the main
run's optimizer settings (max_pulses=1000, num_random_trials=20).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.control import idle

# Gate times for the two experiments; M is resolved at runtime to the blind
# best-over-M winner's repetition number from optimization_data_all_M.npz.
TG_SUBSETS = (640.0, 2560.0)
TG_ARMS = (640.0,)


def winner_M(folder_path, Tg):
    """Repetition number of the best-over-M NT winner at this gate time."""
    d = np.load(os.path.join(folder_path, 'optimization_data_all_M.npz'),
                allow_pickle=True)
    best = None
    for m in (int(x) for x in d['M_values']):
        mg = np.asarray(d[f'M{m}_gate_times'], dtype=float)
        ix = np.where(np.abs(mg - Tg) < 1e-9)[0]
        if not ix.size:
            continue
        inf = float(d[f'M{m}_infs_opt'][int(ix[0])])
        if best is None or inf < best[0]:
            best = (inf, m)
    if best is None:
        raise ValueError(f"no idle winner at Tg={Tg}")
    return best[1]


def make_config(fname=None, M=1):
    return idle.Config(fname=fname, M=M, max_pulses=1000,
                       num_random_trials=20, tau_divisor=160,
                       use_known_as_seed=False, use_simulated=False,
                       gate_time_factors=[])


def run_block(cfg, label, Tg, M):
    """One (Tg, M) block: blind known-library selection + 20 random-restart
    NT optimization on cfg.SMat, winners scored on cfg.SMat_ideal."""
    np.random.seed(idle.RANDOM_SEED)
    idle._OVERLAP_SETUP_CACHE.clear()
    cfg.Tg = Tg
    cfg.M = M
    cfg.T_seq = Tg / M
    cfg.max_pulses_per_rep = int(cfg.max_pulses / M)

    pLib, pDesc = idle.construct_pulse_library(cfg.T_seq, cfg.tau,
                                               cfg.max_pulses_per_rep)
    k_seq, k_char, k_idx = idle.evaluate_known_sequences(cfg, M, pLib)
    k_true = float(idle.calculate_infidelity(k_seq, cfg, M, cfg.T_seq,
                                             use_ideal=True))

    # Random pulse-count draws, exactly as run_optimization_pipeline does.
    max_n_physical = int(cfg.T_seq / cfg.tau) - 1
    effective_max = min(cfg.max_pulses_per_rep, max(1, max_n_physical - 1))
    n_pulses_list = [(np.random.randint(0, effective_max + 1),
                      np.random.randint(0, effective_max + 1))
                     for _ in range(cfg.num_random_trials)]
    nt_seq, nt_char = idle.optimize_random_sequences(cfg, M, n_pulses_list)
    nt_true = float(idle.calculate_infidelity(nt_seq, cfg, M, cfg.T_seq,
                                              use_ideal=True))
    n1, n2 = len(nt_seq[0]) - 2, len(nt_seq[1]) - 2
    print(f"[Tg={Tg:5.0f} M={M:<3d} {label:13s}] known {pDesc[k_idx]:>16s}^{M} "
          f"char {k_char:.3e} true {k_true:.3e} | NT({n1},{n2})^{M} "
          f"char {nt_char:.3e} true {nt_true:.3e}", flush=True)
    return dict(label=label, Tg=float(Tg), M=int(M),
                known_desc=f"{pDesc[k_idx]}^{M}",
                known_char=float(k_char), known_true=k_true, nt_n=(n1, n2),
                nt_char=float(nt_char), nt_true=nt_true)


def zero_channels(SMat, drop):
    out = SMat
    for d in drop:
        out = out.at[d, :, :].set(0).at[:, d, :].set(0)
    return out


def with_cross_only(SMat, keep_pairs):
    out = SMat
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            if r != c and (r, c) not in keep_pairs and (c, r) not in keep_pairs:
                out = out.at[r, c, :].set(0)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default=None,
                    help="run folder supplying the reconstruction for the "
                         "knowledge-subset blocks AND receiving the output "
                         "npz (default: the active regime's NoSPAM folder)")
    a = ap.parse_args()
    results = []

    # ---- Spectral-knowledge subsets ----
    cfg = make_config(fname=a.folder)
    full = cfg.SMat
    variants = [
        ("all-6", full),
        ("robust-4", with_cross_only(full, {(1, 2)})),
        ("diag-3", with_cross_only(full, set())),
        ("1Q-2", zero_channels(with_cross_only(full, set()), [3])),
    ]
    for Tg in TG_SUBSETS:
        M = winner_M(cfg.path, Tg)
        for label, smat in variants:
            cfg.SMat = smat
            results.append(run_block(cfg, label, Tg, M))
    cfg.SMat = full

    # ---- Reference vs raw (SPAM-biased) arm, full 6 spectra ----
    for Tg in TG_ARMS:
        M = winner_M(cfg.path, Tg)
        for arm in ("reference", "raw"):
            cfg_a = make_config(fname=f"DraftRun_SPAM_featured_{arm}", M=M)
            results.append(run_block(cfg_a, f"arm-{arm}", Tg, M))

    out = os.path.join(cfg.path, "report_ask_experiments_idle.npz")
    np.savez(out, results=np.array(results, dtype=object),
             seed=idle.RANDOM_SEED)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
