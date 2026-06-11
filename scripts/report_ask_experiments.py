"""TALK-0609 quantitative artifacts (asks 2 and 3), at the CZ sweet spot.

Ask 2 ("do I need all 6 spectra?"): optimize blind on progressively reduced
CHARACTERIZED spectral knowledge -- the ideal benchmark always keeps the full
truth -- and report the true infidelity of each blind winner:
    all-6   : S11, S22, S1212, S12, S112, S212
    robust-4: drop S112, S212 (what the SPAM-robust protocol loses)
    diag-3  : drop all cross-spectra
    1Q-2    : S11, S22 only (a single-qubit QNS world: no ZZ knowledge)

Ask 3 ("why characterize SPAM?"): same optimization, full 6 spectra, but on
the RAW (SPAM-biased) arm's reconstruction vs the reference arm's -- the true
cost of optimizing against biased spectra.

The RNG is re-seeded before every variant so all variants share identical
restart initializations; the gate-time block mirrors run_optimization's
(library blind selection + 36-restart sweep with identity padding).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.control import cz
from qns2q.control.padding import pad_targets

TG_SUBSETS = (80.0, 320.0)   # short-gate regime + sweet spot
TG_ARMS = (320.0,)
M = 1


def candidates(T_seq, tau, max_pulses):
    max_n = int(T_seq / tau) - 1
    eff = min(max_pulses, max(1, max_n - 1))
    s = {eff, eff - 1, eff - 2, eff // 2, eff // 2 - 1, eff // 2 + 1}
    return sorted(n for n in s if 0 < n <= max_n)


def run_block(cfg, label, Tg):
    """One gate-time block: blind library + restart sweep on cfg.SMat,
    winners scored on cfg.SMat_ideal (full truth)."""
    np.random.seed(cz.RANDOM_SEED)
    cz._OVERLAP_SETUP_CACHE.clear()
    pLib, pDesc = cz.construct_pulse_library(Tg, cfg.tau, cfg.max_pulses)
    k_seq, k_char, k_idx = cz.evaluate_known_sequences_with_T(cfg, M, Tg, pLib)
    k_true = float(cz.calculate_infidelity(k_seq, cfg, M, Tg, use_ideal=True))

    cands = candidates(Tg, cfg.tau, cfg.max_pulses)
    tgt = pad_targets(cands)
    best_char, best_seq = 1.0, None
    for n1 in cands:
        for n2 in cands:
            seq, inf = cz.optimize_sequence(cfg, M, Tg, n1, n2,
                                            pad_to=(tgt[n1 % 2], tgt[n2 % 2]))
            if seq is not None and inf < best_char:
                best_char, best_seq = inf, seq
    nt_true = float(cz.calculate_infidelity(best_seq, cfg, M, Tg, use_ideal=True))
    n1, n2 = len(best_seq[0]) - 2, len(best_seq[1]) - 2
    print(f"[Tg={Tg:5.0f} {label:12s}] known {pDesc[k_idx]:10s} char {k_char:.3e} "
          f"true {k_true:.3e} | NT({n1},{n2}) char {best_char:.3e} "
          f"true {nt_true:.3e}", flush=True)
    return dict(label=label, Tg=float(Tg), known_desc=pDesc[k_idx],
                known_char=float(k_char), known_true=k_true, nt_n=(n1, n2),
                nt_char=float(best_char), nt_true=nt_true)


def zero_channels(SMat, drop):
    """Zero the given channel indices (rows+cols) of a 4x4xN SMat copy."""
    out = SMat
    for d in drop:
        out = out.at[d, :, :].set(0).at[:, d, :].set(0)
    return out


def with_cross_only(SMat, keep_pairs):
    """Zero ALL off-diagonals except the (r, c)/(c, r) pairs listed."""
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

    # ---- Ask 2: spectral-knowledge subsets ----
    cfg = cz.CZOptConfig(fname=a.folder, use_simulated=False,
                         gate_time_factors=[])
    full = cfg.SMat
    variants = [
        ("all-6", full),
        ("robust-4", with_cross_only(full, {(1, 2)})),          # keep S12 only
        ("diag-3", with_cross_only(full, set())),                # no crosses
        ("1Q-2", zero_channels(with_cross_only(full, set()), [3])),  # selfs only
    ]
    for Tg in TG_SUBSETS:
        for label, smat in variants:
            cfg.SMat = smat
            results.append(run_block(cfg, label, Tg))
    cfg.SMat = full

    # ---- Ask 3: reference vs raw (SPAM-biased) arm, full 6 spectra ----
    for Tg in TG_ARMS:
        for arm in ("reference", "raw"):
            cfg_a = cz.CZOptConfig(fname=f"DraftRun_SPAM_featured_{arm}",
                                   gate_time_factors=[])
            results.append(run_block(cfg_a, f"arm-{arm}", Tg))

    out = os.path.join(cfg.path, "report_ask_experiments.npz")
    np.savez(out, results=np.array(results, dtype=object),
             seed=cz.RANDOM_SEED)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
