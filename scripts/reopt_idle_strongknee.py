"""Settle the idle at the strong ZZ knee (1.4e-5): re-optimize the idle full
(2Q-informed) and rung_c (1Q-blind, --self-only) designs on the new ground
truth, best over a focused M-sweep, at 640tau and 2560tau. Tells us whether
the 2Q idle dodges the forced-on-the-CZ knee back to 1e-4-class, and the
emergent idle design gap. Outputs tagged sk_* (no overwrite of published).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from qns2q.control.idle import Config, run_optimization_pipeline

CAP = "DraftRun_NoSPAM_showcase_cap"
GTF = [-1, -3]           # -> 640, 2560 tau
GTS = [640., 2560.]
M_LIST = [1, 2, 4, 8, 16]


def run_design(self_only, tag):
    best_opt = {gt: None for gt in GTS}
    best_cdd = {gt: None for gt in GTS}
    for m in M_LIST:
        cfg = Config(fname=CAP, include_cross_spectra=True, spectral_model='interp',
                     use_known_as_seed=False, M=m, max_pulses=10**9, max_dim=2600,
                     min_sep_factor=8.0, char_self_only=self_only,
                     informed_counts=False, num_random_trials=20, tau_divisor=160,
                     use_simulated=True, gate_time_factors=GTF,
                     output_path_known=f"infs_known_id_M{m}_{tag}.npz",
                     output_path_opt=f"infs_opt_id_M{m}_{tag}.npz",
                     plot_filename=f"x_{tag}_M{m}.pdf",
                     plot_data_name=f"pd_{tag}_M{m}.npz")
        res = run_optimization_pipeline(cfg)
        for i, gt in enumerate(GTS):
            io = float(res['opt'][i][0])
            ik = float(res['known'][i][0])
            if best_opt[gt] is None or io < best_opt[gt][0]:
                best_opt[gt] = (io, m)
            if best_cdd[gt] is None or ik < best_cdd[gt][0]:
                best_cdd[gt] = (ik, m)
        msg = ", ".join(f"{gt:.0f}:{float(res['opt'][i][0]):.2e}"
                        for i, gt in enumerate(GTS))
        print(f"  [{tag}] M={m} done -> {msg}", flush=True)
    return best_opt, best_cdd


print("=== FULL (2Q-informed) ===", flush=True)
full_opt, full_cdd = run_design(False, "sk_full")
print("=== RUNG_C (1Q-blind) ===", flush=True)
rungc_opt, _ = run_design(True, "sk_rungc")

print("\n=== IDLE at strong knee 1.4e-5 ===", flush=True)
for gt in GTS:
    f, mf = full_opt[gt]
    r, mr = rungc_opt[gt]
    c = full_cdd[gt][0]
    print(f"  Tg={gt:.0f}: full(2Q) {f:.3e} (M={mf}) | rung_c(1Q) {r:.3e} (M={mr}) "
          f"| idle gap {r/f:.1f}x | best CDD {c:.3e}", flush=True)
