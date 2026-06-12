"""SHOWCASE-0612 reconstruction acceptance gate.

Compares a reconstructed specs.npz against the analytic showcase truth with
the feature-centric criteria the trap landscape calls for (the generic
rel-dev table of gates_helper misreads a landscape whose gap teeth sit BELOW
the shot floor by design):

  (i)   engineered LINE teeth are resolved: CORE teeth (within 1 sigma of a
        trap-line or coupler-line center) must be DETECTED (value > 3x its
        own bar) with pull <= 3; SHOULDER teeth (1-2 sigma out) must be
        pull-consistent (<= 3) but carry no detection requirement -- their
        truth values sit near the bar level by construction;
  (ii)  gap teeth are CONSISTENT with the quiet floor: |rec| <= max(3 sigma,
        3x truth) -- an upper bound, not a detection requirement;
  (iii) the per-channel medians are quoted for the record (pulls, rel devs,
        within-2-sigma coverage) plus the DC points.

Usage: QNS2Q_REGIME=showcase python scripts/check_showcase_recon.py [folder]
       (default folder: the active regime's NoSPAM folder)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.paths import current_regime, project_root, run_folder
from qns2q.characterize.systematics import analytic_spectra
from qns2q.noise.spectra import line_priors_per_channel


def main(folder=None):
    folder = folder or run_folder()
    specs = np.load(os.path.join(project_root(), folder, "specs.npz"))
    truth = analytic_spectra()
    priors = line_priors_per_channel() or {}
    wk = np.asarray(specs['wk'])
    print(f"[{current_regime()}] {folder}: {wk.size} grid points "
          f"(DC included: {wk[0] == 0.0})")

    overall_ok = True
    for key in ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212'):
        rec = np.asarray(specs[key])
        tr = np.asarray(truth[key](wk))
        err_key = next((k for k in (f"{key}_errtot", f"{key}_err")
                        if k in specs.files), None)
        sig = np.abs(np.asarray(specs[err_key])) if err_key else None
        finite = np.isfinite(rec)
        pull = (np.abs(rec - tr) / (sig + 1e-30)) if sig is not None else None

        core_mask = np.zeros(wk.size, dtype=bool)
        line_mask = np.zeros(wk.size, dtype=bool)
        if key in priors:
            centers, sigmas = priors[key]
            for w0, sg in zip(np.atleast_1d(centers), np.atleast_1d(sigmas)):
                core_mask |= np.abs(wk - w0) < 1.0 * sg
                line_mask |= np.abs(wk - w0) < 2.0 * sg
        core_mask &= wk > 0
        line_mask &= wk > 0
        gap_mask = (~line_mask) & (wk > 0) & finite

        msg = [f"{key:6s}"]
        if pull is not None:
            msg.append(f"median pull {np.nanmedian(pull[finite & (wk > 0)]):5.2f}")
            cov = np.mean(pull[finite & (wk > 0)] <= 2.0)
            msg.append(f"within-2sig {cov:5.1%}")

        if line_mask.any():
            det = np.abs(rec[core_mask]) > 3 * sig[core_mask]
            pl = pull[line_mask]
            ok = bool(np.all(det) and np.all(pl <= 3.0))
            overall_ok &= ok
            msg.append(f"LINE teeth {line_mask.sum()} (core {core_mask.sum()}): "
                       f"core detected {det.sum()}/{det.size}, max pull "
                       f"{pl.max():4.2f} {'OK' if ok else '** FAIL **'}")
        if gap_mask.any() and sig is not None:
            bound = np.maximum(3 * sig[gap_mask], 3 * np.abs(tr[gap_mask]))
            ok = bool(np.all(np.abs(rec[gap_mask]) <= bound))
            overall_ok &= ok
            n_bad = int(np.sum(np.abs(rec[gap_mask]) > bound))
            msg.append(f"gap teeth {gap_mask.sum()}: floor-consistent "
                       f"{gap_mask.sum() - n_bad}/{gap_mask.sum()} "
                       f"{'OK' if ok else '** FAIL **'}")
        dc_ok_key = f"{key}_dc_ok"
        if wk[0] == 0.0 and dc_ok_key in specs.files:
            msg.append(f"DC ok={bool(specs[dc_ok_key])} "
                       f"rec {np.real(rec[0]):.2e} vs truth {np.real(tr[0]):.2e}")
        print("  " + " | ".join(msg))

    print(f"ACCEPTANCE: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
