"""SHOWCASE-0612 reconstruction acceptance gate (capture-grade criterion).

User directive 2026-06-12: the reconstruction must CAPTURE the spectrum --
"at least it should be within error bars". No bounded-only categories.

Per channel, per tooth (omega > 0):
  (i)   accuracy:   pull = |rec - truth| / sigma_tot <= 2 for >= 90% of teeth
        and <= 3 for ALL teeth (the within-bars requirement);
  (ii)  information: SNR = |truth| / sigma_tot >= 1 for ALL teeth -- the bars
        must be small enough that "within bars" is a statement about the
        spectrum, not about ignorance. Line teeth (within 2 sigma of a known
        center) must clear SNR >= 3 (detected, not just consistent).
  (iii) the DC points and per-channel medians are quoted for the record.

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
        pos = (wk > 0) & np.isfinite(rec)
        if sig is None or not pos.any():
            print(f"  {key:6s} -- no bars/teeth; SKIPPED")
            continue
        # complex channels: use the larger of the Re/Im bars against |dev|
        if np.iscomplexobj(sig):
            sig_eff = np.maximum(np.abs(np.real(sig)), np.abs(np.imag(sig)))
        else:
            sig_eff = sig
        pull = np.abs(rec - tr) / (sig_eff + 1e-30)
        snr = np.abs(tr) / (sig_eff + 1e-30)

        line_mask = np.zeros(wk.size, dtype=bool)
        if key in priors:
            centers, sigmas = priors[key]
            for w0, sg in zip(np.atleast_1d(centers), np.atleast_1d(sigmas)):
                line_mask |= np.abs(wk - w0) < 2.0 * sg
        line_mask &= pos

        p = pull[pos]
        s = snr[pos]
        cov2 = float(np.mean(p <= 2.0))
        ok_acc = bool(cov2 >= 0.90 and np.max(p) <= 3.0)
        ok_snr = bool(np.min(s) >= 1.0)
        ok_line = bool(np.min(snr[line_mask]) >= 3.0) if line_mask.any() else True
        overall_ok &= (ok_acc and ok_snr and ok_line)

        msg = (f"  {key:6s} pull med {np.median(p):4.2f} max {np.max(p):4.2f} "
               f"within-2sig {cov2:5.1%} {'OK' if ok_acc else '** ACC FAIL **'}"
               f" | SNR min {np.min(s):4.1f} med {np.median(s):5.1f} "
               f"{'OK' if ok_snr else '** SNR FAIL **'}")
        if line_mask.any():
            msg += (f" | line teeth {int(line_mask.sum())}: min SNR "
                    f"{np.min(snr[line_mask]):4.1f} "
                    f"{'OK' if ok_line else '** LINE FAIL **'}")
        dc_ok_key = f"{key}_dc_ok"
        if wk[0] == 0.0 and dc_ok_key in specs.files:
            msg += (f" | DC ok={bool(specs[dc_ok_key])} "
                    f"rec {np.real(rec[0]):.2e} vs {np.real(tr[0]):.2e}")
        print(msg)

    print(f"ACCEPTANCE: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
