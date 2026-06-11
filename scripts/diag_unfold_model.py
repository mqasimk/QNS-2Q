"""UNFOLD-RESIDUAL diagnostic: is the post-unfold residual pure model mismatch?

Replicates the production unfold (PL-fed, two fixed-point iterations) on one
SPAM arm, additionally computes the TRUTH-fed bias, and decomposes the
post-unfold deviation per spectrum:

    raw_dev   = raw_recon - S_truth          (true bias + stat noise)
    applied   = b2                           (what the unfold subtracts)
    post_meas = raw_dev - b2                 (the measured residual we see)
    post_pred = b_true - b2                  (predicted from model mismatch alone)
    stat_left = raw_dev - b_true             (what a truth-fed unfold would leave)

If post_meas ~= post_pred (and stat_left is at the stat-error scale), the
residual is 100% selfconsistent_spectra model mismatch -> fix the basis.

Usage:  python scripts/diag_unfold_model.py [reference|raw|mitigated]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qns2q.characterize.reconstruct import (SpectraReconConfig, SpectraReconstructor,
                                            _SYS_TO_SPEC, _SYS_TO_ERR)
from qns2q.characterize.systematics import (forward_model_systematic, dc_fit_systematic,
                                            selfconsistent_spectra, analytic_spectra)
from qns2q.noise.spectra import line_priors
from qns2q.paths import run_folder


def main(protocol='reference'):
    cfg = SpectraReconConfig(data_folder=run_folder(spam=True, protocol=protocol))
    rec = SpectraReconstructor(cfg)
    rec.load_observables()
    rec.reconstruct()
    c = cfg

    inv_opts = dict(inversion_method=c.inversion_method, reg_lambda=c.reg_lambda,
                    enforce_nonneg=c.enforce_nonneg)
    n_h = len(c.c_times)
    wk_full = np.concatenate(([0.0], [2*np.pi*(j + 1)/c.T for j in range(n_h)]))

    raw0, nan_mask = {}, {}
    for sk, rk in _SYS_TO_SPEC.items():
        arr = np.asarray(rec.reconstructed_spectra[rk])
        nan_mask[sk] = ~np.isfinite(arr)
        raw0[sk] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def bias_of(spectra_callables):
        b = forward_model_systematic(spectra_callables, c.c_times, c.M, c.T, c.t_vec,
                                     c.w_grain, c.wmax, inv_opts=inv_opts)
        dcb = dc_fit_systematic(spectra_callables, rec.dc_t_sweep,
                                s1212_echo_ct=getattr(rec, 'dc_echo_ct', None),
                                s1212_echo_obs_err=getattr(rec, 'dc_echo_obs_err', None),
                                s1212_echo_wmax=2 * (c.synth_wmax or c.wmax),
                                fid_obs_err=getattr(rec, 'dc_fid_obs_err', None),
                                world_grid=rec._world_grid())
        out = {}
        for sk in _SYS_TO_SPEC:
            full = np.concatenate(([dcb[sk]], np.asarray(b[sk])))
            if not rec.dc_reliable.get(sk, True):
                full[0] = 0.0
            out[sk] = full
        return out

    # Production unfold: two self-consistent-model-fed fixed-point iterations
    # (line+tail-aware when the regime carries line priors, like production).
    lines = line_priors()
    b1 = bias_of(selfconsistent_spectra(wk_full, raw0, lines=lines))
    recon1 = {sk: raw0[sk] - b1[sk] for sk in raw0}
    b2 = bias_of(selfconsistent_spectra(wk_full, recon1, lines=lines))

    # Truth-fed bias (the unfold an oracle would apply).
    b_true = bias_of(analytic_spectra())

    truth_cb = analytic_spectra()
    truth = {sk: np.asarray(truth_cb[sk](wk_full)) for sk in _SYS_TO_SPEC}

    def rms(x):
        x = np.asarray(x)[1:]          # harmonics only
        return float(np.sqrt(np.nanmean(np.abs(x) ** 2)))

    print(f"\n=== unfold-model diagnostic, arm={protocol} "
          f"(RMS over harmonics; rel = RMS/RMS(truth)) ===")
    hdr = (f"{'spec':>6} | {'raw_dev':>9} {'b2(PL)':>9} {'b_true':>9} | "
           f"{'post_meas':>9} {'post_pred':>9} {'corr':>5} | {'stat_left':>9} {'sig_stat':>9}")
    print(hdr)
    print('-' * len(hdr))
    for sk in _SYS_TO_SPEC:
        dev = raw0[sk] - truth[sk]
        dev[nan_mask[sk]] = np.nan
        post_meas = dev - b2[sk]
        post_pred = b_true[sk] - b2[sk]
        stat_left = dev - b_true[sk]
        err = np.nan_to_num(np.asarray(rec.reconstructed_spectra_err[_SYS_TO_ERR[sk]]),
                            nan=0.0)
        pm, pp = post_meas[1:], post_pred[1:]
        ok = np.isfinite(pm) & np.isfinite(pp)
        if ok.sum() > 2:
            a = np.concatenate([np.real(pm[ok]), np.imag(pm[ok])])
            bb = np.concatenate([np.real(pp[ok]), np.imag(pp[ok])])
            corr = float(np.corrcoef(a, bb)[0, 1])
        else:
            corr = np.nan
        tr = rms(truth[sk])
        print(f"{sk:>6} | {rms(dev):9.2e} {rms(b2[sk]):9.2e} {rms(b_true[sk]):9.2e} | "
              f"{rms(post_meas):9.2e} {rms(post_pred):9.2e} {corr:5.2f} | "
              f"{rms(stat_left):9.2e} {rms(err):9.2e}")
        print(f"{'':>6} |  rel:{rms(dev)/tr:7.1%} {'':>20} |  rel:{rms(post_meas)/tr:7.1%} "
              f"{rms(post_pred)/tr:7.1%} {'':>7} | {rms(stat_left)/tr:7.1%} {rms(err)/tr:8.1%}")

    print("\nDC (w=0) rows [value-scale, signed]:")
    print(f"{'spec':>6} | {'raw_dev':>10} {'b2(PL)':>10} {'b_true':>10} | "
          f"{'post_meas':>10} {'post_pred':>10} | {'stat_left':>10} {'sig_stat':>10}")
    for sk in _SYS_TO_SPEC:
        dev0 = float(np.real(raw0[sk][0] - truth[sk][0]))
        if nan_mask[sk][0]:
            continue
        e0 = float(np.real(np.asarray(rec.reconstructed_spectra_err[_SYS_TO_ERR[sk]])[0]))
        print(f"{sk:>6} | {dev0:10.2e} {np.real(b2[sk][0]):10.2e} "
              f"{np.real(b_true[sk][0]):10.2e} | {dev0 - np.real(b2[sk][0]):10.2e} "
              f"{np.real(b_true[sk][0] - b2[sk][0]):10.2e} | "
              f"{dev0 - np.real(b_true[sk][0]):10.2e} {e0:10.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'reference')
