"""DC-sensitivity audit for the optimized CZ sequences (LV0606-WHYSPAM / DC item).

Question: how much does the *predicted* CZ infidelity move when each spectrum's
S(0) shifts within its quoted uncertainty band (+/- 2 sigma_tot of the DC point of
the reference-arm reconstruction)?  Establishes whether the undetermined
S_1212(0) matters at the gate level, and whether the optimized sequences carry
single-qubit DC exposure (unbalanced y_1/y_2).

Usage:  QNS2Q_REGIME=featured python scripts/audit_dc_sensitivity.py
Reads:  DraftRun_NoSPAM_featured/plotting_data/plotting_data_cz_v2.npz  (sequences)
        DraftRun_SPAM_featured_reference/specs.npz                      (spectra+bars)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from qns2q.control.cz import (CZOptConfig, calculate_infidelity, make_tk12,
                              _OVERLAP_SETUP_CACHE)
from qns2q.paths import project_root

SPEC_KEYS = ['S11', 'S22', 'S1212', 'S12', 'S112', 'S212']
SMAT_SLOT = {'S11': (1, 1), 'S22': (2, 2), 'S1212': (3, 3),
             'S12': (1, 2), 'S112': (1, 3), 'S212': (2, 3)}
SELF_KEYS = {'S11', 'S22', 'S1212'}


def signed_area(pt):
    """F(0,T) = integral of the toggling function: signed area of y over [0,T]."""
    diffs = jnp.diff(pt)
    signs = jnp.array((-1.0) ** jnp.arange(len(diffs)))
    return float(jnp.sum(diffs * signs))


def build_smat(cfg, dc_override=None):
    """SMat from cfg.specs with optional DC overrides {key: new S(0)}.

    Mirrors CZOptConfig._build_interpolated_spectra but keeps the RECONSTRUCTED
    DC at the w=0 sample (the class method overwrites that one sample with the
    analytic value -- here the DC must follow the override for the scan)."""
    dc_override = dc_override or {}
    SMat = jnp.zeros((4, 4, cfg.w.size), dtype=jnp.complex128)

    def series(key):
        a = np.array(cfg.specs[key], dtype=complex)
        if key in dc_override:
            a[0] = dc_override[key]
        return (jnp.interp(cfg.w, cfg.wkqns, jnp.real(jnp.asarray(a)), right=0.) +
                1j * jnp.interp(cfg.w, cfg.wkqns, jnp.imag(jnp.asarray(a)), right=0.))

    for key in SPEC_KEYS:
        r, c = SMAT_SLOT[key]
        s = series(key)
        SMat = SMat.at[r, c].set(s)
        if r != c:
            SMat = SMat.at[c, r].set(jnp.conj(s))
    return SMat


def predicted_inf(cfg, seq, T_seq, smat):
    cfg.SMat = smat
    _OVERLAP_SETUP_CACHE.clear()
    return float(calculate_infidelity(seq, cfg, 1, T_seq, use_ideal=False))


def main():
    pd = np.load(os.path.join(project_root(),
                 'DraftRun_NoSPAM_featured/plotting_data/plotting_data_cz_v2.npz'))
    seqs = {
        'NT-optimized': ((jnp.array(pd['best_opt_seq_pt1']),
                          jnp.array(pd['best_opt_seq_pt2'])),
                         float(pd['T_seq_best_opt'])),
        'best known (CDD-lib)': ((jnp.array(pd['best_known_seq_pt1']),
                                  jnp.array(pd['best_known_seq_pt2'])),
                                 float(pd['T_seq_best_known'])),
    }

    cfg = CZOptConfig(fname='DraftRun_SPAM_featured_reference', use_simulated=False)
    specs = cfg.specs
    base_smat = build_smat(cfg)

    print(f"\nSpectra: DraftRun_SPAM_featured_reference (64k SPAM-free arm)")
    print(f"{'key':>6} {'S(0) fit':>11} {'2sig_tot(DC)':>13} {'dc_ok':>6}")
    bands = {}
    for key in SPEC_KEYS:
        v0 = float(np.real(np.asarray(specs[key])[0]))
        e0 = 2.0 * abs(float(np.real(np.asarray(specs[f'{key}_errtot'])[0])))
        ok = bool(np.asarray(specs[f'{key}_dc_ok'])) if f'{key}_dc_ok' in specs.files else True
        bands[key] = (v0, e0)
        print(f"{key:>6} {v0:11.3e} {e0:13.3e} {str(ok):>6}")

    for tag, (seq, T_seq) in seqs.items():
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        F1, F2, F12 = signed_area(pt1), signed_area(pt2), signed_area(pt12)
        base = predicted_inf(cfg, seq, T_seq, base_smat)
        true = None
        cfg.SMat = base_smat
        _OVERLAP_SETUP_CACHE.clear()
        true = float(calculate_infidelity(seq, cfg, 1, T_seq, use_ideal=True))

        print(f"\n=== {tag}  (Tg = {T_seq:.0f} tau, M=1) ===")
        print(f"  DC filter weights F(0,Tg):  F_1 = {F1:+.3f}   F_2 = {F2:+.3f}   "
              f"F_12 = {F12:+.3f}   (Tg = {T_seq:.0f})")
        print(f"  predicted infidelity (reconstructed spectra): {base:.4e}"
              f"   [true/analytic: {true:.4e}]")
        print(f"  {'channel':>8} {'S(0)->':>12} {'pred 1-F':>12} {'delta':>11} {'%of base':>9}")
        for key in SPEC_KEYS:
            v0, e0 = bands[key]
            for sgn, lab in ((+1, '+2sig'), (-1, '-2sig')):
                new_dc = v0 + sgn * e0
                if key in SELF_KEYS:
                    new_dc = max(new_dc, 0.0)   # physical floor for self-spectra
                inf = predicted_inf(cfg, seq, T_seq,
                                    build_smat(cfg, {key: new_dc}))
                d = inf - base
                print(f"  {key:>8} {lab:>5} {new_dc:9.2e} {inf:12.4e} {d:+11.2e} "
                      f"{100 * d / base:8.1f}%")
    print("\n(delta = shift of the PREDICTED gate error when that channel's DC moves "
          "to the edge of its quoted band; the optimizer objective moves identically.)")


if __name__ == '__main__':
    main()
