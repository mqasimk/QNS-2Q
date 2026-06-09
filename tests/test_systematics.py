"""Tests for characterize/systematics.py -- the deterministic comb-inversion systematic.

The central invariant: feeding the *comb-sum* observables (kernel sampled at the
harmonics) back through the inversion must return the input spectrum to machine
precision. That self-check validates that the systematics kernels and normalization
match the forward model exactly; if someone edits a recon kernel in inversion.py
without mirroring it here, the self-check breaks and this test fails.
"""

import numpy as np
import pytest

from qns2q.characterize.systematics import (comb_inversion_systematic, analytic_spectra,
                                            selfconsistent_spectra, dc_systematic, SPEC_KERNELS)

# Small but representative config (keep the grids light so the test stays fast).
TAU = 2.5e-8
T = 160 * TAU
M = 10
TRUNC = 8                       # fewer harmonics -> faster; invariant is grid-independent
GAMMA = T / 14
GAMMA12 = T / 28
C_TIMES = np.array([T / k for k in range(1, TRUNC + 1)])

SELF_KEYS = ['S11', 'S22', 'S1212']
CROSS_KEYS = ['S12', 'S112', 'S212']
ALL_KEYS = SELF_KEYS + CROSS_KEYS


@pytest.fixture(scope="module")
def systematic_result():
    spectra = analytic_spectra(GAMMA, GAMMA12)
    sys, chk = comb_inversion_systematic(spectra, C_TIMES, M, T,
                                         n_wfine=20001, n_tb=6000, return_selfcheck=True)
    return sys, chk


def test_self_check_machine_precision(systematic_result):
    """Comb-sum inversion must return the input spectrum to ~machine precision."""
    _, chk = systematic_result
    for key in ALL_KEYS:
        assert chk[key] < 1e-6, f"{key} self-check residual {chk[key]:.2e} too large"


def test_output_shapes_and_types(systematic_result):
    """Self-spectra systematics are real; cross-spectra are complex; length=truncate."""
    sys, _ = systematic_result
    for key in SELF_KEYS:
        assert sys[key].shape == (TRUNC,)
        assert not np.iscomplexobj(sys[key])
    for key in CROSS_KEYS:
        assert sys[key].shape == (TRUNC,)
        assert np.iscomplexobj(sys[key])


def test_systematic_is_finite_and_bounded(systematic_result):
    """The systematic must be finite and a sane fraction of the signal (not blow-up)."""
    sys, _ = systematic_result
    spectra = analytic_spectra(GAMMA, GAMMA12)
    wk = np.array([2 * np.pi * (j + 1) / T for j in range(TRUNC)])
    for key in ALL_KEYS:
        assert np.all(np.isfinite(sys[key]))
        sig = np.sqrt(np.mean(np.abs(np.asarray(spectra[key](wk))) ** 2))
        rms = np.sqrt(np.mean(np.abs(sys[key]) ** 2))
        assert rms < sig, f"{key} systematic RMS {rms:.1f} exceeds signal RMS {sig:.1f}"


def test_kernels_cover_all_six_spectra():
    """Guard: every reconstructed spectrum has a kernel spec."""
    assert set(SPEC_KERNELS) == set(ALL_KEYS)


def test_dc_systematic_finite_and_bounded():
    """DC bias is finite for all six and a sane fraction (<25%) of the DC value."""
    spectra = analytic_spectra(GAMMA, GAMMA12)
    bias = dc_systematic(spectra, M, T, n_tb_per_period=1000, n_wfine=60001)
    for key in ALL_KEYS:
        assert np.isfinite(bias[key])
        s0 = float(np.real(spectra[key](np.array([0.0]))[0]))
        assert abs(bias[key]) < 0.25 * abs(s0), f"{key} DC bias {bias[key]:.0f} too large"


def test_dc_self_spectra_biased_positive_by_ising_leak():
    """Self-spectra DC bias is positive: the residual S_1212 leak through the partner
    CDD3 (positive) dominates the small negative FID short-time tail."""
    spectra = analytic_spectra(GAMMA, GAMMA12)
    bias = dc_systematic(spectra, M, T, n_tb_per_period=1000, n_wfine=60001)
    assert bias['S11'] > 0, "S_11 DC should be biased high (Ising leak)"
    assert bias['S22'] > 0, "S_22 DC should be biased high (Ising leak)"


def test_selfconsistent_runs_without_truth(systematic_result):
    """The self-consistent estimate (from a reconstructed comb) runs and is finite."""
    sys, _ = systematic_result
    spectra = analytic_spectra(GAMMA, GAMMA12)
    wk_full = np.concatenate(([0.0], [2 * np.pi * (j + 1) / T for j in range(TRUNC)]))
    # Use the analytic comb as a stand-in "reconstruction" to build the s.c. model.
    recon = {k: np.asarray(spectra[k](wk_full)) for k in ALL_KEYS}
    sc = selfconsistent_spectra(wk_full, recon)
    sys_sc = comb_inversion_systematic(sc, C_TIMES, M, T, n_wfine=20001, n_tb=6000)
    for key in ALL_KEYS:
        assert np.all(np.isfinite(sys_sc[key]))
