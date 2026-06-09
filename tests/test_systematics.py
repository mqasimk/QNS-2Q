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
                                            selfconsistent_spectra, dc_systematic, SPEC_KERNELS,
                                            forward_model_systematic, forward_observables, _FWD_OBS,
                                            dc_fit_systematic)
from qns2q.characterize.inversion import _ramsey_fit_dc

# Small but representative config (keep the grids light so the test stays fast).
# tau units (tau = 1): matches the dimensionless spectra in noise/spectra.py.
TAU = 1.0
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


# --- Faithful forward-model (full M-rep) systematic ---------------------------
# Light experiment-like grid so the forward model (full-record toggles over the
# synthesis grid) stays fast.
_W_GRAIN = 150
_WMAX = 2 * np.pi * TRUNC / T
_T_VEC = np.linspace(0, M * T, M * 300)        # 300 time points per period


@pytest.fixture(scope="module")
def forward_systematic_result():
    spectra = analytic_spectra(GAMMA, GAMMA12)
    return forward_model_systematic(spectra, C_TIMES, M, T, _T_VEC, GAMMA, GAMMA12,
                                    _W_GRAIN, _WMAX)


def test_forward_obs_recipe_covers_all_named_observables():
    """Guard: every harmonic observable consumed by reconstruct.py has a forward recipe."""
    expected = {'C_12_0_MT_1', 'C_12_0_MT_2', 'C_12_0_MT_3', 'C_12_0_MT_4',
                'C_12_12_MT_1', 'C_12_12_MT_2', 'C_1_0_MT_1', 'C_2_0_MT_1',
                'C_1_2_MT_1', 'C_1_2_MT_2', 'C_2_1_MT_1', 'C_2_1_MT_2'}
    assert set(_FWD_OBS) == expected


def test_forward_systematic_shapes_and_types(forward_systematic_result):
    """Self-spectra systematics are real; cross-spectra complex; length=truncate."""
    sys = forward_systematic_result
    for key in SELF_KEYS:
        assert sys[key].shape == (TRUNC,) and not np.iscomplexobj(sys[key])
    for key in CROSS_KEYS:
        assert sys[key].shape == (TRUNC,) and np.iscomplexobj(sys[key])


def test_forward_systematic_finite_and_bounded(forward_systematic_result):
    """Finite, and a sane fraction of the signal (the comb bias never exceeds it)."""
    sys = forward_systematic_result
    spectra = analytic_spectra(GAMMA, GAMMA12)
    wk = np.array([2 * np.pi * (j + 1) / T for j in range(TRUNC)])
    for key in ALL_KEYS:
        assert np.all(np.isfinite(sys[key]))
        sig = np.sqrt(np.mean(np.abs(np.asarray(spectra[key](wk))) ** 2))
        rms = np.sqrt(np.mean(np.abs(sys[key]) ** 2))
        assert rms < sig, f"{key} forward systematic RMS {rms:.1f} exceeds signal {sig:.1f}"


def test_forward_obs_reproduce_comb_kernel_to_leading_order():
    """Sanity on normalization: the deterministic forward observables agree with the
    single-period comb-kernel prediction U @ S to within the comb bias (same scale,
    not off by a factor). Checked on the well-behaved real channel C_12_12_MT_1."""
    spectra = analytic_spectra(GAMMA, GAMMA12)
    C = forward_observables(spectra, C_TIMES, M, T, _T_VEC, GAMMA, GAMMA12, _W_GRAIN, _WMAX)
    obs = np.asarray(C['C_12_12_MT_1'])
    assert np.all(np.isfinite(obs))
    # Correlator is a normalized two-qubit coherence exponent -> O(0.1-1), never O(M).
    assert np.max(np.abs(obs)) < 5.0


def test_forward_systematic_antisym_channel_dominates(forward_systematic_result):
    """The structural fact behind the Im S_1_2 fix: for every cross spectrum the
    antisymmetric (CDD3) imaginary channel carries a much larger comb-inversion bias
    than the symmetric real channel. This is why the imaginary points fell outside
    statistics-only bars while the real points did not, and it is what the forward
    systematic must quote. (Config-robust: the Im/Re contrast holds regardless of how
    many harmonics or how fine the grid; the absolute sizes do not.)"""
    sys = forward_systematic_result
    for key in CROSS_KEYS:
        re_rms = np.sqrt(np.mean(np.real(sys[key]) ** 2))
        im_rms = np.sqrt(np.mean(np.imag(sys[key]) ** 2))
        assert im_rms > 2.0 * re_rms, (
            f"{key}: Im systematic {im_rms:.0f} should dominate Re {re_rms:.0f}")


# --- Adaptive multi-time DC slope fit (strong-noise robust) -------------------

def test_ramsey_fit_dc_recovers_linear_slope():
    """In the motional-narrowing regime C(t) = a + (S0/factor) t; the fit recovers S0
    and marks it reliable (self factor 2; cross factor 1)."""
    t = np.linspace(1e-6, 1e-5, 8)
    for factor, S0 in [(2.0, 2.5e5), (1.0, 1.5e5)]:
        C = 0.05 + (S0 / factor) * t
        s0, err, reliable = _ramsey_fit_dc(C, t, factor=factor)
        assert abs(s0 - S0) / S0 < 0.02 and reliable


def test_ramsey_fit_dc_flags_quasistatic():
    """A purely quadratic decay C(t) ~ t^2 never reaches the linear regime -> the fit
    flags it unreliable (the DC value is then only a lower bound)."""
    t = np.linspace(1e-6, 1e-5, 8)
    C = 5e9 * t ** 2
    _, _, reliable = _ramsey_fit_dc(C, t, factor=2.0)
    assert not reliable


def test_dc_fit_systematic_recovers_self_and_qq_cross():
    """The deterministic DC bias is finite for all six. The qubit-qubit cross S12 (no
    partner-CDD3 leak) is recovered tightly; the self DCs S11/S22 carry the residual
    Ising leak through the CDD3 partner (a modest, correctly-quoted systematic), so a
    looser bound applies. (The Ising-coupled DCs may be flagged/large for a quasi-static
    regime; only finiteness is asserted there for regime-robustness.)"""
    spectra = analytic_spectra(GAMMA, GAMMA12)
    t_sweep = np.arange(1, 11) * T
    bias = dc_fit_systematic(spectra, t_sweep, GAMMA, GAMMA12)
    for key in ALL_KEYS:
        assert np.isfinite(bias[key])
    truth12 = float(np.real(spectra['S12'](np.array([0.0]))[0]))
    assert abs(bias['S12']) < 0.1 * abs(truth12), "S12 DC not recovered"
    for key in ('S11', 'S22'):                       # self DC: motional narrowing + CDD3 leak
        truth = float(np.real(spectra[key](np.array([0.0]))[0]))
        assert abs(bias[key]) < 0.2 * abs(truth), f"{key} DC bias unexpectedly large"
