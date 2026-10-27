"""Tests for the SPAM injection / mitigation / robust-estimator machinery.

The deterministic tests run with ZERO noise (all-zero noise matrices): the
propagator is then the identity, every estimator is an exact function of the
injected SPAM parameters, and the paper's closed-form SPAM offsets can be
checked to floating-point accuracy.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest
import qutip as qt

from qns2q.characterize.spam import (confusion_matrix, estimate_spam,
                                     simulate_calibration,
                                     make_c_12_0_mt_robust,
                                     make_c_12_12_mt_robust,
                                     make_c_a_0_mt_robust)
from qns2q.characterize.inversion import regress_observables_over_M
from qns2q.characterize.experiments import QNSExperimentConfig
from qns2q.model.observables import (make_c_12_0_mt, make_c_12_12_mt,
                                     make_c_a_0_mt, make_c_a_b_mt)
from qns2q.model.trajectories import make_init_state, solver_prop

# Injected truth used throughout (visibility, asymmetry, z-SP, transverse-SP).
A_M = np.array([0.97, 0.95])
DELTA = np.array([0.03, -0.02])
A_SP = np.array([0.99, 0.98])
C_SP = np.array([0.02 + 0.04j, 0.03 - 0.02j])

N_SHOTS = 16            # noise-free tests: every shot identical
T_GRAIN = 64


def _zero_noise_mats(t_len, w_grain=8):
    """All-zero noise matrices -> identity propagator (deterministic SPAM tests)."""
    return jnp.zeros((3, 2, t_len, 2 * w_grain))


def _exp_kwargs(t_b, a_sp=A_SP, c=C_SP, m=1, **extra):
    kw = dict(n_shots=N_SHOTS, m=m, t_b=t_b, a_m=A_M, delta=DELTA,
              a_sp=a_sp, c=c, noise_mats=_zero_noise_mats(m * len(t_b)))
    kw.update(extra)
    return kw


@pytest.fixture
def t_b():
    return jnp.linspace(0, 4e-6, T_GRAIN)


class TestConfusionMatrix:
    def test_matches_config_cm(self):
        """spam.confusion_matrix must equal the CM built by QNSExperimentConfig."""
        a1, b1 = 1.00, 0.97
        a2, b2 = 0.965, 0.985
        config = QNSExperimentConfig(a1=a1, b1=b1, a2=a2, b2=b2,
                                     t_grain=8, truncate=2, w_grain=4)
        cm = confusion_matrix(config.a_m, config.delta)
        npt.assert_allclose(cm, np.asarray(config.CM), atol=1e-14)

    def test_columns_sum_to_one(self):
        cm = confusion_matrix(A_M, DELTA)
        npt.assert_allclose(cm.sum(axis=0), np.ones(4), atol=1e-14)


class TestCalibration:
    def test_delta_and_products_exact(self):
        delta_hat, products_hat = simulate_calibration(A_M, DELTA, A_SP, C_SP)
        npt.assert_allclose(delta_hat, DELTA, atol=1e-12)
        npt.assert_allclose(products_hat, A_M * A_SP, atol=1e-12)

    def test_estimate_faithful_split(self):
        est = estimate_spam(A_M, DELTA, A_SP, C_SP, split_error=0.0)
        npt.assert_allclose(est.a_m, A_M, atol=1e-12)
        npt.assert_allclose(est.a_sp, A_SP, atol=1e-12)
        npt.assert_allclose(est.cm, confusion_matrix(A_M, DELTA), atol=1e-12)

    def test_split_error_preserves_products(self):
        est = estimate_spam(A_M, DELTA, A_SP, C_SP, split_error=0.02)
        npt.assert_allclose(est.a_m * est.a_sp, A_M * A_SP, atol=1e-12)
        assert not np.allclose(est.a_m, A_M)


class TestWrungState:
    def test_pp_wrung_is_zz_conjugated_pp(self):
        rho_pp = make_init_state(A_SP, C_SP, state='pp')
        rho_wr = make_init_state(A_SP, C_SP, state='pp_wrung')
        zz = qt.tensor(qt.sigmaz(), qt.sigmaz())
        npt.assert_allclose((zz * rho_pp * zz.dag()).full(), rho_wr.full(),
                            atol=1e-14)

    def test_ideal_pp_wrung_is_xminus_xminus(self):
        rho_wr = make_init_state(np.array([1., 1.]),
                                 np.array([0. + 0j, 0. + 0j]), state='pp_wrung')
        xm = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
        target = qt.tensor(xm * xm.dag(), xm * xm.dag())
        npt.assert_allclose(rho_wr.full(), target.full(), atol=1e-14)


class TestRobustEstimatorsZeroNoise:
    """Zero noise -> coefficients are pure SPAM offsets; check the paper's
    closed forms. Bloch components of the transverse SP error: alpha_x = Re c,
    alpha_y = -Im c."""

    def test_c_12_0_robust_equals_spam_intercept(self, t_b):
        ay = -np.imag(C_SP)
        intercept = -0.5 * np.log((A_M[0] * A_M[1]) ** 2
                                  * (A_SP[0] ** 2 + ay[0] ** 2)
                                  * (A_SP[1] ** 2 + ay[1] ** 2))
        means, errs = make_c_12_0_mt_robust(
            solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
            None, False, state='pp', **_exp_kwargs(t_b))
        npt.assert_allclose(means[0], intercept, atol=1e-8)

    def test_c_12_12_robust_spam_free(self, t_b):
        """The SPAM intercept cancels exactly in C_12_12 = D+ - D-."""
        means, errs = make_c_12_12_mt_robust(
            solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
            None, False, state='pp', **_exp_kwargs(t_b))
        npt.assert_allclose(means[0], 0.0, atol=1e-8)

    def test_c_a_0_robust_equals_spam_intercept(self, t_b):
        ay = -np.imag(C_SP)
        for l in (1, 2):
            intercept = -0.5 * np.log(A_M[l - 1] ** 2
                                      * (A_SP[l - 1] ** 2 + ay[l - 1] ** 2))
            means, errs = make_c_a_0_mt_robust(
                solver_prop, ['CDD1', 'CDD1-1/2'], jnp.array(t_b),
                [t_b[-1] / 2], None, False, l=l, **_exp_kwargs(t_b))
            npt.assert_allclose(means[0], intercept, atol=1e-8,
                                err_msg=f"qubit {l}")

    def test_m_regression_removes_intercept(self, t_b):
        """Zero noise: C(M) = const(SPAM) -> regressed slope*m_ref must be 0."""
        obs_by_M = {}
        for m in (2, 4, 6):
            kw = _exp_kwargs(t_b, m=m)
            means, _ = make_c_12_0_mt_robust(
                solver_prop, ['CPMG', 'CPMG'],
                jnp.tile(jnp.array(t_b), m), [t_b[-1] / 2], None, False,
                state='pp', **kw)
            obs_by_M[m] = means
        c_eff, _ = regress_observables_over_M(obs_by_M, [2, 4, 6], 6)
        npt.assert_allclose(c_eff, 0.0, atol=1e-7)

    def test_robust_matches_legacy_when_no_spam(self, t_b):
        """Without SPAM the robust estimator must agree with the legacy one
        (both exactly zero at zero noise)."""
        ones = np.array([1., 1.])
        zeroc = np.array([0. + 0j, 0. + 0j])
        kw = _exp_kwargs(t_b, a_sp=ones, c=zeroc,
                         a_m=ones.copy(), delta=np.zeros(2))
        means_r, _ = make_c_12_0_mt_robust(
            solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
            None, False, state='pp', **kw)
        means_l, _ = make_c_12_0_mt(
            solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
            None, False, state='pp', **kw)
        npt.assert_allclose(means_r[0], means_l[0], atol=1e-8)
        npt.assert_allclose(means_r[0], 0.0, atol=1e-8)


class TestMitigatedEstimatorsZeroNoise:
    """The mitigated path (estimated CM inversion + estimated a_sp division +
    prep-level twirl) must return the ideal, SPAM-free coefficients."""

    def _kwargs_mitigated(self, t_b, est):
        # Mitigated wiring: prep with twirled transverse SP (c=0), divide by
        # the ESTIMATED a_sp, invert the ESTIMATED confusion matrix.
        return _exp_kwargs(t_b, a_sp=A_SP, c=np.array([0. + 0j, 0. + 0j]),
                           a_sp_div=est.a_sp)

    def test_two_qubit_coefficients_unbiased(self, t_b):
        est = estimate_spam(A_M, DELTA, A_SP, C_SP, split_error=0.0)
        for func in (make_c_12_0_mt, make_c_12_12_mt):
            means, _ = func(
                solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
                jnp.array(est.cm), True, state='pp',
                **self._kwargs_mitigated(t_b, est))
            npt.assert_allclose(means[0], 0.0, atol=1e-8,
                                err_msg=func.__name__)

    def test_single_qubit_coefficients_unbiased(self, t_b):
        est = estimate_spam(A_M, DELTA, A_SP, C_SP, split_error=0.0)
        for func, kw in ((make_c_a_0_mt, {'l': 1}), (make_c_a_b_mt, {'l': 1})):
            means, _ = func(
                solver_prop, ['CDD1', 'CDD1-1/2'], jnp.array(t_b),
                [t_b[-1] / 2], jnp.array(est.cm), True,
                **self._kwargs_mitigated(t_b, est), **kw)
            npt.assert_allclose(means[0], 0.0, atol=1e-8,
                                err_msg=func.__name__)

    def test_raw_estimators_are_biased(self, t_b):
        """Sanity: WITHOUT mitigation the same SPAM injection biases the
        coefficient away from zero -- the thing the protocols must fix."""
        means, _ = make_c_12_0_mt(
            solver_prop, ['CPMG', 'CPMG'], jnp.array(t_b), [t_b[-1] / 2],
            None, False, state='pp',
            **_exp_kwargs(t_b, c=np.array([0. + 0j, 0. + 0j])))
        assert abs(means[0]) > 1e-3


class TestConfigProtocolWiring:
    _common = dict(t_grain=8, truncate=2, w_grain=4,
                   a1=1.00, b1=0.97, a2=0.965, b2=0.985,
                   a_sp=jnp.array(A_SP), c=C_SP.copy())

    def test_mitigated_resolves_estimates(self):
        config = QNSExperimentConfig(spam_protocol='mitigated', **self._common)
        est = config.spam_estimate
        npt.assert_allclose(est.a_m, config.a_m, atol=1e-12)
        npt.assert_allclose(est.a_sp, np.asarray(config.a_sp), atol=1e-12)
        npt.assert_allclose(np.asarray(config.cm_use), est.cm, atol=1e-12)
        npt.assert_allclose(config.c_prep, 0.0, atol=1e-14)   # twirled
        assert config.spMit is True

    def test_robust_defaults_msweep(self):
        config = QNSExperimentConfig(spam_protocol='robust', M=10, **self._common)
        assert config.m_sweep_robust == (6, 8, 10)
        assert config.cm_use is None and config.spMit is False

    def test_raw_disables_mitigation(self):
        config = QNSExperimentConfig(spam_protocol='raw', **self._common)
        assert config.cm_use is None and config.spMit is False

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError):
            QNSExperimentConfig(spam_protocol='bogus', **self._common)
