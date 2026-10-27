"""Nonzero-noise equivalence of the bounded-memory standalone simulation."""
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest
import subprocess
import sys

from qns2q.characterize.single_qubit import (
    QNSExperimentConfig, StreamedNoise, streamed_solver,
)
from qns2q.model.trajectories import (
    make_noise_mat_arr, make_y, make_init_state, solver_prop,
)


@pytest.fixture
def precision():
    return True  # tests/conftest.py enables x64 before importing the model


@pytest.mark.parametrize('m,t_grain,w_grain', [(3, 65, 9), (19, 1500, 8)])
def test_streamed_filters_match_dense(m, t_grain, w_grain, precision):
    config = QNSExperimentConfig(M=m, t_grain=t_grain, w_grain=w_grain)
    y = jnp.asarray(make_y(config.t_b, ['CDD1', 'CDD1-1/2'],
                          ctime=config.T / 3, m=m))
    dense = make_noise_mat_arr('make', t_vec=config.t_vec,
                              w_grain=w_grain, wmax=config.wmax,
                              truncate=config.truncate)
    reference = jnp.stack([
        jax.scipy.integrate.trapezoid(.5 * y[a, a][None, None, :, None] * dense,
                                     config.t_vec, axis=2) for a in range(3)])
    streamed = StreamedNoise(config, chunk_size=128)
    npt.assert_allclose(streamed.filters(y), reference,
                        rtol=2e-11 if precision else 3e-5,
                        atol=1e-13 if precision else 1e-7)
    assert streamed.filters(y) is streamed.cached_filters


def test_same_seed_nonzero_noise_states_match_dense(precision):
    config = QNSExperimentConfig(M=3, t_grain=65, w_grain=9)
    y = jnp.asarray(make_y(config.t_b, ['CDD1', 'CDD1-1/2'],
                          ctime=config.T / 2, m=config.M))
    dense = make_noise_mat_arr('make', t_vec=config.t_vec,
                              w_grain=config.w_grain, wmax=config.wmax,
                              truncate=config.truncate)
    rho = jnp.kron(jnp.asarray(make_init_state(config.a_sp, config.c,
                                             state='p0').full()), jnp.eye(2)/2)
    np.random.seed(793)
    expected = solver_prop(y, dense, config.t_vec, rho, 19)
    expected_next_keys = np.random.randint(0, 10000, (3, 2))
    np.random.seed(793)
    actual = streamed_solver(y, StreamedNoise(config, chunk_size=31, shot_batch=7),
                             config.t_vec, rho, 19)
    actual_states = actual.u[:, :, None] * actual.rho * actual.u.conj()[:, None, :]
    npt.assert_allclose(actual_states, expected,
                        atol=1e-12 if precision else 2e-7,
                        rtol=1e-12 if precision else 2e-7)
    npt.assert_array_equal(np.random.randint(0, 10000, (3, 2)), expected_next_keys)


def test_original_float32_precision_in_fresh_process():
    # Spectra retain arrays created at import time. Switching precision after
    # importing the model creates mixed JAX dtypes; the real CLI starts fresh.
    code = f'''
import jax, runpy
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")
t = runpy.run_path({__file__!r})
for m, nt, nw in [(3, 65, 9), (19, 1500, 8)]:
    t["test_streamed_filters_match_dense"](m, nt, nw, False)
t["test_same_seed_nonzero_noise_states_match_dense"](False)
'''
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def _exact_single_qubit_estimate(m, a_m=(.97, .95), delta=(.01, -.02),
                                  a_sp=(.99, .98), c=(.02+.03j, -.02j),
                                  noise_slope=.004):
    from qns2q.characterize.single_qubit import make_single_qubit_robust
    config = QNSExperimentConfig(M=m, t_grain=65, truncate=1, n_shots=16)

    def gaussian_ensemble_solver(y, noise_mats, t_vec, rho, n_shots):
        # Independent exact Gaussian channel, not a fitted/sample reference.
        npt.assert_array_equal(y[1:], 0)
        assert np.any(np.asarray(y[0, 0]) != 0)
        z = jnp.repeat(jnp.array([1., -1.]), 4)
        variance = noise_slope * m / 2
        averaged = rho * jnp.exp(-.5 * variance * (z[:, None]-z[None, :])**2)
        return jnp.broadcast_to(averaged, (n_shots, 8, 8))

    return make_single_qubit_robust(
        gaussian_ensemble_solver, ['CPMG', 'FID'], config.t_vec, [config.T],
        object(), False,  # a non-matrix sentinel proves CM is not used
        t_b=config.t_b, m=m, n_shots=16, noise_mats=None, l=1,
        a_m=np.array(a_m), delta=np.array(delta),
        a_sp=np.array(a_sp), c=np.array(c))[0][0]


def test_raw_single_qubit_offset_and_free_slope_recovery():
    from qns2q.characterize.single_qubit import fit_repetition_series
    m = np.array([5, 10, 19])
    intercept = -.5*np.log(.97**2*(.99**2+.03**2))
    values = np.array([_exact_single_qubit_estimate(x) for x in m])
    npt.assert_allclose(values, intercept+.004*m, atol=2e-12)
    fit, _ = fit_repetition_series(m, values[:, None], np.full((3, 1), 1e-5))
    npt.assert_allclose(fit[0], [.004, intercept], atol=2e-12)
    npt.assert_allclose(_exact_single_qubit_estimate(5, noise_slope=0),
                        intercept, atol=2e-12)


def test_twisting_cancels_asymmetry_without_removing_visibility():
    base = _exact_single_qubit_estimate(10)
    changed_asymmetry = _exact_single_qubit_estimate(10, delta=(-.02, .03))
    changed_visibility = _exact_single_qubit_estimate(10, a_m=(.8, .95))
    npt.assert_allclose(changed_asymmetry, base, atol=2e-12)
    npt.assert_allclose(changed_visibility-base, -np.log(.8/.97), atol=2e-12)


def test_uncoupled_spectator_does_not_change_single_qubit_estimate():
    base = _exact_single_qubit_estimate(10)
    changed = _exact_single_qubit_estimate(
        10, a_m=(.97, .6), delta=(.01, -.13),
        a_sp=(.99, .5), c=(.02+.03j, .1+.2j))
    npt.assert_allclose(changed, base, atol=2e-12)


def test_single_qubit_noise_reference_matches_direct_S11_filter():
    from qns2q.characterize.single_qubit import (
        single_qubit_controls, single_qubit_model_coefficient)
    from qns2q.model.trajectories import make_noise_mat
    from qns2q.noise.spectra import S_11
    config = QNSExperimentConfig(M=3, t_grain=65, w_grain=9)
    y = single_qubit_controls(config.t_b, config.T/2, config.M)
    npt.assert_array_equal(y[1:], 0)
    expected_y = make_y(config.t_b, ['CPMG', 'FID'], ctime=config.T/2, m=3)
    npt.assert_array_equal(y[0, 0], expected_y[0, 0])
    filters = StreamedNoise(config, chunk_size=32).filters(y)
    direct = jnp.asarray(make_noise_mat(S_11, config.t_vec,
                          w_grain=config.w_grain, wmax=config.wmax, gamma=0.))
    f = jax.scipy.integrate.trapezoid(
        .5*y[0, 0][None, :, None]*direct, config.t_vec, axis=1)
    npt.assert_allclose(single_qubit_model_coefficient(filters),
                        2*np.sum(np.asarray(f)**2), rtol=1e-12, atol=1e-14)
