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
