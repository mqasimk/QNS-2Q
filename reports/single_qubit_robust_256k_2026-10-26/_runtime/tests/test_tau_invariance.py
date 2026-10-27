"""Scale invariance of the QNS pipeline under a change of the time unit.

The physics depends only on dimensionless products (omega*tau, S*tau, J*tau, ...):
rescaling every time by k (tau -> k*tau, so T, c_times, gamma, t_vec all scale)
while substituting the rescaled spectrum

    S'(omega) = S(k*omega) / k

leaves every measured observable EXACTLY invariant -- the noise-synthesis matrices
satisfy b'(k t) = b(t)/k for the same random draw, so the accumulated dephasing
phases are equal to floating-point accuracy.

This is the enabling fact for moving the codebase to tau-anchored (absolute-time
agnostic) units with the minimum pulse separation tau as the unit of time: any
consistent re-anchoring of the absolute scales reproduces the same dimensionless
results, so the SI anchors in spectra.py / Jmax / plot conversions are pure
bookkeeping, not physics.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import qutip as qt
import pytest

from qns2q.model.trajectories import (make_noise_mat_arr, make_y, make_init_state,
                                      solver_prop)
from qns2q.model.observables import e_xx_hat_with_stderr, e_x_hat_with_stderr
from qns2q.noise.spectra import S_el_A, S_el_B, S_nuc_1, S_nuc_2, DT_SHIFT

SEED = 1234
K = 10.0               # time-unit rescaling factor
N_SHOTS = 32
T_GRAIN = 400
W_GRAIN = 64
TRUNCATE = 4
M = 2


def _run_arm(tau, components, dt_shift):
    """One miniature QNS measurement: returns E[X1X2] and E[X1] after a CPMG/CPMG
    block sequence under synthesized noise with a fixed RNG seed."""
    T = 16 * tau
    wmax = 2 * np.pi * TRUNCATE / T
    t_b = jnp.linspace(0, T, T_GRAIN)
    t_vec = jnp.linspace(0, M * T, M * T_GRAIN)

    noise_mats = jnp.array(make_noise_mat_arr(
        'make', t_vec=t_vec, w_grain=W_GRAIN, wmax=wmax,
        truncate=TRUNCATE, midpoint=True,
        components=components, dt_shift=dt_shift))

    y_uv = jnp.array(make_y(t_b, ['CPMG', 'CPMG'], ctime=T / 2, m=M))
    rho0 = make_init_state(np.array([1., 1.]), np.array([0. + 0j, 0. + 0j]),
                           state='pp')
    rho = jnp.array((qt.tensor(rho0, 0.5 * qt.identity(2))).full())

    np.random.seed(SEED)
    sol = solver_prop(y_uv, noise_mats, t_vec, rho, N_SHOTS)
    a_m, delta = np.array([1., 1.]), np.array([0., 0.])
    exx, _ = e_xx_hat_with_stderr(sol, a_m, delta, None)
    ex1, _ = e_x_hat_with_stderr(1, sol, a_m, delta, None)
    return exx, ex1


def test_observables_invariant_under_time_rescaling():
    tau1 = 2.5e-8
    comps = (S_el_A, S_el_B, S_nuc_1, S_nuc_2)
    exx1, ex11 = _run_arm(tau1, comps, DT_SHIFT * tau1 / 2.5e-8)

    # tau -> K*tau with consistently rescaled components S'(w) = S(K*w)/K and a
    # rescaled cross-spectrum lag dt' = K*dt (the mixing constants C2_SHARE, A_J,
    # B_J are dimensionless and invariant).
    comps_scaled = tuple(lambda w, f=f: f(K * w) / K for f in comps)
    exx2, ex12 = _run_arm(K * tau1, comps_scaled, K * DT_SHIFT * tau1 / 2.5e-8)

    npt.assert_allclose(exx2, exx1, rtol=1e-10)
    npt.assert_allclose(ex12, ex11, rtol=1e-10)
