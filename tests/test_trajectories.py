"""Tests for trajectories.py -- pulse sequences and quantum evolution."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import qutip as qt
import pytest

from trajectories import (
    f, cpmg, cdd1, cdd3, make_y, custom_y,
    make_init_state, make_noise_mat, make_noise_traj,
    make_Hamiltonian, make_propagator, sinM, cosM,
)
from spectra_input import S_11


class TestToggleFunction:
    """Tests for the toggle function f()."""

    def test_single_interval(self):
        """f(t, [0, T]) should be +1 everywhere."""
        t = jnp.linspace(0, 1.0, 100)
        tk = [0.0, 1.0]
        vals = f(t, tk)
        npt.assert_allclose(np.array(vals), 1.0, atol=1e-10)

    def test_two_intervals(self):
        """f(t, [0, 0.5, 1]) should be +1 then -1."""
        t = jnp.linspace(0, 1.0, 1000)
        tk = [0.0, 0.5, 1.0]
        vals = f(t, tk)
        # First half should be +1, second half -1
        mid = 500
        assert float(jnp.mean(vals[:mid - 10])) > 0.9
        assert float(jnp.mean(vals[mid + 10:])) < -0.9

    def test_values_are_plus_minus_one(self):
        """Toggle function should only take values +1 or -1 (ignoring boundaries)."""
        t = jnp.linspace(0, 1.0, 10000)
        tk = [0.0, 0.25, 0.5, 0.75, 1.0]
        vals = f(t, tk)
        # Interior points should be close to +/-1
        interior = vals[10:-10]
        npt.assert_allclose(np.abs(np.array(interior)), 1.0, atol=0.1)


class TestCPMG:
    """Tests for the CPMG pulse sequence."""

    def test_output_shape(self, time_vector):
        t = jnp.array(time_vector)
        vals = cpmg(t, 4)
        assert vals.shape == t.shape

    def test_starts_positive(self, time_vector):
        t = jnp.array(time_vector)
        vals = cpmg(t, 2)
        assert float(vals[0]) > 0

    def test_n_equals_1(self):
        """CPMG with n=1 should have 2 sign changes."""
        t = jnp.linspace(0, 1.0, 10000)
        vals = cpmg(t, 1)
        sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(vals))) > 1)
        assert int(sign_changes) == 2


class TestCDD1:
    """Tests for the CDD1 pulse sequence."""

    def test_output_shape(self, time_vector):
        t = jnp.array(time_vector)
        vals = cdd1(t, 4)
        assert vals.shape == t.shape

    def test_starts_positive(self, time_vector):
        t = jnp.array(time_vector)
        vals = cdd1(t, 2)
        assert float(vals[0]) > 0


class TestCDD3:
    """Tests for the CDD3 pulse sequence."""

    def test_output_shape(self, time_vector):
        t = jnp.array(time_vector)
        vals = cdd3(t, 1)
        assert vals.shape == t.shape

    def test_m_equals_1(self):
        """CDD3 with m=1 is a primitive cycle."""
        t = jnp.linspace(0, 1.0, 10000)
        vals = cdd3(t, 1)
        assert vals.shape == t.shape
        # Should have sign changes
        sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(vals))) > 1)
        assert int(sign_changes) > 0


class TestMakeY:
    """Tests for the control matrix construction make_y()."""

    def test_shape(self, small_time_vector):
        t = small_time_vector
        T = t[-1]
        y = make_y(t, ['CPMG', 'CDD1'], ctime=T, m=1)
        assert y.shape == (3, 3, len(t))

    def test_shape_with_repetitions(self, small_time_vector):
        t = small_time_vector
        T = t[-1]
        M = 3
        y = make_y(t, ['CPMG', 'CDD1'], ctime=T, m=M)
        assert y.shape == (3, 3, len(t) * M)

    def test_ising_is_product(self, small_time_vector):
        """y[2,2] should equal y[0,0] * y[1,1]."""
        t = small_time_vector
        T = t[-1]
        y = make_y(t, ['CPMG', 'CDD1'], ctime=T, m=1)
        npt.assert_allclose(y[2, 2], y[0, 0] * y[1, 1], atol=1e-12)

    def test_fid_is_ones(self, small_time_vector):
        """FID pulse should give all ones."""
        t = small_time_vector
        T = t[-1]
        y = make_y(t, ['FID', 'FID'], ctime=T, m=1)
        npt.assert_allclose(y[0, 0], 1.0, atol=1e-12)
        npt.assert_allclose(y[1, 1], 1.0, atol=1e-12)
        npt.assert_allclose(y[2, 2], 1.0, atol=1e-12)

    def test_invalid_pulse_raises(self, small_time_vector):
        t = small_time_vector
        T = t[-1]
        with pytest.raises(ValueError):
            make_y(t, ['INVALID', 'CPMG'], ctime=T, m=1)


class TestCustomY:
    """Tests for custom_y()."""

    def test_shape(self):
        t = jnp.linspace(0, 1.0, 100)
        vt = [jnp.array([0.0, 0.5, 1.0]), jnp.array([0.0, 0.25, 0.75, 1.0])]
        y = custom_y(vt, t, M=1)
        assert y.shape == (3, 3, 100)

    def test_ising_is_product(self):
        t = jnp.linspace(0, 1.0, 100)
        vt = [jnp.array([0.0, 0.3, 0.7, 1.0]), jnp.array([0.0, 0.5, 1.0])]
        y = custom_y(vt, t, M=1)
        npt.assert_allclose(np.array(y[2, 2]), np.array(y[0, 0] * y[1, 1]), atol=1e-10)


class TestMakeInitState:
    """Tests for initial state generation."""

    @pytest.mark.parametrize("state", ['p0', 'p1', '0p', '1p', 'pp'])
    def test_valid_density_matrix(self, state):
        """Initial state should be a valid density matrix (trace=1, Hermitian)."""
        a_sp = np.array([1.0, 1.0])
        c = np.array([0.0, 0.0])
        rho = make_init_state(a_sp, c, state=state)
        # Check it's a QuTiP Qobj
        assert isinstance(rho, qt.Qobj)
        # Trace should be 1
        npt.assert_allclose(float(np.real(rho.tr())), 1.0, atol=1e-12)
        # Hermitian: rho == rho.dag()
        diff = (rho - rho.dag()).norm()
        assert float(diff) < 1e-12

    def test_invalid_state_raises(self):
        with pytest.raises(Exception):
            make_init_state(np.array([1., 1.]), np.array([0., 0.]), state='invalid')

    def test_dimension(self):
        """Initial state should be a 4x4 density matrix (2 qubits)."""
        rho = make_init_state(np.array([1., 1.]), np.array([0., 0.]), state='pp')
        assert rho.shape == (4, 4)


class TestNoiseMat:
    """Tests for noise matrix generation."""

    def test_make_noise_mat_shapes(self):
        t = jnp.linspace(0, 1e-6, 50)
        S_mat, C_mat = make_noise_mat(S_11, t, w_grain=100, wmax=1e8, gamma=0.0)
        assert S_mat.shape[0] == len(t)  # time dimension
        assert C_mat.shape[0] == len(t)
        assert S_mat.shape[1] == 200  # 2 * w_grain
        assert C_mat.shape[1] == 200

    def test_make_noise_traj_shape(self):
        t = jnp.linspace(0, 1e-6, 50)
        S_mat, C_mat = make_noise_mat(S_11, t, w_grain=100, wmax=1e8, gamma=0.0)
        key = jnp.array([42, 99])
        traj = make_noise_traj(S_mat, C_mat, key)
        assert traj.shape == (len(t),)


class TestHamiltonian:
    """Tests for Hamiltonian construction."""

    def test_shape(self):
        n_t = 50
        y_uv = jnp.zeros((3, 3, n_t))
        y_uv = y_uv.at[0, 0].set(1.0)
        y_uv = y_uv.at[1, 1].set(1.0)
        y_uv = y_uv.at[2, 2].set(1.0)
        b_t = jnp.ones((3, n_t)) * 0.01
        H = make_Hamiltonian(y_uv, b_t)
        assert H.shape == (n_t, 8, 8)

    def test_hermitian(self):
        """Hamiltonian should be Hermitian at each time step."""
        n_t = 20
        y_uv = jnp.zeros((3, 3, n_t))
        y_uv = y_uv.at[0, 0].set(1.0)
        y_uv = y_uv.at[1, 1].set(-1.0)
        y_uv = y_uv.at[2, 2].set(-1.0)
        b_t = jnp.ones((3, n_t)) * 0.05
        H = make_Hamiltonian(y_uv, b_t)
        for i in range(n_t):
            npt.assert_allclose(
                np.array(H[i]), np.array(jnp.conj(H[i].T)), atol=1e-12
            )

    def test_diagonal(self):
        """Dephasing Hamiltonian should be diagonal."""
        n_t = 10
        y_uv = jnp.ones((3, 3, n_t))
        b_t = jnp.ones((3, n_t)) * 0.1
        H = make_Hamiltonian(y_uv, b_t)
        for i in range(n_t):
            off_diag = H[i] - jnp.diag(jnp.diag(H[i]))
            npt.assert_allclose(np.array(off_diag), 0.0, atol=1e-12)


class TestPropagator:
    """Tests for the time-evolution propagator."""

    def test_shape(self):
        n_t = 100
        t_vec = jnp.linspace(0, 1e-6, n_t)
        H = jnp.zeros((n_t, 8, 8))
        # Small diagonal Hamiltonian
        for i in range(8):
            H = H.at[:, i, i].set(0.01 * (i + 1))
        U = make_propagator(H, t_vec)
        assert U.shape == (8, 8)

    def test_unitarity(self):
        """Propagator should be unitary: U^dag U = I."""
        n_t = 200
        t_vec = jnp.linspace(0, 1e-6, n_t)
        H = jnp.zeros((n_t, 8, 8))
        for i in range(8):
            H = H.at[:, i, i].set(1e6 * (i + 1))
        U = make_propagator(H, t_vec)
        product = jnp.matmul(jnp.conj(U.T), U)
        npt.assert_allclose(np.array(product), np.eye(8), atol=1e-10)

    def test_zero_hamiltonian_gives_identity(self):
        """Zero Hamiltonian should give identity propagator."""
        n_t = 50
        t_vec = jnp.linspace(0, 1e-6, n_t)
        H = jnp.zeros((n_t, 8, 8))
        U = make_propagator(H, t_vec)
        npt.assert_allclose(np.array(U), np.eye(8), atol=1e-12)
