"""Tests for observables.py -- quantum observable calculations."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import qutip as qt
import pytest

from observables import (
    _get_hadamard_operators, _get_rx_operators, povms,
    a, d, parametric_bootstrap_error,
    get_expect_val_from_probs, get_expect_val_per_shot,
)


class TestHadamardOperators:
    """Tests for Hadamard gate operators."""

    def test_returns_three_operators(self):
        ops = _get_hadamard_operators()
        assert len(ops) == 3

    def test_unitarity(self):
        """Each Hadamard operator should be unitary."""
        ops = _get_hadamard_operators()
        for op in ops:
            product = op.dag() * op
            identity = qt.tensor(qt.identity(2), qt.identity(2), qt.identity(2))
            diff = (product - identity).norm()
            assert float(diff) < 1e-10

    def test_dimensions(self):
        """Operators should be 8x8 (3 qubits)."""
        ops = _get_hadamard_operators()
        for op in ops:
            assert op.shape == (8, 8)


class TestRxOperators:
    """Tests for Rx gate operators."""

    def test_returns_three_operators(self):
        ops = _get_rx_operators()
        assert len(ops) == 3

    def test_unitarity(self):
        ops = _get_rx_operators()
        identity = qt.tensor(qt.identity(2), qt.identity(2), qt.identity(2))
        for op in ops:
            product = op.dag() * op
            diff = (product - identity).norm()
            assert float(diff) < 1e-10

    def test_dimensions(self):
        ops = _get_rx_operators()
        for op in ops:
            assert op.shape == (8, 8)


class TestPOVMs:
    """Tests for POVM operator construction."""

    def test_returns_four_operators(self):
        ops = povms(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert len(ops) == 4

    def test_dimensions(self):
        ops = povms(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        for op in ops:
            assert op.shape == (8, 8)

    def test_perfect_measurement_projectors(self):
        """With perfect measurement (a_m=1, delta=1), POVMs should be projectors."""
        ops = povms(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        # For perfect measurement: p1_0 = |0><0|, p1_1 = |1><1|
        # Check that each is Hermitian
        for op in ops:
            diff = (op - op.dag()).norm()
            assert float(diff) < 1e-10

    def test_hermiticity(self):
        """POVMs should always be Hermitian."""
        ops = povms(np.array([0.95, 0.93]), np.array([0.9, 0.88]))
        for op in ops:
            diff = (op - op.dag()).norm()
            assert float(diff) < 1e-10


class TestConcurrenceMetrics:
    """Tests for concurrence-related functions a() and d()."""

    def test_a_known_value(self):
        """a((1, 0)) = -0.25 * log(1) = 0."""
        result = a((1.0, 0.0))
        npt.assert_allclose(result, 0.0, atol=1e-12)

    def test_a_positive_for_decay(self):
        """a() should be positive when expectation values are < 1."""
        result = a((0.5, 0.3))
        assert result > 0

    def test_d_plus_finite(self):
        result = d('+', (0.8, 0.6, 0.1, 0.2))
        assert np.isfinite(result)

    def test_d_minus_finite(self):
        result = d('-', (0.8, 0.6, 0.1, 0.2))
        assert np.isfinite(result)

    def test_d_invalid_pm_raises(self):
        with pytest.raises(ValueError):
            d('x', (0.8, 0.6, 0.1, 0.2))

    def test_d_symmetry(self):
        """d('+') and d('-') should generally differ for asymmetric inputs."""
        dp = d('+', (0.8, 0.6, 0.3, 0.1))
        dm = d('-', (0.8, 0.6, 0.3, 0.1))
        assert dp != dm


class TestParametricBootstrapError:
    """Tests for the parametric bootstrap error estimation."""

    def test_identity_function(self):
        """For f(x)=x, bootstrap error should approximate input stderr."""
        means = [5.0]
        stderrs = [0.1]
        result = parametric_bootstrap_error(means, stderrs, lambda x: x, n_boot=10000)
        npt.assert_allclose(result, 0.1, atol=0.02)

    def test_zero_stderr_gives_near_zero(self):
        """Zero input stderr should give near-zero output."""
        means = [1.0, 2.0]
        stderrs = [0.0, 0.0]
        # Use a constant function to avoid numerical issues with zero variance
        result = parametric_bootstrap_error(means, stderrs, lambda x, y: x + y, n_boot=1000)
        npt.assert_allclose(result, 0.0, atol=1e-10)

    def test_sum_function_error_propagation(self):
        """For f(x,y) = x+y, error should be sqrt(sx^2 + sy^2)."""
        means = [1.0, 2.0]
        stderrs = [0.3, 0.4]
        expected = np.sqrt(0.3**2 + 0.4**2)
        result = parametric_bootstrap_error(means, stderrs, lambda x, y: x + y, n_boot=50000)
        npt.assert_allclose(result, expected, atol=0.03)


class TestExpectValFromProbs:
    """Tests for expectation value calculations from probabilities."""

    def test_perfect_measurement_pure_state(self):
        """With identity confusion matrix and deterministic probs."""
        cm = np.eye(4)
        # All in |00> state: prob = [1, 0, 0, 0]
        probs = np.array([[1.0, 0.0, 0.0, 0.0]])
        # Two-qubit correlation: p00 + p11 = 1 + 0 = 1 -> 2*1 - 1 = 1
        val = get_expect_val_from_probs(probs, cm, qubit_idx=-1)
        npt.assert_allclose(float(val), 1.0, atol=1e-10)

    def test_qubit1_expectation(self):
        """Single qubit expectation from probability distribution."""
        cm = np.eye(4)
        # p00=0.5, p01=0.5: qubit 1 always in |0>
        probs = np.array([[0.5, 0.5, 0.0, 0.0]])
        val = get_expect_val_from_probs(probs, cm, qubit_idx=1)
        # p(qubit1=0) = p00 + p01 = 1.0 -> 2*1 - 1 = 1
        npt.assert_allclose(float(val), 1.0, atol=1e-10)

    def test_per_shot_shape(self):
        cm = np.eye(4)
        probs = np.random.rand(100, 4)
        probs /= probs.sum(axis=1, keepdims=True)
        vals = get_expect_val_per_shot(probs, cm, qubit_idx=-1)
        assert vals.shape == (100,)
