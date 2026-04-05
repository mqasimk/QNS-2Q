"""Tests for spectral_inversion.py -- spectral reconstruction algorithms."""

import numpy as np
import numpy.testing as npt
import pytest

from spectral_inversion import ff, propagate_linear_error, f1_fid


class TestFilterFunction:
    """Tests for the filter function ff()."""

    def test_fid_at_zero_frequency(self):
        """FID (y=1) at w=0: integral of 1 over [0, T] = T."""
        T = 1e-6
        t = np.linspace(0, T, 10000)
        y = np.ones_like(t)
        result = ff(y, t, 0.0)
        npt.assert_allclose(np.real(result), T, rtol=1e-4)

    def test_returns_complex(self):
        """Filter function should return a complex value."""
        t = np.linspace(0, 1e-6, 1000)
        y = np.ones_like(t)
        result = ff(y, t, 1e7)
        assert np.iscomplex(result) or isinstance(result, (complex, np.complexfloating))

    def test_alternating_toggle_suppresses_dc(self):
        """A toggle function that alternates +-1 should have small ff at w=0."""
        t = np.linspace(0, 1e-6, 10000)
        # Simple alternating: +1 for first half, -1 for second half
        y = np.ones_like(t)
        y[len(t) // 2:] = -1.0
        result = ff(y, t, 0.0)
        # The integral of the alternating function at w=0 should be ~0
        npt.assert_allclose(np.abs(result), 0.0, atol=1e-10)

    def test_parseval_like_consistency(self):
        """The filter function magnitude should be bounded by T * max(|y|)."""
        T = 2e-6
        t = np.linspace(0, T, 5000)
        y = np.ones_like(t)
        w = 5e6
        result = ff(y, t, w)
        assert np.abs(result) <= T * 1.01  # small tolerance for numerical integration


class TestPropagateLinearError:
    """Tests for linear error propagation."""

    def test_identity_matrix(self):
        """Identity A_inv should pass through errors unchanged."""
        A_inv = np.eye(3)
        obs_err = np.array([0.1, 0.2, 0.3])
        result = propagate_linear_error(A_inv, obs_err)
        npt.assert_allclose(result, obs_err, atol=1e-12)

    def test_zero_errors(self):
        """Zero input errors should give zero output errors."""
        A_inv = np.array([[1, 2], [3, 4]], dtype=float)
        obs_err = np.array([0.0, 0.0])
        result = propagate_linear_error(A_inv, obs_err)
        npt.assert_allclose(result, 0.0, atol=1e-12)

    def test_scaling(self):
        """Scaling A_inv by factor k should scale errors by |k|."""
        A_inv = 2.0 * np.eye(3)
        obs_err = np.array([0.1, 0.2, 0.3])
        result = propagate_linear_error(A_inv, obs_err)
        npt.assert_allclose(result, 2.0 * obs_err, atol=1e-12)

    def test_output_shape(self):
        A_inv = np.random.rand(4, 4)
        obs_err = np.array([0.1, 0.2, 0.3, 0.4])
        result = propagate_linear_error(A_inv, obs_err)
        assert result.shape == (4,)

    def test_non_negative(self):
        """Propagated errors should always be non-negative."""
        A_inv = np.random.randn(5, 5)
        obs_err = np.abs(np.random.randn(5)) * 0.1
        result = propagate_linear_error(A_inv, obs_err)
        assert np.all(result >= 0)


class TestF1Fid:
    """Tests for the FID filter function f1_fid()."""

    def test_at_zero_frequency(self):
        """f1_fid(T, 0) should equal T."""
        T = 1e-6
        result = f1_fid(T, 0.0)
        npt.assert_allclose(np.real(result), T, rtol=1e-3)

    def test_returns_complex(self):
        result = f1_fid(1e-6, 1e7)
        assert isinstance(result, (complex, np.complexfloating, np.generic))

    def test_magnitude_bounded(self):
        """Magnitude should be bounded by T."""
        T = 2e-6
        result = f1_fid(T, 5e7)
        assert np.abs(result) <= T * 1.01
