"""Tests for spectra_input.py -- noise power spectral density definitions."""

import jax.numpy as jnp
import numpy as np
import pytest

from qns2q.noise.spectra import L, Gauss, S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12


class TestLorentzian:
    """Tests for the symmetric Lorentzian function L()."""

    def test_peak_near_w0(self):
        """Lorentzian should have local maximum near w0 (sharp peak, mirror far away)."""
        w0 = 1e8  # large w0 so mirror at -w0 is far away
        tc = 1e-6  # sharp peak (narrow width ~ 1/tc = 1e6 << w0)
        w = jnp.linspace(w0 - 5e6, w0 + 5e6, 10001)
        vals = L(w, w0, tc)
        peak_idx = int(jnp.argmax(vals))
        dw = float(w[1] - w[0])
        assert abs(float(w[peak_idx]) - w0) < 2 * dw

    def test_symmetry_centered(self):
        """L(w, 0, tc) should be even: L(w) == L(-w)."""
        w = jnp.linspace(-1e8, 1e8, 201)
        vals = L(w, 0.0, 1e-7)
        np.testing.assert_allclose(np.array(vals), np.array(vals[::-1]), atol=1e-12)

    def test_non_negative(self, frequency_grid):
        vals = L(frequency_grid, 5e6, 3e-7)
        assert jnp.all(vals >= 0)

    def test_value_at_zero_frequency_zero_w0(self):
        """L(0, 0, tc) should equal 1.0 for any tc."""
        val = L(jnp.float64(0.0), 0.0, 1e-7)
        np.testing.assert_allclose(float(val), 1.0, atol=1e-12)

    def test_width_scales_with_tc(self):
        """Larger tc should give a narrower peak."""
        w = jnp.float64(1e6)
        val_narrow = float(L(w, 0.0, 1e-5))  # large tc = narrow
        val_wide = float(L(w, 0.0, 1e-8))    # small tc = wide
        assert val_wide > val_narrow


class TestGaussian:
    """Tests for the non-normalized Gaussian function Gauss()."""

    def test_peak_at_w0(self):
        w0 = 2e7
        sig = 3e6
        w = jnp.linspace(0, 4e7, 10001)  # fine grid around w0
        vals = Gauss(w, w0, sig)
        peak_idx = int(jnp.argmax(vals))
        dw = float(w[1] - w[0])
        assert abs(float(w[peak_idx]) - w0) < 2 * dw

    def test_symmetry_centered(self):
        """Gauss(w, 0, sig) should be even."""
        w = jnp.linspace(-1e8, 1e8, 201)
        vals = Gauss(w, 0.0, 1e6)
        np.testing.assert_allclose(np.array(vals), np.array(vals[::-1]), atol=1e-12)

    def test_non_negative(self, frequency_grid):
        vals = Gauss(frequency_grid, 1e7, 2e6)
        assert jnp.all(vals >= 0)

    def test_decays_far_from_peak(self):
        """Value far from peak should be much smaller than at peak."""
        sig = 1e6
        val_peak = float(Gauss(jnp.float64(1e7), 1e7, sig))
        val_far = float(Gauss(jnp.float64(1e9), 1e7, sig))
        assert val_peak > 100 * val_far


class TestSelfSpectra:
    """Tests for self-spectra S_11, S_22, S_1212."""

    @pytest.mark.parametrize("spectrum_fn", [S_11, S_22, S_1212])
    def test_non_negative(self, frequency_grid, spectrum_fn):
        """Power spectral densities must be non-negative."""
        vals = spectrum_fn(frequency_grid)
        assert jnp.all(vals >= -1e-10)  # allow tiny numerical noise

    @pytest.mark.parametrize("spectrum_fn", [S_11, S_22, S_1212])
    def test_even_symmetry(self, spectrum_fn):
        """Self-spectra must be even: S(w) == S(-w) (tau-unit band scale)."""
        w = jnp.linspace(-2.0, 2.0, 301)
        vals = spectrum_fn(w)
        np.testing.assert_allclose(np.array(vals), np.array(vals[::-1]), rtol=1e-10)

    @pytest.mark.parametrize("spectrum_fn", [S_11, S_22, S_1212])
    def test_output_shape(self, frequency_grid, spectrum_fn):
        vals = spectrum_fn(frequency_grid)
        assert vals.shape == frequency_grid.shape

    @pytest.mark.parametrize("spectrum_fn", [S_11, S_22, S_1212])
    def test_positive_at_dc(self, spectrum_fn):
        """Spectra should have nonzero DC component."""
        val = spectrum_fn(jnp.float64(0.0))
        assert float(val) > 0


class TestCrossSpectra:
    """Tests for cross-spectra S_1_2, S_1_12, S_2_12 (no-lag API: phases are
    internal to the anchored model, see NOISE_MODEL_SPEC.md)."""

    @pytest.mark.parametrize("cross_fn", [S_1_2, S_1_12, S_2_12])
    def test_hermitian_symmetry(self, cross_fn):
        """Cross-spectra of real processes: S_ab(-w) == conj(S_ab(w))."""
        w = jnp.linspace(0.01, 2.0, 101)
        np.testing.assert_allclose(np.array(cross_fn(-w)),
                                   np.conj(np.array(cross_fn(w))), rtol=1e-10)

    @pytest.mark.parametrize("cross_fn,self_fn1,self_fn2", [
        (S_1_2, S_11, S_22),
        (S_1_12, S_11, S_1212),
        (S_2_12, S_22, S_1212),
    ])
    def test_cauchy_schwarz(self, cross_fn, self_fn1, self_fn2):
        """At every w: |S_ab(w)|^2 <= S_aa(w) * S_bb(w) (PSD-matrix positivity)."""
        w = jnp.linspace(0.0, 2.0, 501)
        lhs = jnp.abs(cross_fn(w)) ** 2
        rhs = self_fn1(w) * self_fn2(w)
        assert jnp.all(lhs <= rhs * (1 + 1e-12) + 1e-30)

    @pytest.mark.parametrize("cross_fn", [S_1_2, S_1_12, S_2_12])
    def test_output_shape(self, small_frequency_grid, cross_fn):
        vals = cross_fn(small_frequency_grid)
        assert vals.shape == small_frequency_grid.shape

    def test_imaginary_part_in_band(self):
        """The DT_SHIFT lag produces nonzero Im parts inside the comb band."""
        w = jnp.linspace(0.025, 1.25, 50)
        vals = S_1_2(w)
        assert jnp.max(jnp.abs(jnp.imag(vals))) > 1e-3 * jnp.max(jnp.abs(vals))
