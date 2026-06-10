"""Invariants of the experimentally-anchored noise model (NOISE_MODEL_SPEC.md).

Covers the spec's acceptance gates that are cheap enough for CI:
  (i)  T2* calibration target (260 tau, window 230-290) per qubit;
  (ii) PSD-matrix positivity over the band;
  (iii) the synthesized trajectories reproduce the analytic covariances,
        including the measured (+, +, -) coherence sign pattern;
  (iv) Class-F lines live on the local (qubit) channels only, so the ZZ
       channel stays smooth and the inter-qubit coherence dips at the lines.

The suite runs under the active QNS2Q_REGIME (default: featured = Class F).
Class-M-only behavior is exercised by the bland-regime CI run.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from qns2q.model.trajectories import make_noise_mat_arr, make_channel_trajs
from qns2q.noise import spectra as sp

_FEATURED = sp._LINES_ON

T = 160.0
TRUNCATE = 20
WMAX = 2*np.pi*TRUNCATE/T          # band edge; synthesis covers [0, 2*WMAX]


def _chi(spec, t, n=40001):
    """FID dephasing exponent chi(t) = (2/pi) int_0^{2wmax} S sin^2(wt/2)/w^2 dw,
    matching the H = (1/2) b Z convention of make_Hamiltonian and the synthesis
    covariance <b b> = (1/pi) int_0^{2wmax} S cos."""
    w = np.linspace(1e-9, 2*WMAX, n)
    s = np.real(np.asarray(spec(jnp.asarray(w))))
    return (2/np.pi)*np.trapezoid(s*np.sin(w*t/2)**2/w**2, w)


def test_t2_calibration_target():
    """chi(T2*) = 1 at T2* = 260 tau (window 230-290) for both qubits."""
    for spec in (sp.S_11, sp.S_22):
        t = np.linspace(100, 500, 401)
        chi = np.array([_chi(spec, ti) for ti in t])
        t2 = t[np.argmin(np.abs(chi - 1))]
        assert 230 <= t2 <= 290, f"T2* = {t2} tau outside the calibration window"


def test_psd_matrix_positive_over_band():
    """The 3x3 Hermitian spectral matrix is positive semidefinite at every w."""
    w = np.linspace(0.0, 2*WMAX, 800)
    s11 = np.real(np.asarray(sp.S_11(jnp.asarray(w))))
    s22 = np.real(np.asarray(sp.S_22(jnp.asarray(w))))
    s1212 = np.real(np.asarray(sp.S_1212(jnp.asarray(w))))
    s12 = np.asarray(sp.S_1_2(jnp.asarray(w)))
    s112 = np.asarray(sp.S_1_12(jnp.asarray(w)))
    s212 = np.asarray(sp.S_2_12(jnp.asarray(w)))
    min_eig = np.inf
    for i in range(len(w)):
        mat = np.array([[s11[i], s12[i], s112[i]],
                        [np.conj(s12[i]), s22[i], s212[i]],
                        [np.conj(s112[i]), np.conj(s212[i]), s1212[i]]])
        min_eig = min(min_eig, np.linalg.eigvalsh(mat)[0])
    assert min_eig >= -1e-15, f"PSD matrix has negative eigenvalue {min_eig:.3e}"


def test_coherence_sign_pattern():
    """In-band coherences carry the measured (+, +, -) real-part sign pattern
    (Yoneda 2023), with |c_12| at the calibrated ~0.67 level."""
    w = jnp.linspace(0.3, 0.45, 40)     # mid-band, away from the Class-F lines
    s11, s22, s1212 = sp.S_11(w), sp.S_22(w), sp.S_1212(w)
    c12 = jnp.abs(sp.S_1_2(w))/jnp.sqrt(s11*s22)
    c112 = jnp.real(sp.S_1_12(w))/jnp.sqrt(s11*s1212)
    c212 = jnp.real(sp.S_2_12(w))/jnp.sqrt(s22*s1212)
    assert 0.5 <= float(jnp.mean(c12)) <= 0.8
    assert float(jnp.mean(c112)) > 0.1, "c_{1,12} real part should be positive"
    assert float(jnp.mean(c212)) < -0.1, "c_{2,12} real part should be negative"


def test_synthesis_reproduces_analytic_covariances():
    """Monte-Carlo instantaneous covariances of the synthesized channels match the
    band integrals of the analytic spectra (within MC error), signs included."""
    t_vec = jnp.linspace(0, T, 200)
    mats = make_noise_mat_arr('make', t_vec=t_vec, w_grain=300, wmax=WMAX,
                              truncate=TRUNCATE, midpoint=True)
    rng = np.random.default_rng(20260610)
    keys = rng.integers(0, 10000, (300, 2))
    b = np.array([np.array(make_channel_trajs(mats, jnp.array(k))) for k in keys])

    w = np.linspace(1e-9, 2*WMAX, 60001)
    wj = jnp.asarray(w)
    th = {
        (0, 0): np.trapezoid(np.real(np.asarray(sp.S_11(wj))), w)/np.pi,
        (1, 1): np.trapezoid(np.real(np.asarray(sp.S_22(wj))), w)/np.pi,
        (2, 2): np.trapezoid(np.real(np.asarray(sp.S_1212(wj))), w)/np.pi,
        (0, 1): np.trapezoid(np.real(np.asarray(sp.S_1_2(wj))), w)/np.pi,
        (0, 2): np.trapezoid(np.real(np.asarray(sp.S_1_12(wj))), w)/np.pi,
        (1, 2): np.trapezoid(np.real(np.asarray(sp.S_2_12(wj))), w)/np.pi,
    }
    for (i, j), expected in th.items():
        emp = float(np.mean(b[:, i, :]*b[:, j, :]))
        assert abs(emp - expected) < 0.3*abs(expected) + 1e-7, \
            f"cov[{i},{j}] = {emp:.3e} vs analytic {expected:.3e}"
    assert float(np.mean(b[:, 1, :]*b[:, 2, :])) < 0, "qubit2-J covariance must be negative"


@pytest.mark.skipif(not _FEATURED, reason="Class-F lines only in the featured regime")
def test_lines_local_only_and_coherence_dip():
    """Class F: the nuclear-difference triplet sits on the qubit channels only --
    the ZZ channel stays smooth and c_12(w) dips at the line frequencies."""
    w_line, w_ref = 0.273, 0.36
    # line visible on S_22 (full triplet weight)
    ratio_22 = float(sp.S_22(jnp.array([w_line]))[0]/sp.S_22(jnp.array([w_ref]))[0])
    assert ratio_22 > 5, f"S_22 line contrast {ratio_22:.1f} too small"
    # ZZ channel smooth across the line (no nuclear contribution)
    ratio_zz = float(sp.S_1212(jnp.array([w_line]))[0]/sp.S_1212(jnp.array([w_ref]))[0])
    assert ratio_zz < 2, f"S_1212 should be smooth at the line (ratio {ratio_zz:.1f})"
    # coherence dip: local lines dilute the shared (electrical) fraction
    def c12(wv):
        wv = jnp.array([wv])
        return float(jnp.abs(sp.S_1_2(wv))[0]/jnp.sqrt(sp.S_11(wv)*sp.S_22(wv))[0])
    assert c12(w_line) < 0.4*c12(w_ref), \
        f"coherence should dip at the line: {c12(w_line):.2f} vs {c12(w_ref):.2f}"
