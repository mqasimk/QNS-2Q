"""Regression guard for the infidelity-accumulation exponent eta (eq::accumulation).

Pins the depth-scaling law of main_v10.tex that COMPANION.md flagged as the one
paper equation "not pinned to a committed test" (verified ad hoc 2026-06-27):

    1 - F^{(M)} ~ M^eta (1 - F^{(1)}),   eta in [1, 2]         (eq::accumulation)

with eta = 1 for white (Markovian) noise, eta -> 2 in the quasi-static limit, and
eta < 1 once decoupling/tailoring make the lagged overlaps partially cancel.  At
leading order 1 - F^{(M)} is linear in the self-overlap I_{a,a}(MT) (the leading
Pauli decay rate Gamma = 1/2 Theta_0 in the PTM model), so eta is the log-log slope
of I_{a,a}(MT)/I_{a,a}(T) vs M -- which this test measures through the repo's own
time-folding evaluator (control.idle.prepare_time_domain_overlap /
evaluate_overlap_folded), the machinery App. E.7 (Sec. app::time_folding) derives:

    I_{a,b}(MT) = sum_{p=-(M-1)}^{M-1} (M - |p|) g_p,     g_p = 1/2 int C_{ya,yb}(u) R(u+pT) du

so the accumulated overlap runs between M I(T) (uncorrelated periods, g_{p!=0}->0,
eta=1) and M^2 I(T) (fully correlated / quasi-static, g_p->g_0, eta=2).  Decoupling
drives the low-frequency g_{p!=0} negative, pushing the accumulated overlap below
M I(T) and the effective exponent below unity.

Self-contained: builds its own analytic self-spectra on a coarse comb, so it never
touches a run folder and is independent of QNS2Q_REGIME.  Units: tau = 1 (repo
CLAUDE.md), so times are in tau and the frequency grid is the dimensionless w*tau.
The evaluator returns int C R_folded du WITHOUT the 1/2 of eq::overlap_def (that
factor is applied downstream in calculate_idling_fidelity); it is a constant
prefactor that cancels in the log-log slope, so the eta measured here is unaffected.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from qns2q.control.idle import (prepare_time_domain_overlap,
                                 evaluate_overlap_folded)

TAU = 1.0
T_BASE = 8.0              # base-sequence period, in tau
W_MAX = np.pi            # pulse-spacing Nyquist reach, pi/tau
N_W = 512
W = jnp.linspace(0.0, W_MAX, N_W)                      # one-sided comb, ascending from 0
M_LIST = np.array([1, 2, 4, 8, 16, 32, 64])

# Free induction: y = +1 on [0, T] (no interior pi-pulse).
FID = jnp.array([0.0, T_BASE])
# Repeated spin echo: one pi-pulse at T/2 per base period -> CPMG under repetition.
ECHO = jnp.array([0.0, T_BASE / 2, T_BASE])


def _overlap_series(S_diag, pulse_times, Ms=M_LIST):
    """I_{11}(MT) for each M via the repo's folded evaluator.

    NB: build SMat and reuse the module-level W ONCE, then vary only M.
    prepare_time_domain_overlap now memoizes on the CONTENTS of SMat/w_grid, so
    this is no longer required for correctness (an earlier id()-keyed cache
    could hand a transient per-call SMat a stale folded matrix under recycled
    ids -- observed as a corrupted single I(MT) point; fixed via
    control.idle._array_fingerprint).  Reusing one stable SMat is still the
    right pattern here: it avoids re-hashing the same spectrum on every M."""
    SMat = jnp.zeros((4, 4, N_W), dtype=jnp.complex128).at[1, 1].set(jnp.asarray(S_diag))
    out = []
    for M in Ms:
        RMat, dt, n_base = prepare_time_domain_overlap(SMat, W, TAU, T_BASE, int(M))
        out.append(float(np.real(evaluate_overlap_folded(pulse_times, pulse_times,
                                                          RMat[1, 1], dt, n_base))))
    return np.array(out)


def _eta(S_diag, pulse_times, Ms=M_LIST):
    """Accumulation exponent eta = d log I(MT) / d log M, plus the I(MT)/I(T) ratios."""
    Is = _overlap_series(S_diag, pulse_times, Ms)
    slope = np.polyfit(np.log(Ms), np.log(np.abs(Is)), 1)[0]
    return slope, Is / Is[0]


def _lorentzian(gamma):
    """Low-frequency Lorentzian S(w) = gamma / (gamma^2 + w^2), peak-normalized.
    Correlation time ~ 1/gamma: gamma << 1/(M T) is the quasi-static limit."""
    S = gamma / (gamma ** 2 + np.asarray(W) ** 2)
    return S / S.max()


def test_fold_weights_sum_to_M_squared():
    """The triangular lag weights (M - |p|) sum to M^2 (green text at eq 1610)."""
    for M in (1, 2, 5, 10, 37, 64):
        wsum = sum(M - abs(p) for p in range(-(M - 1), M))
        assert wsum == M * M


def test_white_noise_accumulates_linearly():
    """White (flat) spectrum under free induction: eta = 1 (Markovian limit)."""
    eta, ratios = _eta(np.ones(N_W), FID)
    assert 0.90 < eta < 1.12, f"white-noise eta={eta:.3f} not ~1"
    # I(MT)/I(T) tracks M, not M^2.
    assert np.allclose(ratios, M_LIST.astype(float), rtol=0.10)


def test_quasistatic_noise_accumulates_quadratically():
    """Slow (narrow low-frequency) noise under free induction: eta -> 2.

    Correlation time 1/gamma = 2000 tau >> M*T = 512 tau, so successive periods are
    fully correlated and the overlap grows as M^2 (g_p -> g_0)."""
    eta, ratios = _eta(_lorentzian(5e-4), FID)
    assert eta > 1.85, f"quasi-static eta={eta:.3f} not approaching 2"
    # I(MT)/I(T) tracks M^2 to within a few percent over the M range.
    assert np.allclose(ratios / M_LIST.astype(float) ** 2, 1.0, atol=0.05)


def test_decoupling_makes_accumulation_sublinear():
    """Decoupling (repeated echo) on low-frequency noise: eta < 1, and the accumulated
    overlap falls below M I(T) -- the regime the tailored sequences operate in."""
    eta, ratios = _eta(_lorentzian(0.1), ECHO)
    assert eta < 0.97, f"decoupled eta={eta:.3f} not sub-linear"
    # I(MT) < M I(T) strictly for every repeated (M >= 2) sequence.
    assert np.all(ratios[1:] < M_LIST[1:].astype(float))


def test_eta_ordering_across_regimes():
    """The three regimes are strictly ordered: decoupled < white < quasi-static,
    the physical content of eta in [1,2] with sub-unity under tailoring."""
    eta_dec, _ = _eta(_lorentzian(0.1), ECHO)
    eta_white, _ = _eta(np.ones(N_W), FID)
    eta_qs, _ = _eta(_lorentzian(5e-4), FID)
    assert eta_dec < eta_white < eta_qs
