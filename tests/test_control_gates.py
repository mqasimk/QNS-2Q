"""Unit coverage for the gate-optimization pipeline (OPT-MISC-0611).

Pins the pieces the 6/11 audit found untested: the power-law tail fit and its
fallbacks, the diagonal-exp CZ fidelity map's exact equivalence to the expm
formulation it replaced, comb-vs-folded overlap agreement on a smooth
spectrum, and the SMat construction semantics (measured-DC consumption,
analytic-DC fallback, robust NaN channel drop, corrupted-selfs error).
"""
import types

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from qns2q.control import cz, idle
from qns2q.control.tails import fit_powerlaw_tail, tail_extend_interp
from qns2q.noise.spectra import S_11, S_1212


# ---------------------------------------------------------------- tails ----

def test_tail_fit_recovers_powerlaw():
    wk = np.linspace(0.1, 0.8, 20)
    A_true, p_true = 3.7e-4, 2.3
    comp = A_true * (wk / wk[-1]) ** (-p_true)
    A, p = fit_powerlaw_tail(wk, comp)
    assert np.isclose(A, A_true, rtol=1e-10)
    assert np.isclose(p, p_true, rtol=1e-10)


def test_tail_fit_sign_change_returns_none():
    wk = np.linspace(0.1, 0.8, 20)
    comp = np.ones_like(wk)
    comp[-2] = -1.0          # oscillating tail -> unfittable
    assert fit_powerlaw_tail(wk, comp) is None


def test_tail_fit_rising_tail_clips_flat():
    wk = np.linspace(0.1, 0.8, 20)
    comp = (wk / wk[-1]) ** 1.5          # rising: p_fit = -1.5 -> clipped to 0
    A, p = fit_powerlaw_tail(wk, comp)
    assert p == 0.0


def test_tail_extend_matches_inside_and_extends_outside():
    wk = np.linspace(0.1, 0.8, 20)
    comp = 2e-3 * (wk / wk[-1]) ** (-1.7)
    w_dense = jnp.linspace(0.0, 2.0, 500)
    out = np.asarray(tail_extend_interp(w_dense, wk, comp))
    inside = (np.asarray(w_dense) >= wk[0]) & (np.asarray(w_dense) <= wk[-1])
    expect_in = np.interp(np.asarray(w_dense)[inside], wk, comp)
    assert np.allclose(out[inside], expect_in, rtol=1e-12)
    outside = np.asarray(w_dense) > wk[-1]
    expect_out = 2e-3 * (np.asarray(w_dense)[outside] / wk[-1]) ** (-1.7)
    assert np.allclose(out[outside], expect_out, rtol=1e-6)


# ----------------------------------------------- CZ fidelity map (exact) ----

def _cz_fidelity_expm_reference(I_matrix, J, M, dc_12):
    """The pre-058a5a0 formulation: full matrix expm per Pauli row."""
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]),
                     jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]],
                     [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])

    R = np.zeros((16, 16))
    for i in range(16):
        Oi = p2q[i]
        val = jnp.zeros((4, 4), dtype=jnp.complex128)
        for a in range(3):
            for b in range(3):
                c_a = 0.5 * (1.0 - cz.sgn(Oi, a + 1, 0))
                c_b = 0.5 * (1.0 - cz.sgn(Oi, b + 1, 0))
                coeff = 0.5 * c_a * c_b
                val = val + coeff * I_matrix[a + 1, b + 1] * (z2q[a + 1] @ z2q[b + 1])
        rot = (1.0 - cz.sgn(Oi, 1, 2)) * M * J * dc_12
        G = jax.scipy.linalg.expm(-1j * rot * z2q[3] - val)
        for j in range(16):
            R[i, j] = float(jnp.real(jnp.trace(Oi @ G @ p2q[j]) * 0.25))
    fid = np.trace(np.asarray(cz.zzPTM()).T @ R) / 16.0
    return fid


def test_cz_fidelity_diagonal_exp_matches_expm():
    rng = np.random.default_rng(7)
    I = rng.normal(scale=1e-2, size=(4, 4)) + 1j * rng.normal(scale=1e-3, size=(4, 4))
    I = jnp.asarray(I + I.conj().T)      # Hermitian-ish overlap matrix
    J, M, dc12 = 0.04, 1, 19.6
    fast = float(cz.calculate_cz_fidelity(I, J, M, dc12))
    ref = float(_cz_fidelity_expm_reference(I, J, M, dc12))
    assert np.isclose(fast, ref, rtol=0, atol=1e-12)


# ------------------------------------------------- comb vs folded (smooth) --

def test_comb_matches_folded_on_smooth_spectrum():
    """M = 16 comb approximation agrees with the folded evaluator when the
    spectrum is smooth (no sub-tooth structure)."""
    M, T_seq, tau = 16, 40.0, 1.0
    w = jnp.linspace(0, 2 * jnp.pi, 8000)
    lor = 1e-3 / (1.0 + (np.asarray(w) / 0.3) ** 2)
    SMat = jnp.zeros((4, 4, w.size), dtype=jnp.complex128)
    for d in (1, 2, 3):
        SMat = SMat.at[d, d].set(jnp.asarray(lor * (1 + 0.2 * d)))

    d1 = idle.pulse_times_to_delays(jnp.array(idle.cddn(0., T_seq, 2)))
    d2 = idle.pulse_times_to_delays(jnp.array(idle.cddn(0., T_seq, 3)))
    pt1 = idle.delays_to_pulse_times(d1, T_seq)
    pt2 = idle.delays_to_pulse_times(d2, T_seq)
    pt12 = idle.make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    pts = [pt0, pt1, pt2, pt12]

    idle._OVERLAP_SETUP_CACHE.clear()
    R_fold, dt, nbs = idle.prepare_time_domain_overlap(SMat, w, tau, T_seq, M)
    w0 = 2 * jnp.pi / T_seq
    max_k = int(float(w[-1]) / float(w0))
    omega_k = jnp.arange(1, max_k + 1) * w0
    S_flat = SMat.reshape(-1, SMat.shape[-1])
    S_h = jax.vmap(lambda fp: jnp.interp(omega_k, w, jnp.real(fp), right=0.)
                   + 1j * jnp.interp(omega_k, w, jnp.imag(fp), right=0.))(S_flat)
    S_packed = jnp.concatenate([S_flat[:, :1], S_h], axis=1).reshape(4, 4, -1)

    def inf(comb):
        I = jnp.array([[
            (idle.evaluate_overlap_comb(pts[r], pts[c], S_packed[r, c],
                                        omega_k, T_seq, M) if comb else
             idle.evaluate_overlap_folded(pts[r], pts[c], R_fold[r, c], dt, nbs))
            for c in range(4)] for r in range(4)])
        return float(1.0 - idle.calculate_idling_fidelity(I) / 16.0)

    f, c = inf(False), inf(True)
    assert abs(c - f) / f < 0.02, (f, c)


def test_use_comb_crossover_respects_lines():
    """OPT-COMB-M16: the comb approximation must stay off while the M-fold
    filter tooth width is comparable to the regime's narrowest line width.
    The hard-coded boundary points were measured on the FEATURED model
    (sigma = 0.02, scripts/diag_comb_vs_folded.py); other line-carrying
    regimes (showcase: sigma_min = 0.016) get the generic rule checks."""
    from qns2q.noise.spectra import line_priors
    pri = line_priors()
    if pri is None:
        pytest.skip("bland regime: no lines, legacy M cutoff applies")
    assert not idle.use_comb_approximation(8, 1000.0)   # legacy M <= 10 cutoff
    assert not idle.use_comb_approximation(16, 20.0)    # tooth width >> sigma/8
    assert (cz.use_comb_approximation(16, 20.0)
            == idle.use_comb_approximation(16, 20.0))
    _, sigma = pri
    # generic rule: crossover exactly at 2*pi/(M*T_seq) < sigma/8
    for M_, Ts in ((16, 80.0), (16, 160.0), (128, 80.0), (64, 320.0)):
        expect = (2 * np.pi / (M_ * Ts)) < sigma / 8
        assert idle.use_comb_approximation(M_, Ts) == expect, (M_, Ts)
    if abs(sigma - 0.02) < 1e-12:   # featured: the measured boundary sweep
        assert not idle.use_comb_approximation(16, 80.0)   # up to 3.2%
        assert idle.use_comb_approximation(16, 160.0)      # <= 1.7% measured
        assert idle.use_comb_approximation(128, 80.0)      # <= 0.3%


# ----------------------------------------------------- identity padding ----

def test_pad_targets_counts_and_parity_guard():
    from qns2q.control.padding import pad_targets, pad_count, pad_delays
    t = pad_targets([150, 149, 148, 76, 75, 74])
    assert t == {0: 150, 1: 149}
    assert pad_count(74, t) == 150 and pad_count(75, t) == 149
    assert pad_count(150, t) == 150
    with pytest.raises(ValueError):
        pad_delays(np.ones(3), 4)                      # odd pad count
    assert np.array_equal(pad_delays(np.ones(3), 5),
                          np.array([1., 1., 1., 0., 0.]))


def test_identity_padding_exact_value_and_grad():
    """The padded cost equals the unpadded cost, and the gradient w.r.t. the
    REAL delays is unchanged -- for both gate modules and both evaluators."""
    T_seq, M, tau = 40.0, 1, 1.0
    nbs = int(np.ceil(T_seq / (tau / 4)))
    rng = np.random.default_rng(3)
    RMat = jnp.asarray(rng.normal(size=(4, 4, 2 * nbs - 1)) * 1e-6 + 0j)
    omega_k = jnp.arange(1, 41) * (2 * jnp.pi / T_seq)
    S_packed = jnp.asarray(rng.normal(size=(4, 4, 41)) * 1e-6 + 0j)
    n1, n2 = 5, 8
    d1 = np.asarray(cz.get_random_delays(n1, T_seq, tau))
    d2 = np.asarray(cz.get_random_delays(n2, T_seq, tau))
    x = np.concatenate([d1, d2])
    xp = np.concatenate([d1, np.zeros(4), d2, np.zeros(2)])
    n1p, n2p = n1 + 4, n2 + 2

    def strip(g):
        g = np.asarray(g)
        return np.concatenate([g[:n1], g[n1p:n1p + n2]])

    cases = [
        ("cz folded",
         lambda z, n: cz.cost_vag_folded(jnp.asarray(z), RMat, tau / 4, T_seq,
                                         0.05, M, n_pulses1=n, n_base_steps=nbs)),
        ("cz comb",
         lambda z, n: cz.cost_vag_comb(jnp.asarray(z), S_packed, omega_k,
                                       T_seq, 0.05, n_pulses1=n, M=M)),
        ("idle folded",
         lambda z, n: idle.cost_vag_folded(jnp.asarray(z), RMat, tau / 4,
                                           T_seq, n_pulses1=n, n_base_steps=nbs)),
        ("idle comb",
         lambda z, n: idle.cost_vag_comb(jnp.asarray(z), S_packed, omega_k,
                                         T_seq, n_pulses1=n, M=M)),
    ]
    for label, vag in cases:
        v0, g0 = vag(x, n1)
        v1, g1 = vag(xp, n1p)
        assert np.isclose(float(v0), float(v1), rtol=1e-12, atol=1e-15), label
        assert np.allclose(np.asarray(g0), strip(g1), rtol=1e-9, atol=1e-13), label


# --------------------------------------------------- SMat build semantics --

def _fake_cfg(builder_cls, wk, specs, use_simulated=False, cross=True,
              spectral_model='interp', self_only=False):
    ns = types.SimpleNamespace()
    ns.w = jnp.linspace(0, 2 * jnp.pi, 4000)
    ns.wkqns = jnp.asarray(wk)
    ns.specs = specs
    ns.include_cross_spectra = cross
    ns.use_simulated = use_simulated
    ns.spectral_model = spectral_model
    ns.char_self_only = self_only
    return builder_cls._build_interpolated_spectra(ns)


def _toy_specs(wk, dc=None):
    base = 1e-3 * (1 + np.exp(-np.asarray(wk)))
    specs = {'S11': base, 'S22': 2 * base, 'S1212': 0.5 * base,
             'S12': base * (0.3 + 0.1j), 'S112': base * 0.2, 'S212': base * 0.1}
    if dc is not None:
        for k in specs:
            specs[k] = np.concatenate([[dc if k == 'S1212' else specs[k][0]],
                                       specs[k]])
    return specs


@pytest.mark.parametrize("builder", [cz.CZOptConfig, idle.Config])
def test_smat_consumes_measured_dc(builder):
    wk_teeth = np.linspace(2 * np.pi / 160, np.pi / 4, 20)
    dc_meas = 4.2e-4                       # != analytic S_1212(0)
    wk = np.concatenate([[0.0], wk_teeth])
    SMat = _fake_cfg(builder, wk, _toy_specs(wk_teeth, dc=dc_meas))
    assert np.isclose(complex(SMat[3, 3, 0]).real, dc_meas, rtol=1e-14)
    assert not np.isclose(complex(SMat[3, 3, 0]).real,
                          float(S_1212(jnp.array([0.0]))[0]), rtol=1e-3)


@pytest.mark.parametrize("builder", [cz.CZOptConfig, idle.Config])
def test_smat_dcless_grid_injects_analytic(builder):
    wk_teeth = np.linspace(2 * np.pi / 160, np.pi / 4, 20)
    SMat = _fake_cfg(builder, wk_teeth, _toy_specs(wk_teeth), use_simulated=True)
    assert complex(SMat[1, 1, 0]) == complex(S_11(jnp.array([0.0]))[0])


@pytest.mark.parametrize("builder", [cz.CZOptConfig, idle.Config])
def test_smat_drops_nan_cross_keeps_rest(builder):
    wk = np.concatenate([[0.0], np.linspace(2 * np.pi / 160, np.pi / 4, 20)])
    specs = _toy_specs(np.linspace(2 * np.pi / 160, np.pi / 4, 20), dc=1e-4)
    specs['S112'] = np.full_like(specs['S112'], np.nan, dtype=complex)
    SMat = _fake_cfg(builder, wk, specs)
    assert np.all(np.asarray(SMat[1, 3]) == 0)      # dropped
    assert np.all(np.asarray(SMat[3, 1]) == 0)
    assert np.any(np.asarray(SMat[1, 2]) != 0)      # S12 retained
    assert not np.any(np.isnan(np.asarray(SMat)))


@pytest.mark.parametrize("builder", [cz.CZOptConfig, idle.Config])
def test_smat_nan_self_raises(builder):
    wk = np.concatenate([[0.0], np.linspace(2 * np.pi / 160, np.pi / 4, 20)])
    specs = _toy_specs(np.linspace(2 * np.pi / 160, np.pi / 4, 20), dc=1e-4)
    specs['S22'] = np.full_like(specs['S22'], np.nan)
    with pytest.raises(ValueError, match="corrupted"):
        _fake_cfg(builder, wk, specs)


# ----------------------------------------- SHOWCASE-0612 ablation switches --

@pytest.mark.parametrize("builder", [cz.CZOptConfig, idle.Config])
def test_smat_self_only_drops_zz_and_crosses(builder):
    """char_self_only (ablation rung c): the characterized model keeps S11/S22
    only; the ZZ self-spectrum and every cross are zero (1Q-only QNS world)."""
    wk = np.concatenate([[0.0], np.linspace(2 * np.pi / 160, np.pi / 4, 20)])
    specs = _toy_specs(np.linspace(2 * np.pi / 160, np.pi / 4, 20), dc=1e-4)
    SMat = _fake_cfg(builder, wk, specs, self_only=True)
    assert np.any(np.asarray(SMat[1, 1]) != 0)
    assert np.any(np.asarray(SMat[2, 2]) != 0)
    assert np.all(np.asarray(SMat[3, 3]) == 0)      # ZZ dropped
    for r, c in ((1, 2), (1, 3), (2, 3)):
        assert np.all(np.asarray(SMat[r, c]) == 0)  # crosses dropped
        assert np.all(np.asarray(SMat[c, r]) == 0)


def test_smoothfit_recovers_powerlaw_and_caps_at_dc():
    from qns2q.control.tails import smoothfit_curve
    wk = np.concatenate([[0.0], np.linspace(0.05, 0.8, 20)])
    A_true, p_true, dc = 5e-6, 0.9, 3e-4
    comp = np.concatenate([[dc], A_true * wk[1:] ** (-p_true)])
    w_dense = jnp.linspace(0.0, 2.0, 1000)
    out = np.asarray(smoothfit_curve(w_dense, wk, comp, dc_val=dc))
    mid = (np.asarray(w_dense) > 0.1) & (np.asarray(w_dense) < 1.5)
    expect = A_true * np.asarray(w_dense)[mid] ** (-p_true)
    assert np.allclose(out[mid], expect, rtol=1e-6)
    assert np.isclose(out[0], dc, rtol=1e-12)        # capped, not divergent


def test_smoothfit_is_line_blind():
    """A narrow line on a flat background must NOT survive the smooth fit: the
    fitted curve at the line center stays near the background level (within
    the line's small pull on the global fit), nowhere near the line peak."""
    from qns2q.control.tails import smoothfit_curve
    wk = np.linspace(0.04, 0.8, 20)
    bg = 1e-7 * (wk / 0.3) ** (-0.9)
    line = 2e-5 * np.exp(-(wk - 0.204) ** 2 / (2 * 0.022 ** 2))
    out = np.asarray(smoothfit_curve(jnp.array([0.204]), wk, bg + line,
                                     dc_val=1e-4))
    bg_at_line = 1e-7 * (0.204 / 0.3) ** (-0.9)
    assert out[0] < 10 * bg_at_line, "smooth fit should average the line away"
    assert out[0] < 0.05 * 2e-5, "smooth fit must not track the line peak"


def test_min_sep_prunes_library_and_caps_density():
    """The pulse library under a min-sep floor: every spacing >= min_sep, and
    the deep-nesting CDD orders that violate it are pruned."""
    T_seq, min_sep = 320.0, 8.0
    lib_legacy, _ = cz.construct_pulse_library(T_seq, 1.0, 10**9)
    lib_bw, desc_bw = cz.construct_pulse_library(T_seq, min_sep, 10**9)
    assert len(lib_bw) < len(lib_legacy)
    for d1, d2 in lib_bw:
        for d in (d1, d2):
            if np.asarray(d).size:
                assert float(np.min(np.asarray(d))) >= min_sep - 1e-9
    # CDD6 needs spacing T/2^6 = 5 tau < 8 tau -> the deepest order present
    # must be CDD5
    orders = [int(s.split('(')[1].split(',')[0]) for s in desc_bw
              if s.startswith('CDD')]
    assert max(orders) == 5
