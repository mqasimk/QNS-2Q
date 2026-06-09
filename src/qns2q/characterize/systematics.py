"""
Deterministic comb-inversion systematic error for QNS spectral reconstruction.

Why this exists
---------------
The simulated dephasing noise is a *Gaussian* process (``trajectories.make_noise_traj``
builds it as a linear combination of sinusoids with Gaussian-random amplitudes), so the
second-order cumulant expansion the reconstruction is built on is **exact** -- every
cumulant above the second vanishes identically. The forward map

    chi(t_c) = (1/2pi) \\int dw S(w) |F(w, t_c)|^2

is therefore exact, and the *only* non-statistical reconstruction error is the
**harmonic-comb inversion**: it approximates that continuous integral by a finite sum
over the comb omega_k = 2 pi k / T, k = 1..truncate. Three things that sum misses, all
independent of the number of shots:

  * finite-M comb teeth have non-zero width ~1/(MT) (the integral != the sum),
  * truncation at omega_kmax = 2 pi*truncate/T drops/aliases out-of-band weight,
  * the band (0, omega_1) below the first tooth is never sampled.

How it is quantified (no Monte Carlo)
-------------------------------------
For each reconstruction we rebuild its kernel ``K_i(w)`` on a fine, wide w-grid, form
the EXACT observable ``C*_i = (T/2pi) \\int dw K_i(w) S(w)`` and invert it with the SAME
comb matrix ``U`` the recon uses (built on the same time grid here, so the inversion is
internally exact -- the comb-sum self-check returns S to machine precision). The
residual ``recon_via_U(C*) - S(w_k)`` is the pure systematic.

``S(w)`` may be the analytic ground truth (``analytic_spectra``; the exact method
systematic, appropriate for the simulation figures) or a self-consistent model built
from the reconstructed comb (``selfconsistent_spectra``; the estimate available without
knowing the answer). The kernel definitions mirror ``inversion.py`` one-to-one; the
test-suite guards against drift.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from qns2q.model.trajectories import make_y


# --- which pulses / filter indices feed each reconstruction (mirror inversion.py) ---
# kind: 'self_diff' -> U = (M/T)(|F_a|^2 - |F_b|^2);  'self_sq' -> U = (2M/T)|F_a|^2;
#       'cross'     -> Re channel U1 = (2M/T) F_a(+w) F_b(-w); Im channel U2 = Im(U1-form).
SPEC_KERNELS = {
    'S11':   dict(kind='self_diff', pulse=['CPMG', 'CDD3'], a=0, b=1),
    'S22':   dict(kind='self_diff', pulse=['CPMG', 'CDD3'], a=0, b=1),
    'S1212': dict(kind='self_sq',   pulse=['CPMG', 'CDD3'], a=0),
    'S12':   dict(kind='cross', re=(['CPMG', 'CPMG'], 0, 1), im=(['CDD3', 'CPMG'], 0, 1)),
    'S112':  dict(kind='cross', re=(['CPMG', 'FID'], 0, 2),  im=(['CPMG', 'CDD3'], 0, 1)),
    'S212':  dict(kind='cross', re=(['FID', 'CPMG'], 1, 2),  im=(['CDD3', 'CPMG'], 1, 0)),
}


def _ff_grid(toggle, tb_j, wgrid, sign, chunk=4000):
    """int exp(i*sign*w*t) toggle(t) dt over a w array, on the GPU, chunked for memory."""
    y_j = jnp.asarray(np.asarray(toggle))
    out = []
    for s in range(0, len(wgrid), chunk):
        wc = jnp.asarray(wgrid[s:s + chunk])
        phase = jnp.exp(1j * sign * wc[:, None] * tb_j[None, :])
        out.append(jnp.trapezoid(phase * y_j[None, :], tb_j, axis=1))
    return np.asarray(jnp.concatenate(out))


def comb_inversion_systematic(spectra, c_times, M, T, n_wfine=80001, n_tb=8000,
                              wmax_factor=8.0, return_selfcheck=False):
    """Deterministic per-harmonic comb-inversion systematic for all six spectra.

    Parameters
    ----------
    spectra : dict
        Callables S(w) on a (rad/s) angular-frequency array. Keys 'S11','S22','S1212'
        return real arrays; 'S12','S112','S212' return complex arrays (phase included).
    c_times, M, T : as in the experiment/reconstruction config.
    n_wfine, n_tb, wmax_factor : fine integration-grid controls. Defaults resolve the
        comb teeth (~1/MT) and integrate to ``wmax_factor`` * the comb cutoff.
    return_selfcheck : if True, also return the comb-sum self-check residual per spectrum
        (must be ~0 to machine precision; validates the kernel + normalization).

    Returns
    -------
    sys : dict[str, np.ndarray]
        Per-harmonic systematic bias ``recon(C*) - S(w_k)`` (k=1..truncate). Real arrays
        for the self-spectra, complex (Re+iIm) for the cross-spectra.
    (optionally) chk : dict[str, float]
    """
    c_times = np.asarray(c_times)
    n = len(c_times)
    wk = np.array([2 * np.pi * (j + 1) / T for j in range(n)])
    wfine = np.linspace(wk[-1] / 2000, wmax_factor * wk[-1], n_wfine)
    tb = np.linspace(0, T, n_tb)
    tb_j = jnp.asarray(tb)
    T2pi = T / (2 * np.pi)

    def ys(pulse):
        return [make_y(tb, pulse, ctime=c_times[i], m=1) for i in range(n)]

    sys, chk = {}, {}

    def do_self(key):
        spec = SPEC_KERNELS[key]
        Sk = np.real(np.asarray(spectra[key](wk)))
        Sf = np.real(np.asarray(spectra[key](wfine)))
        Y = ys(spec['pulse'])
        U = np.zeros((n, n)); Cf = np.zeros(n); Cc = np.zeros(n)
        for i in range(n):
            f0k = _ff_grid(Y[i][spec['a'], spec['a']], tb_j, wk, +1)
            f0f = _ff_grid(Y[i][spec['a'], spec['a']], tb_j, wfine, +1)
            if spec['kind'] == 'self_diff':
                f1k = _ff_grid(Y[i][spec['b'], spec['b']], tb_j, wk, +1)
                f1f = _ff_grid(Y[i][spec['b'], spec['b']], tb_j, wfine, +1)
                Kk = (M / T) * (np.abs(f0k) ** 2 - np.abs(f1k) ** 2)
                Kf = (M / T) * (np.abs(f0f) ** 2 - np.abs(f1f) ** 2)
            else:  # self_sq
                Kk = (2 * M / T) * np.abs(f0k) ** 2
                Kf = (2 * M / T) * np.abs(f0f) ** 2
            U[i, :] = np.real(Kk)
            Cf[i] = T2pi * np.trapezoid(np.real(Kf) * Sf, wfine)
            Cc[i] = np.sum(np.real(Kk) * Sk)
        sys[key] = np.linalg.solve(U, Cf) - Sk
        chk[key] = float(np.max(np.abs(np.linalg.solve(U, Cc) - Sk)))

    def do_cross(key):
        spec = SPEC_KERNELS[key]
        Sk = np.asarray(spectra[key](wk)); Sf = np.asarray(spectra[key](wfine))
        (p_re, are, bre) = spec['re']; (p_im, aim, bim) = spec['im']
        Yre = ys(p_re); Yim = ys(p_im)
        U1 = np.zeros((n, n), complex); C1f = np.zeros(n); C1c = np.zeros(n)
        U2 = np.zeros((n, n)); C2f = np.zeros(n); C2c = np.zeros(n)
        for i in range(n):
            k1k = (2 * M / T) * _ff_grid(Yre[i][are, are], tb_j, wk, +1) * _ff_grid(Yre[i][bre, bre], tb_j, wk, -1)
            k1f = (2 * M / T) * _ff_grid(Yre[i][are, are], tb_j, wfine, +1) * _ff_grid(Yre[i][bre, bre], tb_j, wfine, -1)
            U1[i, :] = k1k
            C1f[i] = T2pi * np.trapezoid(np.real(k1f * Sf), wfine)
            C1c[i] = np.real(np.sum(k1k * Sk))
            k2k = np.imag((2 * M / T) * _ff_grid(Yim[i][aim, aim], tb_j, wk, +1) * _ff_grid(Yim[i][bim, bim], tb_j, wk, -1))
            k2f = np.imag((2 * M / T) * _ff_grid(Yim[i][aim, aim], tb_j, wfine, +1) * _ff_grid(Yim[i][bim, bim], tb_j, wfine, -1))
            U2[i, :] = k2k
            C2f[i] = T2pi * np.trapezoid(k2f * (-np.imag(Sf)), wfine)
            C2c[i] = np.sum(k2k * (-np.imag(Sk)))
        re_sys = np.real(np.linalg.solve(U1, C1f)) - np.real(Sk)
        im_sys = -np.real(np.linalg.solve(U2, C2f)) - np.imag(Sk)
        sys[key] = re_sys + 1j * im_sys
        re_chk = np.real(np.linalg.solve(U1, C1c)) - np.real(Sk)
        im_chk = -np.real(np.linalg.solve(U2, C2c)) - np.imag(Sk)
        chk[key] = float(max(np.max(np.abs(re_chk)), np.max(np.abs(im_chk))))

    for key in ('S11', 'S22', 'S1212'):
        do_self(key)
    for key in ('S12', 'S112', 'S212'):
        do_cross(key)

    return (sys, chk) if return_selfcheck else sys


def dc_systematic(spectra, M, T, dc_cts=None, n_tb_per_period=2000,
                  n_wfine=300001, wmax_fid_factor=200e6, wmax_leak_factor=60e6):
    """Deterministic bias of the w=0 (DC) reconstruction point, per spectrum.

    The DC points come from a separate Ramsey-slope estimator, not the comb, so they
    carry their own (deterministic, n_shots-independent) bias:

      * Self-spectra (['FID','CDD3'] / ['CDD3','FID']): the partner-CDD3 a(+)+a(-)
        combination gives  C_{a,0} = chi_aa^FID(MT) + chi_1212^CDD3(MT), where the
        first term is the FID short-time tail (small, slightly negative) and the
        second is the residual S_1212 leak through the CDD3 partner toggle (positive,
        dominant). Estimate = 2 C/(MT). Bias = estimate - S_aa(0).
      * Cross-spectra (['FID','FID']): pure FID short-time tail of the (real) cross
        value. Bias = 2 chi_xy^FID(MT)/(MT) - Re S_xy(0).
      * S_1212: built by subtracting the (leak-contaminated) self-DC from the FID/FID
        ZZ observable; the leak largely cancels, leaving ~the FID-tail bias, which we
        use as a deterministic (conservative) proxy.

    Validated against the simulated data: the self-spectra model reproduces the
    measured DC offsets (e.g. S_11 ratio ~1.07). Returns a dict of signed DC biases.
    """
    if dc_cts is None:
        dc_cts = np.array([T / 8, T / 10, T / 12])   # matches experiments.py DC block
    MT = M * T

    # --- FID short-time-tail "apparent DC": 2 chi^FID(MT)/MT, chi=(1/2pi)int S |F_FID|^2
    wf = np.linspace(0.0, 2 * np.pi * wmax_fid_factor, 2_000_001)
    F2 = np.empty_like(wf); F2[0] = MT ** 2; F2[1:] = (2 - 2 * np.cos(wf[1:] * MT)) / wf[1:] ** 2

    def apparent_fid(real_S_on_wf):
        chi = (1 / (2 * np.pi)) * np.trapezoid(real_S_on_wf * F2, wf)
        return 2 * chi / MT

    # --- residual S_1212 leak through the partner CDD3 toggle, averaged over dc_cts
    N = n_tb_per_period
    t_b = np.linspace(0, T, N)
    t_vec = np.linspace(0, MT, M * N)
    t_j = jnp.asarray(t_vec)
    wL = np.linspace((2 * np.pi * wmax_leak_factor) / n_wfine, 2 * np.pi * wmax_leak_factor, n_wfine)
    S1212_L = np.real(np.asarray(spectra['S1212'](wL)))
    leaks = []
    for ct in dc_cts:
        y = make_y(t_b, ['FID', 'CDD3'], ctime=float(ct), m=M)   # y[2,2] = Ising toggle, tiled
        FI = _ff_grid(np.asarray(y[2, 2]), t_j, wL, +1)
        leaks.append((1 / (2 * np.pi)) * np.trapezoid(S1212_L * np.abs(FI) ** 2, wL))
    leak_dc = 2 * float(np.mean(leaks)) / MT     # the 2/MT estimator factor

    # --- assemble per-spectrum signed DC bias = estimate - truth(0)
    out = {}
    S11_0 = float(np.real(spectra['S11'](np.array([0.0]))[0]))
    S22_0 = float(np.real(spectra['S22'](np.array([0.0]))[0]))
    S1212_0 = float(np.real(spectra['S1212'](np.array([0.0]))[0]))
    out['S11'] = apparent_fid(np.real(np.asarray(spectra['S11'](wf)))) + leak_dc - S11_0
    out['S22'] = apparent_fid(np.real(np.asarray(spectra['S22'](wf)))) + leak_dc - S22_0
    out['S1212'] = apparent_fid(np.real(np.asarray(spectra['S1212'](wf)))) - S1212_0
    for key in ('S12', 'S112', 'S212'):
        S0 = float(np.real(spectra[key](np.array([0.0]))[0]))
        out[key] = apparent_fid(np.real(np.asarray(spectra[key](wf)))) - S0
    return out


def analytic_spectra(gamma, gamma_12):
    """Ground-truth spectrum callables S(w) for the active regime (exact systematic)."""
    from qns2q.noise.spectra import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
    return {
        'S11':   lambda w: np.asarray(S_11(jnp.asarray(w))),
        'S22':   lambda w: np.asarray(S_22(jnp.asarray(w))),
        'S1212': lambda w: np.asarray(S_1212(jnp.asarray(w))),
        'S12':   lambda w: np.asarray(S_1_2(jnp.asarray(w), gamma)),
        'S112':  lambda w: np.asarray(S_1_12(jnp.asarray(w), gamma_12)),
        'S212':  lambda w: np.asarray(S_2_12(jnp.asarray(w), gamma_12 - gamma)),
    }


def selfconsistent_spectra(wk_full, recon):
    """Spectrum callables built from the reconstructed comb (no ground-truth knowledge).

    The reconstruction yields S only at the comb points wk_full = [0, w_1, ..., w_kmax].
    To feed the forward integral we must model S on the *continuous* axis, including the
    unsampled bands. We use a piecewise-linear interpolation through the reconstructed
    points, extended to 0 at w=0..(DC point) and decaying linearly to 0 from w_kmax over
    one extra comb cutoff. This is the deliberately-simple, assumption-light model whose
    only purpose is to show the systematic is *estimable* without the answer; it is not
    used for the quoted figure bars.
    """
    wk_full = np.asarray(wk_full)
    wmax = wk_full[-1]

    def make(key):
        vals = recon[key]
        re = np.real(vals); im = np.imag(vals)

        def fn(w):
            w = np.asarray(w, dtype=float)
            r = np.interp(w, wk_full, re, left=re[0], right=0.0)
            # linear taper to zero over (wmax, 2*wmax) so out-of-band weight is bounded
            taper = np.clip(1.0 - (w - wmax) / wmax, 0.0, 1.0)
            r = np.where(w > wmax, re[-1] * taper, r)
            if np.allclose(im, 0.0):
                return r
            ii = np.interp(w, wk_full, im, left=im[0], right=0.0)
            ii = np.where(w > wmax, im[-1] * taper, ii)
            return r + 1j * ii
        return fn

    return {k: make(k) for k in ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212')}
