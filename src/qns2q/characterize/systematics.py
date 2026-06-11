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
                  n_wfine=300001, wmax_fid_factor=5.0, wmax_leak_factor=1.5):
    # wmax_*_factor are ordinary-frequency integration cutoffs in tau units
    # (cycles per tau): 5.0 and 1.5 = the legacy 200 MHz and 60 MHz at tau = 25 ns.
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


# ==============================================================================
# Faithful forward-model (comb-inversion) systematic
# ==============================================================================
#
# ``comb_inversion_systematic`` above approximates the forward map with a
# SINGLE-PERIOD filter and a continuous integral. That under-quotes the bias on
# the antisymmetric cross channels (the ['CDD3','CPMG'] imaginary channels), whose
# off-tooth/grating weight the single-period kernel misses -- e.g. for S_1_2 the
# true comb-inversion bias on Im is ~25%, but the single-period proxy reports ~half.
#
# This block computes the GENUINE forward observable. ``trajectories.make_propagator``
# is exact pure dephasing (diagonal H, phase by trapezoid), so every measured
# correlator is exactly a fixed combination of the toggled-phase covariances
#
#     Cov(Phi_a, Phi_b),   Phi_a = \int_0^{MT} y_a(t) b_a(t) dt,
#
# evaluated with the FULL M-rep toggles over the DISCRETE noise-synthesis grid --
# i.e. the very quantity the simulator averages. The d(+/-) / a(+/-) extraction
# combinations reduce to (validated against the simulated data to within shot noise):
#
#     C_12_0  = 0.5 (Var Phi_1 + Var Phi_2)     (selfs; qubit-qubit cross + Ising cancel)
#     C_12_12 =      Cov(Phi_1, Phi_2)          (qubit-qubit cross)
#     C_a_0   = 0.5 (Var Phi_l + Var Phi_12)    (self-l + Ising self)
#     C_a_b   =      Cov(Phi_l, Phi_12)         (qubit-l <-> Ising cross)
#
# Channels: 1 -> (S11, toggle y[0,0]); 2 -> (S22, y[1,1]); 12 -> (S1212, y[2,2] =
# y[0,0]*y[1,1]). Cross weights use the model's TRUE complex cross-spectra:
# Cov(Phi_a, Phi_b) = sum_j (dw/pi) Re[S_ab(w_j) ff_a(w_j) conj(ff_b(w_j))], which
# is exact for the component synthesis of model.trajectories (and reduces to the
# legacy sqrt(S_a S_b) e^{-i w dgamma} rule for the old shared-draw model).

# observable name -> (kind, [pulse_q1, pulse_q2], l)   -- mirrors experiments.py
_FWD_OBS = {
    'C_12_0_MT_1':  ('C120',  ['CPMG', 'CPMG'],      None),
    'C_12_0_MT_2':  ('C120',  ['CDD3', 'CPMG'],      None),
    'C_12_0_MT_3':  ('C120',  ['CPMG', 'CDD3'],      None),
    'C_12_0_MT_4':  ('C120',  ['CDD1', 'CDD1'],      None),
    'C_12_12_MT_1': ('C1212', ['CPMG', 'CPMG'],      None),
    'C_12_12_MT_2': ('C1212', ['CDD3', 'CPMG'],      None),
    'C_1_0_MT_1':   ('Ca0',   ['CDD1', 'CDD1-1/2'],  1),
    'C_2_0_MT_1':   ('Ca0',   ['CDD1-1/2', 'CDD1'],  2),
    'C_1_2_MT_1':   ('Cab',   ['CPMG', 'FID'],       1),
    'C_1_2_MT_2':   ('Cab',   ['CPMG', 'CDD1-1/4'],  1),
    'C_2_1_MT_1':   ('Cab',   ['FID', 'CPMG'],       2),
    'C_2_1_MT_2':   ('Cab',   ['CDD1-1/4', 'CPMG'],  2),
}

_CH_TOGGLE = {1: (0, 0), 2: (1, 1), 12: (2, 2)}


def forward_observables(spectra, c_times, M, T, t_vec, w_grain, wmax,
                        chunk=2000):
    """Exact deterministic 2nd-cumulant value of every harmonic QNS observable.

    Returns ``{obs_name: ndarray(n_ctimes)}`` for the names in ``_FWD_OBS``. The
    values are the noiseless (infinite-shot) forward-model observables for the
    ``spectra`` (an ``analytic_spectra`` dict), computed with the full M-rep toggles
    on the experiment's time grid ``t_vec`` and summed over the discrete synthesis
    grid -- identical to what ``trajectories.solver_prop`` averages.
    """
    c_times = np.asarray(c_times)
    n = len(c_times)
    t_vec = np.asarray(t_vec)
    t_j = jnp.asarray(t_vec)
    tgrain = t_vec.size // M
    tb_base = np.linspace(0, T, tgrain)

    size_w = 2 * w_grain
    dw = wmax / w_grain
    wj = (np.arange(size_w) + 0.5) * (2 * wmax) / size_w        # midpoint synthesis grid
    Sw = {1: np.real(np.asarray(spectra['S11'](wj))),
          2: np.real(np.asarray(spectra['S22'](wj))),
          12: np.real(np.asarray(spectra['S1212'](wj)))}
    Sx = {(1, 2): np.asarray(spectra['S12'](wj)),
          (1, 12): np.asarray(spectra['S112'](wj)),
          (2, 12): np.asarray(spectra['S212'](wj))}

    def cov(Ga, cha, Gb, chb):
        # Cov(Phi_a, Phi_b) = sum_j (dw/pi) Re[ S_ab(w_j) ff_a(w_j) ff_b(w_j)* ];
        # for a == b this is the usual (dw/pi) S_aa |ff_a|^2.
        if cha == chb:
            return float(np.sum((dw / np.pi) * Sw[cha] * np.real(np.conj(Ga) * Gb)))
        W = Sx[(cha, chb)]
        return float(np.sum((dw / np.pi) * np.real(W * Ga * np.conj(Gb))))

    out = {k: np.zeros(n) for k in _FWD_OBS}
    for i in range(n):
        y_cache = {}        # pulse-tuple -> make_y result
        G_cache = {}        # (pulse-tuple, ch) -> G(w_j)

        def getG(pulse, ch):
            key = (tuple(pulse), ch)
            if key not in G_cache:
                pk = tuple(pulse)
                if pk not in y_cache:
                    y_cache[pk] = make_y(tb_base, list(pulse), ctime=float(c_times[i]), m=M)
                toggle = np.asarray(y_cache[pk][_CH_TOGGLE[ch]])
                # ff_a(w) = \int toggle(t) e^{i w t} dt   (full record; the
                # cross-spectrum phases live in S_ab, not in per-channel shifts)
                G_cache[key] = _ff_grid(toggle, t_j, wj, +1, chunk=chunk)
            return G_cache[key]

        for name, (kind, pulse, l) in _FWD_OBS.items():
            if kind == 'C120':
                v = 0.5 * (cov(getG(pulse, 1), 1, getG(pulse, 1), 1)
                           + cov(getG(pulse, 2), 2, getG(pulse, 2), 2))
            elif kind == 'C1212':
                v = cov(getG(pulse, 1), 1, getG(pulse, 2), 2)
            elif kind == 'Ca0':
                v = 0.5 * (cov(getG(pulse, l), l, getG(pulse, l), l)
                           + cov(getG(pulse, 12), 12, getG(pulse, 12), 12))
            else:  # 'Cab'
                v = cov(getG(pulse, l), l, getG(pulse, 12), 12)
            out[name][i] = v
    return out


def forward_model_systematic(spectra, c_times, M, T, t_vec,
                             w_grain, wmax, inv_opts=None):
    """Honest comb-inversion systematic for the harmonic reconstruction points.

    Computes the exact deterministic forward observables for the analytic ``spectra``
    (``forward_observables``), runs them through the SAME reconstruction kernels the
    pipeline uses, and returns the residual ``recon(C_fwd) - S_theory(w_k)`` as the
    per-spectrum bias: real arrays for 'S11'/'S22'/'S1212', complex (Re+iIm) for
    'S12'/'S112'/'S212'. Supersedes ``comb_inversion_systematic`` -- it captures the
    full finite-M comb response (not just the single-period proxy), so it does not
    under-quote the antisymmetric (CDD3) cross-channel bias.
    """
    from qns2q.characterize.inversion import (recon_S_11, recon_S_22, recon_S_12_12,
                                              recon_S_1_2, recon_S_1_12, recon_S_2_12)
    opts = dict(inv_opts or {})
    opts['diagnostics'] = False        # never spam diagnostics from inside the systematic
    C = forward_observables(spectra, c_times, M, T, t_vec, w_grain, wmax)
    kw = dict(c_times=np.asarray(c_times), m=M, T=T, **opts)

    rec = {
        'S11':   recon_S_11([C['C_12_0_MT_1'], C['C_12_0_MT_2']], **kw),
        'S22':   recon_S_22([C['C_12_0_MT_1'], C['C_12_0_MT_3']], **kw),
        'S1212': recon_S_12_12([C['C_1_0_MT_1'], C['C_2_0_MT_1'], C['C_12_0_MT_4']], **kw),
        'S12':   recon_S_1_2([C['C_12_12_MT_1'], C['C_12_12_MT_2']], **kw),
        'S112':  recon_S_1_12([C['C_1_2_MT_1'], C['C_1_2_MT_2']], **kw),
        'S212':  recon_S_2_12([C['C_2_1_MT_1'], C['C_2_1_MT_2']], **kw),
    }
    wk = np.array([2 * np.pi * (k + 1) / T for k in range(len(c_times))])
    theory = {'S11': spectra['S11'](wk), 'S22': spectra['S22'](wk),
              'S1212': spectra['S1212'](wk), 'S12': spectra['S12'](wk),
              'S112': spectra['S112'](wk), 'S212': spectra['S212'](wk)}

    sys = {}
    for key in ('S11', 'S22', 'S1212'):
        sys[key] = np.real(np.asarray(rec[key])) - np.real(np.asarray(theory[key]))
    for key in ('S12', 'S112', 'S212'):
        d = np.asarray(rec[key]) - np.asarray(theory[key])
        sys[key] = np.real(d) + 1j * np.imag(d)
    return sys


_ECHO_DC_FILTER_CACHE = {}


def _echo_dc_filters(t_tuple, ct, n_wl=80001, wmax=12.5):
    """|F(w,t_k)|^2 grids for the double-echo Ising-DC observables.

    Cached: the filters depend only on the sweep times and echo cycle time, not on
    the spectra, and the self-consistent unfold evaluates the same mirror
    repeatedly. Returns (wl, |F1_CDD1|^2, |F12_CDD1xCDD1|^2, |F12_CDD1xCPMG|^2),
    each (n_t, n_wl); the y_1 filter is identical in both arms by construction."""
    key = (t_tuple, float(ct), n_wl, wmax)
    if key in _ECHO_DC_FILTER_CACHE:
        return _ECHO_DC_FILTER_CACHE[key]
    from qns2q.model.trajectories import make_y
    wl = np.linspace(wmax / n_wl, wmax, n_wl)
    F1sq, F12b_sq, F12r_sq = [], [], []
    for tk in t_tuple:
        n_tb = max(4000, int(40 * tk / ct))
        tb = np.linspace(0.0, float(tk), n_tb)
        yb = make_y(tb, ['CDD1', 'CDD1'], ctime=ct, m=1)
        yr = make_y(tb, ['CDD1', 'CPMG'], ctime=ct, m=1)
        F1sq.append(np.abs(_ff_grid(np.asarray(yb[0, 0]), jnp.asarray(tb), wl, +1)) ** 2)
        F12b_sq.append(np.abs(_ff_grid(np.asarray(yb[2, 2]), jnp.asarray(tb), wl, +1)) ** 2)
        F12r_sq.append(np.abs(_ff_grid(np.asarray(yr[2, 2]), jnp.asarray(tb), wl, +1)) ** 2)
    out = (wl, np.array(F1sq), np.array(F12b_sq), np.array(F12r_sq))
    _ECHO_DC_FILTER_CACHE[key] = out
    return out


def dc_fit_systematic(spectra, t_sweep, n_w=400001, wmax=12.5, s1212_echo_ct=None,
                      s1212_echo_obs_err=None, s1212_echo_wmax=None):
    # wmax is an angular-frequency integration cutoff in tau units (rad/tau):
    # 12.5 = the legacy 5e8 rad/s at tau = 25 ns.
    """Deterministic bias of the multi-time DC slope fit, per spectrum.

    Mirrors ``inversion._ramsey_fit_dc``: builds the EXACT forward FID-decay curves
    C(t_k) over the sweep for the analytic spectra, runs them through the same DC
    reconstructors, and returns ``recon_dc(C_fwd) - S(0)`` per spectrum. For the
    self/cross spectra whose noise reaches the motional-narrowing regime this bias is
    tiny (~0.1%); for quasi-static / sub-comb-cusp noise (e.g. a 1/|w| Ising spectrum)
    the linear regime is never reached and the bias is large -- exactly the inflated
    bar the flagged DC point should carry. (FID filter |F(w,t)|^2 = sin^2(wt/2)/(w/2)^2;
    self C(t)=(1/2pi)int S|F|^2, cross C(t)=(1/pi)int Re[S_ab(w)]|F|^2.)

    ``s1212_echo_ct``: when the run carries the double-echo Ising-DC observables
    (C_1_0_CDD1CDD1 / C_1_0_CDD1CPMG at echo cycle time ct), pass that ct so the
    S1212 bias mirrors ``recon_S_1212_dc_echo`` (dominant systematic: the
    reference arm's mixed-filter S_1212 pickup) instead of the legacy FF combo.
    ``s1212_echo_obs_err``: the run's per-point error arrays [err_both, err_ref]
    for those observables. The echo signal sits far below the absolute c_min
    floor of the no-error fit path, so the mirror MUST window the exact curves
    the same (SNR-based) way the data fit does -- without this the mirror quotes
    the bias of a different estimator (validated: -27% phantom bias at 16k
    shots where the data fit is accurate to ~1%).
    ``s1212_echo_wmax``: the run's noise-synthesis cutoff (params wmax =
    2*pi*truncate/T). The echo and mixed-filter passbands sit ABOVE that cutoff,
    where the simulated world has no spectral weight at all -- integrating the
    mirror to the generic 12.5 charges the reference arm for pickup that does
    not exist in the run (validated: -23% phantom bias). A real experiment
    re-acquires that term as a genuine out-of-band systematic to be quoted from
    a spectral-tail assumption.
    """
    from qns2q.characterize.inversion import (recon_S_11_dc, recon_S_22_dc, recon_S_1212_dc,
                                              recon_S_1212_dc_echo,
                                              recon_S_1_2_dc, recon_S_1_12_dc, recon_S_2_12_dc)
    t = np.asarray(t_sweep, dtype=float)
    w = np.linspace(wmax / n_w, wmax, n_w)
    F2 = np.sin(w[None, :] * t[:, None] / 2) ** 2 / (w[None, :] / 2) ** 2   # (n_t, n_w)

    def Sr(key):
        return np.real(np.asarray(spectra[key](w)))
    S = {k: Sr(k) for k in ('S11', 'S22', 'S1212')}

    def c_self(key):
        return (1 / (2 * np.pi)) * np.trapezoid(S[key][None, :] * F2, w, axis=1)

    def c_cross(key):
        g = np.real(np.asarray(spectra[key](w)))
        return (1 / np.pi) * np.trapezoid(g[None, :] * F2, w, axis=1)

    # Residual Ising leak through the CDD3 partner of the self-DC observables: the high-
    # order CDD3 nulls most of S_1212 but not all, biasing C_1_0_FIDCDD3 = Cself(S11) +
    # 1/2 Var(Phi_12 through CDD3). Built from the actual CDD3 toggle over each t_k.
    from qns2q.model.trajectories import make_y
    Tper = float(t[0])                                   # sweep starts at m=1 -> t = T
    dc_ct = Tper / 8.0
    wl = np.linspace(wmax / 40001, wmax, 40001)
    S1212_l = np.real(np.asarray(spectra['S1212'](wl)))

    def leak(tk):
        tb = np.linspace(0.0, float(tk), 4000)
        y = make_y(tb, ['FID', 'CDD3'], ctime=dc_ct, m=1)
        FI = _ff_grid(np.asarray(y[2, 2]), jnp.asarray(tb), wl, +1)
        return (1 / (2 * np.pi)) * np.trapezoid(S1212_l * np.abs(FI) ** 2, wl)

    leak_arr = np.array([leak(tk) for tk in t])

    # forward DC observables over the sweep (match experiments.py recipes)
    cs11, cs22, cs1212 = c_self('S11'), c_self('S22'), c_self('S1212')
    C_10 = cs11 + leak_arr                               # C_1_0_FIDCDD3 (+ residual Ising leak)
    C_20 = cs22 + leak_arr                               # C_2_0_CDD3FID
    C_10ff = cs11 + cs1212                               # C_1_0_FIDFID = Cself11 + Cself1212
    C_20ff = cs22 + cs1212                               # C_2_0_FIDFID
    C_120 = cs11 + cs22                                  # C_12_0_FID_FID = Cself11 + Cself22
    C_1212 = c_cross('S12')                              # C_12_12_FID
    C_112 = c_cross('S112')                              # C_1_12_FID
    C_212 = c_cross('S212')                              # C_2_12_FID

    s11, _, _ = recon_S_11_dc([C_10], t_sweep=t)
    s22, _, _ = recon_S_22_dc([C_20], t_sweep=t)
    if s1212_echo_ct is not None:
        # Mirror the double-echo estimator: exact C_1_0 curves under
        # ['CDD1','CDD1'] and ['CDD1','CPMG'] (partner-averaged, so variances add
        # and the S_1_12 cross term cancels), then the same difference fit.
        wle, F1sq, F12b_sq, F12r_sq = _echo_dc_filters(
            tuple(float(x) for x in t), float(s1212_echo_ct),
            wmax=float(s1212_echo_wmax) if s1212_echo_wmax else wmax)
        S11_e = np.real(np.asarray(spectra['S11'](wle)))
        S1212_e = np.real(np.asarray(spectra['S1212'](wle)))
        v1_e = (1 / (2 * np.pi)) * np.trapezoid(S11_e[None, :] * F1sq, wle, axis=1)
        C_eb = v1_e + (1 / (2 * np.pi)) * np.trapezoid(S1212_e[None, :] * F12b_sq, wle, axis=1)
        C_er = v1_e + (1 / (2 * np.pi)) * np.trapezoid(S1212_e[None, :] * F12r_sq, wle, axis=1)
        s1212, _, _ = recon_S_1212_dc_echo([C_eb, C_er], t_sweep=t,
                                           obs_err=s1212_echo_obs_err)
    else:
        s1212, _, _ = recon_S_1212_dc([C_10ff, C_20ff, C_120], t_sweep=t)
    s12, _, _ = recon_S_1_2_dc([C_1212], t_sweep=t)
    s112, _, _ = recon_S_1_12_dc([C_112], t_sweep=t)
    s212, _, _ = recon_S_2_12_dc([C_212], t_sweep=t)

    def truth(key):
        return float(np.real(spectra[key](np.array([0.0]))[0]))
    return {'S11': s11 - truth('S11'), 'S22': s22 - truth('S22'),
            'S1212': s1212 - truth('S1212'), 'S12': s12 - truth('S12'),
            'S112': s112 - truth('S112'), 'S212': s212 - truth('S212')}


def analytic_spectra():
    """Ground-truth spectrum callables S(w) for the active regime (exact systematic)."""
    from qns2q.noise.spectra import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
    return {
        'S11':   lambda w: np.asarray(S_11(jnp.asarray(w))),
        'S22':   lambda w: np.asarray(S_22(jnp.asarray(w))),
        'S1212': lambda w: np.asarray(S_1212(jnp.asarray(w))),
        'S12':   lambda w: np.asarray(S_1_2(jnp.asarray(w))),
        'S112':  lambda w: np.asarray(S_1_12(jnp.asarray(w))),
        'S212':  lambda w: np.asarray(S_2_12(jnp.asarray(w))),
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
