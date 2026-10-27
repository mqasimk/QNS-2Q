"""
Spectral inversion and reconstruction algorithms.

Where this sits in the pipeline
--------------------------------
This is the linear-algebra core of Stage 2 (spectral reconstruction), the second
step of the "characterize" arm of the two-arm pipeline described in the repo's
CLAUDE.md (characterize: QNS experiments -> spectral reconstruction; control: CZ/
idle gate optimization). It does not run experiments or hold noise-model physics
itself -- it turns already-measured correlation-function *coefficients* (the
outputs of running QNS pulse sequences many times, at a repetition count ``M``
and per-repetition time ``T``) into estimates of the noise power spectral density
(PSD) matrix, at either a comb of harmonic frequencies or at zero frequency (DC).

Concretely, every QNS experiment applies a control ("toggling") sequence to the
qubits and measures a correlation function ``C`` that is a *linear functional* of
the true spectrum ``S(w)`` (exact here because the simulated noise is Gaussian --
see ``characterize/systematics.py`` for why the 2nd-order cumulant expansion is
exact). Discretizing that linear functional at a set of control times gives a
square (or overdetermined) linear system ``U @ S = C``; this module builds ``U``
from the pulse sequences (via ``ff()``/``model.trajectories.make_y``) and solves
for ``S`` (``solve_inverse``), propagating the observables' error bars through the
same linear map (``propagate_linear_error``). Two families of solvers are
provided: the harmonic ``recon_S_*`` functions (comb frequencies omega_k = 2*pi*k/T)
and the DC ``recon_S_*_dc`` / ``_ramsey_fit_dc`` functions (zero frequency, from
the slope of a Ramsey/FID decay curve swept over total evolution time).

Callers and companions
-----------------------
- ``characterize/experiments.py`` runs the QNS pulse sequences and produces the
  raw correlation coefficients (``C_12_0_MT_1`` etc.) that feed every function
  here.
- ``characterize/reconstruct.py`` is the sole caller of the six harmonic
  ``recon_S_*`` functions and the six DC ``recon_S_*_dc``/``recon_S_1212_dc*``
  functions (see its ``call_recon``/``call_recon_dc`` helpers); it assembles their
  outputs into the ``specs.npz`` file that ``control/cz.py`` and ``control/idle.py``
  read as the reconstructed noise spectrum for pulse optimization.
- ``characterize/systematics.py`` independently re-derives the same kernels
  (``ff``-style Fourier overlaps) to compute the *forward-model* bias of this
  comb inversion analytically, without Monte Carlo; its docstring explains the
  three sources of that bias (finite-M tooth width, comb truncation, unsampled
  sub-comb band). Its test (``tests/test_systematics.py``) guards against the two
  modules' kernels silently drifting apart.
- ``characterize/spam.py`` calls ``regress_observables_over_M`` to remove a
  SPAM-induced offset via M-scaling before handing coefficients to the
  ``recon_S_*`` functions above.

Two families of variable names recur across this file's functions and mirror the
manuscript's notation directly (do not rename them): ``c_times`` (the swept
control/block times, one per reconstructed harmonic), ``m``/``M`` (repetition
count of the pulse block), ``T`` (time per repetition), ``wk`` (the harmonic comb
``2*pi*k/T``), and ``y``/``y_uv`` (the +-1 "toggling function" control matrix from
``make_y`` that flips sign every time a pulse fires -- see ``ff()`` below for what
it is used for).
"""

import numpy as np
from qns2q.model.trajectories import make_y


# ==============================================================================
# Inversion backends (selectable; default 'direct' preserves legacy behavior)
# ==============================================================================
#
# Every harmonic reconstruction solves a linear system  U @ S = C  for the
# spectral samples S.  The legacy code uses a square, exactly-determined U and
# np.linalg.inv -- which amplifies observation noise by cond(U) and can return
# unphysical (negative) self-spectra.  ``solve_inverse`` adds robustness options
# that the caller selects via the ``inversion_method`` kwarg:
#
#   'direct'   : S = inv(U) @ C                       (legacy; U must be square)
#   'lstsq'    : S = pinv(U) @ C                      (works for tall/overdetermined U)
#   'tikhonov' : S = (U^H U + lam I)^-1 U^H @ C       (ridge-regularized)
#   'nnls'     : non-negative least squares           (self-spectra only; S >= 0)
#
# All paths return (S, J) where J is the *effective linear map* (S = J @ C) used
# for analytic error propagation via ``propagate_linear_error``.  For 'nnls' the
# map is nonlinear at active constraints, so J is the unconstrained least-squares
# map -- error bars are then a (slightly conservative) approximation.
#
# Plain-language recap for readers new to numerical linear algebra: cond(U) (the
# matrix condition number, ratio of its largest to smallest singular value) says
# how much a small measurement error in C gets amplified when you divide it out
# again to get S -- a large cond(U) means a noisy experiment can flip the sign of
# a physically-required-positive self-spectrum. 'lstsq'/'tikhonov'/'nnls' exist as
# ways to trade a little bias for a lot less noise amplification when that happens;
# 'direct' (exact inversion of a square U) is what the published reconstructions use
# and is kept as the default so old results reproduce exactly.

def solve_inverse(U, rhs, method='direct', reg_lambda=0.0, nonneg=False,
                  diagnostics=False, label=''):
    """Solve the linear system U @ S = rhs for the spectral samples S.

    This is the shared backend behind every ``recon_S_*`` harmonic reconstructor
    below: they each build a kernel matrix ``U`` (from filter-function overlaps,
    see ``ff()``) and a measured coefficient vector ``rhs``, then call this
    function to get the spectrum samples back out. See the module-level comment
    above this function for what each ``method`` does and when to reach for it.

    Parameters
    ----------
    U : array_like
        The (n, n) [or (n_obs, n) if overdetermined] kernel/measurement matrix
        mapping true spectral samples to the noiseless coefficient.
    rhs : array_like
        The measured coefficient vector C (the right-hand side of U @ S = C).
    method : {'direct', 'lstsq', 'tikhonov'}
        Which linear-solve backend to use (ignored if ``nonneg=True``).
    reg_lambda : float
        Ridge/Tikhonov regularization strength (only used by ``method='tikhonov'``).
    nonneg : bool
        If True, solve via non-negative least squares instead of ``method``
        (physically appropriate for self-spectra, which cannot be negative).
    diagnostics : bool
        If True, print U's shape, condition number, and extreme singular values
        (a quick way to see whether a given experiment design is numerically
        well-posed) before solving.
    label : str
        Human-readable tag included in the diagnostics printout (e.g. 'S_11').

    Returns
    -------
    (S, J) : tuple of np.ndarray
        ``S`` is the solved spectral sample vector; ``J`` is the effective linear
        map with ``S = J @ rhs``, used by ``propagate_linear_error`` to turn
        observable error bars into spectral error bars.
    """
    U = np.asarray(U)
    rhs = np.asarray(rhs)

    if diagnostics:
        try:
            cond = np.linalg.cond(U)
        except np.linalg.LinAlgError:
            cond = np.inf
        sv = np.linalg.svd(U, compute_uv=False)
        print(f"    [diag] {label or 'U'}: shape={U.shape} cond={cond:.3e} "
              f"sigma_max={sv[0]:.3e} sigma_min={sv[-1]:.3e}")

    if nonneg:
        # Non-negative LS for real self-spectra. Operate on the real part.
        # (Imported here, not at module top, purely so this optional scipy
        # dependency is only touched by code paths that actually request it.)
        from scipy.optimize import nnls
        S, _ = nnls(np.real(U), np.real(rhs))
        J = np.linalg.pinv(U)            # for (approximate) error propagation
        return S.astype(U.dtype), J

    if method == 'direct':
        J = np.linalg.inv(U)
        return J @ rhs, J
    if method == 'lstsq':
        J = np.linalg.pinv(U)
        return J @ rhs, J
    if method == 'tikhonov':
        UH = U.conj().T
        J = np.linalg.inv(UH @ U + reg_lambda * np.eye(U.shape[1], dtype=U.dtype)) @ UH
        return J @ rhs, J
    raise ValueError(f"Unknown inversion_method '{method}'")


def _inv_opts(kwargs):
    """Extract the inversion-backend options from a recon function's kwargs."""
    return dict(
        method=kwargs.get('inversion_method', 'direct'),
        reg_lambda=kwargs.get('reg_lambda', 0.0),
        diagnostics=kwargs.get('diagnostics', False),
    )


def regress_observables_over_M(obs_by_M, M_values, m_ref):
    """SPAM-free M-scaling regression of a single comb observable.

    The comb coefficient scales as  C(MT) ~ (M/T) sum_k U'_k S_k + b , where the
    M-independent intercept b absorbs the SPAM term and the O(1/M) comb-collapse
    systematic.  Fitting C(M_i) linearly in M_i and keeping the SLOPE removes
    both.  The returned ``C_eff = slope * m_ref`` is the SPAM-free coefficient at
    the reference repetition ``m_ref`` (the M used to build U in the recon_*
    functions), so it can be dropped straight into the existing reconstructors.

    Plain-language version: SPAM = State Preparation And Measurement error, a
    roughly constant offset added to every measured coefficient regardless of how
    long the experiment runs. The true noise signal grows in proportion to the
    number of pulse-block repetitions M, but the SPAM offset does not -- so
    repeating the same experiment at several different M and fitting a straight
    line C vs. M lets you throw away the (M-independent) intercept and keep only
    the physically meaningful slope. This is the mechanism behind the
    ``spam_protocol='robust'`` estimators in ``characterize/spam.py`` (see
    CLAUDE.md's SPAM-pipeline section).

    Parameters
    ----------
    obs_by_M : dict[int, array_like]   mapping M -> observable vector (over c_times)
    M_values : sequence of int          the M's to regress over (>= 2, all >> 1)
    m_ref : int                          repetition at which U is constructed

    Returns
    -------
    (C_eff, slope_err) : the SPAM-free coefficient and the per-point slope stderr.
    """
    M = np.asarray(sorted(M_values), dtype=float)
    if M.size < 2:
        raise ValueError("M-scaling regression needs >= 2 distinct M values.")
    Y = np.array([np.asarray(obs_by_M[int(m)]) for m in M])     # (n_M, n_ctimes)
    # Per-c_time linear fit C = slope*M + intercept (least squares, vectorized).
    A = np.vstack([M, np.ones_like(M)]).T                       # (n_M, 2)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)                # (2, n_ctimes)
    slope = coef[0]
    resid = Y - A @ coef
    dof = max(M.size - 2, 1)
    s2 = np.sum(resid ** 2, axis=0) / dof
    Sxx = np.sum((M - M.mean()) ** 2)
    slope_err = np.sqrt(s2 / Sxx)
    return slope * m_ref, slope_err * m_ref


def truncation_bias_estimate(spec_fn, T, truncate, wmax_factor=8.0, args=()):
    """Estimate the fraction of spectral weight beyond the comb cutoff omega_kmax.

    The comb reconstructs S at omega_k = 2*pi*k/T, k=1..truncate; weight above
    omega_kmax = 2*pi*truncate/T is unsampled and biases the overlap integrals.
    Returns (weight_above / weight_total) using the analytic spectrum spec_fn.

    This is a diagnostic-only sanity check, not part of the reconstruction itself:
    ``characterize/reconstruct.py`` calls it (only when its ``diagnostics`` flag is
    on) to print how much of the true spectrum's weight the harmonic comb never
    samples at all -- a large fraction is a warning that ``truncate`` (the number
    of harmonics kept) or ``T`` should be revisited for a given noise spectrum.
    """
    w_cut = 2 * np.pi * truncate / T
    w = np.linspace(0, wmax_factor * w_cut, 200001)
    Sw = np.real(spec_fn(w, *args))
    total = np.trapezoid(Sw, w)
    above = np.trapezoid(Sw[w > w_cut], w[w > w_cut])
    return float(above / total) if total > 0 else 0.0


def ff(y, t, w):
    """
    Calculate the filter function (Fourier transform of the toggle function).

    Physics background: a dynamical-decoupling/QNS pulse sequence can be encoded
    as a "toggling function" y(t) that is +1 or -1 depending on which side of the
    accumulated pulses the qubit's phase currently sits on (built by
    ``model.trajectories.make_y``). The measured decoherence/correlation signal is
    a linear (here exact, since the noise is Gaussian) functional of the true
    noise spectrum S(w), weighted by |ff(y, t, w)|^2 -- this is the "filter
    function" of the QNS/dynamical-decoupling literature. Every ``U`` matrix built
    below is just this ``ff`` evaluated at the harmonic frequencies for a specific
    pulse sequence, so this one function is the shared building block of every
    kernel matrix in this module.

    Parameters
    ----------
    y : array_like
        Toggle function values (+-1 valued; from ``make_y``).
    t : array_like
        Time vector.
    w : float
        Frequency.

    Returns
    -------
    complex
        Filter function value at frequency `w`.
    """
    return np.trapezoid(np.exp(1j*w*t)*y, t)

# The four f1_* helpers below are single-qubit filter-function shortcuts (they
# call ff()/make_y() internally so callers don't have to build the time grid and
# toggle function themselves). Only f1_fid is currently exercised by anything in
# the repo (tests/test_spectral_inversion.py); f1_cpmg/f1_cdd1/f1_cdd3 have no
# in-repo callers today but are kept as ready-made single-qubit reference filter
# functions for future diagnostics (analogous to the ones systematics.py builds
# for the two-qubit combinations actually used in reconstruction).

def f1_cpmg(ct, T, w):
    """
    Calculate the single-qubit CPMG filter function value.

    Parameters
    ----------
    ct : float
        Control time for one block.
    T : float
        Total experiment time.
    w : float
        Frequency.

    Returns
    -------
    complex
        Filter function value.
    """
    y = make_y(np.linspace(0, T, 10**5), ['CPMG', 'CPMG'], ctime=ct, m=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapezoid(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def f1_fid(T, w):
    """
    Calculate the single-qubit FID filter function value.

    FID ("free induction decay") means no decoupling pulses at all, so the toggle
    function is identically +1 and this reduces to a plain Fourier transform of a
    constant -- the simplest possible filter function, used as a baseline/sanity
    check for the others (see the ``w=0`` test in ``tests/test_spectral_inversion.py``,
    where it should equal exactly ``T``).

    Parameters
    ----------
    T : float
        Total time.
    w : float
        Frequency.

    Returns
    -------
    complex
        Filter function value.
    """
    t_vec = np.linspace(0, T, 10**5)
    return np.trapezoid(np.exp(1j*w*t_vec), t_vec)

def f1_cdd1(ct, T, w):
    """
    Calculate the single-qubit CDD1 filter function value.

    Parameters
    ----------
    ct : float
        Control time.
    T : float
        Total time.
    w : float
        Frequency.

    Returns
    -------
    complex
        Filter function value.
    """
    y = make_y(np.linspace(0, T, 10**5), ['CDD1', 'CDD1'], ctime=ct, m=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapezoid(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def f1_cdd3(ct, T, w):
    """
    Calculate the single-qubit CDD3 filter function value.

    Parameters
    ----------
    ct : float
        Control time.
    T : float
        Total time.
    w : float
        Frequency.

    Returns
    -------
    complex
        Filter function value.
    """
    # CDD3's toggling function is perfectly balanced (equal total time on each
    # side of the phase flip) by construction, so its DC (w=0) filter-function
    # value is exactly zero; short-circuiting here avoids doing a full numerical
    # integration just to recover that known-exact answer.
    if w == 0:
        return 0
    y = make_y(np.linspace(0, T, 10**5), ['CDD3', 'CDD3'], ctime=ct, m=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapezoid(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def propagate_linear_error(A_inv, obs_err):
    """
    Propagates independent observation errors through linear system S = A_inv @ C.
    Sigma_S_i = sqrt( sum_j |A_inv_ij|^2 * Sigma_C_j^2 )

    Plain-language version: this is the standard "propagation of errors" formula
    for a linear transformation (here, the spectrum-from-coefficients inversion
    ``S = A_inv @ C``) of independent, Gaussian-distributed measurement errors --
    each output error bar is a quadrature sum (root of sum of squares) of the input
    error bars, weighted by how much the inversion matrix amplifies each one. It
    assumes the C_j errors are independent of each other; every ``recon_S_*``
    function below calls this once per real/imaginary channel with ``A_inv`` set
    to the ``J`` returned by ``solve_inverse``.

    Args:
        A_inv: (N, N) inverse matrix.
        obs_err: (N,) array of observation standard errors.

    Returns:
        (N,) array of propagated errors.
    """
    # |A_inv|^2
    A_sq = np.abs(A_inv)**2
    # obs_err^2
    obs_var = np.array(obs_err)**2
    # Sigma_S^2 = A_sq @ obs_var
    S_var = A_sq @ obs_var
    return np.sqrt(S_var)


def recon_S_11(coefs, **kwargs):
    """
    Reconstruct the self-spectrum S_11 at harmonic frequencies.

    Physics of the combination (general pattern -- see below for how the other
    five ``recon_S_*`` functions in this file reuse it): ``C_12_0_MT_1`` and
    ``C_12_0_MT_2`` are the same two-qubit Pauli-correlator combination
    (``model.observables.c_12_0_mt_i``, computed by ``characterize/experiments.py``)
    measured under two pulse sequences that are IDENTICAL on qubit 2 (CPMG both
    times) and differ only on qubit 1 (CPMG vs. CDD3). Because qubit 2's
    contribution is the same in both measurements, it cancels in the difference
    ``C_12_0_MT_1 - C_12_0_MT_2``, leaving a quantity that depends only on qubit
    1's noise spectrum S_11 -- the harmonic-comb analogue of how the DC estimator
    ``recon_S_11_dc`` isolates S_11 via a decoupled partner qubit. The linear
    system solved here is exactly that difference expressed as ``U @ S_11 = C``.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_0_MT_1, C_12_0_MT_2).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed S_11 spectral values at harmonics.
        If obs_err is provided, returns (S_11_k, S_11_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_12_0_MT_1 = coefs[0]
    C_12_0_MT_2 = coefs[1]
    # wk: the harmonic comb omega_k = 2*pi*k/T, k=1..len(c_times) -- one frequency
    # sample per control time, the same comb the module docstring describes.
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    # tb: a fine, purely-numerical time grid used only to evaluate the Fourier
    # overlap integral (ff()) accurately -- NOT the experiment's physical control
    # time (that's c_times/ctime, passed into make_y below).
    tb = np.linspace(0, T, 10**4)
    # y_arr: one toggle-function matrix per control time. Passing pulse=['CPMG',
    # 'CDD3'] is a shorthand to get BOTH single-qubit filter functions (CPMG in
    # channel [0,0], CDD3 in channel [1,1]) out of one make_y() call -- these
    # functions only depend on the pulse *type*, not on which physical qubit they
    # are eventually assigned to, so the same y_arr is reused as "qubit 1 under
    # CPMG vs. CDD3" below.
    y_arr = [make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[i], m=1) for i in range(np.size(c_times))]
    # U: the kernel/measurement matrix (see solve_inverse's docstring) such that
    # U @ S_11 equals the noiseless version of C_12_0_MT_1 - C_12_0_MT_2 -- built
    # here as the difference of squared filter-function magnitudes (CPMG minus
    # CDD3) at each harmonic.
    U = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U[i, j] = ((m/T)*(np.square(np.absolute(ff(y_arr[i][0, 0], tb, wk[j])))
                              - np.square(np.absolute(ff(y_arr[i][1, 1], tb, wk[j])))))
    S_11_k, inv_U = solve_inverse(U, C_12_0_MT_1 - C_12_0_MT_2,
                                  nonneg=kwargs.get('enforce_nonneg', False),
                                  label='S_11', **_inv_opts(kwargs))

    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        # Error of difference is sqrt(err1^2 + err2^2)
        combined_err = np.sqrt(np.array(errs[0])**2 + np.array(errs[1])**2)
        S_11_err = propagate_linear_error(inv_U, combined_err)
        return np.real(S_11_k), S_11_err
        
    return np.real(S_11_k)


def recon_S_22(coefs, **kwargs):
    """
    Reconstruct the self-spectrum S_22 at harmonic frequencies.

    Mirror image of ``recon_S_11`` above: ``C_12_0_MT_1`` and ``C_12_0_MT_3`` are
    identical on qubit 1 (CPMG both times) and differ only on qubit 2 (CPMG vs.
    CDD3), so qubit 1's contribution cancels in the difference and what remains
    depends only on S_22. See ``recon_S_11`` for the full explanation of the
    ``wk``/``tb``/``y_arr``/``U`` variables reused identically here.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_0_MT_1, C_12_0_MT_3).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed S_22 spectral values at harmonics.
        If obs_err is provided, returns (S_22_k, S_22_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_12_0_MT_1 = coefs[0]
    C_12_0_MT_3 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**4)
    y_arr = [make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[i], m=1) for i in range(np.size(c_times))]
    U = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U[i, j] = (m/T)*(np.square(np.absolute(ff(y_arr[i][0, 0], tb, wk[j])))
                             - np.square(np.absolute(ff(y_arr[i][1, 1], tb, wk[j]))))
    S_22_k, inv_U = solve_inverse(U, C_12_0_MT_1 - C_12_0_MT_3,
                                  nonneg=kwargs.get('enforce_nonneg', False),
                                  label='S_22', **_inv_opts(kwargs))

    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        combined_err = np.sqrt(np.array(errs[0])**2 + np.array(errs[1])**2)
        S_22_err = propagate_linear_error(inv_U, combined_err)
        return np.real(S_22_k), S_22_err
        
    return np.real(S_22_k)


def recon_S_1_2(coefs, **kwargs):
    """
    Reconstruct the cross-spectrum S_1_2 at harmonic frequencies.

    Unlike the self-spectra above, a cross-spectrum measures CORRELATION between
    qubit 1's and qubit 2's noise, so its kernel is built from the PRODUCT of the
    two qubits' filter functions (``ff(qubit1) * ff(qubit2)``) rather than a
    difference of squared magnitudes. The real and imaginary parts of a complex
    cross-spectrum need different pulse combinations to isolate cleanly: the real
    part comes from the symmetric CPMG-CPMG combination (``U_1``, ``C_12_12_MT_1``);
    the imaginary part needs one qubit's sequence to be antisymmetric under time-
    reversal (CDD3 on qubit 1, ``U_2``/``C_12_12_MT_2``) since a purely real kernel
    cannot pick up an imaginary spectral component. (This Im channel is the one
    with the largest forward-model/comb-truncation bias of the six spectra --
    ``characterize/systematics.py`` quantifies and corrects it.)

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_12_MT_1, C_12_12_MT_2).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed complex S_1_2 spectral values at harmonics.
        If obs_err is provided, returns (S_1_2_k, S_1_2_err).
        S_1_2_err is complex, with real part being error of real part of spectrum.
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_12_12_MT_1 = coefs[0]
    C_12_12_MT_2 = coefs[1]
    tb = np.linspace(0, T, 10**5)
    # y1_arr/y2_arr: toggle-function matrices for the Re-channel (CPMG-CPMG) and
    # Im-channel (CDD3-CPMG) experiments respectively -- here BOTH diagonal
    # entries ([0,0] for qubit 1, [1,1] for qubit 2) are used, since the kernel
    # needs the product of the two qubits' actual filter functions.
    y1_arr = np.array([make_y(tb, ['CPMG', 'CPMG'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    y2_arr = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    U_1 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    U_2 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U_1[i, j] = (2*m/T)*(ff(y1_arr[i][0, 0], tb, wk[j])*ff(y1_arr[i][1, 1], tb, -wk[j]))
            U_2[i, j] = np.imag((2*m/T)*(ff(y2_arr[i][0, 0], tb, wk[j])*ff(y2_arr[i][1, 1], tb, -wk[j])))
    
    opts = _inv_opts(kwargs)
    s1, inv_U1 = solve_inverse(U_1, C_12_12_MT_1, label='Re S_1_2', **opts)
    s2, inv_U2 = solve_inverse(U_2, C_12_12_MT_2, label='Im S_1_2', **opts)
    Re_S_1_2_k = np.real(s1)
    Im_S_1_2_k = -np.real(s2)
    S_val = Re_S_1_2_k + 1j*Im_S_1_2_k
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        err_Re = propagate_linear_error(inv_U1, errs[0])
        err_Im = propagate_linear_error(inv_U2, errs[1])
        return S_val, err_Re + 1j*err_Im

    return S_val


def recon_S_12_12(coefs, **kwargs):
    """
    Reconstruct the Ising self-spectrum S_1212 at harmonic frequencies.

    S_1212 is the PSD of the two-qubit (ZZ/Ising) coupling channel, ``y[2,2] =
    y[0,0]*y[1,1]`` in ``make_y``'s convention -- it does not correspond to
    either physical qubit alone. It cannot be read off a single measurement
    because the Ising phase cancels out of the two-qubit joint coefficient
    (``C_12_0``); instead it is recovered from the single-qubit FID-type
    coefficients ``C_1_0_MT_1``/``C_2_0_MT_1`` (which each retain a mix of a
    single-qubit self-term and the Ising term) minus the joint coefficient
    ``C_12_0_MT_4`` (which retains only the two single-qubit self-terms):
    the single-qubit self-terms cancel in the combination
    ``C_1_0_MT_1 + C_2_0_MT_1 - C_12_0_MT_4``, leaving only S_1212. This is the
    harmonic-comb counterpart of the DC combination documented in detail in
    ``recon_S_1212_dc`` below (same algebraic structure, different pulse family
    and frequency).

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_1_0_MT_1, C_2_0_MT_1, C_12_0_MT_4).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed S_1212 spectral values at harmonics.
        If obs_err is provided, returns (S_1212_k, S_1212_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_1_0_MT_1 = coefs[0]
    C_2_0_MT_1 = coefs[1]
    C_12_0_MT_4 = coefs[2]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    U = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.float64)
    tb = np.linspace(0, T, 10**5)
    y_arr = np.array([make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U[i, j] = (2*m/T)*(np.real(np.square(np.absolute(ff(y_arr[i][0, 0], tb, wk[j])))))
    S_1212_k, inv_U = solve_inverse(U, np.real(C_1_0_MT_1 + C_2_0_MT_1 - C_12_0_MT_4),
                                    nonneg=kwargs.get('enforce_nonneg', False),
                                    label='S_1212', **_inv_opts(kwargs))

    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        # err sum = sqrt(err1^2 + err2^2 + err3^2)
        combined_err = np.sqrt(np.array(errs[0])**2 + np.array(errs[1])**2 + np.array(errs[2])**2)
        S_1212_err = propagate_linear_error(inv_U, combined_err)
        return S_1212_k, S_1212_err

    return S_1212_k


def recon_S_1_12(coefs, **kwargs):
    """
    Reconstruct the cross-spectrum S_1_12 at harmonic frequencies.

    S_1_12 is the correlation between qubit 1's individual noise and the Ising
    (12) channel, so -- like ``recon_S_1_2`` above -- its kernel is a PRODUCT of
    filter functions, but here between qubit 1's channel (``y[0,0]``) and the
    Ising channel (``y[2,2] = y[0,0]*y[1,1]``, see ``make_y``) instead of between
    the two individual qubits. As in ``recon_S_1_2``, the real part (``U1``) uses
    a symmetric pulse combo (CPMG/FID) and the imaginary part (``U2``) needs an
    antisymmetric one (CPMG/CDD3) to be sensitive to Im(S_1_12) at all.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_1_2_MT_1, C_1_2_MT_2).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed complex S_1_12 spectral values at harmonics.
        If obs_err is provided, returns (S_1_12_k, S_1_12_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_1_2_MT_1 = coefs[0]
    C_1_2_MT_2 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**5)
    U1 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    U2 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    y1_arr = np.array([make_y(tb, ['CPMG', 'FID'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    y2_arr = np.array([make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U1[i, j]=(2*m/T)*(ff(y1_arr[i][0, 0], tb, wk[j])*ff(y1_arr[i][2, 2], tb, -wk[j]))
            U2[i, j]=np.imag((2*m/T)*ff(y2_arr[i][0, 0], tb, wk[j])*ff(y2_arr[i][1, 1], tb, -wk[j]))
    
    opts = _inv_opts(kwargs)
    s1, inv_U1 = solve_inverse(U1, C_1_2_MT_1, label='Re S_1_12', **opts)
    s2, inv_U2 = solve_inverse(U2, C_1_2_MT_2, label='Im S_1_12', **opts)
    Re_S_1_12_k = np.real(s1)
    Im_S_1_12_k = -np.real(s2)
    S_val = Re_S_1_12_k + 1j*Im_S_1_12_k
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        err_Re = propagate_linear_error(inv_U1, errs[0])
        err_Im = propagate_linear_error(inv_U2, errs[1])
        return S_val, err_Re + 1j*err_Im
        
    return S_val


def recon_S_2_12(coefs, **kwargs):
    """
    Reconstruct the cross-spectrum S_2_12 at harmonic frequencies.

    Mirror image of ``recon_S_1_12`` above, swapping the roles of qubit 1 and
    qubit 2: correlation between qubit 2's individual noise and the Ising
    channel, with the same symmetric/antisymmetric pulse-combo logic for the
    real/imaginary parts.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_2_1_MT_1, C_2_1_MT_2).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Reconstructed complex S_2_12 spectral values at harmonics.
        If obs_err is provided, returns (S_2_12_k, S_2_12_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    C_2_1_MT_1 = coefs[0]
    C_2_1_MT_2 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**5)
    U1 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    U2 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    y1_arr = np.array([make_y(tb, ['FID', 'CPMG'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    y2_arr = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=c_times[i], m=1) for i in range(np.size(c_times))])
    for i in range(np.size(c_times)):
        for j in range(np.size(c_times)):
            U1[i, j]=(2*m/T)*(ff(y1_arr[i][1, 1], tb, wk[j])*ff(y1_arr[i][2, 2], tb, -wk[j]))
            U2[i, j]=np.imag((2*m/T)*ff(y2_arr[i][1, 1], tb, wk[j])*ff(y2_arr[i][0, 0], tb, -wk[j]))
    
    opts = _inv_opts(kwargs)
    s1, inv_U1 = solve_inverse(U1, C_2_1_MT_1, label='Re S_2_12', **opts)
    s2, inv_U2 = solve_inverse(U2, C_2_1_MT_2, label='Im S_2_12', **opts)
    Re_S_2_12_k = np.real(s1)
    Im_S_2_12_k = -np.real(s2)
    S_val = Re_S_2_12_k + 1j*Im_S_2_12_k
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        err_Re = propagate_linear_error(inv_U1, errs[0])
        err_Im = propagate_linear_error(inv_U2, errs[1])
        return S_val, err_Re + 1j*err_Im

    return S_val



# --- DC reconstruction functions from FID experiments ---
#
# Notation used throughout this section: Phi_1, Phi_2, Phi_12 are the random
# accumulated dephasing PHASES of qubit 1, qubit 2, and the Ising (ZZ) channel
# respectively, picked up during a free-induction-decay (FID, i.e. no pulses)
# experiment of duration t. Their variances/covariances (Var Phi_1 = <Phi_1^2>,
# etc.) grow linearly in t with slope equal to the corresponding zero-frequency
# spectral density -- e.g. Var Phi_1(t) -> S_11(0) * t -- which is the physical
# fact every ``recon_S_*_dc`` function below exploits via ``_ramsey_fit_dc``.

def _ramsey_fit_dc(C, t, factor, obs_err=None, c_max=3.0, c_min=0.05, curv_tol=0.30,
                   selfspec=True):
    """Strong-noise-robust zero-frequency S(0) from the FID/Ramsey decay slope.

    The decay exponent ``C(t)`` of the partner-decoupled FID (self) or the cross
    FID/FID coefficient (cross), measured over a sweep of total evolution times
    ``t_k``, grows in the motional-narrowing (linear) regime as

        C(t)  ->  a + (S(0)/factor) * t ,   factor = 2 (self, C=Var/2), 1 (cross, C=Cov),

    so ``S(0) = factor * slope``. ("Motional narrowing" is standard NMR/dephasing-
    noise language for the short/intermediate-time behavior where the qubit has
    not yet sampled enough of the noise spectrum for its full lineshape to
    matter, so the decay looks like this plain S(0)-only, linear-in-t process.
    "Quasi-static" noise, referenced further down, is the opposite limit --
    noise so slow that the qubit barely dephases within the experiment, so
    ``C(t)`` grows quadratically at first ["curvature"] instead of linearly,
    and the S(0)-from-slope trick below is not yet reliable within the measured
    time window.) The legacy single-point estimator ``2<C(MT)>/(MT)``
    fails for strong noise because the coherence at the full record MT is fully decayed
    (C >> 1, below the shot-noise floor). Here the slope is fit over the window where
    ``C(t)`` is both MEASURABLE and LINEAR, selected ADAPTIVELY -- so the effective
    measurement time tracks the noise strength with no per-spectrum tuning (change the
    spectra and the window moves itself).

    Window and reliability criteria adapt to whether per-point errors are known:

    - measurable: with ``obs_err`` the criterion is statistical, ``C > 2 sigma_C``
      (an absolute floor would discard the perfectly informative low-C points of
      weak cross channels); without errors (exact forward-model curves) the legacy
      absolute floor ``c_min`` applies. ``C < c_max`` guards saturation either way.
    - curvature: with ``obs_err`` the quadratic coefficient must be statistically
      significant (>2 sigma) AND practically large to flag -- a magnitude-only test
      trips on noise for low-SNR channels; exact curves keep the magnitude test.
    - sign: self-spectra flag a non-positive S(0) as undetermined (S >= 0 is
      physical); cross-spectra DC may be legitimately negative, so only the
      ``|S0| < 2 sigma`` significance test applies (set ``selfspec=False``).

    Parameters
    ----------
    C, t : array_like
        Decay exponent ``C(t_k)`` over the sweep of total evolution times ``t_k``.
    factor : float
        2 for self-spectra, 1 for cross-spectra.
    obs_err : array_like, optional
        Per-time standard error of ``C(t_k)``; propagated to the slope error.
    selfspec : bool
        True for the three self-spectra (nonnegativity enforced in the flag).

    Returns
    -------
    (S0, S0_err, reliable)
        ``reliable=False`` when the decay has not reached the linear regime within the
        measurable window (quasi-static / sub-comb-cusp noise -- detected as significant
        curvature), the sweep does not bracket the window, or the slope is
        insignificant: ``S0`` is then only a lower bound (self) or indeterminate
        (cross) and should be quoted with an inflated systematic bar.
    """
    C = np.asarray(C, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    C, t = C[order], t[order]
    e = None if obs_err is None else np.asarray(obs_err, dtype=float)[order]

    # Measurable window: statistically resolved (or above the absolute floor when
    # no errors are known) but not saturated.
    lo = c_min if e is None else 2.0 * e
    meas = (C > lo) & (C < c_max)
    # Sweep brackets the window only if it is neither all-decayed nor all-tiny.
    lo_last = c_min if e is None else 2.0 * e[-1]
    covered = bool(meas.sum() >= 2 and C[0] < c_max and C[-1] > lo_last)
    if meas.sum() < 2:                       # degrade gracefully: use the un-saturated tail
        meas = C < c_max
        if meas.sum() < 2:
            return float(factor * np.mean(C) / t[-1]), 0.0, False

    tm, Cm = t[meas], C[meas]
    em = None if e is None else e[meas]
    lin = tm >= 0.4 * tm.max()               # linear regime = upper part of measurable window
    if lin.sum() < 3:                        # too few late points: keep the whole window
        lin = np.ones(tm.size, dtype=bool)

    x, y = tm[lin], Cm[lin]
    w = np.ones_like(x) if em is None else 1.0 / np.maximum(em[lin], 1e-12) ** 2
    xb = np.sum(w * x) / np.sum(w)
    Sxx = np.sum(w * (x - xb) ** 2)
    slope = np.sum(w * (x - xb) * y) / Sxx
    S0 = factor * slope
    S0_err = float(factor * np.sqrt(1.0 / Sxx)) if em is not None else 0.0

    reliable = covered
    if em is None and meas.sum() >= 3:       # curvature -> not yet linear (quasi-static)
        c2, c1, _ = np.polyfit(tm, Cm, 2)
        if abs(c2 * tm.max() ** 2) > curv_tol * abs(c1 * tm.max()):
            reliable = False
    elif em is not None and meas.sum() >= 4:
        # Noise-aware curvature: weighted quadratic fit; flag only when c2 is both
        # statistically significant and practically large vs the fitted slope.
        X = np.vander(tm, 3)                 # columns t^2, t, 1
        Wd = 1.0 / np.maximum(em, 1e-12) ** 2
        cov = np.linalg.inv((X.T * Wd) @ X)
        beta = cov @ ((X.T * Wd) @ Cm)
        c2, sig_c2 = beta[0], np.sqrt(cov[0, 0])
        if (abs(c2) > 2.0 * sig_c2
                and abs(c2 * tm.max() ** 2) > curv_tol * abs(slope * tm.max())):
            reliable = False
    # An undetermined slope means the signal is swamped (e.g. a strong-noise cross
    # channel whose self-decay dwarfs the cross term): not determined -> flag so the
    # figure quotes an inflated bar. Nonpositivity flags self-spectra only.
    if selfspec and S0 <= 0:
        reliable = False
    if S0_err > 0 and abs(S0) < 2.0 * S0_err:
        reliable = False
    return float(S0), S0_err, bool(reliable)


def _dc_obs_err(kwargs):
    """Per-time obs error for the DC sweep (kwargs['obs_err'] is a list of one array)."""
    oe = kwargs.get('obs_err')
    return None if oe is None else oe[0]


def recon_S_11_dc(coefs, **kwargs):
    """Reconstruct S_11(0) from the partner-decoupled FID/CDD3 decay-slope fit.

    Under ['FID','CDD3'] (qubit 1 free, qubit 2 CDD3-decoupled) the coefficient
    C_{1,0}(t) is the qubit-1 free-induction-decay exponent governed by S_11 alone
    (the partner CDD3 nulls the Ising term). S_11(0) = 2 * slope of C_{1,0}(t),
    fit over the adaptively-selected measurable+linear window (strong-noise robust;
    see ``_ramsey_fit_dc``).

    coefs : [C_1_0_FIDCDD3 over the time sweep]   kwargs : t_sweep, obs_err (optional)
    Returns (S0, S0_err, reliable).
    """
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 2.0, _dc_obs_err(kwargs))


def recon_S_22_dc(coefs, **kwargs):
    """Reconstruct S_22(0): symmetric counterpart of ``recon_S_11_dc`` using
    ['CDD3','FID']. S_22(0) = 2 * slope of C_{2,0}(t). Returns (S0, S0_err, reliable)."""
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 2.0, _dc_obs_err(kwargs))


def recon_S_1_2_dc(coefs, **kwargs):
    """Reconstruct S_1_2(0) from the cross FID/FID coefficient C_12_12(t).

    C_12_12(t) = Cov(Phi_1, Phi_2) -> a + S_1_2(0) * t in the linear regime, so the
    cross-spectrum carries NO factor of 2: S_1_2(0) = slope of C_12_12(t).
    Returns (S0, S0_err, reliable).

    coefs : [C_12_12_FID over the time sweep]    kwargs : t_sweep, obs_err (optional)
    """
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs),
                          selfspec=False)


def recon_S_1_12_dc(coefs, **kwargs):
    """Reconstruct S_1_12(0) from C_a_b(l=1) FID/FID: S_1_12(0) = slope of C_1_12(t).
    Returns (S0, S0_err, reliable). (For quasi-static Ising noise the linear regime is
    not reached and ``reliable`` is False -- the value is then a lower bound.)"""
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs),
                          selfspec=False)


def recon_S_2_12_dc(coefs, **kwargs):
    """Reconstruct S_2_12(0) from C_a_b(l=2) FID/FID: S_2_12(0) = slope of C_2_12(t).
    Returns (S0, S0_err, reliable)."""
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs),
                          selfspec=False)


def recon_S_1212_dc_echo(coefs, **kwargs):
    """Reconstruct the Ising self-DC S_1212(0) from the double-echo difference.

    Simultaneous CDD1 on both qubits echoes the single-qubit phase (y_1 balanced)
    while y_12 = y_1*y_2 = +1 retains FULL DC weight; the CDD1/CPMG reference has
    the SAME y_1 filter but a fast-toggled, balanced y_12. Their difference is
    (1/2)Var Phi_12 at first order: Var Phi_1 cancels EXACTLY (identical y_1
    filter), so -- unlike recon_S_1212_dc's FID/FID combination, which extracts
    Var Phi_12 as a ~25x-smaller difference of single-qubit variances -- the
    subtraction terms are echo-small and the estimator is not statistically
    swamped. (In plain terms: because the two experiments being subtracted share
    an *identical* qubit-1 dephasing contribution, that large, noisy quantity
    cancels exactly rather than being subtracted approximately -- so the
    remaining signal is not a small difference of two large, independently-noisy
    numbers, and the statistical error stays small.) The residual mixed-filter
    (CDD1xCPMG) pickup of S_1212 is a deterministic systematic mirrored/corrected
    by the DC forward model.

    coefs : [C_1_0_CDD1CDD1, C_1_0_CDD1CPMG] over the time sweep.
    kwargs : t_sweep, obs_err (optional).  Returns (S0, S0_err, reliable).
    """
    cb, cr = (np.asarray(x, dtype=float) for x in coefs)
    oe = kwargs.get('obs_err')
    err = None
    if oe is not None:
        eb, er = (np.asarray(oe[i], dtype=float) for i in range(2))
        err = np.sqrt(eb ** 2 + er ** 2)
    return _ramsey_fit_dc(cb - cr, kwargs['t_sweep'], 2.0, err)


def recon_S_1212_dc(coefs, **kwargs):
    """Reconstruct the Ising self-DC S_1212(0) from the FID/FID single-qubit + ZZ combo.

    The Ising phase cancels in the two-qubit (ZZ) coherences, so C_12_0 alone holds no
    S_1212. The single-qubit FID/FID coefficients DO retain it:
        C_1_0_FF = 1/2 (Var Phi_1 + Var Phi_12),  C_2_0_FF = 1/2 (Var Phi_2 + Var Phi_12),
        C_12_0_FF = 1/2 (Var Phi_1 + Var Phi_2),
    so  Var Phi_12(t) = C_1_0_FF + C_2_0_FF - C_12_0_FF  ->  S_1212(0) * t  (factor 1).
    This mirrors the harmonic recon_S_12_12 combination at DC, and -- like the others --
    is fit over the adaptive measurable+linear window (reliable=False for quasi-static
    Ising noise whose DC cusp never reaches the linear regime).

    coefs : [C_1_0_FIDFID, C_2_0_FIDFID, C_12_0_FID_FID] over the time sweep.
    kwargs : t_sweep, obs_err (optional).  Returns (S0, S0_err, reliable).
    """
    c10, c20, c120 = (np.asarray(x, dtype=float) for x in coefs)
    V12 = c10 + c20 - c120
    oe = kwargs.get('obs_err')
    err = None
    if oe is not None:
        e10, e20, e120 = (np.asarray(oe[i], dtype=float) for i in range(3))
        err = np.sqrt(e10 ** 2 + e20 ** 2 + e120 ** 2)
    return _ramsey_fit_dc(V12, kwargs['t_sweep'], 1.0, err)

