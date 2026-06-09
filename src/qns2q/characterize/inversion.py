"""
Spectral inversion and reconstruction algorithms.

This module provides tools to reconstruct noise power spectral densities (PSDs)
from experimental observables via least-squares inversion. It supports both
harmonic reconstruction (at specific frequency combs) and DC (zero-frequency)
reconstruction from FID and CPMG/CDD experiments.
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

def solve_inverse(U, rhs, method='direct', reg_lambda=0.0, nonneg=False,
                  diagnostics=False, label=''):
    """Solve U @ S = rhs with a selectable backend. Returns (S, J)."""
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

    Parameters
    ----------
    y : array_like
        Toggle function values.
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
    if w == 0:
        return 0
    y = make_y(np.linspace(0, T, 10**5), ['CDD3', 'CDD3'], ctime=ct, m=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapezoid(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def propagate_linear_error(A_inv, obs_err):
    """
    Propagates independent observation errors through linear system S = A_inv @ C.
    Sigma_S_i = sqrt( sum_j |A_inv_ij|^2 * Sigma_C_j^2 )

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
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**4)
    y_arr = [make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[i], m=1) for i in range(np.size(c_times))]
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

def _ramsey_fit_dc(C, t, factor, obs_err=None, c_max=3.0, c_min=0.05, curv_tol=0.30):
    """Strong-noise-robust zero-frequency S(0) from the FID/Ramsey decay slope.

    The decay exponent ``C(t)`` of the partner-decoupled FID (self) or the cross
    FID/FID coefficient (cross), measured over a sweep of total evolution times
    ``t_k``, grows in the motional-narrowing (linear) regime as

        C(t)  ->  a + (S(0)/factor) * t ,   factor = 2 (self, C=Var/2), 1 (cross, C=Cov),

    so ``S(0) = factor * slope``. The legacy single-point estimator ``2<C(MT)>/(MT)``
    fails for strong noise because the coherence at the full record MT is fully decayed
    (C >> 1, below the shot-noise floor). Here the slope is fit over the window where
    ``C(t)`` is both MEASURABLE (``c_min < C < c_max``) and LINEAR, selected
    ADAPTIVELY -- so the effective measurement time tracks the noise strength with no
    per-spectrum tuning (change the spectra and the window moves itself).

    Parameters
    ----------
    C, t : array_like
        Decay exponent ``C(t_k)`` over the sweep of total evolution times ``t_k``.
    factor : float
        2 for self-spectra, 1 for cross-spectra.
    obs_err : array_like, optional
        Per-time standard error of ``C(t_k)``; propagated to the slope error.

    Returns
    -------
    (S0, S0_err, reliable)
        ``reliable=False`` when the decay has not reached the linear regime within the
        measurable window (quasi-static / sub-comb-cusp noise -- detected as significant
        curvature) or the sweep does not bracket the window: ``S0`` is then only a lower
        bound and should be quoted with an inflated systematic bar.
    """
    C = np.asarray(C, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    C, t = C[order], t[order]
    e = None if obs_err is None else np.asarray(obs_err, dtype=float)[order]

    meas = (C > c_min) & (C < c_max)
    # Sweep brackets the window only if it is neither all-decayed nor all-tiny.
    covered = bool(meas.sum() >= 2 and C[0] < c_max and C[-1] > c_min)
    if meas.sum() < 2:                       # degrade gracefully: use the un-saturated tail
        meas = C < c_max
        if meas.sum() < 2:
            return float(factor * np.mean(C) / t[-1]), 0.0, False

    tm, Cm = t[meas], C[meas]
    em = None if e is None else e[meas]
    lin = tm >= 0.4 * tm.max()               # linear regime = upper part of measurable window
    if lin.sum() < 2:
        lin = np.ones(tm.size, dtype=bool)

    x, y = tm[lin], Cm[lin]
    w = np.ones_like(x) if em is None else 1.0 / np.maximum(em[lin], 1e-12) ** 2
    xb = np.sum(w * x) / np.sum(w)
    Sxx = np.sum(w * (x - xb) ** 2)
    slope = np.sum(w * (x - xb) * y) / Sxx
    S0 = factor * slope
    S0_err = float(factor * np.sqrt(1.0 / Sxx)) if em is not None else 0.0

    reliable = covered
    if meas.sum() >= 3:                      # curvature -> not yet linear (quasi-static)
        c2, c1, _ = np.polyfit(tm, Cm, 2)
        if abs(c2 * tm.max() ** 2) > curv_tol * abs(c1 * tm.max()):
            reliable = False
    # An unphysical (negative) or statistically-insignificant slope means the signal is
    # swamped (e.g. a strong-noise cross channel whose self-decay dwarfs the cross term):
    # not determined -> flag so the figure quotes an inflated bar.
    if S0 <= 0 or (S0_err > 0 and S0 < 2.0 * S0_err):
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
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs))


def recon_S_1_12_dc(coefs, **kwargs):
    """Reconstruct S_1_12(0) from C_a_b(l=1) FID/FID: S_1_12(0) = slope of C_1_12(t).
    Returns (S0, S0_err, reliable). (For quasi-static Ising noise the linear regime is
    not reached and ``reliable`` is False -- the value is then a lower bound.)"""
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs))


def recon_S_2_12_dc(coefs, **kwargs):
    """Reconstruct S_2_12(0) from C_a_b(l=2) FID/FID: S_2_12(0) = slope of C_2_12(t).
    Returns (S0, S0_err, reliable)."""
    return _ramsey_fit_dc(coefs[0], kwargs['t_sweep'], 1.0, _dc_obs_err(kwargs))


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

