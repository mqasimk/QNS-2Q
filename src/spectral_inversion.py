"""
Spectral inversion and reconstruction algorithms.

This module provides tools to reconstruct noise power spectral densities (PSDs)
from experimental observables via least-squares inversion. It supports both
harmonic reconstruction (at specific frequency combs) and DC (zero-frequency)
reconstruction from FID and CPMG/CDD experiments.
"""

import numpy as np
from trajectories import make_y

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
    inv_U = np.linalg.inv(U)
    S_11_k = inv_U@(C_12_0_MT_1-C_12_0_MT_2)
    
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
    inv_U = np.linalg.inv(U)
    S_22_k = inv_U@(C_12_0_MT_1-C_12_0_MT_3)
    
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
    
    inv_U1 = np.linalg.inv(U_1)
    inv_U2 = np.linalg.inv(U_2)
    
    Re_S_1_2_k = np.real(inv_U1@C_12_12_MT_1)
    Im_S_1_2_k = -np.real(inv_U2@C_12_12_MT_2)
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
    inv_U = np.linalg.inv(U)
    S_1212_k = inv_U@np.real(C_1_0_MT_1+C_2_0_MT_1-C_12_0_MT_4)
    
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
    
    inv_U1 = np.linalg.inv(U1)
    inv_U2 = np.linalg.inv(U2)
    
    Re_S_1_12_k = np.real(inv_U1@C_1_2_MT_1)
    Im_S_1_12_k = -np.real(inv_U2@C_1_2_MT_2)
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
    
    inv_U1 = np.linalg.inv(U1)
    inv_U2 = np.linalg.inv(U2)

    Re_S_2_12_k = np.real(inv_U1@C_2_1_MT_1)
    Im_S_2_12_k = -np.real(inv_U2@C_2_1_MT_2)
    S_val = Re_S_2_12_k + 1j*Im_S_2_12_k
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err']
        err_Re = propagate_linear_error(inv_U1, errs[0])
        err_Im = propagate_linear_error(inv_U2, errs[1])
        return S_val, err_Re + 1j*err_Im

    return S_val



# --- DC reconstruction functions from FID experiments ---

def recon_S_11_dc(coefs, **kwargs):
    """
    Reconstruct S_11(0) from C_12_0 with ['FID','CPMG'] pulse sequence.

    Subtracts harmonic contributions of S_22 and S_1212 at CPMG harmonics,
    then averages over c_times to extract the DC value.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_0_FID_CPMG).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        S_22_k : array_like
            Already reconstructed S_22 harmonic values.
        S_1212_k : array_like
            Already reconstructed S_1212 harmonic values.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_11(0).
        If obs_err is provided, returns (S_11_dc, S_11_dc_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    S_22_k = kwargs['S_22_k']
    S_1212_k = kwargs['S_1212_k']
    C_12_0_FID_CPMG = coefs[0]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**5)
    dc_estimates = np.zeros(np.size(c_times))
    for i in range(np.size(c_times)):
        y = make_y(tb, ['FID', 'CPMG'], ctime=c_times[i], m=1)
        harmonic_sum = 0.0
        for j in range(np.size(c_times)):
            F_cpmg_sq = np.square(np.absolute(ff(y[1, 1], tb, wk[j])))
            harmonic_sum += (m/T) * F_cpmg_sq * (S_22_k[j] + 2*S_1212_k[j])
        dc_estimates[i] = (C_12_0_FID_CPMG[i] - harmonic_sum) / (m*T)
    
    val = np.mean(dc_estimates)
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        # err_i = errs[i] / (m*T)
        # err_mean = sqrt(sum(err_i^2)) / N
        err_mean = np.sqrt(np.sum((np.array(errs) / (m*T))**2)) / np.size(c_times)
        return val, err_mean
        
    return val


def recon_S_22_dc(coefs, **kwargs):
    """
    Reconstruct S_22(0) from C_12_0 with ['CPMG','FID'] pulse sequence.

    Subtracts harmonic contributions of S_11 and S_1212 at CPMG harmonics,
    then averages over c_times to extract the DC value.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_0_CPMG_FID).
    **kwargs
        c_times : array_like
            Control times used.
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        S_11_k : array_like
            Already reconstructed S_11 harmonic values.
        S_1212_k : array_like
            Already reconstructed S_1212 harmonic values.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_22(0).
        If obs_err is provided, returns (S_22_dc, S_22_dc_err).
    """
    c_times = kwargs['c_times']
    m = kwargs['m']
    T = kwargs['T']
    S_11_k = kwargs['S_11_k']
    S_1212_k = kwargs['S_1212_k']
    C_12_0_CPMG_FID = coefs[0]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    tb = np.linspace(0, T, 10**5)
    dc_estimates = np.zeros(np.size(c_times))
    for i in range(np.size(c_times)):
        y = make_y(tb, ['CPMG', 'FID'], ctime=c_times[i], m=1)
        harmonic_sum = 0.0
        for j in range(np.size(c_times)):
            F_cpmg_sq = np.square(np.absolute(ff(y[0, 0], tb, wk[j])))
            harmonic_sum += (m/T) * F_cpmg_sq * (S_11_k[j] + 2*S_1212_k[j])
        dc_estimates[i] = (C_12_0_CPMG_FID[i] - harmonic_sum) / (m*T)
    
    val = np.mean(dc_estimates)

    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        err_mean = np.sqrt(np.sum((np.array(errs) / (m*T))**2)) / np.size(c_times)
        return val, err_mean

    return val


def recon_S_1212_dc(coefs, **kwargs):
    """
    Reconstruct S_1212(0) from C_12_0 with ['FID','FID'] pulse sequence.

    Uses already-reconstructed S_11(0) and S_22(0) to isolate S_1212(0).

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_0_FID_FID).
    **kwargs
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        S_11_dc : float
            Reconstructed S_11(0).
        S_22_dc : float
            Reconstructed S_22(0).
        obs_err : list of array_like, optional
            Standard errors for the observables.
        S_11_dc_err : float, optional
            Error of S_11(0).
        S_22_dc_err : float, optional
            Error of S_22(0).

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_1212(0).
        If obs_err is provided, returns (S_1212_dc, S_1212_dc_err).
    """
    m = kwargs['m']
    T = kwargs['T']
    S_11_dc = kwargs['S_11_dc']
    S_22_dc = kwargs['S_22_dc']
    C_12_0_FID_FID = coefs[0]
    # ['FID','FID'] C_12_0 is c_time-independent; average for noise reduction
    meas = np.mean(C_12_0_FID_FID)
    val = -(meas / (m*T) - S_11_dc - S_22_dc) / 2
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        # err_meas = sqrt(sum(errs^2)) / N
        err_meas = np.sqrt(np.sum(np.array(errs)**2)) / len(errs)
        
        s11_err = kwargs.get('S_11_dc_err', 0.0)
        s22_err = kwargs.get('S_22_dc_err', 0.0)
        
        # err = 0.5 * sqrt( (err_meas/(m*T))^2 + s11_err^2 + s22_err^2 )
        total_err = 0.5 * np.sqrt((err_meas / (m*T))**2 + s11_err**2 + s22_err**2)
        return val, total_err

    return val


def recon_S_1_2_dc(coefs, **kwargs):
    """
    Reconstruct S_1_2(0) from C_12_12 with ['FID','FID'] pulse sequence.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_12_12_FID).
    **kwargs
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_1_2(0).
        If obs_err is provided, returns (S_1_2_dc, S_1_2_dc_err).
    """
    m = kwargs['m']
    T = kwargs['T']
    C_12_12_FID = coefs[0]
    # ['FID','FID'] C_12_12 is c_time-independent; average for noise reduction
    val = np.mean(C_12_12_FID) / (2*m*T)
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        err_meas = np.sqrt(np.sum(np.array(errs)**2)) / len(errs)
        return val, err_meas / (2*m*T)

    return val


def recon_S_1_12_dc(coefs, **kwargs):
    """
    Reconstruct S_1_12(0) from C_a_b(l=1) with ['FID','FID'] pulse sequence.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_1_12_FID).
    **kwargs
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_1_12(0).
        If obs_err is provided, returns (S_1_12_dc, S_1_12_dc_err).
    """
    m = kwargs['m']
    T = kwargs['T']
    C_1_12_FID = coefs[0]
    # ['FID','FID'] C_a_b is c_time-independent; average for noise reduction
    val = np.mean(C_1_12_FID) / (2*m*T)
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        err_meas = np.sqrt(np.sum(np.array(errs)**2)) / len(errs)
        return val, err_meas / (2*m*T)

    return val


def recon_S_2_12_dc(coefs, **kwargs):
    """
    Reconstruct S_2_12(0) from C_a_b(l=2) with ['FID','FID'] pulse sequence.

    Parameters
    ----------
    coefs : list of array_like
        Observables (C_2_12_FID).
    **kwargs
        m : int
            Number of repetitions.
        T : float
            Time per repetition.
        obs_err : list of array_like, optional
            Standard errors for the observables.

    Returns
    -------
    float or (float, float)
        Reconstructed DC value S_2_12(0).
        If obs_err is provided, returns (S_2_12_dc, S_2_12_dc_err).
    """
    m = kwargs['m']
    T = kwargs['T']
    C_2_12_FID = coefs[0]
    # ['FID','FID'] C_a_b is c_time-independent; average for noise reduction
    val = -np.mean(C_2_12_FID) / (2*m*T)
    
    if 'obs_err' in kwargs and kwargs['obs_err'] is not None:
        errs = kwargs['obs_err'][0]
        err_meas = np.sqrt(np.sum(np.array(errs)**2)) / len(errs)
        return val, err_meas / (2*m*T)

    return val

