"""Spectral tail extrapolation for the gate-model SMat construction.

The reconstruction stops at the comb's last tooth (the protocol Nyquist
pi/4tau, set by the most-nested probe sequence); interpolating the spectra
with ``right=0`` told the pulse optimizer that the band above is noise-free,
and the optimizer parks its filter passband exactly there (the 76-pulse/80tau
NT solution has its fundamental at w ~ 3/tau): 85% of that gate's TRUE error
lived in the unmeasured band, and the predicted infidelity under-read 6.6x.

Fix: extend each reconstructed spectrum component beyond the last tooth with a
power law fitted to the top teeth -- physically motivated (the charge-noise
components are power laws; everything above the last nuclear line is a smooth
tail), measured where it can be, and conservative where the fit is
indeterminate (oscillating / sign-changing cross components fall back to the
legacy right=0, which the gate-level audit shows contribute ~0 there).
"""
import numpy as np
import jax.numpy as jnp

# Teeth used for the tail fit. 5 keeps the window above the last Class-F
# nuclear line (tooth 15 of the T=160 comb sits on the 2x line's shoulder).
TAIL_N_FIT = 5


def fit_powerlaw_tail(wk, comp, n_fit=TAIL_N_FIT):
    """(A, p) of comp ~ A*(w/wk[-1])^(-p) from log-log LSQ over the last n_fit
    teeth, or None when no one-signed resolved tail exists there.

    The exponent is clipped to [0, 6]: a rising fitted tail (noise artifact)
    degrades to a flat extension at the fitted level rather than growing.
    """
    wk = np.asarray(wk, dtype=float)
    y = np.asarray(comp, dtype=float)
    if wk.size < n_fit or y.size != wk.size:
        return None
    yf, wf = y[-n_fit:], wk[-n_fit:]
    s = np.sign(yf[-1])
    if s == 0 or np.any(np.sign(yf) != s) or np.any(wf <= 0):
        return None
    ly = np.log(np.abs(yf))
    lw = np.log(wf / wk[-1])
    p = float(-np.polyfit(lw, ly, 1)[0])
    if not np.isfinite(p):
        return None
    p = float(np.clip(p, 0.0, 6.0))
    A = float(s * np.exp(np.mean(ly + p * lw)))
    return A, p


def tail_extend_interp(w_dense, wk, comp, n_fit=TAIL_N_FIT):
    """Linear interpolation of a real spectrum component inside the comb band,
    power-law tail beyond the last tooth (legacy right=0 when unfittable)."""
    wk = jnp.asarray(wk)
    comp = jnp.asarray(comp)
    base = jnp.interp(w_dense, wk, comp, right=0.)
    fit = fit_powerlaw_tail(np.asarray(wk), np.asarray(comp), n_fit)
    if fit is None:
        return base
    A, p = fit
    wlast = float(np.asarray(wk)[-1])
    # jnp.where evaluates both branches: clamp w below the band edge so the
    # power law stays finite at w -> 0 (those samples are masked anyway).
    ws = jnp.maximum(w_dense, wlast)
    tail = A * (ws / wlast) ** (-p)
    return jnp.where(w_dense > wlast, tail, base)


def tail_extend_interp_complex(w_dense, wk, fp, n_fit=TAIL_N_FIT):
    """Complex version: Re and Im components are fitted/extended separately."""
    return (tail_extend_interp(w_dense, wk, jnp.real(fp), n_fit)
            + 1j * tail_extend_interp(w_dense, wk, jnp.imag(fp), n_fit))


def smoothfit_curve(w_dense, wk, comp, dc_val=None):
    """LINE-BLIND smooth model of a self-spectrum: one power law A*w^-p,
    log-log least-squares over ALL positive teeth, saturated toward DC.

    This is the ablation-ladder rung-(b) characterization model ("coarse
    knowledge"): an experimenter who fits a single 1/f^p charge-noise law
    through the comb -- lines included, because a line-blind fitter cannot
    exclude what it does not know is there -- and anchors the low-frequency
    plateau at the measured DC point. Deliberately NOT assumption-light: the
    point is to price what the line-aware reconstruction adds.

    Parameters
    ----------
    w_dense : jax array -- evaluation grid (>= 0).
    wk : array -- comb grid (may carry a DC point at index 0).
    comp : array -- reconstructed spectrum samples on wk (real part used).
    dc_val : float, optional -- measured S(0); caps the rising power law.

    Returns the fitted curve on ``w_dense`` (real jnp array).
    """
    wk = np.asarray(wk, dtype=float)
    y = np.real(np.asarray(comp)).astype(float)
    pos = wk > 0
    yp, wp = y[pos], wk[pos]
    good = yp > 0
    if good.sum() < 3:
        # degenerate comb: fall back to a flat line at the median level
        level = float(np.median(np.abs(yp))) if yp.size else 0.0
        return jnp.full(w_dense.shape, level)
    p = float(-np.polyfit(np.log(wp[good]), np.log(yp[good]), 1)[0])
    p = float(np.clip(p, 0.0, 6.0))
    logA = float(np.mean(np.log(yp[good]) + p * np.log(wp[good])))
    A = float(np.exp(logA))
    cap = float(dc_val) if (dc_val is not None and np.isfinite(dc_val)
                            and dc_val > 0) else A * wp[0] ** (-p)
    ws = jnp.maximum(w_dense, 1e-12)
    return jnp.minimum(A * ws ** (-p), cap)
