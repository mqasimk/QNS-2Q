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
