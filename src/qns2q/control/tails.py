"""Spectral tail extrapolation for the gate-model SMat construction.

Small, self-contained helper module (no other qns2q imports) used by BOTH gate
optimizers in the **control** arm of the pipeline. It has one job: given the
noise power-spectral-density values that Stage 2 reconstruction (
``characterize/reconstruct.py`` -> ``specs.npz``) measured only at the QNS
comb's discrete "teeth" (frequencies ``wk``), produce a value of that spectrum
at ANY frequency on the dense grid the optimizers integrate their predicted
gate infidelity over (``w_dense``, called ``self.w`` in the callers). Inside
the comb band that is a plain linear interpolation; above the comb's last
tooth (where there is no data) it is a fitted power-law extrapolation --
the reasoning for that choice is the whole story below.

Callers: ``control/cz.py`` (``CZOptConfig._build_interpolated_spectra``) and
``control/idle.py`` (the analogous ``combine()`` helper) both call
``tail_extend_interp_complex`` once per spectral channel (S11, S22, S1212,
S12, S112, S212) while assembling their 4x4 ``SMat`` -- the array of noise
spectra, one entry per Pauli/Ising channel, that the predicted-infidelity
integral is built from. ``characterize/systematics.py``'s
``selfconsistent_spectra`` (used by the unfold bias correction, and by the
optimizers' ``--spectral-model selfconsistent`` mode) separately reuses just
the fitting piece, ``fit_powerlaw_tail``, for the same physical reason (a
smooth power-law background above the last resolved nuclear line) but with
its own line-subtraction and low-frequency treatment -- see that file's
``GATE-TAILS`` cross-reference comment.

Why extrapolate at all, instead of just returning 0 above the last tooth (the
original, simpler choice): the reconstruction stops at the comb's last tooth
(the protocol Nyquist pi/4tau, set by the most-nested probe sequence);
interpolating the spectra with ``right=0`` told the pulse optimizer that the
band above is noise-free, and the optimizer parks its filter passband exactly
there (the 76-pulse/80tau NT solution has its fundamental at w ~ 3/tau): 85%
of that gate's TRUE error lived in the unmeasured band, and the predicted
infidelity under-read 6.6x relative to a time-domain (non-spectral) check.
In short: silently treating "unmeasured" as "zero noise" let the optimizer
game the objective by hiding a real noise band from itself.

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
# nuclear line (tooth 15 of the T=160 comb sits on the 2x line's shoulder):
# fitting with fewer teeth would just be fitting noise on top of that line's
# falling shoulder rather than the smooth background above it.
TAIL_N_FIT = 5


def fit_powerlaw_tail(wk, comp, n_fit=TAIL_N_FIT):
    """Fit ``comp ~ A*(w/wk[-1])^(-p)`` to the top ``n_fit`` comb teeth.

    Parameters
    ----------
    wk : array
        Comb frequencies (rad/tau, angular; tau=1 units per CLAUDE.md) at
        which the spectrum was actually measured by Stage 2 reconstruction --
        the "teeth" of the QNS comb. Must be sorted ascending, as produced by
        the reconstruction code.
    comp : array
        One real-valued component (real part or imaginary part -- this
        function only ever sees a real array) of a reconstructed spectral
        channel (e.g. S11, S22, S1212, or one of the cross-spectra) sampled
        at ``wk``. Same length as ``wk``.
    n_fit : int
        How many of the highest-frequency teeth to use for the fit (default
        ``TAIL_N_FIT``); see the module-level comment on that constant for
        why 5 is the right window for this comb.

    Returns
    -------
    (A, p) or None
        The fitted amplitude and decay exponent, or ``None`` when the last
        ``n_fit`` teeth do not describe a single well-behaved tail (too few
        points, the highest-frequency tooth landing at exactly 0, teeth that
        change sign across the window -- e.g. an oscillating cross-spectrum
        -- or a non-finite fit). Callers fall back to the old ``right=0``
        (noise-free) tail in that case, which is conservative for a
        component whose sign is not even settled.

    The exponent is clipped to [0, 6]: a rising fitted tail (noise artifact --
    physically the charge-noise components this model is built from are
    non-increasing power laws, so a positive slope here is a fit to noise, not
    signal) degrades to a flat extension at the fitted level rather than
    growing without bound as w increases.
    """
    wk = np.asarray(wk, dtype=float)
    y = np.asarray(comp, dtype=float)
    if wk.size < n_fit or y.size != wk.size:
        return None
    yf, wf = y[-n_fit:], wk[-n_fit:]
    s = np.sign(yf[-1])
    if s == 0 or np.any(np.sign(yf) != s) or np.any(wf <= 0):
        return None
    # Fit in log-log space: a power law A*(w/wk[-1])^(-p) is a straight line
    # of slope -p in log(comp) vs log(w/wk[-1]), so an ordinary least-squares
    # polyfit of degree 1 gives the exponent directly.
    ly = np.log(np.abs(yf))
    lw = np.log(wf / wk[-1])
    p = float(-np.polyfit(lw, ly, 1)[0])
    if not np.isfinite(p):
        return None
    p = float(np.clip(p, 0.0, 6.0))
    A = float(s * np.exp(np.mean(ly + p * lw)))
    return A, p


def tail_extend_interp(w_dense, wk, comp, n_fit=TAIL_N_FIT):
    """Evaluate a real spectrum component on the optimizer's dense grid.

    Parameters
    ----------
    w_dense : array
        The frequency grid the caller actually needs values on -- the
        optimizer's dense integration grid (``self.w`` in ``cz.py``/
        ``idle.py``), which extends well past the comb's last tooth.
    wk, comp : array
        As in ``fit_powerlaw_tail``: the measured comb frequencies and the
        real-valued spectral component sampled there.
    n_fit : int
        Passed straight through to ``fit_powerlaw_tail``.

    Inside the comb band (``wk[0] <= w <= wk[-1]``) this is a plain linear
    interpolation through the measured teeth. Beyond the last tooth it is the
    power-law tail from ``fit_powerlaw_tail`` when one exists, else the
    legacy flat ``right=0`` (i.e. "no more data -> assume no noise there").
    """
    wk = jnp.asarray(wk)
    comp = jnp.asarray(comp)
    base = jnp.interp(w_dense, wk, comp, right=0.)
    fit = fit_powerlaw_tail(np.asarray(wk), np.asarray(comp), n_fit)
    if fit is None:
        return base
    A, p = fit
    wlast = float(np.asarray(wk)[-1])
    # jnp.where evaluates BOTH branches eagerly (it is not a lazy if/else --
    # a JAX/numpy idiom for building an array under a mask with everything
    # traceable/vmap-able), so `tail` must be finite everywhere `w_dense` can
    # be, even below the band edge. Clamp w there with jnp.maximum so the
    # power law never sees w=0 (those clamped samples are the ones masked
    # out by jnp.where below anyway, so their actual value doesn't matter).
    ws = jnp.maximum(w_dense, wlast)
    tail = A * (ws / wlast) ** (-p)
    return jnp.where(w_dense > wlast, tail, base)


def tail_extend_interp_complex(w_dense, wk, fp, n_fit=TAIL_N_FIT):
    """Complex-valued wrapper around `tail_extend_interp` (the function the
    gate optimizers actually import).

    A reconstructed cross-spectrum channel (e.g. S12, S112, S212) is complex
    in general, while `fit_powerlaw_tail`/`tail_extend_interp` only handle a
    single real array. Here the real and imaginary parts of `fp` (the
    spectral values at the comb teeth `wk`) are each independently
    interpolated/extended -- they can have very different shapes (e.g. one
    part settling into a clean power law, the other oscillating and falling
    back to `right=0`) -- and recombined into one complex array on `w_dense`.
    """
    return (tail_extend_interp(w_dense, wk, jnp.real(fp), n_fit)
            + 1j * tail_extend_interp(w_dense, wk, jnp.imag(fp), n_fit))
