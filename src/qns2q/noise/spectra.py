"""
Analytical definitions of noise power spectral densities (PSDs).

**Units: the minimum pulse separation tau is the unit of time (tau = 1).**
Frequencies are dimensionless angular frequencies w~ = w*tau and the spectra are
dimensionless S~ = tau*S; every observable depends only on such products, so the
physics is identical to the legacy SI-anchored model (see
``tests/test_tau_invariance.py``). All constants below are written as explicit
products of the legacy SI values and ``_TAU_SI`` (the historical tau = 25 ns), so
the provenance stays visible and the conversion is exact: e.g. a peak at
5e6 rad/s becomes w~0 = 5e6*_TAU_SI = 0.125, an amplitude of 5e4 /s becomes
S~ = 5e4*_TAU_SI = 1.25e-3, a correlation time of 3e-7 s becomes t~c = 12.
``_TAU_SI`` is ONLY this conversion constant -- no pipeline code uses it as a
timescale.

Two noise models are provided and selected at import time by the ``QNS2Q_REGIME``
environment variable (see ``run_paths.current_regime``):

* ``featured`` (default) -- multi-peak self-spectra; used for the noise-tailoring
  advantage demonstrations.
* ``bland`` -- monotonic Lorentzian self-spectra; the original model used for the
  in-paper QNS reconstruction figures (recovered from commit ``77e516a^``).

Switching is global for the run, e.g. ``QNS2Q_REGIME=bland python id_optimize.py``.
The public API (``S_11``, ``S_22``, ``S_1212`` self-spectra and ``S_1_2``, ``S_1_12``,
``S_2_12`` cross-spectra) is identical in both regimes, so importing code needs no
changes. Cross-spectra are derived from the selected self-spectra.
"""

import jax.numpy as jnp
import jax

from qns2q.paths import current_regime, run_folder, project_root

# Legacy SI anchor (tau = 25 ns): conversion constant for the dimensionless
# (tau-unit) spectral definitions below. Frequencies x _TAU_SI, times / _TAU_SI,
# spectral amplitudes x _TAU_SI.
_TAU_SI = 2.5e-8


@jax.jit
def L(w, w0, tc):
    """Symmetric Lorentzian: peak at +/-w0, width set by correlation time tc."""
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def Gauss(w, w0, sig):
    """Symmetric (non-normalized) Gaussian: peak at +/-w0, std dev sig."""
    return 0.5*(jnp.exp(-(w-w0)**2/(2*sig**2))+jnp.exp(-(w+w0)**2/(2*sig**2)))


# --- Self-spectra: selected by regime -------------------------------------------
# Both branches define S_11/S_22/S_1212 with identical signatures and __name__, so
# downstream code (and the spec_vec_names stored in params.npz) is regime-agnostic.

_REGIME = current_regime()

if _REGIME == "bland":
    # Monotonic model -- shape families from 77e516a^:src/spectra_input.py, weakened
    # ~5x (and the Ising cusp broadened 10x) so the w=0 point is measurable by the
    # multi-time FID decay-slope DC protocol. The original amplitudes (S11/S22/S1212 =
    # 2.5e5/1e5/1e6, Ising tc=1e-5) put the joint FID/FID coherences -- which decay
    # with the SUM of all self variances -- below the n_shots=2000 ensemble floor by
    # the 3rd of 10 sweep points, and the 5e4 rad/s Ising cusp needed t >> 130 us to
    # reach the linear (slope = S(0)) regime vs the 40 us sweep. With these values the
    # worst decay exponent at t_max = 1600 tau is ~3.4 (signal >= 1.5x floor) and
    # every channel is >= 95% linear over the fit window.
    # All arguments are in tau units: w is w*tau, constants are SI x/_TAU_SI.
    @jax.jit
    def S_11(w):
        """Self spectrum for qubit 1 (bland). w and S in tau units."""
        tc = 1e-6 / _TAU_SI
        S0 = 0 * 3e4 * _TAU_SI
        St2 = 5e4 * _TAU_SI
        w0 = 1.4e7 * _TAU_SI
        return (S0*(0*Gauss(w, 1.75*w0, 10/tc)+L(w, w0, 1*tc))
                +St2*L(w, 0, tc))

    @jax.jit
    def S_22(w):
        """Self spectrum for qubit 2 (bland). w and S in tau units."""
        tc = 1.5e-6 / _TAU_SI
        S0 = 0 * 2.5e4 * _TAU_SI
        St2 = 2e4 * _TAU_SI
        w0 = 0.8e7 * _TAU_SI
        return (S0*(Gauss(w, 1.8*w0, 10/tc)+Gauss(w, 2.5*w0, 20/tc))
                +St2*L(w, 0, tc))

    @jax.jit
    def S_1212(w):
        """Self spectrum for the ZZ (Ising) interaction (bland). w, S in tau units."""
        tc = 1e-6 / _TAU_SI
        St2 = 4e4 * _TAU_SI
        return St2/(1+(2*tc*jnp.abs(w)))

else:  # "featured"
    # Multi-peak model -- current HEAD definitions.
    # All arguments are in tau units: w is w*tau, constants are SI x/_TAU_SI.
    @jax.jit
    def S_11(w):
        """Self spectrum for qubit 1 (featured: peak@0.125/tau + plateau + peak + DC)."""
        peak1 = (5e4 * _TAU_SI) * L(w, 5e6 * _TAU_SI, 3e-7 / _TAU_SI)
        plateau = (4e4 * _TAU_SI) * Gauss(w, 1.6e7 * _TAU_SI, 3e6 * _TAU_SI)
        peak2 = (3e4 * _TAU_SI) * Gauss(w, 2.7e7 * _TAU_SI, 2e6 * _TAU_SI)
        dc = (5e3 * _TAU_SI) * L(w, 0, 1e-6 / _TAU_SI)
        return peak1 + plateau + peak2 + dc

    @jax.jit
    def S_22(w):
        """Self spectrum for qubit 2 (featured: plateau + peak@0.5/tau + bump + DC)."""
        plateau = (3e4 * _TAU_SI) * Gauss(w, 8e6 * _TAU_SI, 6e6 * _TAU_SI)
        peak = (6e4 * _TAU_SI) * L(w, 2e7 * _TAU_SI, 2.5e-7 / _TAU_SI)
        bump = (2e4 * _TAU_SI) * Gauss(w, 2.8e7 * _TAU_SI, 1.5e6 * _TAU_SI)
        dc = (8e3 * _TAU_SI) * L(w, 0, 1.5e-6 / _TAU_SI)
        return plateau + peak + bump + dc

    @jax.jit
    def S_1212(w):
        """Self spectrum for the ZZ (Ising) interaction (featured: peak + hump + DC)."""
        peak1 = (4e4 * _TAU_SI) * L(w, 1.2e7 * _TAU_SI, 4e-7 / _TAU_SI)
        hump = (2e4 * _TAU_SI) * Gauss(w, 2.3e7 * _TAU_SI, 4e6 * _TAU_SI)
        dc = (1e4 * _TAU_SI) * L(w, 0, 2e-6 / _TAU_SI)
        return peak1 + hump + dc


# --- Cross-spectra: derived from the selected self-spectra (regime-agnostic) -----
#
# These are the MAXIMALLY-COHERENT cross-spectra: |S_ab(w)| = sqrt(S_aa S_bb), i.e.
# magnitude-squared coherence gamma^2(w) = |S_ab|^2/(S_aa S_bb) = 1 at every frequency,
# with a pure linear phase exp(-i w gamma). This is realized EXACTLY by the synthesis
# in trajectories.make_noise_mat_arr: b_1, b_2, b_12 are built from one shared Gaussian
# draw (A, B), scaled by sqrt(S_aa) and time-shifted by gamma_a, so the true synthesized
# cross-PSD is exactly this form at each synthesis line. (A naive FFT of the trajectories
# estimates coherence ~0.93, but that deficit is a finite-window leakage artifact of the
# time-shift gamma -- it vanishes at gamma=0 -- not a real decorrelation; the QNS forward
# model and reconstruction both see coherence 1.)

@jax.jit
def S_1_2(w, gamma):
    """Cross spectrum for qubits 1 and 2 (phase offset gamma)."""
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


@jax.jit
def S_1_12(w, gamma12):
    """Cross spectrum between qubit 1 and the Ising interaction."""
    return jnp.sqrt(S_11(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)


@jax.jit
def S_2_12(w, gamma12):
    """Cross spectrum between qubit 2 and the Ising interaction."""
    return jnp.sqrt(S_22(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Parameters (tau units: tau = 1)
    tau = 1.0
    T = 160 * tau
    truncate = 20
    gamma = T / 14
    gamma12 = T / 28

    # Define frequency range
    w = jnp.linspace(-2 * np.pi * truncate / T, 2 * np.pi * truncate / T, 5001)
    wk = jnp.linspace(-2 * np.pi * truncate / T, 2 * np.pi * truncate / T, 2 * truncate + 1)

    # Compute spectra
    spectra = {
        "S_11": S_11(w),
        "S_22": S_22(w),
        "S_1212": S_1212(w),
        "S_1_2": S_1_2(w, gamma),
        "S_1_12": S_1_12(w, gamma12),
        "S_2_12": S_2_12(w, gamma12)
    }

    # Compute spectra at harmonic frequencies
    spectra_k = {
        "S_11": S_11(wk),
        "S_22": S_22(wk),
        "S_1212": S_1212(wk),
        "S_1_2": S_1_2(wk, gamma),
        "S_1_12": S_1_12(wk, gamma12),
        "S_2_12": S_2_12(wk, gamma12)
    }

    # Plotting (dimensionless tau-unit axes: x = w*tau, y = S*tau)
    xunits = 1.0
    plot_params = {
        'lw': 1,
        'legendfont': 12,
        'xlabelfont': 16,
        'ylabelfont': 16,
        'tickfont': 12,
    }

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    # Convert jax arrays to numpy for plotting
    w_np = np.array(w)
    wk_np = np.array(wk)

    # S_11
    axs[0, 0].plot(wk_np / xunits, np.real(spectra_k['S_11']), 'r^')
    axs[0, 0].plot(w_np / xunits, np.real(spectra['S_11']), 'r--', lw=1.5 * plot_params['lw'])
    axs[0, 0].legend([r'$\hat{S}_{1,1}(\omega_k)$', r'$S_{1,1}(\omega)$'], fontsize=plot_params['legendfont'])

    # S_22
    axs[0, 1].plot(wk_np / xunits, np.real(spectra_k['S_22']), 'r^')
    axs[0, 1].plot(w_np / xunits, np.real(spectra['S_22']), 'r--', lw=1.5 * plot_params['lw'])
    axs[0, 1].legend([r'$\hat{S}_{2,2}(\omega_k)$', r'$S_{2,2}(\omega)$'], fontsize=plot_params['legendfont'])

    # S_1212
    axs[0, 2].plot(wk_np / xunits, np.real(spectra_k['S_1212']), 'r^')
    axs[0, 2].plot(w_np / xunits, np.real(spectra['S_1212']), 'r--', lw=1.5 * plot_params['lw'])
    axs[0, 2].legend([r'$\hat{S}_{12,12}(\omega_k)$', r'$S_{12,12}(\omega)$'], fontsize=plot_params['legendfont'])

    # S_1_2
    axs[1, 0].plot(wk_np / xunits, np.real(spectra_k['S_1_2']), 'r^')
    axs[1, 0].plot(w_np / xunits, np.real(spectra['S_1_2']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 0].plot(wk_np / xunits, np.imag(spectra_k['S_1_2']), 'b^')
    axs[1, 0].plot(w_np / xunits, np.imag(spectra['S_1_2']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 0].legend([r'Re[$\hat{S}_{1,2}(\omega_k)$]', r'Re[$S_{1,2}(\omega)$]', r'Im[$\hat{S}_{1,2}(\omega_k)$]', r'Im[$S_{1,2}(\omega)$]'], fontsize=plot_params['legendfont'])

    # S_1_12
    axs[1, 1].plot(wk_np / xunits, np.real(spectra_k['S_1_12']), 'r^')
    axs[1, 1].plot(w_np / xunits, np.real(spectra['S_1_12']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 1].plot(wk_np / xunits, np.imag(spectra_k['S_1_12']), 'b^')
    axs[1, 1].plot(w_np / xunits, np.imag(spectra['S_1_12']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 1].legend([r'Re[$\hat{S}_{1,12}(\omega_k)$]', r'Re[$S_{1,12}(\omega)$]', r'Im[$\hat{S}_{1,12}(\omega_k)$]', r'Im[$S_{1,12}(\omega)$]'], fontsize=plot_params['legendfont'])

    # S_2_12
    axs[1, 2].plot(wk_np / xunits, np.real(spectra_k['S_2_12']), 'r^')
    axs[1, 2].plot(w_np / xunits, np.real(spectra['S_2_12']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 2].plot(wk_np / xunits, np.imag(spectra_k['S_2_12']), 'b^')
    axs[1, 2].plot(w_np / xunits, np.imag(spectra['S_2_12']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 2].legend([r'Re[$\hat{S}_{2,12}(\omega_k)$]', r'Re[$S_{2,12}(\omega)$]', r'Im[$\hat{S}_{2,12}(\omega_k)$]', r'Im[$S_{2,12}(\omega)$]'], fontsize=plot_params['legendfont'])

    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel(r'$\omega\tau$', fontsize=plot_params['xlabelfont'])
            ax.tick_params(direction='in', labelsize=plot_params['tickfont'])
            ax.grid(True, alpha=0.3)
            ax.set_yscale('asinh')

    plt.tight_layout()

    # Save the simulated spectra into the active regime's run folder
    fname = run_folder()
    path = os.path.join(project_root(), fname)
    if not os.path.exists(path):
        os.mkdir(path)

    # Map keys to match specs.npz format
    key_map = {
        "S_11": "S11",
        "S_22": "S22",
        "S_1212": "S1212",
        "S_1_2": "S12",
        "S_1_12": "S112",
        "S_2_12": "S212"
    }

    save_data = {key_map[k]: np.array(v) for k, v in spectra_k.items()}
    save_data['wk'] = np.array(wk)

    # Save parameters as well
    save_data['T'] = T
    save_data['truncate'] = truncate
    save_data['gamma'] = gamma
    save_data['gamma_12'] = gamma12

    np.savez(os.path.join(path, 'simulated_spectra.npz'), **save_data)
    print(f"Simulated spectra saved to {os.path.join(path, 'simulated_spectra.npz')}")

    plt.show()
