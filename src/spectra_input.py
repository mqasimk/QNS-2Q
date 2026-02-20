"""
Analytical definitions of noise power spectral densities (PSDs).

This module provides various functions to generate common noise spectra used in
quantum noise spectroscopy simulations, including Lorentzian and Gaussian
models. It also defines specific spectra for two-qubit systems, including
self-spectra for individual qubits and the Ising interaction, as well as
cross-correlated spectra.
"""

import jax.numpy as jnp
import jax


@jax.jit
def L(w, w0, tc):
    """
    Generate a symmetric Lorentzian function.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the Lorentzian.
    w0 : float
        Location of the peak in frequency space.
    tc : float
        Correlation time for the noise process (determines the width).

    Returns
    -------
    jax.Array
        Value of the symmetric Lorentzian at frequencies `w`.
    """
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def Gauss(w, w0, sig):
    """
    A non-normalized Gaussian function.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the Gaussian.
    w0 : float
        Peak frequency of the Gaussian.
    sig : float
        Standard deviation (width) of the Gaussian.

    Returns
    -------
    jax.Array
        Value of the Gaussian at frequencies `w`.
    """
    return 0.5*(jnp.exp(-(w-w0)**2/(2*sig**2))+jnp.exp(-(w+w0)**2/(2*sig**2)))


@jax.jit
def S_11(w):
    """
    Self spectrum for qubit 1.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.

    Returns
    -------
    jax.Array
        PSD value for qubit 1.
    """
    tc = 1e-5
    S0 = 0*3e4
    St2 = 2.5e5
    w0 = 1.4e7
    return (S0*(0*Gauss(w, 1.75*w0, 10/tc)+L(w, w0, 1*tc))
            +St2*L(w, 0, 1.5*tc))


@jax.jit
def S_22(w):
    """
    Self spectrum for qubit 2.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.

    Returns
    -------
    jax.Array
        PSD value for qubit 2.
    """
    tc=1e-5
    S0 = 0*2.5e4
    St2 = 1e5
    w0=0.8e7
    return (S0*(Gauss(w, 1.8*w0, 10/tc)+Gauss(w, 2.5*w0, 20/tc))
            +St2*L(w, 0, 2*tc))


@jax.jit
def S_1212(w):
    """
    Self spectrum for the ZZ (Ising) interaction.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.

    Returns
    -------
    jax.Array
        PSD value for the Ising interaction.
    """
    tc = 1e-3
    S0 = 0*1e3
    St2 = 1e6
    w0 = 1.5*10**7
    #return S0*L(w, w0, 0.2*tc)+ St2*L(w, 0, 0.2*tc)
    return St2/(1+(2*tc*jnp.abs(w)))


@jax.jit
def S_1_2(w, gamma):
    """
    Cross spectrum for qubits 1 and 2.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.
    gamma : float
        Phase offset (time delay) for the cross-correlation.

    Returns
    -------
    jax.Array
        Complex-valued cross-spectrum.
    """
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


@jax.jit
def S_1_12(w, gamma12):
    """
    Cross spectrum between qubit 1 and the Ising interaction.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.
    gamma12 : float
        Phase offset (time delay) for the cross-correlation.

    Returns
    -------
    jax.Array
        Complex-valued cross-spectrum.
    """
    return jnp.sqrt(S_11(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)


@jax.jit
def S_2_12(w, gamma12):
    """
    Cross spectrum between qubit 2 and the Ising interaction.

    Parameters
    ----------
    w : jax.Array
        Frequencies at which to evaluate the spectrum.
    gamma12 : float
        Phase offset (time delay) for the cross-correlation.

    Returns
    -------
    jax.Array
        Complex-valued cross-spectrum.
    """
    return jnp.sqrt(S_22(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Parameters
    tau = 2.5e-8
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

    # Plotting
    xunits = 1e6
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
    axs[0, 0].legend([r'$\hat{S}_{1,1}^+(\omega_k)$', r'$S_{1,1}^+(\omega)$'], fontsize=plot_params['legendfont'])

    # S_22
    axs[0, 1].plot(wk_np / xunits, np.real(spectra_k['S_22']), 'r^')
    axs[0, 1].plot(w_np / xunits, np.real(spectra['S_22']), 'r--', lw=1.5 * plot_params['lw'])
    axs[0, 1].legend([r'$\hat{S}_{2,2}^+(\omega_k)$', r'$S_{2,2}^+(\omega)$'], fontsize=plot_params['legendfont'])

    # S_1212
    axs[0, 2].plot(wk_np / xunits, np.real(spectra_k['S_1212']), 'r^')
    axs[0, 2].plot(w_np / xunits, np.real(spectra['S_1212']), 'r--', lw=1.5 * plot_params['lw'])
    axs[0, 2].legend([r'$\hat{S}_{12,12}^+(\omega_k)$', r'$S_{12,12}^+(\omega)$'], fontsize=plot_params['legendfont'])

    # S_1_2
    axs[1, 0].plot(wk_np / xunits, np.real(spectra_k['S_1_2']), 'r^')
    axs[1, 0].plot(w_np / xunits, np.real(spectra['S_1_2']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 0].plot(wk_np / xunits, np.imag(spectra_k['S_1_2']), 'b^')
    axs[1, 0].plot(w_np / xunits, np.imag(spectra['S_1_2']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 0].legend([r'Re[$\hat{S}_{1,2}^+(\omega_k)$]', r'Re[$S_{1,2}^+(\omega)$]', r'Im[$\hat{S}_{1,2}^+(\omega_k)$]', r'Im[$S_{1,2}^+(\omega)$]'], fontsize=plot_params['legendfont'])

    # S_1_12
    axs[1, 1].plot(wk_np / xunits, np.real(spectra_k['S_1_12']), 'r^')
    axs[1, 1].plot(w_np / xunits, np.real(spectra['S_1_12']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 1].plot(wk_np / xunits, np.imag(spectra_k['S_1_12']), 'b^')
    axs[1, 1].plot(w_np / xunits, np.imag(spectra['S_1_12']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 1].legend([r'Re[$\hat{S}_{1,12}^+(\omega_k)$]', r'Re[$S_{1,12}^+(\omega)$]', r'Im[$\hat{S}_{1,12}^+(\omega_k)$]', r'Im[$S_{1,12}^+(\omega)$]'], fontsize=plot_params['legendfont'])

    # S_2_12
    axs[1, 2].plot(wk_np / xunits, np.real(spectra_k['S_2_12']), 'r^')
    axs[1, 2].plot(w_np / xunits, np.real(spectra['S_2_12']), 'r--', lw=1.5 * plot_params['lw'])
    axs[1, 2].plot(wk_np / xunits, np.imag(spectra_k['S_2_12']), 'b^')
    axs[1, 2].plot(w_np / xunits, np.imag(spectra['S_2_12']), 'b--', lw=1.5 * plot_params['lw'])
    axs[1, 2].legend([r'Re[$\hat{S}_{2,12}^+(\omega_k)$]', r'Re[$S_{2,12}^+(\omega)$]', r'Im[$\hat{S}_{2,12}^+(\omega_k)$]', r'Im[$S_{2,12}^+(\omega)$]'], fontsize=plot_params['legendfont'])

    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel(r'$\omega$(MHz)', fontsize=plot_params['xlabelfont'])
            ax.tick_params(direction='in', labelsize=plot_params['tickfont'])
            ax.grid(True, alpha=0.3)
            ax.set_yscale('asinh')

    plt.tight_layout()

    # Save the simulated spectra
    fname = "DraftRun_NoSPAM_Boring"
    parent_dir = os.pardir
    path = os.path.join(parent_dir, fname)
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
