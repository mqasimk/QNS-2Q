import jax.numpy as jnp
import jax



@jax.jit
def L(w, w0, tc):
    """
    Generate a symmetric Lorentzian function.
    :param w: ndarray of the frequencies at which to evaluate the Lorentzian.
    :param w0: float with the location of the peak in frequency space.
    :param tc: float with the correlation time for the noise process, which determines the width of the Lorentzian.
    :return: ndarray of the value of the function.
    """
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def Gauss(w, w0, sig):
    """
    A Gaussian function that is not normalized.
    :param w: ndarray of the frequencies at which to evaluate the Gaussian.
    :param w0: float with the peak of the Gaussian.
    :param sig: float with the standard deviation of the Gaussian.
    :return: ndarray of the value of the function.
    """
    return 0.5*(jnp.exp(-(w-w0)**2/(2*sig**2))+jnp.exp(-(w+w0)**2/(2*sig**2)))


@jax.jit
def S_11(w):
    """
    Self spectrum for qubit 1
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc = 1e-5
    S0 = 6e4
    St2 = 1e5
    w0 = 1.25e7
    return (S0*(Gauss(w, 1.75*w0, 10/tc))
            +St2*L(w, 0, 1.5*tc))


@jax.jit
def S_22(w):
    """
    Self spectrum for qubit 2
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc=1e-5
    S0 = 5e4
    St2 = 1e5
    w0=0.75e7
    return (S0*(Gauss(w, 1.5*w0, 10/tc)+Gauss(w, 2.5*w0, 10/tc))
            +St2*L(w, 0, 2*tc))


@jax.jit
def S_1212(w):
    """
    Self spectrum for the ZZ (Ising) interaction
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc = 1e-4
    S0 = 0*1e3
    St2 = 1e6
    w0 = 1.5*10**7
    return S0*L(w, w0, 0.2*tc)+ St2*L(w, 0, 0.2*tc)#St2/(1+(2*tc*jnp.abs(w)))


@jax.jit
def S_1_2(w, gamma):
    """
    Cross spectrum for qubits 1,2
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: complex valued ndarray of the value of the function.
    """
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


@jax.jit
def S_1_12(w, gamma12):
    """
    Cross spectrum between qubit 1 and the Ising interaction
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: complex valued ndarray of the value of the function.
    """
    return jnp.sqrt(S_11(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)


@jax.jit
def S_2_12(w, gamma12):
    """
    Cross spectrum between qubit 2 and the Ising interaction
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: complex valued ndarray of the value of the function.
    """
    return jnp.sqrt(S_22(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Define frequency range
    w = jnp.linspace(-2 * np.pi * 20 / (160*2.5e-8), 2 * np.pi * 20 / (160*2.5e-8), 5001)
    wk = jnp.linspace(-2 * np.pi * 20 / (160*2.5e-8), 2 * np.pi * 20 / (160*2.5e-8), 41)
    # Parameters
    gamma = 160*2.5e-8/14
    gamma12 = 160*2.5e-8/28

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
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Spectra with arsinh scaled y-axis')
    axes = axes.flatten()

    for i, (name, s) in enumerate(spectra.items()):
        ax = axes[i]
        # Convert to numpy array to ensure compatibility with matplotlib
        s_np = np.array(s)

        line_real, = ax.plot(w, np.real(s_np), label='Real')
        line_imag, = ax.plot(w, np.imag(s_np), label='Imag', linestyle='--')

        # Plot harmonic frequencies
        s_k = np.array(spectra_k[name])
        ax.plot(wk, np.real(s_k), 'o', color=line_real.get_color(), label='Real (Harmonics)')
        ax.plot(wk, np.imag(s_k), 'x', color=line_imag.get_color(), label='Imag (Harmonics)')

        ax.set_title(name)
        ax.set_xlabel('Frequency (w)')
        ax.set_ylabel('arsinh(S)')
        ax.set_yscale('asinh')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
