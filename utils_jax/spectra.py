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
    return jnp.exp(-(w-w0)**2/(2*sig**2))


@jax.jit
def S_11(w):
    """
    Self spectrum for qubit 1
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc=6e-7
    S0 =1e3
    St2 = 1e6
    w0=1*10**7
    return (S0*(2*L(w, 0, tc)+L(w, w0, 3*tc)+Gauss(w, 1.5*w0, 1/tc)+1.3*L(w, 2.5*w0, 0.75*tc))
            +St2*L(w, 0, 1e3*tc))


@jax.jit
def S_22(w):
    """
    Self spectrum for qubit 2
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc=6e-7
    S0 = 1.25e3
    St2 = 1e6
    w0=1*10**7
    return (S0*(2*L(w, 0, tc)+L(w, 1.2*w0, tc)+(2/3)*Gauss(w, 2.5*w0, 2/tc))
            +St2*L(w, 0, 1e3*tc))


@jax.jit
def S_1212(w):
    """
    Self spectrum for the ZZ (Ising) interaction
    :param w: ndarray of the frequencies at which to evaluate the spectrum.
    :return: ndarray of the value of the function.
    """
    tc=6e-7
    S0 = 0.6e3
    w0=2*10**7
    return S0*L(w, w0, tc)+2*S0/(1+(2*tc*w))


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

