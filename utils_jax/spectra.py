import jax.numpy as jnp
import jax


@jax.jit
def L(w, w0, tc):
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def Gauss(w, w0, sig):
    return jnp.exp(-(w-w0)**2/(2*sig**2))


@jax.jit
def S_11(w):
    tc=6e-7
    S0 =1e3
    St2 = 1e6
    w0=2*10**7
    return (S0*(6*L(w, 0, 0.5*tc)+L(w, w0, tc)+Gauss(w, 1.25*w0, 2/tc)+4*L(w, 2.25*w0, 0.5*tc))
            +St2*L(w, 0, 1e3*tc))


@jax.jit
def S_22(w):
    tc=6e-7
    S0 = 1.25e3
    St2 = 1e6
    w0=3*10**7
    return (S0*(6*L(w, 0, 0.5*tc)+2*L(w, w0, 0.5*tc)+Gauss(w, 1.5*w0, 3/tc))
            +St2*L(w, 0, 1e3*tc))


@jax.jit
def S_1212(w):
    tc=1e-6
    S0 = 1e3
    w0=0*10**9
    return S0/(1+(tc*w))#L(w, w0, 0.5*tc)


@jax.jit
def S_1_2(w, gamma):
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


@jax.jit
def S_1_12(w, gamma12):
    return jnp.sqrt(S_11(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)


@jax.jit
def S_2_12(w, gamma12):
    return jnp.sqrt(S_22(w)*S_1212(w))*jnp.exp(-1j*w*gamma12)

