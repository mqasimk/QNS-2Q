import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
# from trajectories import custom_y
from jax.scipy.linalg import expm
import scipy as sp
import jaxopt

def S_11(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=1*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))#+S0*(1/(1+(tc**2)*(jnp.abs(w)-(6e6))**2))


def S_22(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=1*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))


def S_1_2(w, gamma):
    return S_11(w)*np.exp(-1j*w*gamma)

def S_12(w):
    tc=2/(1*10**6)
    S0 = 3e3
    w0=4*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))

#######################################################
# Utility functions

def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O)@z2q[a]@z2q[b]@O@z2q[a]@z2q[b])/4


def ff(tk, w):
    return jnp.sum(jnp.array([1j*((-1)**k)*(jnp.exp(-1j*w*tk[k+1])-jnp.exp(-1j*w*tk[k]))/w for k in range(jnp.size(tk)-1)]), axis=0)


def Gp_re(t_i, t_j, w, M):
    return jnp.real(ff(t_i, w)*ff(t_j, -w)*jnp.sin(w*M*t_i[-1]*0.5)**2/jnp.sin(w*t_i[-1]*0.5)**2)


def Gp_im(t_i, t_j, w, M):
    return jnp.imag(ff(t_i, w)*ff(t_j, -w)*jnp.sin(w*M*t_i[-1]*0.5)**2/jnp.sin(w*t_i[-1]*0.5)**2)


def y_t(t, tk):
    return jnp.sum(jnp.array([((-1)**i)*jnp.heaviside(t-tk[i], 1)*jnp.heaviside(tk[i+1] - t, 1) for i in range(jnp.size(tk) - 1)]), axis=0)

def makeCO(vt, O, SMat, M, **kwargs):
    w = kwargs.get('w_axis')
    vt12 = jnp.unique(jnp.concatenate((vt[0], vt[1])))
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    i=0
    for ti in [vt[0], vt[1], vt12]:
        j = 0
        for tj in [vt[0], vt[1], vt12]:
            Gp = Gp.at[i, j].set(Gp_re_map(ti, tj, w, M) + 1j*Gp_im_map(ti, tj, w, M))
            j+=1
        i+=1
    CO = 1j*0.
    for i in range(3):
        for j in range(3):
            CO += -z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], w)/(4*jnp.pi)
    return jnp.real(jax.scipy.linalg.expm(CO))

def inf_ID(vt_in, SMat, M, T, w):
    vt = params_to_tk(vt_in, T)
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    vt12 = jnp.unique(jnp.concatenate((vt[0], vt[1])))
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    i=0
    for ti in [vt[0], vt[1], vt12]:
        j = 0
        for tj in [vt[0], vt[1], vt12]:
            Gp = Gp.at[i, j].set(Gp_re_map(ti, tj, w, M) + 1j*Gp_im_map(ti, tj, w, M))
            j+=1
        i+=1
    L_diag = jnp.zeros(16)
    k=0
    for O in p2q:
        CO = 1j*0.
        for i in range(3):
            for j in range(3):
                CO += -z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], w)/(4*jnp.pi)
        L_diag = L_diag.at[k].set(jnp.real(jnp.trace(O@jax.scipy.linalg.expm(-CO*0.5)@O)*0.25))
        k += 1
    dt = T/32.
    return (1.-jnp.sum(L_diag)/16)+jnp.sum(jnp.array([(dt-jnp.abs((vt[i][j]-vt[i][j+1])))**2 for i in range(2) for j in range(vt.shape[1]-1)]), axis=0)*1e5


def optimize_ctrl(vt, SMat, M, step, max_iter, w):
    for i in range(max_iter):
        vt = vt - step*jax.grad(inf_ID, argnums=0)(vt, SMat, M, w)
        print(f'Iteration {i+1} completed')
    return vt


def hyperOpt(SMat, nP, M, T, w):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID, maxiter=15, jit=False, method='L-BFGS-B',
                                            options={'disp': True, 'gtol': 1e-9, 'ftol': 1e-9,
                                                     'maxfun': 1000, 'maxls': 20})
    opt_out = []
    init_params = []
    iter = 0
    for i in nP:
        vt = jnp.array([np.random.rand(i)*T, np.random.rand(i)*T])
        init_params.append(vt)
        lower_bnd = jnp.zeros_like(vt)
        upper_bnd = jnp.ones_like(vt)*T
        bnds = (lower_bnd, upper_bnd)
        opt = optimizer.run(vt, bnds, SMat, M, T, w)
        opt_out.append(opt)
        iter += 1
    inf_ID_out = jnp.array([inf_ID(opt_out[i].params, SMat, M, T, w) for i in range(iter)])
    vt_min = opt_out[jnp.argmin(inf_ID_out)].params
    vt_opt = params_to_tk(vt_min, T)
    vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt, inf_ID(vt_opt, SMat, M, T, w), vt_init, inf_ID(vt_init, SMat, M, T, w)

def Lambda(Oi, Oj, vt, SMat, M, w):
    vt12 = jnp.unique(jnp.concatenate((vt[0], vt[1])))
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    i=0
    for ti in [vt[0], vt[1], vt12]:
        j = 0
        for tj in [vt[0], vt[1], vt12]:
            Gp = Gp.at[i, j].set(Gp_re_map(ti, tj, w, M) + 1j*Gp_im_map(ti, tj, w, M))
            j+=1
        i+=1
    CO = 1j*0.
    for i in range(3):
        for j in range(3):
            CO += -z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(Oi, i+1, 0)-1)*Gp[i, j], w)/(4*jnp.pi)
    return jnp.real(jnp.trace(Oi@jax.scipy.linalg.expm(-CO*0.5)@Oj)*0.25)


def params_to_tk(params, T):
    vt = jnp.zeros((params.shape[0], params.shape[1]+2))
    vt = vt.at[0, 0].set(0.)
    vt = vt.at[1, 0].set(0.)
    vt = vt.at[0, -1].set(T)
    vt = vt.at[1, -1].set(T)
    vt = vt.at[0, 1:params.shape[1]+1].set(jnp.sort(params[0]))
    vt = vt.at[1, 1:params.shape[1]+1].set(jnp.sort(params[1]))
    return vt

T = 1e-5
mc = 14
M=10

p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
w = jnp.linspace(0.1, 2*jnp.pi*mc/T, 2000)
wk = jnp.array([2*jnp.pi*(n)/T for n in range(mc+1)])
gamma = T/5
S_11_k = S_11(wk)
S_22_k = S_22(wk)
S_12_k = S_12(wk)
SMat = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
SMat = SMat.at[0, 0].set(jnp.interp(w, wk, S_11_k))
SMat = SMat.at[1, 1].set(jnp.interp(w, wk, S_22_k))
SMat = SMat.at[2, 2].set(jnp.interp(w, wk, S_12_k))
SMat = SMat.at[0, 1].set(S_1_2(w, gamma))
SMat = SMat.at[1, 0].set(jnp.conj(S_1_2(w, gamma)))

L_map = jax.vmap(jax.vmap(Lambda, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, None, None, None, None))

vt_opt, inf_min, vt_init, inf_init = hyperOpt(SMat, [18, 19, 20], M, T, w)
L_opt = L_map(p2q, p2q, vt_opt, SMat, M, w)
L_init = L_map(p2q, p2q, vt_init, SMat, M, w)
print('End')
