import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
# from trajectories import custom_y
from jax.scipy.linalg import expm
import scipy as sp
import jaxopt

def L(w, w0, tc):
    return 1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2)

def S_11(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=6*10**6
    return S0*(L(w, 0, 0.5*tc)+L(w, w0, tc))


def S_22(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=6*10**6
    return S0*(L(w, 0, 0.5*tc)+L(w, w0, 0.5*tc))


def S_1_2(w, gamma):
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


def S_12(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=5*10**6
    return 0.5*S0*L(w, w0, tc*0.5)

#######################################################
# Utility functions

def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O)@z2q[a]@z2q[b]@O@z2q[a]@z2q[b])/4


def ff(tk, w):
    return jnp.sum(jnp.array([1j*((-1)**k)*(jnp.exp(-1j*w*tk[k+1])-jnp.exp(-1j*w*tk[k]))/w for k in range(jnp.size(tk)-1)]), axis=0)


def Gp_re(i, j, vt_list, w, M):
    return jnp.real(ff(vt_list[i], w)*ff(vt_list[j], -w)*jnp.sin(w*M*vt_list[0][-1]*0.5)**2/jnp.sin(w*vt_list[0][-1]*0.5)**2)


def Gp_im(i, j, vt_list, w, M):
    return jnp.imag(ff(vt_list[i], w)*ff(vt_list[j], -w)*jnp.sin(w*M*vt_list[0][-1]*0.5)**2/jnp.sin(w*vt_list[0][-1]*0.5)**2)


def y_t(t, tk):
    return jnp.sum(jnp.array([((-1)**i)*jnp.heaviside(t-tk[i], 1)*jnp.heaviside(tk[i+1] - t, 1) for i in range(jnp.size(tk) - 1)]), axis=0)

def make_tk12(tk1, tk2):
    x = tk1[~jnp.isin(tk1, tk2)]
    y = tk2[~jnp.isin(tk2, tk1)]
    z = jnp.zeros(x.size+y.size+2)
    z = z.at[0].set(0.)
    z = z.at[-1].set(tk1[-1])
    z = z.at[1:z.size-1].set(jnp.sort(jnp.concatenate((x, y))))
    return z

def makeCO(vt, O, SMat, M, **kwargs):
    w = kwargs.get('w_axis')
    vt12 = make_tk12(vt[0], vt[1])
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


def CO_sum_els(O, SMat, Gp, i, j, w):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], w)/(4*jnp.pi)


def Lambda_diags(SMat, Gp, w):
    inds = jnp.array([0, 1, 2])
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els,
                                            in_axes=(None, None, None, None, 0, None)),
                                   in_axes=(None, None, None, 0, None, None)),
                          in_axes=(0, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q, SMat, Gp, inds, inds, w), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(p2q[i]@jax.scipy.linalg.expm(-CO[i]*0.5)@p2q[i])*0.25 for i in range(CO.shape[0])]))


def inf_ID(params, nPs, SMat, M, T, w):
    vt1 = params_to_tk(params, T, [0, nPs[0]])
    vt2 = params_to_tk(params, T, [nPs[0], nPs[0]+nPs[1]])
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(i, j, vt, w, M) + 1j*Gp_im_map(i, j, vt, w, M))
    L_diag = Lambda_diags(SMat, Gp, w)
    dt = T/32
    return (1.-jnp.sum(L_diag)/16)+jnp.sum(jnp.array([(dt-jnp.abs((vt[i][j]-vt[i][j+1])))**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)*1e7

def infidelity(params, SMat, M, w):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(i, j, vt, w, M) + 1j*Gp_im_map(i, j, vt, w, M))
    L_diag = Lambda_diags(SMat, Gp, w)
    return 1.-jnp.sum(L_diag)/16

def hyperOpt(SMat, nPs, M, T, w):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID, maxiter=30, jit=False, method='L-BFGS-B',
                                            options={'disp': True, 'gtol': 1e-9, 'ftol': 1e-6,
                                                     'maxfun': 1000, 'maxls': 10})
    opt_out = []
    init_params = []
    for i in nPs[0]:
        opt_out_temp = []
        for j in nPs[1]:
            vt = jnp.array(np.random.rand(i+j)*T)
            init_params.append(vt)
            lower_bnd = jnp.zeros_like(vt)
            upper_bnd = jnp.ones_like(vt)*T
            bnds = (lower_bnd, upper_bnd)
            opt = optimizer.run(vt, bnds, jnp.array([i, j]), SMat, M, T, w)
            opt_out_temp.append(opt)
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt = [params_to_tk(vt_min_arr, T, [0, nPs[0][inds_min[0]]]), params_to_tk(vt_min_arr, T, [nPs[0][inds_min[0]], nPs[0][inds_min[0]]+nPs[1][inds_min[1]]])]
    vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
    #vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt, inf_ID_out[inds_min]


def Lambda(Oi, Oj, vt, SMat, M, w):
    vt12 = make_tk12(vt[0], vt[1])
    vt.append(vt12)
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(i, j, vt, w, M) + 1j*Gp_im_map(i, j, vt, w, M))
    CO = 1j*0.
    for i in range(3):
        for j in range(3):
            CO += -z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(Oi, i+1, 0)-1)*Gp[i, j], w)/(4*jnp.pi)
    return jnp.real(jnp.trace(Oi@jax.scipy.linalg.expm(-CO*0.5)@Oj)*0.25)


def params_to_tk(params, T, shape):
    vt = jnp.zeros(shape[1]-shape[0]+2)
    vt = vt.at[0].set(0.)
    vt = vt.at[-1].set(T)
    vt = vt.at[1:shape[1]-shape[0]+1].set(jnp.sort(params[shape[0]:shape[1]]))
    return vt


def cpmg_vt(T, n):
    tk = [(k+0.50)*T/(2*n) for k in range(int(2*n))]
    tk.append(T)
    tk.insert(0,0.)
    return jnp.array(tk)


def cdd1_vt(T, n):
    tk = [(k+1)*T/(2*n) for k in range(int(2*n-1))]
    tk.append(T)
    tk.insert(0,0.)
    return jnp.array(tk)


def opt_known_pulses(nCs, SMat, M, T, w):
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    seq = [cpmg_vt, cdd1_vt]
    L_map = jax.vmap(jax.vmap(Lambda, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, None, None, None, None))
    infidelity_arr = jnp.zeros((2, 2, len(nCs[0]), len(nCs[1])))
    for i in range(2):
        for j in range(2):
            for k in range(len(nCs[0])):
                for l in range(len(nCs[1])):
                    tk1 = seq[i](T, nCs[0][k])
                    tk2 = seq[j](T, nCs[1][l])
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-L_map(p2q, p2q, vt, SMat, M, w).trace()/16)
    inds_min = jnp.unravel_index(jnp.argmin(infidelity_arr), infidelity_arr.shape)
    tk1 = seq[inds_min[0]](T, nCs[0][inds_min[2]])
    tk2 = seq[inds_min[1]](T, nCs[1][inds_min[3]])
    tk12 = make_tk12(tk1, tk2)
    vt_opt = [tk1, tk2, tk12]
    return vt_opt, infidelity_arr[inds_min[0], inds_min[1], inds_min[2], inds_min[3]]


T = 1e-5
mc = 16
M = 20
gamma = T/25

p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
w = jnp.linspace(0.1, 2*jnp.pi*mc/T, 2000)
wk = jnp.array([2*jnp.pi*(n)/T for n in range(mc+1)])
S_11_k = S_11(wk)
S_22_k = S_22(wk)
S_12_k = S_12(wk)
S_1_2_k = S_1_2(wk, gamma)
SMat = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)

SMat = SMat.at[0, 0].set(jnp.interp(w, wk, S_11_k))
SMat = SMat.at[1, 1].set(jnp.interp(w, wk, S_22_k))
SMat = SMat.at[2, 2].set(jnp.interp(w, wk, S_12_k))
SMat = SMat.at[0, 1].set(jnp.interp(w, wk, S_1_2_k))
SMat = SMat.at[1, 0].set(jnp.interp(w, wk, jnp.conj(S_1_2_k)))

SMat_ideal = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
SMat_ideal = SMat_ideal.at[0, 0].set(S_11(w))
SMat_ideal = SMat_ideal.at[1, 1].set(S_22(w))
SMat_ideal = SMat_ideal.at[2, 2].set(S_12(w))
SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(w, gamma))
SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(w, gamma)))


L_map = jax.vmap(jax.vmap(Lambda, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, None, None, None, None))

nCs = [[4,5,8,9,12], [4,5,8,9,12]]
known_opt, known_inf = opt_known_pulses(nCs, SMat, M, T, w)
L_known = L_map(p2q, p2q, known_opt, SMat_ideal, M, w)
inf_known = infidelity(known_opt, SMat_ideal, M, w)

print('infidelity over known seqs: ')
print(inf_known)
print('number of pulses: ')
print([known_opt[i].shape[0]-2 for i in range(2)])



nPs = [[11,23,24], [12,23,24]]
vt_opt, inf_min = hyperOpt(SMat, nPs, M, T, w)
L_opt = L_map(p2q, p2q, vt_opt, SMat_ideal, M, w)
inf_opt = infidelity(vt_opt, SMat_ideal, M, w)

print('infidelity over optimized seqs: ')
print(inf_opt)
print('number of pulses: ')
print([vt_opt[i].shape[0]-2 for i in range(2)])



plt.plot(w, Gp_re(0, 0, known_opt, w, M))
plt.plot(w, Gp_im(0, 0, known_opt, w, M))
plt.show()
plt.plot(w, Gp_re(1, 1, known_opt, w, M))
plt.plot(w, Gp_im(1, 1, known_opt, w, M))
plt.show()
plt.plot(w, Gp_re(0, 0, vt_opt, w, M))
plt.plot(w, Gp_im(0, 0, vt_opt, w, M))
plt.show()
plt.plot(w, Gp_re(1, 1, vt_opt, w, M))
plt.plot(w, Gp_im(1, 1, vt_opt, w, M))
plt.show()
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), known_opt[0]))
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), known_opt[1])+3)
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), known_opt[2])+6)
plt.legend([r'$y_{1,1}(t)$', r'$y_{2,2}(t)$', r'$y_{12,12}(t)$'])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), vt_opt[0]))
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), vt_opt[1])+3)
plt.plot(jnp.linspace(0, T, 1000), y_t(jnp.linspace(0, T, 1000), vt_opt[2])+6)
plt.legend([r'$y_{1,1}(t)$', r'$y_{2,2}(t)$', r'$y_{12,12}(t)$'])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()
print('End')

