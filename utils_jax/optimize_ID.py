import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
# from trajectories import custom_y
from jax.scipy.linalg import expm
import scipy as sp
import jaxopt


########################################################################################################################
########################################################################################################################
####################################### Define the spectra #############################################################
########################################################################################################################
########################################################################################################################



@jax.jit
def L(w, w0, tc):
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def S_11(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=5*10**6
    return S0*(L(w, 0, 0.5*tc)+L(w, w0, tc))


@jax.jit
def S_22(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=4*10**6
    return S0*(L(w, 0, tc)+L(w, w0, 0.5*tc))


@jax.jit
def S_1_2(w, gamma):
    return jnp.sqrt(S_11(w)*S_22(w))*jnp.exp(-1j*w*gamma)


@jax.jit
def S_12(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=0*10**6
    return 0.5*S0*L(w, w0, tc*0.5)

@jax.jit
def S_1_12(w, gamma12):
    return 0*jnp.sqrt(S_11(w)*S_12(w))*jnp.exp(-1j*w*gamma12)


@jax.jit
def S_2_12(w, gamma12):
    return 0*jnp.sqrt(S_22(w)*S_12(w))*jnp.exp(-1j*w*gamma12)


########################################################################################################################
########################################################################################################################
####################################### Utility functions ##############################################################
########################################################################################################################
########################################################################################################################


@jax.jit
def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O)@z2q[a]@z2q[b]@O@z2q[a]@z2q[b])/4


@jax.jit
def ff(tk, w):
    return jnp.sum(jnp.array([1j*((-1)**k)*(jnp.exp(-1j*w*tk[k+1])-jnp.exp(-1j*w*tk[k]))/w for k in range(jnp.size(tk)-1)]), axis=0)


def Gp_re(i, j, vt_list, w, M):
    return jnp.real(ff(vt_list[i], w)*ff(vt_list[j], -w)*jnp.sin(w*M*vt_list[0][-1]*0.5)**2/jnp.sin(w*vt_list[0][-1]*0.5)**2)


def Gp_im(i, j, vt_list, w, M):
    return jnp.imag(ff(vt_list[i], w)*ff(vt_list[j], -w)*jnp.sin(w*M*vt_list[0][-1]*0.5)**2/jnp.sin(w*vt_list[0][-1]*0.5)**2)


@jax.jit
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

@jax.jit
def makeCO(vt, O, SMat, M, w):
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


@jax.jit
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
    return 2-jnp.sum(L_diag, axis=0)/16-(1/(vt[0].shape[0]-1+vt[1].shape[0]-1))*jnp.sum(jnp.array([jnp.tanh(10*((vt[i][j+1]-vt[i][j])/dt-1.)) for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)


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
    return 1.-jnp.sum(L_diag, axis=0)/16


def T2(params, SMat, M, w, qubit):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = params[0]
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(i, j, vt, w, M) + 1j*Gp_im_map(i, j, vt, w, M))
    L_diag = Lambda_diags(SMat, Gp, w)
    if qubit == 2:
        return L_diag[1]
    elif qubit == 1:
        return L_diag[4]
    else:
        raise ValueError("qubit must be an integer 1 or 2")


def hyperOpt(SMat, nPs, M, T, w):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID, maxiter=50, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-6,
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
            print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt = [params_to_tk(vt_min_arr, T, [0, nPs[0][inds_min[0]]]), params_to_tk(vt_min_arr, T, [nPs[0][inds_min[0]], nPs[0][inds_min[0]]+nPs[1][inds_min[1]]])]
    vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
    #vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt, infidelity(vt_opt, SMat, M, w) #inf_ID_out[inds_min]


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


def uddn(T, n):
    tk = [T*jnp.sin(jnp.pi*k/(2*n+2))**2 for k in range(n)]
    tk.append(T)
    tk.insert(0, 0)
    return jnp.array(tk)


def comb_vks(vk1, vk2):
    vk1 = np.array(vk1)
    vk2 = np.array(vk2)
    if vk1.size == 0:
        return vk2
    if vk2.size == 0:
        return vk1
    if vk1[-1] == vk2[0]:
        if vk1.size == 1 and vk2.size > 1:
            return vk2[1:]
        if vk2.size == 1 and vk1.size > 1:
            return vk1[:-1]
        if vk2.size == 1 and vk1.size == 1:
            return np.array([])
        return np.concatenate((vk1[:-1], vk2[1:]))
    return np.concatenate((vk1, vk2))


def cddn_rec(T, n):
    if n == 1:
        return np.array([0., T/2])
    return comb_vks([0.], comb_vks(comb_vks(cddn_rec(T/2, n-1), [T/2]), cddn_rec(T/2, n-1) + T/2))

def cddn(T, n):
    return np.concatenate((np.array([0]), comb_vks(cddn_rec(T, n), [T])))


# t = np.linspace(0, 1, 1000)
# plt.plot(t, y_t(t, cddn(1, 1)))
# plt.plot(t, y_t(t, cddn(1, 2))+3)
# plt.plot(t, y_t(t, cddn(1, 3))+6)
# plt.plot(t, y_t(t, cddn(1, 4))+9)
# plt.show()
# plt.plot(t, y_t(t, uddn(1, 2)))
# plt.plot(t, y_t(t, uddn(1, 4))+3)
# plt.plot(t, y_t(t, uddn(1, 8))+6)
# plt.plot(t, y_t(t, uddn(1, 16))+9)
# plt.show()
# print("End Test")


def opt_known_pulses(nCs, SMat, M, T, w):
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    seq = [cddn, uddn]
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


########################################################################################################################
########################################################################################################################
####################################### Set and run the optimization ###################################################
########################################################################################################################
########################################################################################################################


T = 1e-5
mc = 14
M = 1
gamma = T/25
gamma12 = T/30


p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
w = jnp.linspace(0.1, 2*jnp.pi*mc/T, 2000)
wk = jnp.array([2*jnp.pi*n/T for n in range(mc+1)])
S_11_k = S_11(wk)
S_22_k = S_22(wk)
S_12_k = S_12(wk)
S_1_2_k = S_1_2(wk, gamma)
S_1_12_k = S_1_12(wk, gamma)
S_2_12_k = S_2_12(wk, gamma)
SMat = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)


SMat = SMat.at[0, 0].set(jnp.interp(w, wk, S_11_k))
SMat = SMat.at[1, 1].set(jnp.interp(w, wk, S_22_k))
SMat = SMat.at[2, 2].set(jnp.interp(w, wk, S_12_k))
SMat = SMat.at[0, 1].set(jnp.interp(w, wk, S_1_2_k))
SMat = SMat.at[1, 0].set(jnp.interp(w, wk, jnp.conj(S_1_2_k)))
SMat = SMat.at[0, 2].set(jnp.interp(w, wk, S_1_12_k))
SMat = SMat.at[2, 0].set(jnp.interp(w, wk, jnp.conj(S_1_12_k)))
SMat = SMat.at[1, 2].set(jnp.interp(w, wk, S_2_12_k))
SMat = SMat.at[2, 1].set(jnp.interp(w, wk, jnp.conj(S_2_12_k)))


SMat_ideal = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
SMat_ideal = SMat_ideal.at[0, 0].set(S_11(w))
SMat_ideal = SMat_ideal.at[1, 1].set(S_22(w))
SMat_ideal = SMat_ideal.at[2, 2].set(S_12(w))
SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(w, gamma))
SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(w, gamma)))


L_map = jax.vmap(jax.vmap(Lambda, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, None, None, None, None))
taxis = jnp.linspace(T, 100*T, 100)
vt_T2 = jnp.array([jnp.array([[0, taxis[i]], [0, taxis[i]]]) for i in range(taxis.shape[0])])
# print(f"T2 for qubit 1 is: {T2(vt_T2, SMat, M, w, 1)}")
# print(f"T2 for qubit 1 is: {T2(vt_T2, SMat, M, w, 2)}")
q1T2 = jnp.array([T2(vt_T2[i], SMat, M, w, 1) for i in range(vt_T2.shape[0])])
q2T2 = jnp.array([T2(vt_T2[i], SMat, M, w, 2) for i in range(vt_T2.shape[0])])
print(f"The base sequence time T = {T/1e-6} microseconds")
print(f"T2 time for qubit 1 is {(1/np.polyfit(taxis, -np.log(q1T2), 1)[0])/1e-6} microseconds")
print(f"T2 time for qubit 2 is {(1/np.polyfit(taxis, -np.log(q2T2), 1)[0])/1e-6} microseconds")
plt.plot(taxis, -np.log(q1T2), 'r.')
plt.plot(taxis, -np.log(q2T2), 'b.')
plt.show()


nCs = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
known_opt, known_inf = opt_known_pulses(nCs, SMat, M, T, w)
L_known = L_map(p2q, p2q, known_opt, SMat, M, w)
inf_known = infidelity(known_opt, SMat, M, w)


print('infidelity over known seqs: ')
print(inf_known)
print('number of pulses: ')
print([known_opt[i].shape[0]-2 for i in range(2)])


nPs = [[1, 2, 3], [1, 2, 3]]
vt_opt, inf_min = hyperOpt(SMat, nPs, M, T, w)
L_opt = L_map(p2q, p2q, vt_opt, SMat, M, w)
inf_opt = infidelity(vt_opt, SMat, M, w)


print('infidelity over optimized seqs: ')
print(inf_opt)
print('number of pulses: ')
print([vt_opt[i].shape[0]-2 for i in range(2)])


########################################################################################################################
########################################################################################################################
####################################### Plotting code for the paper ####################################################
########################################################################################################################
########################################################################################################################


fig, axs = plt.subplots(2, 3, figsize=(12,14))
alp = 0.4
lw = 0.5
Mplot = 1


max_lim_row1_left = np.max(np.array([np.abs(np.real(S_11(w))/1e3).max(), np.abs(np.imag(S_22(w))/1e3).max(),
                                     np.abs(np.imag(S_12(w))/1e3).max()]))
max_lim_row1_right = np.max(np.array([np.abs(Gp_re(0, 0, vt_opt, w, Mplot)).max(),
                                      np.abs(Gp_re(1, 1, vt_opt, w, Mplot)).max(),
                                      np.abs(Gp_re(2, 2, vt_opt, w, Mplot)).max()]))


max_lim_row2_left = np.max(np.array([np.abs(np.real(S_1_2(w, gamma))/1e3).max(),
                                     np.abs(np.imag(S_1_12(w, gamma12))/1e3).max(),
                                     np.abs(np.imag(S_2_12(w, gamma-gamma12))/1e3).max()]))
max_lim_row2_right = np.max(np.array([np.abs(Gp_re(0, 1, vt_opt, w, Mplot)).max(),
                                      np.abs(Gp_re(0, 2, vt_opt, w, Mplot)).max(),
                                      np.abs(Gp_re(1, 2, vt_opt, w, Mplot)).max()]))


axs[0, 0].plot(w/1e6, S_11(w)/1e3, 'r-', alpha=alp)
axs[0, 0].legend([r'$S^+_{1,1}(\omega)$'], fontsize=10, loc='upper left')
axs[0, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=16)
axs[0, 0].set_ylim(0, max_lim_row1_left*1.01)
axs[0, 0].tick_params(direction='in')
axs[0, 0].set_xlabel('$\omega$ (MHz)')
bx = axs[0, 0].twinx()
bx.plot(w/1e6, Gp_re(0, 0, vt_opt, w, Mplot), 'g--', lw=lw*1.5)
bx.plot(w/1e6, Gp_re(0, 0, known_opt, w, Mplot), 'm-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
bx.set_ylim(0)
bx.tick_params(direction='in')
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;1,1}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{1,1;1,1}(\omega, T)$]']
          , fontsize=10, loc='upper right')
axs[0, 1].plot(w/1e6, S_22(w)/1e3, 'r-', alpha=alp)
# axs[0, 1].set_yticks([])
axs[0, 1].set_yticklabels([])
axs[0, 1].legend([r'$S^+_{2,2}(\omega)$'], fontsize=10, loc='upper left')
axs[0, 1].set_ylim(0, max_lim_row1_left*1.01)
axs[0, 1].tick_params(direction='in')
axs[0, 1].set_xlabel('$\omega$ (MHz)')
bx = axs[0, 1].twinx()
bx.plot(w/1e6, Gp_re(1, 1, vt_opt, w, Mplot), 'g--', lw=lw*1.5)
bx.plot(w/1e6, Gp_re(1, 1, known_opt, w, Mplot), 'm-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
bx.set_ylim(0)
bx.tick_params(direction='in')
bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;2,2}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{2,2;2,2}(\omega, T)$]']
          , fontsize=10, loc='upper right')
axs[0, 2].plot(2*w/1e6, S_12(2*w)/1e3, 'r-', alpha=alp)
axs[0, 2].legend([r'$S^+_{12,12}(\omega)$'], fontsize=10, loc='upper left')
# axs[0, 2].set_yticks([])
axs[0, 2].set_yticklabels([])
axs[0, 2].set_ylim(0)
axs[0, 2].tick_params(direction='in')
axs[0, 2].set_xlabel('$\omega$ (MHz)')
bx = axs[0, 2].twinx()
bx.plot(2*w/1e6, Gp_re(2, 2, vt_opt, 2*w, Mplot), 'g--', lw=lw*1.5)
bx.plot(2*w/1e6, Gp_re(2, 2, known_opt, 2*w, Mplot), 'm-', lw=lw)
bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=16)
bx.legend([r'Re[$G^{+,\text{opt}}_{12,12;12,12}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{12,12;12,12}(\omega, T)$]']
          , fontsize=10, loc='upper right')
bx.set_ylim(0)
bx.tick_params(direction='in')



axs[1, 0].plot(w/1e6, np.real(S_1_2(w, gamma))/1e3, 'r-', alpha=alp)
axs[1, 0].plot(w/1e6, np.imag(S_1_2(w, gamma))/1e3, 'b-', alpha=alp)
axs[1, 0].legend([r'Re[$S^+_{1,2}(\omega)$]', r'Im[$S^+_{1,2}(\omega)$]'], fontsize=10, loc='upper left')
axs[1, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=16)
# max_lim = np.maximum(np.abs(np.real(S_1_2(w, gamma))/1e3).max(), np.abs(np.imag(S_1_2(w, gamma))/1e3).max())
axs[1, 0].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 0].tick_params(direction='in')
axs[1, 0].set_xlabel('$\omega$ (MHz)')
bx = axs[1, 0].twinx()
bx.tick_params(direction='in')
bx.plot(w/1e6, Gp_re(0, 1, vt_opt, w, Mplot), 'g--', lw=lw*1.5)
bx.plot(w/1e6, Gp_im(0, 1, vt_opt, w, Mplot), 'k--', lw=lw*1.5)
bx.plot(w/1e6, Gp_re(0, 1, known_opt, w, Mplot), 'm-', lw=lw)
bx.plot(w/1e6, Gp_im(0, 1, known_opt, w, Mplot), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(0, 1, vt_opt, w, Mplot)).max(),
                           np.abs(Gp_im(0, 1, vt_opt, w, Mplot)).max(),
                           np.abs(Gp_re(0, 1, known_opt, w, Mplot)).max(),
                           np.abs(Gp_im(0, 1, known_opt, w, Mplot)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]']
          , fontsize=10, loc='lower right')
axs[1, 1].plot(2*w/1e6, np.real(S_1_12(2*w, gamma12))/1e3, 'r-', alpha=alp)
axs[1, 1].plot(2*w/1e6, np.imag(S_1_12(2*w, gamma12))/1e3, 'b-', alpha=alp)
# axs[1, 1].set_yticks([])
axs[1, 1].set_yticklabels([])
# max_lim = np.maximum(np.abs(np.real(S_1_12(w, gamma12))/1e3).max(), np.abs(np.imag(S_1_12(w, gamma12))/1e3).max())
axs[1, 1].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 1].legend([r'Re[$S^+_{1,12}(\omega)$]', r'Im[$S^+_{1,12}(\omega)$]'], fontsize=10, loc='upper left')
axs[1, 1].tick_params(direction='in')
axs[1, 1].set_xlabel('$\omega$ (MHz)')
bx = axs[1, 1].twinx()
bx.tick_params(direction='in')
bx.plot(2*w/1e6, Gp_re(0, 2, vt_opt, 2*w, Mplot), 'g--', lw=lw*1.5)
bx.plot(2*w/1e6, Gp_im(0, 2, vt_opt, 2*w, Mplot), 'k--', lw=lw*1.5)
bx.plot(2*w/1e6, Gp_re(0, 2, known_opt, 2*w, Mplot), 'm-', lw=lw)
bx.plot(2*w/1e6, Gp_im(0, 2, known_opt, 2*w, Mplot), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(0, 2, vt_opt, w, Mplot)).max(),
                                     np.abs(Gp_im(0, 2, vt_opt, w, Mplot)).max(),
                                     np.abs(Gp_re(0, 2, known_opt, w, Mplot)).max(),
                                     np.abs(Gp_im(0, 2, known_opt, w, Mplot)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]']
          , fontsize=10, loc='lower right')
axs[1, 2].plot(2*w/1e6, np.real(S_2_12(2*w, gamma-gamma12))/1e3, 'r-', alpha=alp)
axs[1, 2].plot(2*w/1e6, np.imag(S_2_12(2*w, gamma-gamma12))/1e3, 'b-', alpha=alp)
# axs[1, 2].set_yticks([])
axs[1, 2].set_yticklabels([])
# max_lim = np.maximum(np.abs(np.real(S_2_12(w, gamma-gamma12))/1e3).max(), np.abs(np.imag(S_2_12(w, gamma-gamma12))/1e3).max())
axs[1, 2].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 2].legend([r'Re[$S^+_{2,12}(\omega)$]', r'Im[$S^+_{2,12}(\omega)$]'], fontsize=10, loc='upper left')
axs[1, 2].tick_params(direction='in')
axs[1, 2].set_xlabel('$\omega$ (MHz)')
bx = axs[1, 2].twinx()
bx.tick_params(direction='in')
bx.plot(2*w/1e6, Gp_re(1, 2, vt_opt, 2*w, Mplot), 'g--', lw=lw*1.5)
bx.plot(2*w/1e6, Gp_im(1, 2, vt_opt, 2*w, Mplot), 'k--', lw=lw*1.5)
bx.plot(2*w/1e6, Gp_re(1, 2, known_opt, 2*w, Mplot), 'm-', lw=lw)
bx.plot(2*w/1e6, Gp_im(1, 2, known_opt, 2*w, Mplot), 'c-', lw=lw)
bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=16)
max_lim = np.max(np.array([np.abs(Gp_re(1, 2, vt_opt, w, Mplot)).max(),
                                     np.abs(Gp_im(1, 2, vt_opt, w, Mplot)).max(),
                                     np.abs(Gp_re(1, 2, known_opt, w, Mplot)).max(),
                                     np.abs(Gp_im(1, 2, known_opt, w, Mplot)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]']
          , fontsize=10, loc='lower right')
plt.show()
print('End')
