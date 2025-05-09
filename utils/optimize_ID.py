import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import jaxopt
import os
from spectraIn import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12


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


# @jax.jit
def Gp_re(vti, vtj, w, M):
    return jnp.real(ff(vti, w)*ff(vtj, -w)*jnp.sin(w*M*vti[-1]*0.5)**2/jnp.sin(w*vti[-1]*0.5)**2)


# @jax.jit
def Gp_im(vti, vtj, w, M):
    return jnp.imag(ff(vti, w)*ff(vtj, -w)*jnp.sin(w*M*vti[-1]*0.5)**2/jnp.sin(w*vti[-1]*0.5)**2)


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
def CO_sum_els(O, SMat, Gp, i, j, w):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5*(sgn(O, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], w)/jnp.pi


@jax.jit
def CO_sum_els_wk(O, SMat_k, Gp, i, j, M, T):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5*(sgn(O, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jnp.sum(SMat_k[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], axis=0)*M/T


@jax.jit
def Lambda_diags(SMat, Gp, w):
    inds = jnp.array([0, 1, 2])
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els,
                                            in_axes=(None, None, None, None, 0, None)),
                                   in_axes=(None, None, None, 0, None, None)),
                          in_axes=(0, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q, SMat, Gp, inds, inds, w), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(p2q[i]@jax.scipy.linalg.expm(-CO[i])@p2q[i])*0.25 for i in range(CO.shape[0])]))


@jax.jit
def Lambda_diags_wk(SMat_k, Gp, M, T):
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    #CO = jnp.array([CO_sum_els_wk(p2q[i], SMat_k, Gp, j, k, M, T) for i in range(p2q.size) for j in range(3) for k in range(3)])
    inds = jnp.array([0, 1, 2])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els_wk,
                                            in_axes=(None, None, None, None, 0, None, None)),
                                   in_axes=(None, None, None, 0, None, None, None)),
                          in_axes=(0, None, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q, SMat_k, Gp, inds, inds, M, T), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(p2q[i]@jax.scipy.linalg.expm(-CO[i])@p2q[i])*0.25 for i in range(CO.shape[0])]))


def inf_ID(params, i, j, SMat, M, T, w):
    vt1 = jnp.sort(jnp.concatenate((jnp.array([0]), params[:i], jnp.array([T]))))
    vt2 = jnp.sort(jnp.concatenate((jnp.array([0]), params[i:], jnp.array([T]))))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    L_diag = Lambda_diags(SMat, Gp, w)
    dt = tau
    fid=jnp.sum(L_diag, axis=0)/16.
    clustering=jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    #(1/(vt[0].shape[0]+vt[1].shape[0]-2))*jnp.sum(jnp.array([jnp.tanh(vt[i].shape[0]*((vt[i][j+1]-vt[i][j])/dt-1)) for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0))
    return -fid-clustering*1e-2


def inf_ID_wk(params, i, j, SMat_k, M, T, wk):
    vt1 = jnp.sort(jnp.concatenate((jnp.array([0]), params[:i], jnp.array([T]))))
    vt2 = jnp.sort(jnp.concatenate((jnp.array([0]), params[i:], jnp.array([T]))))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, wk.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk, 1) + 1j*Gp_im_map(vt[i], vt[j], wk, 1))
    L_diag = Lambda_diags_wk(SMat_k, Gp, M, T)
    dt = tau
    fid=jnp.sum(L_diag, axis=0)/16.
    clustering=jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)#jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    return -fid-clustering*1e-2


def infidelity(params, SMat, M, w):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    L_diag = Lambda_diags(SMat, Gp, w)
    return 1.-jnp.sum(L_diag, axis=0)/16


def infidelity_k(params, SMat_k, M, wk):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, wk.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk, 1) + 1j*Gp_im_map(vt[i], vt[j], wk, 1))
    L_diag = Lambda_diags_wk(SMat_k, Gp, M, vt1[-1])
    return 1.-jnp.sum(L_diag, axis=0)/16


def T2(params, SMat, M, w, qubit):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = params[0]
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, 1) + 1j*Gp_im_map(vt[i], vt[j], w, 1))
    x1q = jnp.array([[0, 1], [1, 0]])
    z1q = jnp.array([[1, 0], [0, -1]])
    q1 = jnp.kron(x1q, z1q)
    q2 = jnp.kron(z1q, x1q)
    if qubit == 2:
        return Lambda(q2, q2, vt, SMat, M, w)
    elif qubit == 1:
        return Lambda(q1, q1, vt, SMat, M, w)
    else:
        raise ValueError("qubit must be an integer 1 or 2")


def hyperOpt(SMat, nPs, M, T, w):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID, maxiter=200, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-6,
                                                     'maxfun': 1000, 'maxls': 20})
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
            opt = optimizer.run(vt, bnds, i, j, SMat, M, T, w)
            opt_out_temp.append(opt)
            print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt = [params_to_tk(vt_min_arr, T, jnp.array([0, nPs[0][inds_min[0]]])), params_to_tk(vt_min_arr, T, jnp.array([nPs[0][inds_min[0]], nPs[0][inds_min[0]]+nPs[1][inds_min[1]]]))]
    vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
    #vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt, infidelity(vt_opt, SMat, M, w) #inf_ID_out[inds_min]


def hyperOpt_k(SMat_k, nPs, M, T, wk):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID_wk, maxiter=200, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-6,
                                                     'maxfun': 1000, 'maxls': 20})
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
            opt = optimizer.run(vt, bnds, i, j, SMat_k, M, T, wk)
            opt_out_temp.append(opt)
            print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt = [params_to_tk(vt_min_arr, T, jnp.array([0, nPs[0][inds_min[0]]])), params_to_tk(vt_min_arr, T, jnp.array([nPs[0][inds_min[0]], nPs[0][inds_min[0]]+nPs[1][inds_min[1]]]))]
    vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
    return vt_opt, infidelity_k(vt_opt, SMat_k, M, wk)


def Lambda(Oi, Oj, vt, SMat, M, w):
    vt12 = make_tk12(vt[0], vt[1])
    vt.append(vt12)
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    CO = 0
    for i in range(3):
        for j in range(3):
            CO += -0.5*(sgn(Oi, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(Oi, i+1, 0)-1)*Gp[i, j], w)/jnp.pi
    return jnp.real(jnp.trace(Oi@jax.scipy.linalg.expm(-CO)@Oj)*0.25)


def params_to_tk(params, T, shape: jnp.ndarray):
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


def pddn(T, n, M):
    out = cddn(T,n)
    if M == 1:
        return out
    for i in range(M):
        out = np.concatenate((out, cddn(T,n)+(i+1)*T))
    return  out


def opt_known_pulses(pLib, SMat, M, w):
    infmin = jnp.inf
    inds_min=0
    for i in range(len(pLib)):
        for j in range(len(pLib)):
            for k in range(len(pLib[i])):
                for l in range(len(pLib[j])):
                    tk1 = pLib[i][k]
                    tk2 = pLib[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
                    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
                    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)
                    for m in range(3):
                        for n in range(3):
                            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], w, M) + 1j*Gp_im_map(vt[m], vt[n], w, M))
                    L_diag = Lambda_diags(SMat, Gp, w)
                    # infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-jnp.sum(L_diag, axis=0)/16)
                    inf_ijkl = 1.-jnp.sum(L_diag, axis=0)/16
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i,j,k,l)
    tk1 = pLib[inds_min[0]][inds_min[2]]
    tk2 = pLib[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt = [tk1, tk2, tk12]
    return vt_opt, infmin


def opt_known_pulses_k(pLib, SMat_k, M, wk):
    infmin = jnp.inf
    inds_min=0
    for i in range(len(pLib)):
        for j in range(len(pLib)):
            for k in range(len(pLib[i])):
                for l in range(len(pLib[j])):
                    tk1 = pLib[i][k]
                    tk2 = pLib[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
                    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
                    Gp = jnp.zeros((3, 3, wk.size), dtype=jnp.complex64)
                    for m in range(3):
                        for n in range(3):
                            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], wk, 1) + 1j*Gp_im_map(vt[m], vt[n], wk, 1))
                    L_diag = Lambda_diags_wk(SMat_k, Gp, M, T)
                    # infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-jnp.sum(L_diag, axis=0)/16)
                    inf_ijkl = 1.-jnp.sum(L_diag, axis=0)/16
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i,j,k,l)
    tk1 = pLib[inds_min[0]][inds_min[2]]
    tk2 = pLib[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt = [tk1, tk2, tk12]
    return vt_opt, infmin


def makeSMat_k(specs, wk, wkqns, gamma, gamma12):
    SMat = jnp.zeros((3, 3, wk.size), dtype=jnp.complex64)
    SMat = SMat.at[0, 0].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_11(wk[0])]), specs["S11"]))))
    SMat = SMat.at[1, 1].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_22(wk[0])]), specs["S22"]))))
    SMat = SMat.at[2, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1212(wk[0])]), specs["S1212"]))))
    SMat = SMat.at[0, 1].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"]))))
    SMat = SMat.at[1, 0,].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"])))))
    SMat = SMat.at[0, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1_12(wk[0], gamma12)]), specs["S112"]))))
    SMat = SMat.at[2, 0].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_1_12(wk[0], gamma12)]), specs["S112"])))))
    SMat = SMat.at[1, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_2_12(wk[0], gamma12-gamma)]), specs["S212"]))))
    SMat = SMat.at[2, 1].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_2_12(wk[0], gamma12-gamma)]), specs["S212"])))))
    return SMat


def makeSMat_k_ideal(wk, gamma, gamma12):
    SMat_ideal = jnp.zeros((3, 3, wk.size), dtype=jnp.complex64)
    SMat_ideal = SMat_ideal.at[0, 0].set(S_11(wk))
    SMat_ideal = SMat_ideal.at[1, 1].set(S_22(wk))
    SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(wk))
    SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(wk, gamma))
    SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(wk, gamma)))
    SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(wk, gamma12))
    SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(wk, gamma12)))
    SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(wk, gamma12-gamma))
    SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(wk, gamma12-gamma)))
    return SMat_ideal


########################################################################################################################
########################################################################################################################
####################################### Set and run the optimization ###################################################
########################################################################################################################
########################################################################################################################


# Load the system parameters
parent_dir = os.pardir
fname = "DraftRun_NoSPAM_hat"
path = os.path.join(parent_dir, fname)
specs = np.load(os.path.join(path, "specs.npz"))
params = np.load(os.path.join(path, "params.npz"))


t_vec = params['t_vec']
w_grain = params['w_grain']
wmax = params['wmax']
mc = params['truncate']
gamma = params['gamma']
gamma12 = params['gamma_12']
t_b = params['t_b']
a_m = params['a_m']
delta = params['delta']
c_times = params['c_times']
n_shots = params['n_shots']
# M = params['M']
a_sp = params['a_sp']
c = params['c']
Tqns = params['T']



T=Tqns
M=40
p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
w = jnp.linspace(0.01, 2*jnp.pi*mc/Tqns, 4000)
w_ideal = jnp.linspace(0.01, 2*jnp.pi*2*mc/Tqns, 8000)
wkqns = jnp.array([2*jnp.pi*(n+1)/Tqns for n in range(mc)])
wkqns_ideal = jnp.array([2*jnp.pi*(n)/Tqns for n in range(2*mc+1)])
SMat = jnp.zeros((3, 3, w.size), dtype=jnp.complex64)


# Interpolate the loaded spectra to use in the optimization
SMat = SMat.at[0, 0].set(jnp.interp(w, wkqns, specs["S11"]))
SMat = SMat.at[1, 1].set(jnp.interp(w, wkqns, specs["S22"]))
SMat = SMat.at[2, 2].set(jnp.interp(w, wkqns, specs["S1212"]))
SMat = SMat.at[0, 1].set(jnp.interp(w, wkqns, specs["S12"]))
SMat = SMat.at[1, 0].set(jnp.interp(w, wkqns, np.conj(specs["S12"])))
SMat = SMat.at[0, 2].set(jnp.interp(w, wkqns, specs["S112"]))
SMat = SMat.at[2, 0].set(jnp.interp(w, wkqns, np.conj(specs["S11"])))
SMat = SMat.at[1, 2].set(jnp.interp(w, wkqns, specs["S212"]))
SMat = SMat.at[2, 1].set(jnp.interp(w, wkqns, np.conj(specs["S212"])))



# Create a matrix to ideal spectra to validate the optimiaation over
SMat_ideal = jnp.zeros((3, 3, w_ideal.size), dtype=jnp.complex64)
SMat_ideal = SMat_ideal.at[0, 0].set(S_11(w_ideal))
SMat_ideal = SMat_ideal.at[1, 1].set(S_22(w_ideal))
SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(w_ideal))
SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(w_ideal, gamma))
SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(w_ideal, gamma)))
SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(w_ideal, gamma12))
SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(w_ideal, gamma12)))
SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(w_ideal, gamma12-gamma))
SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(w_ideal, gamma12-gamma)))


# Calculate the T2 time for each qubit to be used as a point of reference
L_map = jax.vmap(jax.vmap(Lambda, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, None, None, None, None))
taxis = jnp.linspace(1*T, 20*T, 20)
vt_T2 = jnp.array([jnp.array([[0, taxis[i]], [0, taxis[i]]]) for i in range(taxis.shape[0])])
q1T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, w_ideal, 1) for i in range(vt_T2.shape[0])])
q2T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, w_ideal, 2) for i in range(vt_T2.shape[0])])
T2q1= jnp.inf
T2q2 = jnp.inf
for i in range(q1T2.size):
    if q1T2[i]<0.5:
        T2q1 = (taxis[i] + taxis[i-1])*0.5
        break
for i in range(q2T2.size):
    if q2T2[i]<0.5:
        T2q2 = (taxis[i] + taxis[i-1])*0.5
        break
print("###########################################################################################")
print(f"The base sequence time T = {T/1e-6} us")
print(f"T2 time for qubit 1 is {np.round(T2q1/1e-6, 2)} us")
print(f"T2 time for qubit 2 is {np.round(T2q2/1e-6, 2)} us")
print("###########################################################################################")
print("In terms of T,")
print(f"T2 time for qubit 1 is {np.round(T2q1/T,2)} T")
print(f"T2 time for qubit 2 is {np.round(T2q2/T, 2)} T")
print("###########################################################################################")




uddLib = []


# Parameters for the optimization over known pulse sequences
tau = T/80
Tknown = M*T
Mknown = 1


print("###########################################################################################")
print(f"Optimizing the Idling gate for {np.round(Mknown*Tknown/T2q1, 2)} T2q1 or {np.round(Mknown*Tknown/T2q2, 2)} T2q2")
print("###########################################################################################")


best_seq = 0
best_inf = np.inf
best_M = 0


for i in [1/20,1/10,1/5,1/4,1/2,1]:
    pLib=[]
    cddLib = []
    Tknown = i*T
    Mknown = int(M/i)
    if Mknown >= 10:
        wk = jnp.array([0.01]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(mc*i)))])
        wk_ideal = jnp.array([0.01]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(4*mc*i)))])
        SMat_k = makeSMat_k(specs, wk, jnp.concatenate((jnp.array([wk[0]]), wkqns)), gamma, gamma12)
        SMat_k_ideal = makeSMat_k_ideal(wk_ideal, gamma, gamma12)
        # Make CDD_n libraries that respect the minimum pulse separation constraint
        cddOrd = 1
        make = True
        while make:
            pul = cddn(Tknown, cddOrd)
            cddOrd += 1
            for j in range(1, pul.size-2):
                if pul[j+1] - pul[j] < tau:
                    make = False
            if make == False:
                break
            cddLib.append(pul)
        pLib.append(cddLib)
        # Generate an Idling gate that is optimized using known sequences
        known_opt, known_inf = opt_known_pulses_k(pLib, SMat_k, Mknown, wk)
        inf_known = infidelity_k(known_opt, SMat_k_ideal, Mknown, wk_ideal)
    else:
        cddOrd = 5
        make = True
        while make:
            pul = cddn(Tknown, cddOrd)
            cddOrd += 1
            for j in range(1, pul.size-2):
                if pul[j+1] - pul[j] < tau:
                    make = False
            if make == False:
                break
            cddLib.append(pul)
        pLib.append(cddLib)
        # Generate an Idling gate that is optimized using known sequences
        known_opt, known_inf = opt_known_pulses(pLib, SMat, Mknown, w)
        inf_known = infidelity(known_opt, SMat_ideal, Mknown, w_ideal)
    if known_inf <= best_inf:
        best_seq = known_opt
        best_inf = inf_known
        best_M = Mknown
        print(f"The best infidelity till now is {best_inf}; number of pulses {[best_seq[0].shape[0]-2, best_seq[1].shape[0]-2]}")
    print(f"# repetitions considered = {Mknown}")


print('infidelity over known seqs: ')
print(best_inf)
print('number of pulses: ')
print([best_seq[i].shape[0]-2 for i in range(2)])


opt_seq = 0
opt_inf = np.inf
opt_M = 0
for i in [1/10]:
    Topt = i*T
    Mopt = M/i
    nPs = [[4,6],[4,6]]
    if Mopt >= 10:
        wk = jnp.array([0.01]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(mc*i)))])
        wk_ideal = jnp.array([0.01]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(4*mc*i)))])
        SMat_k = makeSMat_k(specs, wk, jnp.concatenate((jnp.array([wk[0]]), wkqns)), gamma, gamma12)
        SMat_k_ideal = makeSMat_k_ideal(wk_ideal, gamma, gamma12)
        # Generate an Idling gate that is optimized over a given number of pulses on each qubit
        vt_opt, inf_min = hyperOpt_k(SMat_k, nPs, Mopt, Topt, wk)
        inf_opt = infidelity_k(vt_opt, SMat_k_ideal, Mopt, wk_ideal)
    else:
        vt_opt, inf_min = hyperOpt(SMat, nPs, Mopt, Topt, w)
        inf_opt = infidelity(vt_opt, SMat_ideal, Mopt, w_ideal)
    if inf_min <= opt_inf:
        opt_seq = vt_opt
        opt_inf = inf_opt
        opt_M = Mopt
        print(f"The best infidelity till now is {opt_inf}; number of pulses {[opt_seq[0].shape[0]-2, opt_seq[1].shape[0]-2]}")
    print(f"# repetitions considered = {Mopt}")


print('infidelity over optimized seqs: ')
print(opt_inf)
print('number of pulses: ')
print([opt_seq[i].shape[0]-2 for i in range(2)])


########################################################################################################################
########################################################################################################################
####################################### Plotting code for the paper ####################################################
########################################################################################################################
########################################################################################################################


fig, axs = plt.subplots(2, 3, figsize=(16,9))
alp = 0.4
lw = 0.5
Mplot_opt = opt_M
Mplot_known = best_M
legendfont = 10
labelfont = 16
xunits = 1e6
yunits = 1e3
wk=wkqns
vt_opt=opt_seq
known_opt = best_seq

max_lim_row1_left = np.max(np.array([np.abs(np.real(S_11(w))/yunits).max(), np.abs(np.imag(S_22(w))/yunits).max(),
                                     np.abs(np.imag(S_1212(w))/yunits).max()]))
max_lim_row1_right = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[0], w, Mplot_opt)).max(),
                                      np.abs(Gp_re(vt_opt[1], vt_opt[1], w, Mplot_opt)).max(),
                                      np.abs(Gp_re(vt_opt[2], vt_opt[2], w, Mplot_opt)).max()]))


max_lim_row2_left = np.max(np.array([np.abs(np.real(S_1_2(w[int(w.size/mc):], gamma))/yunits).max(),
                                     np.abs(np.imag(S_1_12(w[int(w.size/mc):], gamma12))/yunits).max(),
                                     np.abs(np.imag(S_2_12(w[int(w.size/mc):], gamma-gamma12))/yunits).max()]))
max_lim_row2_right = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[1], w, Mplot_opt)).max(),
                                      np.abs(Gp_re(vt_opt[0], vt_opt[2], w, Mplot_opt)).max(),
                                      np.abs(Gp_re(vt_opt[1], vt_opt[2], w, Mplot_opt)).max()]))


axs[0, 0].plot(w/xunits, S_11(w)/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[0, 0].plot(wk/xunits, specs["S11"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[0, 0].legend([r'$S^+_{1,1}(\omega)$', r'$\hat{S}^+_{1,1}(\omega)$'], fontsize=legendfont, loc='upper left')
axs[0, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=labelfont)
axs[0, 0].set_ylim(0, max_lim_row1_left*1.01)
axs[0, 0].tick_params(direction='in')
axs[0, 0].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
axs[0, 0].set_yscale('asinh')
bx = axs[0, 0].twinx()
bx.plot(w/xunits, Gp_re(vt_opt[0], vt_opt[0], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[0], known_opt[0], w, Mplot_known), 'm-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
bx.set_ylim(0)
bx.tick_params(direction='in')
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;1,1}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{1,1;1,1}(\omega, T)$]']
          , fontsize=legendfont, loc='upper right')
axs[0, 1].plot(w/xunits, S_22(w)/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[0, 1].plot(wk/xunits, specs["S22"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
# axs[0, 1].set_yticks([])
axs[0, 1].set_yticklabels([])
axs[0, 1].legend([r'$S^+_{2,2}(\omega)$', r'$\hat{S}^+_{2,2}(\omega)$'], fontsize=legendfont, loc='upper left')
axs[0, 1].set_ylim(0, max_lim_row1_left*1.01)
axs[0, 1].tick_params(direction='in')
axs[0, 1].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
axs[0, 1].set_yscale('asinh')
bx = axs[0, 1].twinx()
bx.plot(w/xunits, Gp_re(vt_opt[1], vt_opt[1], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[1], known_opt[1], w, Mplot_known), 'm-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
bx.set_ylim(0)
bx.tick_params(direction='in')
bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;2,2}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{2,2;2,2}(\omega, T)$]']
          , fontsize=legendfont, loc='upper right')
axs[0, 2].plot(w/xunits, S_1212(w)/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[0, 2].plot(wk/xunits, specs["S1212"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[0, 2].legend([r'$S^+_{12,12}(\omega)$', r'$\hat{S}^+_{12,12}(\omega)$'], fontsize=legendfont, loc='upper left')
# axs[0, 2].set_yticks([])
axs[0, 2].set_yticklabels([])
axs[0, 2].set_ylim(0)
axs[0, 2].tick_params(direction='in')
axs[0, 2].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
axs[0, 2].set_yscale('asinh')
bx = axs[0, 2].twinx()
bx.plot(w/xunits, Gp_re(vt_opt[2], vt_opt[2], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[2], known_opt[2], w, Mplot_known), 'm-', lw=lw)
bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=labelfont)
bx.legend([r'Re[$G^{+,\text{opt}}_{12,12;12,12}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{12,12;12,12}(\omega, T)$]']
          , fontsize=legendfont, loc='upper right')
bx.set_ylim(0)
bx.tick_params(direction='in')

########################################################################################################################
w = w[300:]
axs[1, 0].plot(w/xunits, np.real(S_1_2(w, gamma))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(wk/xunits, np.real(specs["S12"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(w/xunits, np.imag(S_1_2(w, gamma))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(wk/xunits, np.imag(specs["S12"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
axs[1, 0].legend([r'Re[$S^+_{1,2}(\omega)$]', r'Re[$\hat{S}^+_{1,2}(\omega)$]', r'Im[$S^+_{1,2}(\omega)$]'
                     , r'Im[$\hat{S}^+_{1,2}(\omega)$]'], fontsize=legendfont, loc='upper left')
axs[1, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=labelfont)
# max_lim = np.maximum(np.abs(np.real(S_1_2(w, gamma))/1e3).max(), np.abs(np.imag(S_1_2(w, gamma))/1e3).max())
axs[1, 0].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 0].tick_params(direction='in')
axs[1, 0].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
bx = axs[1, 0].twinx()
bx.tick_params(direction='in')
bx.plot(w/xunits, Gp_re(vt_opt[0], vt_opt[1], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_im(vt_opt[0], vt_opt[1], w, Mplot_opt), 'k--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[0], known_opt[1], w, Mplot_known), 'm-', lw=lw)
bx.plot(w/xunits, Gp_im(known_opt[0], known_opt[1], w, Mplot_known), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[1], w, Mplot_opt)).max(),
                           np.abs(Gp_im(vt_opt[0], vt_opt[1], w, Mplot_opt)).max(),
                           np.abs(Gp_re(known_opt[0], known_opt[1], w, Mplot_known)).max(),
                           np.abs(Gp_im(known_opt[0], known_opt[1], w, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
axs[1, 1].plot(w/xunits, np.real(S_1_12(w, gamma12))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(wk/xunits, np.real(specs["S112"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(w/xunits, np.imag(S_1_12(w, gamma12))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(wk/xunits, np.imag(specs["S112"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
# axs[1, 1].set_yticks([])
axs[1, 1].set_yticklabels([])
# max_lim = np.maximum(np.abs(np.real(S_1_12(w, gamma12))/1e3).max(), np.abs(np.imag(S_1_12(w, gamma12))/1e3).max())
axs[1, 1].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 1].legend([r'Re[$S^+_{1,12}(\omega)$]', r'Re[$\hat{S}^+_{1,12}(\omega)$]', r'Im[$S^+_{1,12}(\omega)$]',
                  r'Im[$\hat{S}^+_{1,2}(\omega)$]'], fontsize=legendfont, loc='upper left')
axs[1, 1].tick_params(direction='in')
axs[1, 1].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
bx = axs[1, 1].twinx()
bx.tick_params(direction='in')
bx.plot(w/xunits, Gp_re(vt_opt[0], vt_opt[2], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_im(vt_opt[0], vt_opt[2], w, Mplot_opt), 'k--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[0], known_opt[2], w, Mplot_known), 'm-', lw=lw)
bx.plot(w/xunits, Gp_im(known_opt[0], known_opt[2], w, Mplot_known), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[2], w, Mplot_opt)).max(),
                                     np.abs(Gp_im(vt_opt[0], vt_opt[2], w, Mplot_opt)).max(),
                                     np.abs(Gp_re(known_opt[0], known_opt[2], w, Mplot_known)).max(),
                                     np.abs(Gp_im(known_opt[0], known_opt[2], w, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
axs[1, 2].plot(w/xunits, np.real(S_2_12(w, gamma12-gamma))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(wk/xunits, np.real(specs["S212"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(w/xunits, np.imag(S_2_12(w, gamma12-gamma))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(wk/xunits, np.imag(specs["S212"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
# axs[1, 2].set_yticks([])
axs[1, 2].set_yticklabels([])
# max_lim = np.maximum(np.abs(np.real(S_2_12(w, gamma-gamma12))/1e3).max(), np.abs(np.imag(S_2_12(w, gamma-gamma12))/1e3).max())
axs[1, 2].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 2].legend([r'Re[$S^+_{2,12}(\omega)$]', r'Re[$\hat{S}^+_{2,12}(\omega)$]', r'Im[$S^+_{2,12}(\omega)$]',
                  r'Im[$\hat{S}^+_{2,12}(\omega)$]'], fontsize=legendfont, loc='upper left')
axs[1, 2].tick_params(direction='in')
axs[1, 2].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
bx = axs[1, 2].twinx()
bx.tick_params(direction='in')
bx.plot(w/xunits, Gp_re(vt_opt[1], vt_opt[2], w, Mplot_opt), 'g--', lw=lw)
bx.plot(w/xunits, Gp_im(vt_opt[1], vt_opt[2], w, Mplot_opt), 'k--', lw=lw)
bx.plot(w/xunits, Gp_re(known_opt[1], known_opt[2], w, Mplot_known), 'm-', lw=lw)
bx.plot(w/xunits, Gp_im(known_opt[1], known_opt[2], w, Mplot_known), 'c-', lw=lw)
bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=labelfont)
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[1], vt_opt[2], w, Mplot_opt)).max(),
                                     np.abs(Gp_im(vt_opt[1], vt_opt[2], w, Mplot_opt)).max(),
                                     np.abs(Gp_re(known_opt[1], known_opt[2], w, Mplot_known)).max(),
                                     np.abs(Gp_im(known_opt[1], known_opt[2], w, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
plt.savefig(os.path.join(path, 'IDGate.png'), dpi = 1200)
# plt.show()
print('End')


