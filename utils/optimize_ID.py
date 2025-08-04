
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
def ff(tk, w_arg):
    return jnp.sum(jnp.array([1j*((-1)**k)*(jnp.exp(-1j*w_arg*tk[k+1])-jnp.exp(-1j*w_arg*tk[k]))/w_arg for k in range(jnp.size(tk)-1)]), axis=0)


def Gp_re(vti, vtj, w_arg, M_arg):
    return jnp.real(ff(vti, w_arg)*ff(vtj, -w_arg)*jnp.sin(w_arg*M_arg*vti[-1]*0.5)**2/jnp.sin(w_arg*vti[-1]*0.5)**2)


def Gp_im(vti, vtj, w_arg, M_arg):
    return jnp.imag(ff(vti, w_arg)*ff(vtj, -w_arg)*jnp.sin(w_arg*M_arg*vti[-1]*0.5)**2/jnp.sin(w_arg*vti[-1]*0.5)**2)


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
def CO_sum_els(O, SMat_arg, Gp, i, j, w_arg):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5*(sgn(O, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat_arg[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], w_arg)/jnp.pi


@jax.jit
def CO_sum_els_wk(O, SMat_k_arg, Gp, i, j, M_arg, T_arg):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5*(sgn(O, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jnp.sum(SMat_k_arg[i, j]*(sgn(O, i+1, 0)-1)*Gp[i, j], axis=0)*M_arg/T_arg


@jax.jit
def Lambda_diags(SMat_arg, Gp, w_arg):
    inds = jnp.array([0, 1, 2])
    p1q_local_lambda = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q_local_lambda = jnp.array([jnp.kron(p1q_local_lambda[i], p1q_local_lambda[j]) for i in range(4) for j in range(4)])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els,
                                            in_axes=(None, None, None, None, 0, None)),
                                   in_axes=(None, None, None, 0, None, None)),
                          in_axes=(0, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q_local_lambda, SMat_arg, Gp, inds, inds, w_arg), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(jax.scipy.linalg.expm(-CO[i]))*0.25 for i in range(CO.shape[0])]))


@jax.jit
def Lambda_diags_wk(SMat_k_arg, Gp, M_arg, T_arg):
    p1q_local_lambda = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q_local_lambda = jnp.array([jnp.kron(p1q_local_lambda[i], p1q_local_lambda[j]) for i in range(4) for j in range(4)])
    #CO = jnp.array([CO_sum_els_wk(p2q[i], SMat_k, Gp, j, k, M, T) for i in range(p2q.size) for j in range(3) for k in range(3)])
    inds = jnp.array([0, 1, 2])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els_wk,
                                            in_axes=(None, None, None, None, 0, None, None)),
                                   in_axes=(None, None, None, 0, None, None, None)),
                          in_axes=(0, None, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q_local_lambda, SMat_k_arg, Gp, inds, inds, M_arg, T_arg), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(jax.scipy.linalg.expm(-CO[i]))*0.25 for i in range(CO.shape[0])]))


def inf_ID(params_arg, i, SMat_arg, M_arg, T_arg, w_arg, tau_arg):
    vt1 = jnp.sort(jnp.concatenate((jnp.array([0]), params_arg[:i], jnp.array([T_arg]))))
    vt2 = jnp.sort(jnp.concatenate((jnp.array([0]), params_arg[i:], jnp.array([T_arg]))))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
    for m in range(3):
        for n in range(3):
            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], w_arg, M_arg) + 1j*Gp_im_map(vt[m], vt[n], w_arg, M_arg))
    L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
    dt = tau_arg
    fid=jnp.sum(L_diag, axis=0)/16.
    #clustering=jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    clustering=-jnp.sum(jnp.array([jnp.exp(-(9/(2*dt**2))*(vt[k][l+1]-vt[k][l])**2)/vt[k].shape[0] for k in range(2) for l in range(vt[k].shape[0]-1)]), axis=0)
    return -fid-clustering


def inf_ID_wk(params_arg, ind, SMat_k_arg, M_arg, T_arg, wk_arg, tau_arg):
    vt1 = jnp.sort(jnp.concatenate((jnp.array([0]), params_arg[:ind], jnp.array([T_arg]))))
    vt2 = jnp.sort(jnp.concatenate((jnp.array([0]), params_arg[ind:], jnp.array([T_arg]))))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk_arg, 1) + 1j*Gp_im_map(vt[i], vt[j], wk_arg, 1))
    L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, T_arg)
    dt = tau_arg
    fid=jnp.sum(L_diag, axis=0)/16.
    # clustering=jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2/(vt[i].shape[0]) for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    clustering=-jnp.sum(jnp.array([jnp.exp(-(9/(2*dt**2))*(vt[k][l+1]-vt[k][l])**2)/vt[k].shape[0] for k in range(2) for l in range(vt[k].shape[0]-1)]), axis=0)
    return -fid-clustering


def infidelity(params_arg, SMat_arg, M_arg, w_arg):
    vt1 = params_arg[0]
    vt2 = params_arg[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, M_arg) + 1j*Gp_im_map(vt[i], vt[j], w_arg, M_arg))
    L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
    return 1.-jnp.sum(L_diag, axis=0)/16


def infidelity_k(params_arg, SMat_k_arg, M_arg, wk_arg):
    vt1 = params_arg[0]
    vt2 = params_arg[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk_arg, 1) + 1j*Gp_im_map(vt[i], vt[j], wk_arg, 1))
    L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, vt1[-1])
    return 1.-jnp.sum(L_diag, axis=0)/16


def T2(params_arg, SMat_arg, M_arg, w_arg, qubit):
    vt1 = params_arg[0]
    vt2 = params_arg[1]
    vt12 = params_arg[0]
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, 1) + 1j*Gp_im_map(vt[i], vt[j], w_arg, 1))
    x1q = jnp.array([[0, 1], [1, 0]])
    z1q = jnp.array([[1, 0], [0, -1]])
    q1 = jnp.kron(x1q, z1q)
    q2 = jnp.kron(z1q, x1q)
    if qubit == 2:
        return Lambda(q2, q2, vt, SMat_arg, M_arg, w_arg)
    elif qubit == 1:
        return Lambda(q1, q1, vt, SMat_arg, M_arg, w_arg)
    else:
        raise ValueError("qubit must be an integer 1 or 2")


def hyperOpt(SMat_arg, nPs_arg, M_arg, T_arg, w_arg, tau_arg):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID, maxiter=200, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-6,
                                                     'maxfun': 1000, 'maxls': 20})
    opt_out = []
    init_params = []
    for i in nPs_arg[0]:
        opt_out_temp = []
        for j in nPs_arg[1]:
            vt = jnp.array(np.random.rand(i+j)*T_arg)
            init_params.append(vt)
            lower_bnd = jnp.zeros_like(vt)
            upper_bnd = jnp.ones_like(vt)*T_arg
            bnds = (lower_bnd, upper_bnd)
            opt = optimizer.run(vt, bnds, i, SMat_arg, M_arg, T_arg, w_arg, tau_arg)
            opt_out_temp.append(opt)
            print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt_0 = params_to_tk(vt_min_arr, T_arg, jnp.array([0, nPs_arg[0][inds_min[0]]]))
    vt_opt_1 = params_to_tk(vt_min_arr, T_arg, jnp.array([nPs_arg[0][inds_min[0]], nPs_arg[0][inds_min[0]]+nPs_arg[1][inds_min[1]]]))
    vt_opt_local = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
    #vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt_local, infidelity(vt_opt_local, SMat_arg, M_arg, w_arg) #inf_ID_out[inds_min]


def hyperOpt_k(SMat_k_arg, nPs_arg, M_arg, T_arg, wk_arg, tau_arg):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID_wk, maxiter=200, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-6,
                                                     'maxfun': 1000, 'maxls': 20})
    opt_out = []
    init_params = []
    for i in nPs_arg[0]:
        opt_out_temp = []
        for j in nPs_arg[1]:
            vt = jnp.array(np.random.rand(i+j)*T_arg)
            init_params.append(vt)
            lower_bnd = jnp.zeros_like(vt)
            upper_bnd = jnp.ones_like(vt)*T_arg
            bnds = (lower_bnd, upper_bnd)
            opt = optimizer.run(vt, bnds, i, SMat_k_arg, M_arg, T_arg, wk_arg, tau_arg)
            opt_out_temp.append(opt)
            print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt_0 = params_to_tk(vt_min_arr, T_arg, jnp.array([0, nPs_arg[0][inds_min[0]]]))
    vt_opt_1 = params_to_tk(vt_min_arr, T_arg, jnp.array([nPs_arg[0][inds_min[0]], nPs_arg[0][inds_min[0]]+nPs_arg[1][inds_min[1]]]))
    vt_opt_local = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
    return vt_opt_local, infidelity_k(vt_opt_local, SMat_k_arg, M_arg, wk_arg)


def Lambda(Oi, Oj, vt, SMat_arg, M_arg, w_arg):
    vt12 = make_tk12(vt[0], vt[1])
    vt.append(vt12)
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, M_arg) + 1j*Gp_im_map(vt[i], vt[j], w_arg, M_arg))
    CO = 0
    for i in range(3):
        for j in range(3):
            CO += -0.5*(sgn(Oi, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat_arg[i, j]*(sgn(Oi, i+1, 0)-1)*Gp[i, j], w_arg)/jnp.pi
    return jnp.real(jnp.trace(Oi@jax.scipy.linalg.expm(-CO)@Oj)*0.25)


def params_to_tk(params_arg, T_arg, shape: jnp.ndarray):
    vt = jnp.zeros(shape[1]-shape[0]+2)
    vt = vt.at[0].set(0.)
    vt = vt.at[-1].set(T_arg)
    vt = vt.at[1:shape[1]-shape[0]+1].set(jnp.sort(params_arg[shape[0]:shape[1]]))
    return vt


def cpmg_vt(T_arg, n):
    tk = [(k+0.50)*T_arg/(2*n) for k in range(int(2*n))]
    tk.append(T_arg)
    tk.insert(0,0.)
    return jnp.array(tk)


def cdd1_vt(T_arg, n):
    tk = [(k+1)*T_arg/(2*n) for k in range(int(2*n-1))]
    tk.append(T_arg)
    tk.insert(0,0.)
    return jnp.array(tk)


def uddn(T_arg, n):
    tk = [T_arg*jnp.sin(jnp.pi*k/(2*n+2))**2 for k in range(n)]
    tk.append(T_arg)
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


def cddn_rec(T_arg, n):
    if n == 1:
        return np.array([0., T_arg/2])
    return comb_vks([0.], comb_vks(comb_vks(cddn_rec(T_arg/2, n-1), [T_arg/2]), cddn_rec(T_arg/2, n-1) + T_arg/2))


def cddn(T_arg, n):
    return np.concatenate((np.array([0]), comb_vks(cddn_rec(T_arg, n), [T_arg])))


def pddn(T_arg, n, M_arg):
    out = cddn(T_arg,n)
    if M_arg == 1:
        return out
    for i in range(M_arg):
        out = np.concatenate((out, cddn(T_arg,n)+(i+1)*T_arg))
    return  out


def opt_known_pulses(pLib_arg, SMat_arg, M_arg, w_arg):
    infmin = jnp.inf
    inds_min=0
    for i in range(len(pLib_arg)):
        for j in range(len(pLib_arg)):
            for k in range(len(pLib_arg[i])):
                for l in range(len(pLib_arg[j])):
                    tk1 = pLib_arg[i][k]
                    tk2 = pLib_arg[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
                    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
                    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
                    for m in range(3):
                        for n in range(3):
                            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], w_arg, M_arg) + 1j*Gp_im_map(vt[m], vt[n], w_arg, M_arg))
                    L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
                    # infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-jnp.sum(L_diag, axis=0)/16)
                    inf_ijkl = 1.-jnp.sum(L_diag, axis=0)/16
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i,j,k,l)
    tk1 = pLib_arg[inds_min[0]][inds_min[2]]
    tk2 = pLib_arg[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt_local = [tk1, tk2, tk12]
    return vt_opt_local, infmin


def opt_known_pulses_k(pLib_arg, SMat_k_arg, M_arg, T_arg, wk_arg):
    infmin = jnp.inf
    inds_min=0
    for i in range(len(pLib_arg)):
        for j in range(len(pLib_arg)):
            for k in range(len(pLib_arg[i])):
                for l in range(len(pLib_arg[j])):
                    tk1 = pLib_arg[i][k]
                    tk2 = pLib_arg[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
                    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
                    Gp = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
                    for m in range(3):
                        for n in range(3):
                            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], wk_arg, 1) + 1j*Gp_im_map(vt[m], vt[n], wk_arg, 1))
                    L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, T_arg)
                    # infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-jnp.sum(L_diag, axis=0)/16)
                    inf_ijkl = 1.-jnp.sum(L_diag, axis=0)/16
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i,j,k,l)
    tk1 = pLib_arg[inds_min[0]][inds_min[2]]
    tk2 = pLib_arg[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt_local = [tk1, tk2, tk12]
    return vt_opt_local, infmin


def makeSMat_k(specs_arg, wk_arg, wkqns_arg, gamma_arg, gamma12_arg):
    SMat_local = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    SMat_local = SMat_local.at[0, 0].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_11(wk_arg[0])]), specs_arg["S11"]))))
    SMat_local = SMat_local.at[1, 1].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_22(wk_arg[0])]), specs_arg["S22"]))))
    SMat_local = SMat_local.at[2, 2].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_1212(wk_arg[0])]), specs_arg["S1212"]))))
    SMat_local = SMat_local.at[0, 1].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_1_2(wk_arg[0], gamma_arg)]), specs_arg["S12"]))))
    SMat_local = SMat_local.at[1, 0,].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(jnp.concatenate((jnp.array([S_1_2(wk_arg[0], gamma_arg)]), specs_arg["S12"])))))
    SMat_local = SMat_local.at[0, 2].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_1_12(wk_arg[0], gamma12_arg)]), specs_arg["S112"]))))
    SMat_local = SMat_local.at[2, 0].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(jnp.concatenate((jnp.array([S_1_12(wk_arg[0], gamma12_arg)]), specs_arg["S112"])))))
    SMat_local = SMat_local.at[1, 2].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_2_12(wk_arg[0], gamma12_arg-gamma_arg)]), specs_arg["S212"]))))
    SMat_local = SMat_local.at[2, 1].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(jnp.concatenate((jnp.array([S_2_12(wk_arg[0], gamma12_arg-gamma_arg)]), specs_arg["S212"])))))
    return SMat_local


def makeSMat_k_ideal(wk_arg, gamma_arg, gamma12_arg):
    SMat_ideal_local = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    SMat_ideal_local = SMat_ideal_local.at[0, 0].set(S_11(wk_arg))
    SMat_ideal_local = SMat_ideal_local.at[1, 1].set(S_22(wk_arg))
    SMat_ideal_local = SMat_ideal_local.at[2, 2].set(S_1212(wk_arg))
    SMat_ideal_local = SMat_ideal_local.at[0, 1].set(S_1_2(wk_arg, gamma_arg))
    SMat_ideal_local = SMat_ideal_local.at[1, 0].set(jnp.conj(S_1_2(wk_arg, gamma_arg)))
    SMat_ideal_local = SMat_ideal_local.at[0, 2].set(S_1_12(wk_arg, gamma12_arg))
    SMat_ideal_local = SMat_ideal_local.at[2, 0].set(jnp.conj(S_1_12(wk_arg, gamma12_arg)))
    SMat_ideal_local = SMat_ideal_local.at[1, 2].set(S_2_12(wk_arg, gamma12_arg-gamma_arg))
    SMat_ideal_local = SMat_ideal_local.at[2, 1].set(jnp.conj(S_2_12(wk_arg, gamma12_arg-gamma_arg)))
    return SMat_ideal_local


########################################################################################################################
########################################################################################################################
####################################### Set and run the optimization ###################################################
########################################################################################################################
########################################################################################################################


# Load the system parameters
parent_dir = os.pardir
fname = "DraftRun_NoSPAM"
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
M = params['M']
a_sp = params['a_sp']
c = params['c']
Tqns = params['T']


T=Tqns
p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
w = jnp.linspace(0.0001, 2*jnp.pi*mc/Tqns, 4000)
w_ideal = jnp.linspace(0.0001, 2*jnp.pi*2*mc/Tqns, 8000)
wkqns = jnp.array([2*jnp.pi*(n+1)/Tqns for n in range(mc)])
wkqns_ideal = jnp.array([2*jnp.pi*n/Tqns for n in range(2*mc+1)])
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



# Parameters for the optimization over known pulse sequences
tau = T/80
Tg = 5*14*1e-6



print("###########################################################################################")
print(f"Optimizing the Idling gate for {np.round(Tg/T2q1, 2)} T2q1 or {np.round(Tg/T2q2, 2)} T2q2")
print("###########################################################################################")



best_seq = 0
best_inf = np.inf
best_M = 0



reps_known = [20, 40, 100, 200, 300, 400, 600]
reps_opt = [20, 40, 100, 200, 300, 400, 600]



for i in reps_known:
    pLib=[]
    cddLib = []
    Tknown = Tg/i
    Mknown = i
    print("####################")
    print("T="+str(Tknown)+" and M="+str(i))
    print("####################")
    if Mknown >= 10:
        wk_local = jnp.array([0.0001]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(Tg*mc/(i*Tqns))))])
        wk_ideal_local = jnp.array([0.0001]+[2*jnp.pi*(n+1)/Tknown for n in range(int(jnp.floor(4*Tg*mc/(i*Tqns))))])
        SMat_k_local = makeSMat_k(specs, wk_local, jnp.concatenate((jnp.array([wk_local[0]]), wkqns)), gamma, gamma12)
        SMat_k_ideal = makeSMat_k_ideal(wk_ideal_local, gamma, gamma12)
        # Make CDD_n libraries that respect the minimum pulse separation constraint
        cddOrd = 1
        make = True
        while make:
            pul = cddn(Tknown, cddOrd)
            cddOrd += 1
            for j in range(1, pul.size-1):
                if pul[j+1] - pul[j] < tau:
                    make = False
            if not make:
                break
            cddLib.append(pul)
        pLib.append(cddLib)
        # Generate an Idling gate that is optimized using known sequences
        known_opt, known_inf = opt_known_pulses_k(pLib, SMat_k_local, Mknown, Tknown, wk_local)
        inf_known = infidelity_k(known_opt, SMat_k_ideal, Mknown, wk_ideal_local)
    else:
        cddOrd = 1
        make = True
        while make:
            pul = cddn(Tknown, cddOrd)
            cddOrd += 1
            for j in range(1, pul.size-2):
                if pul[j+1] - pul[j] < tau:
                    make = False
            if not make:
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
for i in reps_opt:
    Topt = Tg/i
    Mopt = i
    nPs = np.random.randint(1, Topt/tau, (2,4))
    print(nPs)
    if Mopt >= 10:
        wk_local = jnp.array([0.0001]+[2*jnp.pi*(n+1)/Topt for n in range(int(jnp.floor(Tg*mc/(i*Tqns))))])
        wk_ideal_local = jnp.array([0.0001]+[2*jnp.pi*(n+1)/Topt for n in range(int(jnp.floor(4*Tg*mc/(i*Tqns))))])
        SMat_k_local = makeSMat_k(specs, wk_local, jnp.concatenate((jnp.array([wk_local[0]]), wkqns)), gamma, gamma12)
        SMat_k_ideal = makeSMat_k_ideal(wk_ideal_local, gamma, gamma12)
        # Generate an Idling gate that is optimized over a given number of pulses on each qubit
        vt_opt, inf_min = hyperOpt_k(SMat_k_local, nPs, Mopt, Topt, wk_local, tau)
        inf_opt = infidelity_k(vt_opt, SMat_k_ideal, Mopt, wk_ideal_local)
    else:
        vt_opt, inf_min = hyperOpt(SMat, nPs, Mopt, Topt, w, tau)
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



np.savez(os.path.join(path, 'optimizeLog.npz'), gtime=Tg, best_inf=best_inf, best_seq_1=best_seq[0],
         best_seq_2=best_seq[1], best_seq_12=best_seq[2], best_M=best_M, opt_inf=opt_inf, opt_seq_1=opt_seq[0],
         opt_seq_2=opt_seq[1], opt_seq_12=opt_seq[2], opt_M=opt_M)



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
wk_local_plot=wkqns
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
axs[0, 0].plot(wk_local_plot/xunits, specs["S11"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
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
axs[0, 1].plot(wk_local_plot/xunits, specs["S22"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
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
axs[0, 2].plot(wk_local_plot/xunits, specs["S1212"]/yunits, 'r--^', alpha=alp, lw=1.5*lw)
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
w_plot = w[300:]
axs[1, 0].plot(w_plot/xunits, np.real(S_1_2(w_plot, gamma))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(wk_local_plot/xunits, np.real(specs["S12"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(w_plot/xunits, np.imag(S_1_2(w_plot, gamma))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 0].plot(wk_local_plot/xunits, np.imag(specs["S12"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
axs[1, 0].legend([r'Re[$S^+_{1,2}(\omega)$]', r'Re[$\hat{S}^+_{1,2}(\omega)$]', r'Im[$S^+_{1,2}(\omega)$]'
                     , r'Im[$\hat{S}^+_{1,2}(\omega)$]'], fontsize=legendfont, loc='upper left')
axs[1, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=labelfont)
# max_lim = np.maximum(np.abs(np.real(S_1_2(w, gamma))/1e3).max(), np.abs(np.imag(S_1_2(w, gamma))/1e3).max())
axs[1, 0].set_ylim(-max_lim_row2_left*1.01, max_lim_row2_left*1.01)
axs[1, 0].tick_params(direction='in')
axs[1, 0].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
bx = axs[1, 0].twinx()
bx.tick_params(direction='in')
bx.plot(w_plot/xunits, Gp_re(vt_opt[0], vt_opt[1], w_plot, Mplot_opt), 'g--', lw=lw)
bx.plot(w_plot/xunits, Gp_im(vt_opt[0], vt_opt[1], w_plot, Mplot_opt), 'k--', lw=lw)
bx.plot(w_plot/xunits, Gp_re(known_opt[0], known_opt[1], w_plot, Mplot_known), 'm-', lw=lw)
bx.plot(w_plot/xunits, Gp_im(known_opt[0], known_opt[1], w_plot, Mplot_known), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[1], w_plot, Mplot_opt)).max(),
                           np.abs(Gp_im(vt_opt[0], vt_opt[1], w_plot, Mplot_opt)).max(),
                           np.abs(Gp_re(known_opt[0], known_opt[1], w_plot, Mplot_known)).max(),
                           np.abs(Gp_im(known_opt[0], known_opt[1], w_plot, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
axs[1, 1].plot(w_plot/xunits, np.real(S_1_12(w_plot, gamma12))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(wk_local_plot/xunits, np.real(specs["S112"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(w_plot/xunits, np.imag(S_1_12(w_plot, gamma12))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 1].plot(wk_local_plot/xunits, np.imag(specs["S112"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
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
bx.plot(w_plot/xunits, Gp_re(vt_opt[0], vt_opt[2], w_plot, Mplot_opt), 'g--', lw=lw)
bx.plot(w_plot/xunits, Gp_im(vt_opt[0], vt_opt[2], w_plot, Mplot_opt), 'k--', lw=lw)
bx.plot(w_plot/xunits, Gp_re(known_opt[0], known_opt[2], w_plot, Mplot_known), 'm-', lw=lw)
bx.plot(w_plot/xunits, Gp_im(known_opt[0], known_opt[2], w_plot, Mplot_known), 'c-', lw=lw)
# bx.set_yticklabels([])
# bx.set_yticks([])
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[2], w_plot, Mplot_opt)).max(),
                                     np.abs(Gp_im(vt_opt[0], vt_opt[2], w_plot, Mplot_opt)).max(),
                                     np.abs(Gp_re(known_opt[0], known_opt[2], w_plot, Mplot_known)).max(),
                                     np.abs(Gp_im(known_opt[0], known_opt[2], w_plot, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
axs[1, 2].plot(w_plot/xunits, np.real(S_2_12(w_plot, gamma12-gamma))/yunits, 'r-', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(wk_local_plot/xunits, np.real(specs["S212"])/yunits, 'r--^', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(w_plot/xunits, np.imag(S_2_12(w_plot, gamma12-gamma))/yunits, 'b-', alpha=alp, lw=1.5*lw)
axs[1, 2].plot(wk_local_plot/xunits, np.imag(specs["S212"])/yunits, 'b--^', alpha=alp, lw=1.5*lw)
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
bx.plot(w_plot/xunits, Gp_re(vt_opt[1], vt_opt[2], w_plot, Mplot_opt), 'g--', lw=lw)
bx.plot(w_plot/xunits, Gp_im(vt_opt[1], vt_opt[2], w_plot, Mplot_opt), 'k--', lw=lw)
bx.plot(w_plot/xunits, Gp_re(known_opt[1], known_opt[2], w_plot, Mplot_known), 'm-', lw=lw)
bx.plot(w_plot/xunits, Gp_im(known_opt[1], known_opt[2], w_plot, Mplot_known), 'c-', lw=lw)
bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=labelfont)
max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[1], vt_opt[2], w_plot, Mplot_opt)).max(),
                                     np.abs(Gp_im(vt_opt[1], vt_opt[2], w_plot, Mplot_opt)).max(),
                                     np.abs(Gp_re(known_opt[1], known_opt[2], w_plot, Mplot_known)).max(),
                                     np.abs(Gp_im(known_opt[1], known_opt[2], w_plot, Mplot_known)).max()]))
bx.set_ylim(-max_lim*1.01, max_lim*1.01)
bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]',
           r'Re[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]']
          , fontsize=legendfont, loc='lower right')
plt.savefig(os.path.join(path, 'IDGateLog.pdf'))
# plt.show()
print('End')
