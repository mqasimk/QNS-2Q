import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import jaxopt
from spectraIn import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12
import os
jax.config.update("jax_enable_x64", True)


########################################################################################################################
########################################################################################################################
####################################### Utility functions ##############################################################
########################################################################################################################
########################################################################################################################


@jax.jit
def Sc(w):
    tc = 4e-4
    S0 = 1e-6
    return S0/(1+(tc**2)*(w**2))


@jax.jit
def zzPTM():
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    U = jax.scipy.linalg.expm(-1j*jnp.kron(p1q[3], p1q[3])*jnp.pi/4)
    gamma = jnp.array([[(1/4)*jnp.trace(p2q[i]@U@p2q[j]@U.conj().transpose()) for j in range(16)]for i in range(16)])
    return jnp.real(gamma)


# @jax.jit
def pulse(mu, sig, t):
    return jnp.exp(-(t-mu)**2/(2*sig**2))


@jax.jit
def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O)@z2q[a]@z2q[b]@O@z2q[a]@z2q[b])/4


@jax.jit
def ff(tk, w):
    return jnp.sum(jnp.array([1j*((-1)**k)*(jnp.exp(-1j*w*tk[k+1])-jnp.exp(-1j*w*tk[k]))/w for k in range(jnp.size(tk)-1)]), axis=0)


@jax.jit
def ft(f, t, w):
    return jax.scipy.integrate.trapezoid(f*jnp.exp(1j*w*t), t)


@jax.jit
def Gp_re(vti, vtj, w, M):
    return jnp.real(ff(vti, w)*ff(vtj, -w)*jnp.sin(w*M*vti[-1]*0.5)**2/jnp.sin(w*vti[-1]*0.5)**2)


@jax.jit
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


def inf_CZ(params, i, j, SMat, M, T, w, Jmax):
    vt1 = jnp.concatenate((jnp.array([0]), params[:i], jnp.array([T])))
    vt2 = jnp.concatenate((jnp.array([0]), params[i:i+j], jnp.array([T])))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)), in_axes=(0, None, None, None, None, None, None))
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    dt = tau
    # xax = jnp.linspace(0, M*vt[2][-1], M*1000)
    # xbase = jnp.linspace(0, vt[2][-1], 1000)
    # sig = M*vt[2][-1]/2
    # tf = M*vt[2][-1]/2
    # area = jax.scipy.integrate.trapezoid(jnp.tile(y_t(xbase, vt[2]), M)*pulse(tf, sig, xax), xax)
    # J = jnp.maximum(jnp.minimum((jnp.pi*0.25)/area, Jmax), -Jmax)
    tax = np.linspace(0, T, 1000)
    J = jnp.maximum(jnp.minimum(jnp.pi*0.25/(M*jax.scipy.integrate.trapezoid(y_t(tax, vt12), tax)), Jmax), -Jmax)
    fid = jnp.trace(zzPTM().transpose()@L_map(p2q, p2q, vt, SMat, M, w, J))/16.
    clustering=(1/(vt[0].shape[0]+vt[1].shape[0]-2))*jnp.sum(jnp.array([jnp.tanh(vt[i].shape[0]*(vt[i][j+1]-vt[i][j]/dt-1)) for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    #jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    return -fid-clustering*1e-3



def infidelity(params, SMat, M, w, J):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)), in_axes=(0, None, None, None, None, None, None))
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    return 1.-jnp.trace(zzPTM().transpose()@L_map(p2q, p2q, vt, SMat, M, w, J))/16.



def T2(params, SMat, M, w, qubit):
    vt1 = params[0]
    vt2 = params[1]
    vt12 = params[0]
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, 1) + 1j*Gp_im_map(vt[i], vt[j], w, 1))
    x1q = jnp.array([[0, 1], [1, 0]])
    z1q = jnp.array([[1, 0], [0, -1]])
    q1 = jnp.kron(x1q, z1q)
    q2 = jnp.kron(z1q, x1q)
    if qubit == 2:
        return Lambda_CZ(q2, q2, vt, SMat, M, w, 0)
    elif qubit == 1:
        return Lambda_CZ(q1, q1, vt, SMat, M, w, 0)
    else:
        raise ValueError("qubit must be an integer 1 or 2")


def hyperOpt(SMat, nPs, M, T, w, Jmax, vtin = None):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_CZ, maxiter=80, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-8,
                                                     'maxfun': 1000, 'maxls': 20})
    opt_out = []
    init_params = []
    if vtin is None:
        for i in nPs[0]:
            opt_out_temp = []
            for j in nPs[1]:
                vt = jnp.concatenate((jnp.sort(np.random.rand(i)*T),jnp.sort(np.random.rand(j)*T)))
                init_params.append(vt)
                lower_bnd = jnp.zeros(vt.size)
                upper_bnd = jnp.ones(vt.size)*T
                bnds = (lower_bnd, upper_bnd)
                opt = optimizer.run(vt, bnds, i, j, SMat, M, T, w, Jmax)
                opt_out_temp.append(opt)
                print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([i, j]))
            opt_out.append(opt_out_temp)
        inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
        inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
        vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
        vt_opt = [params_to_tk(vt_min_arr, T, jnp.array([0, nPs[0][inds_min[0]]])), params_to_tk(vt_min_arr, T, jnp.array([nPs[0][inds_min[0]], nPs[0][inds_min[0]]+nPs[1][inds_min[1]]]))]
        vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
        # xax = jnp.linspace(0, M*vt_opt[2][-1], M*1000)
        # xbase = jnp.linspace(0, vt_opt[2][-1], 1000)
        # sig = M*vt_opt[2][-1]/2
        # tf = M*vt_opt[2][-1]/2
        # area = jax.scipy.integrate.trapezoid(jnp.tile(y_t(xbase, vt_opt[2]), M)*pulse(tf, sig, xax), xax)
        # Jopt = jnp.maximum(jnp.minimum((jnp.pi*0.25)/area, Jmax), -Jmax)
        tax = jnp.linspace(0, T, 1000)
        Jopt = jnp.pi*0.25/(M*jax.scipy.integrate.trapezoid(y_t(tax, vt_opt[2]), tax))
    else:
        for vt1 in vtin:
            opt_out_temp = []
            for vt2 in vtin:
                vt = jnp.concatenate((vt1[1:-1],vt2[1:-1]))
                init_params.append(vt)
                lower_bnd = jnp.zeros(vt.size)
                upper_bnd = jnp.ones(vt.size)*T
                bnds = (lower_bnd, upper_bnd)
                opt = optimizer.run(vt, bnds, vt1[1:-1].size, vt2[1:-1].size, SMat, M, T, w, Jmax)
                opt_out_temp.append(opt)
                print("Optimized Cost: "+str(opt.state[0])+", No. of pulses on qubits:"+str([vt1[1:-1].size, vt2[1:-1].size]))
            opt_out.append(opt_out_temp)
        inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
        inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape) #jnp.where(inf_ID_out == jnp.min(inf_ID_out))
        vt_opt = [vtin[inds_min[0]], vtin[inds_min[1]]]
        vt_opt.append(make_tk12(vt_opt[0], vt_opt[1]))
        # xax = jnp.linspace(0, M*vt_opt[2][-1], M*1000)
        # xbase = jnp.linspace(0, vt_opt[2][-1], 1000)
        # sig = M*vt_opt[2][-1]/2
        # tf = M*vt_opt[2][-1]/2
        # area = jax.scipy.integrate.trapezoid(jnp.tile(y_t(xbase, vt_opt[2]), M)*pulse(tf, sig, xax), xax)
        # Jopt = jnp.maximum(jnp.minimum((jnp.pi*0.25)/area, Jmax), -Jmax)
        tax = jnp.linspace(0, T, 1000)
        Jopt = np.maximum(np.minimum(jnp.pi*0.25/(M*jax.scipy.integrate.trapezoid(y_t(tax, vt_opt[2]), tax)),Jmax),-Jmax)
    return vt_opt, infidelity(vt_opt, SMat, M, w, Jopt), Jopt


def Lambda_CZ(Oi, Oj, vt, SMat, M, w, J):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array([jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j*Gp_im_map(vt[i], vt[j], w, M))
    tax = jnp.linspace(0, vt[2][-1], 1000)
    CO = 0
    # ft1212 = jax.vmap(ft, in_axes=(None, None, 0))
    # y1212 = J*pulse(tf, sig, xax)*jnp.tile(y_t(xbase, vt[2]), M)
    for i in range(3):
        for j in range(3):
            CO += (-0.5*(sgn(Oi, i+1, j+1)+1)*z2q[i+1]@z2q[j+1]*jax.scipy.integrate.trapezoid(SMat[i, j]*(sgn(Oi, i+1, 0)-1)*Gp[i, j], w)/jnp.pi)
                   #+(1-sgn(Oi, 1, 2))*(jax.scipy.integrate.trapezoid(Sc(w)*jnp.real(np.abs(ft1212(y1212, xax, w))**2), w)/jnp.pi)*z2q[0])
    rot = (1.-sgn(Oi, 1, 2))*M*jax.scipy.integrate.trapezoid(J*y_t(tax, vt[2]), tax)*z2q[3]
    return jnp.real(jnp.trace(Oi@jax.scipy.linalg.expm(-1j*rot-CO)@Oj)*0.25)


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


def opt_known_pulses(pLib, SMat, M, w, Jmax):
    infmin = jnp.inf
    inds_min=0
    Jopt = 0
    for i in range(len(pLib)):
        for j in range(len(pLib)):
            for k in range(len(pLib[i])):
                for l in range(len(pLib[j])):
                    tk1 = pLib[i][k]
                    tk2 = pLib[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    # xax = jnp.linspace(0, M*vt[2][-1], M*1000)
                    # xbase = jnp.linspace(0, vt[2][-1], 1000)
                    # sig = M*vt[2][-1]/2
                    # tf = M*vt[2][-1]/2
                    # area = jax.scipy.integrate.trapezoid(jnp.tile(y_t(xbase, vt[2]), M)*pulse(tf, sig, xax), xax)
                    # J = jnp.maximum(jnp.minimum((jnp.pi*0.25)/area, Jmax), -Jmax)
                    tax = jnp.linspace(0, vt[2][-1], 1000)
                    J = np.maximum(np.minimum(jnp.pi*0.25/(M*jax.scipy.integrate.trapezoid(y_t(tax, vt[2]), tax)),Jmax),-Jmax)
                    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)), in_axes=(0, None, None, None, None, None, None))
                    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
                    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
                    inf_ijkl = 1.-jnp.trace(zzPTM().transpose()@L_map(p2q, p2q, vt, SMat, M, w, J))/16.
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i,j,k,l)
                        Jopt = J
    tk1 = pLib[inds_min[0]][inds_min[2]]
    tk2 = pLib[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt = [tk1, tk2, tk12]
    return vt_opt, infmin, Jopt


def makeSMat_k(specs, wk, wkqns, gamma, gamma12):
    SMat = jnp.zeros((3, 3, wk.size), dtype=jnp.complex128)
    SMat = SMat.at[0, 0].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_11(wk[0])]), specs["S11"]))))
    SMat = SMat.at[1, 1].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_22(wk[0])]), specs["S22"]))))
    # SMat = SMat.at[2, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1212(wk[0])]), specs["S1212"]))))
    SMat = SMat.at[0, 1].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"]))))
    SMat = SMat.at[1, 0,].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"])))))
    # SMat = SMat.at[0, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1_12(wk[0], gamma12)]), specs["S112"]))))
    # SMat = SMat.at[2, 0].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_1_12(wk[0], gamma12)]), specs["S112"])))))
    # SMat = SMat.at[1, 2].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_2_12(wk[0], gamma12-gamma)]), specs["S212"]))))
    # SMat = SMat.at[2, 1].set(jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_2_12(wk[0], gamma12-gamma)]), specs["S212"])))))
    return SMat


def makeSMat_k_ideal(wk, gamma, gamma12):
    SMat_ideal = jnp.zeros((3, 3, wk.size), dtype=jnp.complex128)
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


parent_dir = os.pardir
fname = "DraftRun_Mmit_hat"
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
# M=20


p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
wqns = jnp.linspace(0.001, 2*jnp.pi*mc/T, 4000)
w_ideal = jnp.linspace(0.0001, 2*jnp.pi*2*mc/T, 12000)
wkqns = jnp.array([2*jnp.pi*(n+1)/Tqns for n in range(mc)])
SMat = jnp.zeros((3, 3, wqns.size), dtype=jnp.complex128)


SMat = SMat.at[0, 0].set(jnp.interp(wqns, wkqns, specs["S11"]))
SMat = SMat.at[1, 1].set(jnp.interp(wqns, wkqns, specs["S22"]))
# SMat = SMat.at[2, 2].set(jnp.interp(wqns, wkqns, specs["S1212"]))
SMat = SMat.at[0, 1].set(jnp.interp(wqns, wkqns, specs["S12"]))
SMat = SMat.at[1, 0].set(jnp.interp(wqns, wkqns, np.conj(specs["S12"])))
# SMat = SMat.at[0, 2].set(jnp.interp(wqns, wkqns, specs["S112"]))
# SMat = SMat.at[2, 0].set(jnp.interp(wqns, wkqns, np.conj(specs["S112"])))
# SMat = SMat.at[1, 2].set(jnp.interp(wqns, wkqns, specs["S212"]))
# SMat = SMat.at[2, 1].set(jnp.interp(wqns, wkqns, np.conj(specs["S212"])))


SMat_ideal = jnp.zeros((3, 3, w_ideal.size), dtype=jnp.complex128)
SMat_ideal = SMat_ideal.at[0, 0].set(S_11(w_ideal))
SMat_ideal = SMat_ideal.at[1, 1].set(S_22(w_ideal))
SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(w_ideal))
SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(w_ideal, gamma))
SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(w_ideal, gamma)))
SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(w_ideal, gamma12))
SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(w_ideal, gamma12)))
SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(w_ideal, gamma12-gamma))
SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(w_ideal, gamma12-gamma)))


taxis = jnp.linspace(1*T, 20*T, 20)
vt_T2 = jnp.array([jnp.array([[0, taxis[i]], [0, taxis[i]]]) for i in range(taxis.shape[0])])
q1T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, w_ideal, 1) for i in range(vt_T2.shape[0])])
q2T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, w_ideal, 2) for i in range(vt_T2.shape[0])])
T2q1 = jnp.inf
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
print(f"T2 time for qubit 1 is {np.round(T2q1/1e-6, 2)} us")
print(f"T2 time for qubit 2 is {np.round(T2q2/1e-6, 2)} us")
print("###########################################################################################")
print("In terms of T,")
print(f"T2 time for qubit 1 is {np.round(T2q1/T,2)} T")
print(f"T2 time for qubit 2 is {np.round(T2q2/T, 2)} T")
print("###########################################################################################")


yaxis_base = []
xaxis_base = []
yaxis_Tg = []
seqs = []
xaxis_Tg = []


tau = T/80
Jmax = 3e6

for i in [1,2,3,3.5,4,4.5,5]:
    yaxis = []
    Tg = T/2**(i-1)
    base_gate = [[jnp.array([0, Tg])], [jnp.array([0, Tg])]]
    base_opt, base_inf, Jopt = opt_known_pulses(base_gate, SMat_ideal, 1, w_ideal, Jmax)
    inf_base = infidelity(base_opt, SMat_ideal, 1, w_ideal, Jopt)
    yaxis_base.append(inf_base)
    xaxis_base.append(Tg)

    print("###########################################################################################")
    print(f"Gate Time: {np.round(Tg/T2q1, 2)} T2q1 or {np.round(Tg/T2q2, 2)} T2q2")
    print(f"Jmax: {Jmax}")
    print("###########################################################################################")
    print(f'infidelity of the uncorrected gate: {inf_base}')
    print(f'number of pulses: {[base_opt[i].shape[0]-2 for i in range(2)]}')
    print(f"The coupling strength J: {Jopt}")
    print("###########################################################################################")
    best_inf = np.inf
    best_seq = 0
    Jopt_best = 0
    if Tg < tau:
        best_inf = base_inf
        best_seq = base_gate
        continue
    for j in [1]:
        pLib=[]
        cddLib = []
        Tknown = j*Tg
        Mknown = int(1/j)
        # cddLib.append(np.array([0.,Tknown]))
        if Tknown < tau:
            continue
        # if Mknown < 10:
        cddOrd = 1
        make = True
        while make:
            pul = cddn(Tknown, cddOrd)
            cddOrd += 1
            for i in range(1, pul.size-2):
                if pul[i+1] - pul[i] < tau:
                    make = False
            if make == False:
                break
            cddLib.append(pul)
        pLib.append(cddLib)
        known_opt, known_inf, Jopt = opt_known_pulses(pLib, SMat, Mknown, wqns, Jmax)
        if known_inf <= best_inf:
            inf_known = infidelity(known_opt, SMat_ideal, Mknown, w_ideal, Jopt)
            best_inf = inf_known
            best_seq = known_opt
            Jopt_best = Jopt
            print(f'number of pulses: {[best_seq[i].shape[0]-2 for i in range(2)]}')
            print(f"The coupling strength J: {Jopt_best}")
            print(f'best infidelity till now: {best_inf}, # of repetitions considered: {Mknown}')
            print("###########################################################################################")
    yaxis_Tg.append(best_inf)
    xaxis_Tg.append(Tg)
    seqs.append(best_seq)

yaxis_opt = []
xaxis_opt = []
seqs_opt = []
nplist = [[[158],[158]],[[78],[78]],[[38,39],[38,39]],[[38,39],[38,39]],[[38,39],[38,39]],[[38,39],[38,39]],[[38,39],[38,39]],[[38,39],[38,39]],[[38,39],[38,39]]]
count = 0
for i in [1,2,3,3.5,4,4.5,5]:
    yaxis = []
    Tg = T/2**(i-1)
    best_inf = np.inf
    best_seq = 0
    Jopt_best = 0
    if Tg < tau:
        continue
    for j in [1]:
        Topt = j*Tg
        Mopt = int(1/j)
        if Topt < tau:
            continue
        nps = nplist[count]#[[int((Topt/tau)/2),int((Topt/tau)/2)+1],[int((Topt/tau)/2),int((Topt/tau)/2)+1]]
        # cddLib = []
        # if Topt < tau:
        #     continue
        # cddOrd = 1
        # make = True
        # while make:
        #     pul = cddn(Topt, cddOrd)
        #     cddOrd += 1
        #     for k in range(1, pul.size-2):
        #         if pul[k+1] - pul[k] < tau:
        #             make = False
        #     if make == False:
        #         break
        #     cddLib.append(pul)
        if 1 <= i <= 6:
            vt_opt, opt_inf, Jopt = hyperOpt(SMat, nps, Mopt, Topt, wqns, Jmax, [jnp.linspace(0, Topt, int((Topt/tau)))])
        else:
            vt_opt, opt_inf, Jopt = hyperOpt(SMat, nps, Mopt, Topt, wqns, Jmax)
            count+=1
        # vt_opt, opt_inf, Jopt = hyperOpt(SMat, nps, Mopt, Topt, wqns, Jmax)
        # count+=1
        inf_opt = infidelity(vt_opt, SMat_ideal, Mopt, w_ideal, Jopt)
        if opt_inf <= best_inf:
            best_inf = inf_opt
            best_seq = vt_opt
            Jopt_best = Jopt
            print(f'number of pulses: {[best_seq[i].shape[0]-2 for i in range(2)]}')
            print(f"The coupling strength J: {Jopt_best}")
            print(f'best infidelity till now: {best_inf}, # of repetitions considered: {Mopt}')
            print("###########################################################################################")
    yaxis_opt.append(best_inf)
    xaxis_opt.append(Tg)
    seqs_opt.append(best_seq)


np.savez(os.path.join(path,"infs_known_4.npz"), infs_known=np.array(yaxis_Tg), infs_base=np.array(yaxis_base), taxis=np.array(xaxis_Tg))
np.savez(os.path.join(path,"infs_opt_4.npz"), infs_opt=np.array(yaxis_opt), taxis=np.array(xaxis_opt))


# knowndata = np.load(os.path.join(path,"infs_known_3.npz"))
# optdata = np.load(os.path.join(path,"infs_opt_3.npz"))

# yaxis_Tg = knowndata["infs_known"]
# xaxis_Tg = knowndata["taxis"]
# yaxis_base = knowndata["infs_base"]
# xaxis_base = xaxis_Tg
# xaxis_opt = optdata["taxis"]
# yaxis_opt = optdata["infs_opt"]




legendfont = 12
labelfont = 16
tickfont = 12


fig = plt.figure(figsize=(16,9))
plt.plot(xaxis_base, yaxis_base, "r^-")
plt.plot(xaxis_Tg, yaxis_Tg, "bs-")
plt.plot(xaxis_opt, yaxis_opt, "ko-")
plt.legend(["Uncorrected Gate", "DD Gate", "NT Gate"], fontsize=legendfont)
plt.xlabel('Gate Time (s)', fontsize=labelfont)
plt.ylabel('Gate Infidelity', fontsize=labelfont)
plt.xscale('log')
plt.yscale('log')
# plt.tick_params(axis='both', labelsize=tickfont)
plt.savefig(os.path.join(path,"infs_GateTime_4.png"), dpi=800)
plt.show()
