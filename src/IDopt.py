
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import jaxopt
import os
from spectraIn import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12
import itertools


########################################################################################################################
########################################################################################################################
#
# Configuration Class
#
########################################################################################################################
########################################################################################################################


class PulseOptimizerConfig:
    """A class to hold all the parameters for the optimization."""

    def __init__(self, fname="DraftRun_NoSPAM", parent_dir=os.pardir, Tg=5 * 14 * 1e-6, reps_known=None,
                 reps_opt=None, tau_divisor=80, max_pulses=100):
        """
        Initializes the configuration for the pulse optimization.

        Args:
            fname (str): The name of the file containing the system parameters.
            parent_dir (str): The parent directory of the file.
            Tg (float): The total gate time.
            reps_known (list): A list of repetitions for the known pulse sequences.
            reps_opt (list): A list of repetitions for the optimized pulse sequences.
            tau_divisor (int): The divisor for the pulse separation.
            max_pulses (int): The maximum total number of pulses allowed in the entire sequence (Tg).
        """
        if reps_opt is None:
            reps_opt = [i for i in range(10, 101, 10)]
        if reps_known is None:
            reps_known = [i for i in range(1, 101)]

        self.fname = fname
        self.parent_dir = parent_dir
        self.path = os.path.join(self.parent_dir, self.fname)
        self.specs = np.load(os.path.join(self.path, "specs.npz"))
        self.params = np.load(os.path.join(self.path, "params.npz"))

        self.t_vec = self.params['t_vec']
        self.w_grain = self.params['w_grain']
        self.wmax = self.params['wmax']
        self.mc = self.params['truncate']
        self.gamma = self.params['gamma']
        self.gamma12 = self.params['gamma_12']
        self.t_b = self.params['t_b']
        self.a_m = self.params['a_m']
        self.delta = self.params['delta']
        self.c_times = self.params['c_times']
        self.n_shots = self.params['n_shots']
        self.M = self.params['M']
        self.a_sp = self.params['a_sp']
        self.c = self.params['c']
        self.Tqns = self.params['T']
        self.T = self.Tqns
        self.tau = self.T / tau_divisor
        self.Tg = Tg
        self.reps_known = reps_known
        self.reps_opt = reps_opt
        self.max_pulses = max_pulses

        self.p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        self.p2q = jnp.array([jnp.kron(self.p1q[i], self.p1q[j]) for i in range(4) for j in range(4)])
        self.w = jnp.linspace(0.00001, 2 * jnp.pi * self.mc / self.Tqns, 10000)
        self.w_ideal = jnp.linspace(0.00001, 2 * jnp.pi * 2 * self.mc / self.Tqns, 20000)
        self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])
        self.wkqns_ideal = jnp.array([2 * jnp.pi * n / self.Tqns for n in range(2 * self.mc + 1)])


########################################################################################################################
########################################################################################################################
#
# Utility Functions
#
########################################################################################################################
########################################################################################################################


@jax.jit
def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O) @ z2q[a] @ z2q[b] @ O @ z2q[a] @ z2q[b]) / 4


@jax.jit
def ff(tk, w_arg):
    return jnp.sum(
        jnp.array([1j * ((-1) ** k) * (jnp.exp(-1j * w_arg * tk[k + 1]) - jnp.exp(-1j * w_arg * tk[k])) / w_arg for k in
                   range(jnp.size(tk) - 1)]), axis=0)


def Gp_re(vti, vtj, w_arg, M_arg):
    return jnp.real(
        ff(vti, w_arg) * ff(vtj, -w_arg) * jnp.sin(w_arg * M_arg * vti[-1] * 0.5) ** 2 / jnp.sin(w_arg * vti[-1] * 0.5) ** 2)


def Gp_im(vti, vtj, w_arg, M_arg):
    return jnp.imag(
        ff(vti, w_arg) * ff(vtj, -w_arg) * jnp.sin(w_arg * M_arg * vti[-1] * 0.5) ** 2 / jnp.sin(w_arg * vti[-1] * 0.5) ** 2)


@jax.jit
def y_t(t, tk):
    return jnp.sum(
        jnp.array([((-1) ** i) * jnp.heaviside(t - tk[i], 1) * jnp.heaviside(tk[i + 1] - t, 1) for i in
                   range(jnp.size(tk) - 1)]), axis=0)


def make_tk12(tk1, tk2):
    x = tk1[~jnp.isin(tk1, tk2)]
    y = tk2[~jnp.isin(tk2, tk1)]
    z = jnp.zeros(x.size + y.size + 2)
    z = z.at[0].set(0.)
    z = z.at[-1].set(tk1[-1])
    z = z.at[1:z.size - 1].set(jnp.sort(jnp.concatenate((x, y))))
    return z


@jax.jit
def CO_sum_els(O, SMat_arg, Gp, i, j, w_arg):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5 * (sgn(O, i + 1, j + 1) + 1) * z2q[i + 1] @ z2q[j + 1] * jax.scipy.integrate.trapezoid(
        SMat_arg[i, j] * (sgn(O, i + 1, 0) - 1) * Gp[i, j], w_arg) / jnp.pi


@jax.jit
def CO_sum_els_wk(O, SMat_k_arg, Gp, i, j, M_arg, T_arg):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return -0.5 * (sgn(O, i + 1, j + 1) + 1) * z2q[i + 1] @ z2q[j + 1] * jnp.sum(
        SMat_k_arg[i, j] * (sgn(O, i + 1, 0) - 1) * Gp[i, j], axis=0) * M_arg / T_arg


@jax.jit
def Lambda_diags(SMat_arg, Gp, w_arg):
    inds = jnp.array([0, 1, 2])
    p1q_local_lambda = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q_local_lambda = jnp.array(
        [jnp.kron(p1q_local_lambda[i], p1q_local_lambda[j]) for i in range(4) for j in range(4)])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els,
                                            in_axes=(None, None, None, None, 0, None)),
                                   in_axes=(None, None, None, 0, None, None)),
                          in_axes=(0, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q_local_lambda, SMat_arg, Gp, inds, inds, w_arg), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(jax.scipy.linalg.expm(-CO[i])) * 0.25 for i in range(CO.shape[0])]))


@jax.jit
def Lambda_diags_wk(SMat_k_arg, Gp, M_arg, T_arg):
    p1q_local_lambda = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q_local_lambda = jnp.array(
        [jnp.kron(p1q_local_lambda[i], p1q_local_lambda[j]) for i in range(4) for j in range(4)])
    inds = jnp.array([0, 1, 2])
    CO_sum_map = jax.vmap(jax.vmap(jax.vmap(CO_sum_els_wk,
                                            in_axes=(None, None, None, None, 0, None, None)),
                                   in_axes=(None, None, None, 0, None, None, None)),
                          in_axes=(0, None, None, None, None, None, None))
    CO = jnp.sum(CO_sum_map(p2q_local_lambda, SMat_k_arg, Gp, inds, inds, M_arg, T_arg), axis=(1, 2))
    return jnp.real(jnp.array([jnp.trace(jax.scipy.linalg.expm(-CO[i])) * 0.25 for i in range(CO.shape[0])]))


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
            Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], w_arg, M_arg) + 1j * Gp_im_map(vt[m], vt[n], w_arg, M_arg))
    L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
    dt = tau_arg
    fid = jnp.sum(L_diag, axis=0) / 16.
    # clustering=jnp.sum(jnp.array([((vt[i][j+1]-vt[i][j])-dt)**2 for i in range(2) for j in range(vt[i].shape[0]-1)]), axis=0)
    clustering = -jnp.sum(jnp.array(
        [((vt[k][l + 1] - vt[k][l]) - dt)**2 / vt[k].shape[0]**2 for k in range(2) for l in
         range(vt[k].shape[0] - 1)]), axis=0)
    return -fid + clustering


def inf_ID_wk(params_arg, ind, SMat_k_arg, M_arg, T_arg, wk_arg, tau_arg):
    vt1 = jnp.sort(jnp.concatenate((jnp.array([0.]), params_arg[:ind], jnp.array([T_arg]))))
    vt2 = jnp.sort(jnp.concatenate((jnp.array([0.]), params_arg[ind:], jnp.array([T_arg]))))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk_arg, 1) + 1j * Gp_im_map(vt[i], vt[j], wk_arg, 1))
    L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, T_arg)
    dt = tau_arg
    fid = jnp.sum(L_diag, axis=0) / 16.
    clustering = -jnp.sum(jnp.array(
        [((vt[k][l + 1] - vt[k][l]) - dt)**2 / vt[k].shape[0]**2 for k in range(2) for l in
         range(vt[k].shape[0] - 1)]), axis=0)
    #-jnp.sum(jnp.array(
    #     [jnp.exp(-(9 / (2 * dt ** 2)) * (vt[k][l + 1] - vt[k][l]) ** 2) / vt[k].shape[0] for k in range(2) for l in
    #      range(vt[k].shape[0] - 1)]), axis=0)
    return -fid + clustering


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
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, M_arg) + 1j * Gp_im_map(vt[i], vt[j], w_arg, M_arg))
    L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
    return 1. - jnp.sum(L_diag, axis=0) / 16


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
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], wk_arg, 1) + 1j * Gp_im_map(vt[i], vt[j], wk_arg, 1))
    L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, vt1[-1])
    return 1. - jnp.sum(L_diag, axis=0) / 16


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
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, 1) + 1j * Gp_im_map(vt[i], vt[j], w_arg, 1))
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
            vt = jnp.array(np.random.rand(i + j) * T_arg)
            init_params.append(vt)
            lower_bnd = jnp.zeros_like(vt)
            upper_bnd = jnp.ones_like(vt) * T_arg
            bnds = (lower_bnd, upper_bnd)
            opt = optimizer.run(vt, bnds, i, SMat_arg, M_arg, T_arg, w_arg, tau_arg)
            opt_out_temp.append(opt)
            print(f"    - Optimized with ({i}, {j}) pulses. Cost: {opt.state[0]:.4e}")
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out),
                                 inf_ID_out.shape)  # jnp.where(inf_ID_out == jnp.min(inf_ID_out))
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt_0 = params_to_tk(vt_min_arr, T_arg, jnp.array([0, nPs_arg[0][inds_min[0]]]))
    vt_opt_1 = params_to_tk(vt_min_arr, T_arg, jnp.array(
        [nPs_arg[0][inds_min[0]], nPs_arg[0][inds_min[0]] + nPs_arg[1][inds_min[1]]]))
    vt_opt_local = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
    # vt_init = params_to_tk(init_params[jnp.argmin(inf_ID_out)], T)
    return vt_opt_local, infidelity(vt_opt_local, SMat_arg, M_arg, w_arg)  # inf_ID_out[inds_min]


def hyperOpt_k(SMat_k_arg, nPs_arg, M_arg, T_arg, wk_arg, tau_arg):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_ID_wk, maxiter=400, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-7, 'ftol': 1e-7,
                                                     'maxfun': 1000, 'maxls': 40})
    opt_out = []
    init_params = []
    for i in nPs_arg[0]:
        opt_out_temp = []
        for j in nPs_arg[1]:
            vt = jnp.concatenate((jnp.linspace(0, T_arg, i + 2)[1:-1], jnp.linspace(0, T_arg, j + 2)[1:-1]))
            # jnp.array(np.random.rand(i + j) * T_arg)
            init_params.append(vt)
            lower_bnd = jnp.zeros_like(vt)
            upper_bnd = jnp.ones_like(vt) * T_arg
            bnds = (lower_bnd, upper_bnd)
            opt = optimizer.run(vt, bnds, i, SMat_k_arg, M_arg, T_arg, wk_arg, tau_arg)
            opt_out_temp.append(opt)
            print(f"    - Optimized with ({i}, {j}) pulses. Cost: {opt.state[0]:.4e}")
        opt_out.append(opt_out_temp)
    inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
    inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out),
                                 inf_ID_out.shape)
    vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
    vt_opt_0 = params_to_tk(vt_min_arr, T_arg, jnp.array([0, nPs_arg[0][inds_min[0]]]))
    vt_opt_1 = params_to_tk(vt_min_arr, T_arg, jnp.array(
        [nPs_arg[0][inds_min[0]], nPs_arg[0][inds_min[0]] + nPs_arg[1][inds_min[1]]]))
    vt_opt_local = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
    return vt_opt_local, infidelity_k(vt_opt_local, SMat_k_arg, M_arg, wk_arg)


def Lambda(Oi, Oj, vt, SMat_arg, M_arg, w_arg):
    vt12 = make_tk12(vt[0], vt[1])
    vt.append(vt12)
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w_arg, M_arg) + 1j * Gp_im_map(vt[i], vt[j], w_arg, M_arg))
    CO = 0
    for i in range(3):
        for j in range(3):
            CO += -0.5 * (sgn(Oi, i + 1, j + 1) + 1) * z2q[i + 1] @ z2q[j + 1] * jax.scipy.integrate.trapezoid(
                SMat_arg[i, j] * (sgn(Oi, i + 1, 0) - 1) * Gp[i, j], w_arg) / jnp.pi
    return jnp.real(jnp.trace(Oi @ jax.scipy.linalg.expm(-CO) @ Oj) * 0.25)


def params_to_tk(params_arg, T_arg, shape: jnp.ndarray):
    vt = jnp.zeros(shape[1] - shape[0] + 2)
    vt = vt.at[0].set(0.)
    vt = vt.at[-1].set(T_arg)
    vt = vt.at[1:shape[1] - shape[0] + 1].set(jnp.sort(params_arg[shape[0]:shape[1]]))
    return vt


def cpmg_vt(T_arg, n):
    tk = [(k + 0.50) * T_arg / (2 * n) for k in range(int(2 * n))]
    tk.append(T_arg)
    tk.insert(0, 0.)
    return jnp.array(tk)


def cdd1_vt(T_arg, n):
    tk = [(k + 1) * T_arg / (2 * n) for k in range(int(2 * n - 1))]
    tk.append(T_arg)
    tk.insert(0, 0.)
    return jnp.array(tk)


def uddn(T_arg, n):
    tk = [T_arg * jnp.sin(jnp.pi * k / (2 * n + 2)) ** 2 for k in range(n)]
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

def remove_consecutive_duplicates(input_list):
    output_list = []
    i = 0
    while i < len(input_list):
        if i + 1 < len(input_list) and input_list[i] == input_list[i+1]:
            i += 2 # Skip both duplicates
        else:
            output_list.append(input_list[i])
            i += 1
    return output_list
def cdd(t0, T, n):
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)
def cddn(t0, T, n):
    out = remove_consecutive_duplicates(cdd(t0, T, n))
    if out[0] == 0.:
        return out + [T]
    else:
        return [0.] + out + [T]
def cddn_util(t0, T, n):
    return remove_consecutive_duplicates(cdd(t0, T, n))
def mqCDD(T, n, m):
    tk1 = cddn_util(0., T, n)
    tk2 = []
    for i in range(len(tk1)-1):
        tk2 += cddn_util(tk1[i], tk1[i+1]-tk1[i], m)
    tk2 += cddn_util(tk1[-1], T-tk1[-1], m)
    if tk1[0] != 0.:
        tk1 = [0.] + tk1
    if tk2[0] != 0.:
        tk2 = [0.] + tk2
    return [tk1 + [T], tk2 + [T]]


def pddn(T_arg, n, M_arg):
    out = cddn(0., T_arg, n)
    if M_arg == 1:
        return out
    for i in range(M_arg):
        out = np.concatenate((out, cddn(0., T_arg, n) + (i + 1) * T_arg))
    return out


def opt_known_pulses(pLib_arg, SMat_arg, M_arg, w_arg):
    infmin = jnp.inf
    inds_min = 0
    for i in range(len(pLib_arg)):
        tk1 = pLib_arg[i][0]
        tk2 = pLib_arg[i][1]
        tk12 = make_tk12(tk1, tk2)
        vt = [tk1, tk2, tk12]
        Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
        Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
        Gp = jnp.zeros((3, 3, w_arg.size), dtype=jnp.complex64)
        for m in range(3):
            for n in range(3):
                Gp = Gp.at[m, n].set(
                    Gp_re_map(vt[m], vt[n], w_arg, M_arg) + 1j * Gp_im_map(vt[m], vt[n], w_arg, M_arg))
        L_diag = Lambda_diags(SMat_arg, Gp, w_arg)
        # infidelity_arr = infidelity_arr.at[i, j, k, l].set(1-jnp.sum(L_diag, axis=0)/16)
        inf_ijkl = 1. - jnp.sum(L_diag, axis=0) / 16
        if inf_ijkl < infmin:
            infmin = inf_ijkl
            inds_min = i
    tk1 = pLib_arg[inds_min][0]
    tk2 = pLib_arg[inds_min][1]
    tk12 = make_tk12(tk1, tk2)
    vt_opt_local = [tk1, tk2, tk12]
    return vt_opt_local, infmin, inds_min

def opt_known_pulses_k(pLib_arg, SMat_k_arg, M_arg, wk_arg):
    infmin = jnp.inf
    inds_min = 0
    for i in range(len(pLib_arg)):
        vt = [pLib_arg[i][0], pLib_arg[i][1], make_tk12(pLib_arg[i][0], pLib_arg[i][1])]
        Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
        Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
        Gp = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
        for m in range(3):
            for n in range(3):
                Gp = Gp.at[m, n].set(Gp_re_map(vt[m], vt[n], wk_arg, 1) + 1j * Gp_im_map(vt[m], vt[n], wk_arg, 1))
        L_diag = Lambda_diags_wk(SMat_k_arg, Gp, M_arg, vt[0][-1])
        inf_i = 1. - jnp.sum(L_diag, axis=0) / 16.
        if inf_i < infmin:
            infmin = inf_i
            inds_min = i
        tk1 = pLib_arg[inds_min][0]
        tk2 = pLib_arg[inds_min][1]
        tk12 = make_tk12(tk1, tk2)
        vt_opt_local = [tk1, tk2, tk12]
    return vt_opt_local, infmin, inds_min


def get_cdd_orders_from_index(index, cdd_lib_len, mq_cdd_orders):
    """
    Infers the order of the CDD sequence from its index in pLib.

    Args:
        index (int): The index of the sequence in the pLib list.
        cdd_lib_len (int): The length of the cddLib list.
        mq_cdd_orders (list): A list of (n, m) tuples for mqCDD orders.

    Returns:
        str: A string describing the sequence type and orders.
    """
    num_cdd_permutations = cdd_lib_len * (cdd_lib_len - 1)

    if index < num_cdd_permutations:
        # It's a permutation of two cddn sequences
        # The order of cddn(..., k) is k. cddLib is 0-indexed, so order is index + 1.
        # itertools.permutations('ABC', 2) -> (A,B), (A,C), (B,A), (B,C), (C,A), (C,B)
        # index = i * (cdd_lib_len - 1) + j_prime
        i = index // (cdd_lib_len - 1)
        j_prime = index % (cdd_lib_len - 1)
        j = j_prime if j_prime < i else j_prime + 1
        order1 = i + 1
        order2 = j + 1
        return f"cddn permutation with orders ({order1}, {order2})"
    else:
        # It's an mqCDD sequence
        mq_index = index - num_cdd_permutations
        if mq_index < len(mq_cdd_orders):
            order_n, order_m = mq_cdd_orders[mq_index]
            return f"mqCDD sequence with orders (n={order_n}, m={order_m})"
        else:
            return "Index out of bounds for known sequences."

def makeSMat_k(specs_arg, wk_arg, wkqns_arg, gamma_arg, gamma12_arg):
    SMat_local = jnp.zeros((3, 3, wk_arg.size), dtype=jnp.complex64)
    SMat_local = SMat_local.at[0, 0].set(
        jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_11(wk_arg[0])]), specs_arg["S11"]))))
    SMat_local = SMat_local.at[1, 1].set(
        jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_22(wk_arg[0])]), specs_arg["S22"]))))
    SMat_local = SMat_local.at[2, 2].set(
        jnp.interp(wk_arg, wkqns_arg, jnp.concatenate((jnp.array([S_1212(wk_arg[0])]), specs_arg["S1212"]))))
    SMat_local = SMat_local.at[0, 1].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate(
        (jnp.array([S_1_2(wk_arg[0], gamma_arg)]), specs_arg["S12"]))))
    SMat_local = SMat_local.at[1, 0,].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(
        jnp.concatenate((jnp.array([S_1_2(wk_arg[0], gamma_arg)]), specs_arg["S12"])))))
    SMat_local = SMat_local.at[0, 2].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate(
        (jnp.array([S_1_12(wk_arg[0], gamma12_arg)]), specs_arg["S112"]))))
    SMat_local = SMat_local.at[2, 0].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(
        jnp.concatenate((jnp.array([S_1_12(wk_arg[0], gamma12_arg)]), specs_arg["S112"])))))
    SMat_local = SMat_local.at[1, 2].set(jnp.interp(wk_arg, wkqns_arg, jnp.concatenate(
        (jnp.array([S_2_12(wk_arg[0], gamma12_arg - gamma_arg)]), specs_arg["S212"]))))
    SMat_local = SMat_local.at[2, 1].set(jnp.interp(wk_arg, wkqns_arg, jnp.conj(
        jnp.concatenate((jnp.array([S_2_12(wk_arg[0], gamma12_arg - gamma_arg)]), specs_arg["S212"])))))
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
    SMat_ideal_local = SMat_ideal_local.at[1, 2].set(S_2_12(wk_arg, gamma12_arg - gamma_arg))
    SMat_ideal_local = SMat_ideal_local.at[2, 1].set(jnp.conj(S_2_12(wk_arg, gamma12_arg - gamma_arg)))
    return SMat_ideal_local


def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """
    Constructs a library of known pulse sequences (cddn permutations and mqCDD).

    Args:
        T_seq (float): The total time for a single sequence repetition.
        tau_min (float): The minimum allowed time between pulses.
        max_pulses (int): The maximum number of pulses allowed in a single sequence.

    Returns:
        tuple: A tuple containing:
            - pLib (list): The library of pulse sequences. Each element is a tuple of two pulse time arrays.
            - cddLib (list): The library of single-qubit cddn sequences used.
            - mq_cdd_orders_log (list): A list of (n, m) orders for the mqCDD sequences.
    """
    # 1. Generate single-qubit cddn sequences until pulse separation is too small
    cddLib = []
    cddOrd = 1
    while True:
        pul = jnp.array(cddn(0., T_seq, cddOrd))
        if any(pul[j + 1] - pul[j] < tau_min for j in range(1, len(pul) - 2)):
            break
        cddLib.append(pul)
        cddOrd += 1

    # 2. Create permutations of the single-qubit sequences
    pLib = list(itertools.permutations(cddLib, 2))

    # 3. Generate and add multi-qubit (mqCDD) sequences
    mq_cdd_orders_log = []
    ncddOrd1 = 1
    while True:
        ncddOrd2 = 1
        pul_n = mqCDD(T_seq, ncddOrd1, ncddOrd2)[0]
        if any(pul_n[j + 1] - pul_n[j] < tau_min for j in range(1, len(pul_n) - 2)):
            break  # Stop if the outer sequence pulses are too close

        while True:
            pul = mqCDD(T_seq, ncddOrd1, ncddOrd2)
            if any(pul[1][j + 1] - pul[1][j] < tau_min for j in range(1, len(pul[1]) - 2)):
                break  # Stop if the inner sequence pulses are too close
            pLib.append((jnp.array(pul[0]), jnp.array(pul[1])))
            mq_cdd_orders_log.append((ncddOrd1, ncddOrd2))
            ncddOrd2 += 1
        ncddOrd1 += 1

    # 4. Prune the library based on the maximum number of pulses
    # Note: This is inefficient as it rebuilds lists. For very large libraries, a different approach may be needed.
    pruned_pLib = [p for p in pLib if (len(p[0]) - 2) <= max_pulses and (len(p[1]) - 2) <= max_pulses]
    
    # We need to find which mq_cdd_orders correspond to the pruned pLib
    num_cdd_perms = len(list(itertools.permutations(cddLib, 2)))
    pruned_mq_cdd_orders_log = [mq_cdd_orders_log[i] for i, p in enumerate(pLib[num_cdd_perms:]) if (len(p[0]) - 2) <= max_pulses and (len(p[1]) - 2) <= max_pulses]

    pLib = pruned_pLib
    mq_cdd_orders_log = pruned_mq_cdd_orders_log
    return pLib, cddLib, mq_cdd_orders_log


########################################################################################################################
########################################################################################################################
#
# Main Execution Block
#
########################################################################################################################
########################################################################################################################


def main():
    """Main execution block for the pulse optimization."""

    # Load the configuration
    config = PulseOptimizerConfig()

    # Create the spectral matrices
    SMat = jnp.zeros((3, 3, config.w.size), dtype=jnp.complex64)
    SMat = SMat.at[0, 0].set(jnp.interp(config.w, config.wkqns, config.specs["S11"]))
    SMat = SMat.at[1, 1].set(jnp.interp(config.w, config.wkqns, config.specs["S22"]))
    SMat = SMat.at[2, 2].set(jnp.interp(config.w, config.wkqns, config.specs["S1212"]))
    SMat = SMat.at[0, 1].set(jnp.interp(config.w, config.wkqns, config.specs["S12"]))
    SMat = SMat.at[1, 0].set(jnp.interp(config.w, config.wkqns, np.conj(config.specs["S12"])))
    SMat = SMat.at[0, 2].set(jnp.interp(config.w, config.wkqns, config.specs["S112"]))
    SMat = SMat.at[2, 0].set(jnp.interp(config.w, config.wkqns, np.conj(config.specs["S11"])))
    SMat = SMat.at[1, 2].set(jnp.interp(config.w, config.wkqns, config.specs["S212"]))
    SMat = SMat.at[2, 1].set(jnp.interp(config.w, config.wkqns, np.conj(config.specs["S212"])))

    SMat_ideal = jnp.zeros((3, 3, config.w_ideal.size), dtype=jnp.complex64)
    SMat_ideal = SMat_ideal.at[0, 0].set(S_11(config.w_ideal))
    SMat_ideal = SMat_ideal.at[1, 1].set(S_22(config.w_ideal))
    SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(config.w_ideal))
    SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(config.w_ideal, config.gamma))
    SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(config.w_ideal, config.gamma)))
    SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(config.w_ideal, config.gamma12))
    SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(config.w_ideal, config.gamma12)))
    SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(config.w_ideal, config.gamma12 - config.gamma))
    SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(config.w_ideal, config.gamma12 - config.gamma)))

    # Calculate T2 times
    taxis = jnp.linspace(1 * config.T, 20 * config.T, 20)
    vt_T2 = jnp.array([jnp.array([[0, taxis[i]], [0, taxis[i]]]) for i in range(taxis.shape[0])])
    q1T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, config.w_ideal, 1) for i in range(vt_T2.shape[0])])
    q2T2 = jnp.array([T2(vt_T2[i], SMat_ideal, 1, config.w_ideal, 2) for i in range(vt_T2.shape[0])])
    T2q1 = jnp.inf
    T2q2 = jnp.inf
    for i in range(q1T2.size):
        if q1T2[i] < 0.5:
            T2q1 = (taxis[i] + taxis[i - 1]) * 0.5
            break
    for i in range(q2T2.size):
        if q2T2[i] < 0.5:
            T2q2 = (taxis[i] + taxis[i - 1]) * 0.5
            break

    header_width = 80
    print("\n" + "=" * header_width)
    print(" " * 25 + "Pulse Sequence Optimization")
    print("=" * header_width)
    print(f" Gate Time (Tg): {config.Tg / 1e-6:<.2f} us")
    print(f" Base Sequence Time (T_QNS): {config.T / 1e-6:<.2f} us")
    print("-" * header_width)
    print(" System Coherence:")
    print(f"  - Qubit 1 T2: {T2q1 / 1e-6:<.2f} us  ({T2q1 / config.T:<.2f} T_QNS)")
    print(f"  - Qubit 2 T2: {T2q2 / 1e-6:<.2f} us  ({T2q2 / config.T:<.2f} T_QNS)")
    print(f" Target gate time is {config.Tg / T2q1:.2f}x T2_Q1 and {config.Tg / T2q2:.2f}x T2_Q2.")
    print("=" * header_width + "\n")

    # Optimization over known pulse sequences
    best_seq = [jnp.array([]), jnp.array([])]
    best_inf = np.inf
    best_M = 0
    inf_vs_M_known = []
    for i in config.reps_known:
        Tknown = config.Tg / i
        Mknown = i
        print("-" * header_width)
        print(f"Optimizing Known Sequences for M = {Mknown} (T_seq = {Tknown/1e-6:.4f} us)")
        print("-" * header_width)

        pLib, cddLib, mq_cdd_orders_log = construct_pulse_library(Tknown, config.tau,
                                                                  max_pulses=int(config.max_pulses / Mknown))

        # Check if the library is empty and terminate if so
        if not pLib:
            print("\n" + "!"*header_width)
            print(f"Terminating known sequence optimization at M={Mknown}, T={Tknown} because no valid sequences could be generated.")
            print("The sequence time is likely too short for the given pulse separation and max pulse constraints.")
            break

        # Generate an Idling gate that is optimized using known sequences
        wk_local = jnp.array(
            [0.00001] + [2 * jnp.pi * (n + 1) / Tknown for n in range(int(jnp.floor(config.Tg * config.mc / (i * config.Tqns))))])
        wk_ideal_local = jnp.array(
            [0.00001] + [2 * jnp.pi * (n + 1) / Tknown for n in range(int(jnp.floor(4 * config.Tg * config.mc / (i * config.Tqns))))])
        SMat_k_local = makeSMat_k(config.specs, wk_local, jnp.concatenate((jnp.array([wk_local[0]]), config.wkqns)),
                                  config.gamma, config.gamma12)
        SMat_k_ideal = makeSMat_k_ideal(wk_ideal_local, config.gamma, config.gamma12)
        if i < 10:
            known_opt, known_inf, known_ind = opt_known_pulses(pLib, SMat, Mknown, config.w)
            inf_known = infidelity(known_opt, SMat_ideal, Mknown, config.w_ideal)
        else:
            known_opt, known_inf, known_ind = opt_known_pulses_k(pLib, SMat_k_local, Mknown, wk_local)
            inf_known = infidelity_k(known_opt, SMat_k_ideal, Mknown, wk_ideal_local)
        inf_vs_M_known.append((Mknown, inf_known))
        if known_inf <= best_inf:
            best_seq = known_opt
            best_inf = inf_known
            best_M = Mknown
            best_orders_str = get_cdd_orders_from_index(known_ind, len(cddLib), mq_cdd_orders_log)
            print("\n>>> New Best Known Sequence Found!")
            print(f"    Infidelity: {best_inf:.4e}")
            print(f"    Pulses: [{best_seq[0].shape[0] - 2}, {best_seq[1].shape[0] - 2}]")
            print(f"    Sequence Type: {best_orders_str}\n")

    print("\n" + "=" * header_width)
    print("Summary for Known Sequence Optimization")
    print("-" * header_width)
    print(f"Best Infidelity: {best_inf:.4e}")
    if best_M > 0:
        pulses_q1_per_rep = best_seq[0].shape[0] - 2
        pulses_q2_per_rep = best_seq[1].shape[0] - 2
        total_pulses_known = (pulses_q1_per_rep + pulses_q2_per_rep) * best_M
        print(f"Optimal Repetitions (M): {best_M}")
        print(f"Pulses per Repetition (Q1, Q2): [{pulses_q1_per_rep}, {pulses_q2_per_rep}]")
        print(f"Total Pulses (Q1, Q2): [{pulses_q1_per_rep * best_M}, {pulses_q2_per_rep * best_M}] (Total: {total_pulses_known})")
    else:
        print("No optimal known sequence was found.")
    print("=" * header_width + "\n")

    # Optimization over new pulse sequences
    opt_seq = best_seq  # Initialize with the best known sequence
    opt_inf = np.inf
    opt_M = 0
    inf_vs_M_opt = []
    for i in config.reps_opt:
        Topt = config.Tg / i
        Mopt = i

        max_pulses_per_rep = int(config.max_pulses / Mopt)
        if max_pulses_per_rep < 1:
            print("\n" + "!" * header_width)
            print(f"Terminating new sequence optimization at M={Mopt} because max pulses per repetition is less than 1.")
            break

        # The upper bound for random pulse number selection is the minimum of what's physically possible
        # with tau and what's allowed by the max_pulses_per_rep constraint.
        upper_bound_pulses = min(max(2, int(Topt / config.tau)), max_pulses_per_rep + 1)

        if upper_bound_pulses <= 1:
            print(f"\nSkipping new sequence optimization for M={Mopt} because no valid number of pulses can be generated.")
            print(f"The sequence time (T_seq={Topt / 1e-6:.4f} us) is likely too short for the given constraints.")
            continue

        nPs = np.random.randint(1, upper_bound_pulses, (2, 4))
        print("-" * header_width)
        print(f"Optimizing New Sequences for M = {Mopt} (T_seq = {Topt/1e-6:.4f} us)")
        print(f"Pulse numbers to try: {nPs[0]} for Q1, {nPs[1]} for Q2")
        print("-" * header_width)
        if Mopt >= 10:
            wk_local = jnp.array(
                [0.00001] + [2 * jnp.pi * (n + 1) / Topt for n in range(int(jnp.floor(config.Tg * config.mc / (i * config.Tqns))))])
            wk_ideal_local = jnp.array(
                [0.00001] + [2 * jnp.pi * (n + 1) / Topt for n in range(int(jnp.floor(4 * config.Tg * config.mc / (i * config.Tqns))))])
            SMat_k_local = makeSMat_k(config.specs, wk_local, jnp.concatenate((jnp.array([wk_local[0]]), config.wkqns)),
                                      config.gamma, config.gamma12)
            SMat_k_ideal = makeSMat_k_ideal(wk_ideal_local, config.gamma, config.gamma12)
            # Generate an Idling gate that is optimized over a given number of pulses on each qubit
            vt_opt, inf_min = hyperOpt_k(SMat_k_local, nPs, Mopt, Topt, wk_local, config.tau)
            inf_opt = infidelity_k(vt_opt, SMat_k_ideal, Mopt, wk_ideal_local)
        else:
            vt_opt, inf_min = hyperOpt(SMat, nPs, Mopt, Topt, config.w, config.tau)
            inf_opt = infidelity(vt_opt, SMat_ideal, Mopt, config.w_ideal)
        inf_vs_M_opt.append((Mopt, inf_opt))
        if inf_opt <= opt_inf:
            opt_seq = vt_opt
            opt_inf = inf_opt
            opt_M = Mopt
            print("\n>>> New Best Optimized Sequence Found!")
            print(f"    Infidelity: {opt_inf:.4e}")
            print(f"    Pulses: [{opt_seq[0].shape[0] - 2}, {opt_seq[1].shape[0] - 2}]\n")

    print("\n" + "=" * header_width)
    print("Summary for New Sequence Optimization")
    print("-" * header_width)
    print(f"Best Infidelity: {opt_inf:.4e}")
    if opt_M > 0:
        pulses_q1_per_rep_opt = opt_seq[0].shape[0] - 2
        pulses_q2_per_rep_opt = opt_seq[1].shape[0] - 2
        total_pulses_opt = (pulses_q1_per_rep_opt + pulses_q2_per_rep_opt) * opt_M
        print(f"Optimal Repetitions (M): {opt_M}")
        print(f"Pulses per Repetition (Q1, Q2): [{pulses_q1_per_rep_opt}, {pulses_q2_per_rep_opt}]")
        print(f"Total Pulses (Q1, Q2): [{pulses_q1_per_rep_opt * opt_M}, {pulses_q2_per_rep_opt * opt_M}] (Total: {total_pulses_opt})")
    else:
        print("No optimal new sequence was found.")
    print("=" * header_width + "\n")

    np.savez(os.path.join(config.path, 'optimizeLog.npz'), gtime=config.Tg, best_inf=best_inf, best_seq_1=best_seq[0],
             best_seq_2=best_seq[1], best_seq_12=best_seq[2], best_M=best_M, opt_inf=opt_inf, opt_seq_1=opt_seq[0],
             opt_seq_2=opt_seq[1], opt_seq_12=opt_seq[2], opt_M=opt_M)

    # Plotting results
    plot_results(config, best_seq, best_M, opt_seq, opt_M)
    plot_inf_vs_m(config, inf_vs_M_known, inf_vs_M_opt)


def plot_inf_vs_m(config, inf_vs_m_known, inf_vs_m_opt):
    """
    Plots the best infidelity found for each M against M for known and optimized sequences.

    Args:
        config (PulseOptimizerConfig): The configuration object.
        inf_vs_m_known (list): A list of (M, infidelity) tuples for known sequences.
        inf_vs_m_opt (list): A list of (M, infidelity) tuples for optimized sequences.
    """
    plt.figure(figsize=(10, 6))

    if inf_vs_m_known:
        m_values_known, infidelities_known = zip(*sorted(inf_vs_m_known))
        plt.plot(m_values_known, infidelities_known, 'o-', label='Best Infidelity (Known Seqs)')
    else:
        print("No data to plot for known sequences infidelity vs. M.")

    if inf_vs_m_opt:
        m_values_opt, infidelities_opt = zip(*sorted(inf_vs_m_opt))
        plt.plot(m_values_opt, infidelities_opt, 's--', label='Best Infidelity (Optimized Seqs)')
    else:
        print("No data to plot for optimized sequences infidelity vs. M.")

    if not inf_vs_m_known and not inf_vs_m_opt:
        return

    plt.xlabel('Number of Repetitions (M)')
    plt.ylabel('Best Infidelity')
    plt.title('Infidelity vs. Number of Repetitions (M)')
    plt.grid(True, which='both', linestyle='--')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(config.path, 'infidelity_vs_M.pdf')
    plt.savefig(save_path)
    print(f"Saved infidelity vs. M plot to {save_path}")

def plot_results(config, known_opt, best_M, vt_opt, opt_M):
    """Plots the results of the optimization."""

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    alp = 0.4
    lw = 0.5
    Mplot_opt = opt_M
    Mplot_known = best_M
    legendfont = 10
    labelfont = 16
    xunits = 1e6
    yunits = 1e3
    wk_local_plot = config.wkqns

    max_lim_row1_left = np.max(np.array([np.abs(np.real(S_11(config.w)) / yunits).max(),
                                         np.abs(np.imag(S_22(config.w)) / yunits).max(),
                                         np.abs(np.imag(S_1212(config.w)) / yunits).max()]))
    max_lim_row1_right = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[0], config.w, Mplot_opt)).max(),
                                          np.abs(Gp_re(vt_opt[1], vt_opt[1], config.w, Mplot_opt)).max(),
                                          np.abs(Gp_re(vt_opt[2], vt_opt[2], config.w, Mplot_opt)).max()]))

    max_lim_row2_left = np.max(
        np.array([np.abs(np.real(S_1_2(config.w[int(config.w.size / config.mc):], config.gamma)) / yunits).max(),
                  np.abs(np.imag(S_1_12(config.w[int(config.w.size / config.mc):], config.gamma12)) / yunits).max(),
                  np.abs(np.imag(
                      S_2_12(config.w[int(config.w.size / config.mc):], config.gamma - config.gamma12)) / yunits).max()]))
    max_lim_row2_right = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[1], config.w, Mplot_opt)).max(),
                                          np.abs(Gp_re(vt_opt[0], vt_opt[2], config.w, Mplot_opt)).max(),
                                          np.abs(Gp_re(vt_opt[1], vt_opt[2], config.w, Mplot_opt)).max()]))

    axs[0, 0].plot(config.w / xunits, S_11(config.w) / yunits, 'r-', alpha=alp, lw=1.5 * lw)
    axs[0, 0].plot(wk_local_plot / xunits, config.specs["S11"] / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    axs[0, 0].legend([r'$S^+_{1,1}(\omega)$', r'$\hat{S}^+_{1,1}(\omega)$'], fontsize=legendfont, loc='upper left')
    axs[0, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=labelfont)
    axs[0, 0].set_ylim(0, max_lim_row1_left * 1.01)
    axs[0, 0].tick_params(direction='in')
    axs[0, 0].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    axs[0, 0].set_yscale('asinh')
    bx = axs[0, 0].twinx()
    bx.plot(config.w / xunits, Gp_re(vt_opt[0], vt_opt[0], config.w, Mplot_opt), 'g--', lw=lw)
    bx.plot(config.w / xunits, Gp_re(known_opt[0], known_opt[0], config.w, Mplot_known), 'm-', lw=lw)
    # bx.set_yticklabels([])
    # bx.set_yticks([])
    bx.set_ylim(0)
    bx.tick_params(direction='in')
    bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;1,1}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{1,1;1,1}(\omega, T)$]']
              , fontsize=legendfont, loc='upper right')
    axs[0, 1].plot(config.w / xunits, S_22(config.w) / yunits, 'r-', alpha=alp, lw=1.5 * lw)
    axs[0, 1].plot(wk_local_plot / xunits, config.specs["S22"] / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    # axs[0, 1].set_yticks([])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].legend([r'$S^+_{2,2}(\omega)$', r'$\hat{S}^+_{2,2}(\omega)$'], fontsize=legendfont, loc='upper left')
    axs[0, 1].set_ylim(0, max_lim_row1_left * 1.01)
    axs[0, 1].tick_params(direction='in')
    axs[0, 1].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    axs[0, 1].set_yscale('asinh')
    bx = axs[0, 1].twinx()
    bx.plot(config.w / xunits, Gp_re(vt_opt[1], vt_opt[1], config.w, Mplot_opt), 'g--', lw=lw)
    bx.plot(config.w / xunits, Gp_re(known_opt[1], known_opt[1], config.w, Mplot_known), 'm-', lw=lw)
    # bx.set_yticklabels([])
    # bx.set_yticks([])
    bx.set_ylim(0)
    bx.tick_params(direction='in')
    bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;2,2}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{2,2;2,2}(\omega, T)$]']
              , fontsize=legendfont, loc='upper right')
    axs[0, 2].plot(config.w / xunits, S_1212(config.w) / yunits, 'r-', alpha=alp, lw=1.5 * lw)
    axs[0, 2].plot(wk_local_plot / xunits, config.specs["S1212"] / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    axs[0, 2].legend([r'$S^+_{12,12}(\omega)$', r'$\hat{S}^+_{12,12}(\omega)$'], fontsize=legendfont,
                     loc='upper left')
    # axs[0, 2].set_yticks([])
    axs[0, 2].set_yticklabels([])
    axs[0, 2].set_ylim(0)
    axs[0, 2].tick_params(direction='in')
    axs[0, 2].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    axs[0, 2].set_yscale('asinh')
    bx = axs[0, 2].twinx()
    bx.plot(config.w / xunits, Gp_re(vt_opt[2], vt_opt[2], config.w, Mplot_opt), 'g--', lw=lw)
    bx.plot(config.w / xunits, Gp_re(known_opt[2], known_opt[2], config.w, Mplot_known), 'm-', lw=lw)
    bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=labelfont)
    bx.legend(
        [r'Re[$G^{+,\text{opt}}_{12,12;12,12}(\omega, T)$]', r'Re[$G^{+,\text{known}}_{12,12;12,12}(\omega, T)$]']
        , fontsize=legendfont, loc='upper right')
    bx.set_ylim(0)
    bx.tick_params(direction='in')

    ########################################################################################################################
    w_plot = config.w[300:]
    axs[1, 0].plot(w_plot / xunits, np.real(S_1_2(w_plot, config.gamma)) / yunits, 'r-', alpha=alp, lw=1.5 * lw)
    axs[1, 0].plot(wk_local_plot / xunits, np.real(config.specs["S12"]) / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    axs[1, 0].plot(w_plot / xunits, np.imag(S_1_2(w_plot, config.gamma)) / yunits, 'b-', alpha=alp, lw=1.5 * lw)
    axs[1, 0].plot(wk_local_plot / xunits, np.imag(config.specs["S12"]) / yunits, 'b--^', alpha=alp, lw=1.5 * lw)
    axs[1, 0].legend([r'Re[$S^+_{1,2}(\omega)$]', r'Re[$\hat{S}^+_{1,2}(\omega)$]', r'Im[$S^+_{1,2}(\omega)$]'
                         , r'Im[$\hat{S}^+_{1,2}(\omega)$]'], fontsize=legendfont, loc='upper left')
    axs[1, 0].set_ylabel(r'$S^+_{a,b}(\omega)$ (kHz)', fontsize=labelfont)
    # max_lim = np.maximum(np.abs(np.real(S_1_2(w, gamma))/1e3).max(), np.abs(np.imag(S_1_2(w, gamma))/1e3).max())
    axs[1, 0].set_ylim(-max_lim_row2_left * 1.01, max_lim_row2_left * 1.01)
    axs[1, 0].tick_params(direction='in')
    axs[1, 0].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    bx = axs[1, 0].twinx()
    bx.tick_params(direction='in')
    bx.plot(w_plot / xunits, Gp_re(vt_opt[0], vt_opt[1], w_plot, Mplot_opt), 'g--', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(vt_opt[0], vt_opt[1], w_plot, Mplot_opt), 'k--', lw=lw)
    bx.plot(w_plot / xunits, Gp_re(known_opt[0], known_opt[1], w_plot, Mplot_known), 'm-', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(known_opt[0], known_opt[1], w_plot, Mplot_known), 'c-', lw=lw)
    # bx.set_yticklabels([])
    # bx.set_yticks([])
    max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[1], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_im(vt_opt[0], vt_opt[1], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_re(known_opt[0], known_opt[1], w_plot, Mplot_known)).max(),
                               np.abs(Gp_im(known_opt[0], known_opt[1], w_plot, Mplot_known)).max()]))
    bx.set_ylim(-max_lim * 1.01, max_lim * 1.01)
    bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;2,2}(\omega, T)$]',
               r'Re[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]', r'Im[$G^{+,\text{known}}_{1,1;2,2}(\omega, T)$]']
              , fontsize=legendfont, loc='lower right')
    axs[1, 1].plot(w_plot / xunits, np.real(S_1_12(w_plot, config.gamma12)) / yunits, 'r-', alpha=alp, lw=1.5 * lw)
    axs[1, 1].plot(wk_local_plot / xunits, np.real(config.specs["S112"]) / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    axs[1, 1].plot(w_plot / xunits, np.imag(S_1_12(w_plot, config.gamma12)) / yunits, 'b-', alpha=alp, lw=1.5 * lw)
    axs[1, 1].plot(wk_local_plot / xunits, np.imag(config.specs["S112"]) / yunits, 'b--^', alpha=alp, lw=1.5 * lw)
    # axs[1, 1].set_yticks([])
    axs[1, 1].set_yticklabels([])
    # max_lim = np.maximum(np.abs(np.real(S_1_12(w, gamma12))/1e3).max(), np.abs(np.imag(S_1_12(w, gamma12))/1e3).max())
    axs[1, 1].set_ylim(-max_lim_row2_left * 1.01, max_lim_row2_left * 1.01)
    axs[1, 1].legend([r'Re[$S^+_{1,12}(\omega)$]', r'Re[$\hat{S}^+_{1,12}(\omega)$]', r'Im[$S^+_{1,12}(\omega)$]',
                      r'Im[$\hat{S}^+_{1,2}(\omega)$]'], fontsize=legendfont, loc='upper left')
    axs[1, 1].tick_params(direction='in')
    axs[1, 1].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    bx = axs[1, 1].twinx()
    bx.tick_params(direction='in')
    bx.plot(w_plot / xunits, Gp_re(vt_opt[0], vt_opt[2], w_plot, Mplot_opt), 'g--', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(vt_opt[0], vt_opt[2], w_plot, Mplot_opt), 'k--', lw=lw)
    bx.plot(w_plot / xunits, Gp_re(known_opt[0], known_opt[2], w_plot, Mplot_known), 'm-', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(known_opt[0], known_opt[2], w_plot, Mplot_known), 'c-', lw=lw)
    # bx.set_yticklabels([])
    # bx.set_yticks([])
    max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[0], vt_opt[2], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_im(vt_opt[0], vt_opt[2], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_re(known_opt[0], known_opt[2], w_plot, Mplot_known)).max(),
                               np.abs(Gp_im(known_opt[0], known_opt[2], w_plot, Mplot_known)).max()]))
    bx.set_ylim(-max_lim * 1.01, max_lim * 1.01)
    bx.legend([r'Re[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{1,1;12,12}(\omega, T)$]',
               r'Re[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]',
               r'Im[$G^{+,\text{known}}_{1,1;12,12}(\omega, T)$]']
              , fontsize=legendfont, loc='lower right')
    axs[1, 2].plot(w_plot / xunits, np.real(S_2_12(w_plot, config.gamma12 - config.gamma)) / yunits, 'r-', alpha=alp,
                   lw=1.5 * lw)
    axs[1, 2].plot(wk_local_plot / xunits, np.real(config.specs["S212"]) / yunits, 'r--^', alpha=alp, lw=1.5 * lw)
    axs[1, 2].plot(w_plot / xunits, np.imag(S_2_12(w_plot, config.gamma12 - config.gamma)) / yunits, 'b-', alpha=alp,
                   lw=1.5 * lw)
    axs[1, 2].plot(wk_local_plot / xunits, np.imag(config.specs["S212"]) / yunits, 'b--^', alpha=alp, lw=1.5 * lw)
    # axs[1, 2].set_yticks([])
    axs[1, 2].set_yticklabels([])
    # max_lim = np.maximum(np.abs(np.real(S_2_12(w, gamma-gamma12))/1e3).max(), np.abs(np.imag(S_2_12(w, gamma-gamma12))/1e3).max())
    axs[1, 2].set_ylim(-max_lim_row2_left * 1.01, max_lim_row2_left * 1.01)
    axs[1, 2].legend([r'Re[$S^+_{2,12}(\omega)$]', r'Re[$\hat{S}^+_{2,12}(\omega)$]', r'Im[$S^+_{2,12}(\omega)$]',
                      r'Im[$\hat{S}^+_{2,12}(\omega)$]'], fontsize=legendfont, loc='upper left')
    axs[1, 2].tick_params(direction='in')
    axs[1, 2].set_xlabel('$\omega$ (MHz)', fontsize=labelfont)
    bx = axs[1, 2].twinx()
    bx.tick_params(direction='in')
    bx.plot(w_plot / xunits, Gp_re(vt_opt[1], vt_opt[2], w_plot, Mplot_opt), 'g--', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(vt_opt[1], vt_opt[2], w_plot, Mplot_opt), 'k--', lw=lw)
    bx.plot(w_plot / xunits, Gp_re(known_opt[1], known_opt[2], w_plot, Mplot_known), 'm-', lw=lw)
    bx.plot(w_plot / xunits, Gp_im(known_opt[1], known_opt[2], w_plot, Mplot_known), 'c-', lw=lw)
    bx.set_ylabel(r'$G^+_{a,a;b,b}(\omega, T)$', fontsize=labelfont)
    max_lim = np.max(np.array([np.abs(Gp_re(vt_opt[1], vt_opt[2], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_im(vt_opt[1], vt_opt[2], w_plot, Mplot_opt)).max(),
                               np.abs(Gp_re(known_opt[1], known_opt[2], w_plot, Mplot_known)).max(),
                               np.abs(Gp_im(known_opt[1], known_opt[2], w_plot, Mplot_known)).max()]))
    bx.set_ylim(-max_lim * 1.01, max_lim * 1.01)
    bx.legend([r'Re[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]', r'Im[$G^{+,\text{opt}}_{2,2;12,12}(\omega, T)$]',
               r'Re[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]',
               r'Im[$G^{+,\text{known}}_{2,2;12,12}(\omega, T)$]']
              , fontsize=legendfont, loc='lower right')
    plt.savefig(os.path.join(config.path, 'IDGateLog.pdf'), bbox_inches='tight')
    # plt.show()
    print('End')


if __name__ == '__main__':
    main()
