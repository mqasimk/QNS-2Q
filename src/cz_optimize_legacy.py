import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import jaxopt
from spectra_input import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12
import os
from dataclasses import dataclass, field

jax.config.update("jax_enable_x64", True)


########################################################################################################################
################################## Configuration #####################################################################
########################################################################################################################

@dataclass
class CZOptConfig:
    """Configuration for the CZ gate optimization."""
    run_name: str = "DraftRun_NoSPAM"
    parent_dir: str = os.pardir
    Jmax: float = 3e6
    gate_time_factors: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    nplist: list = field(default_factory=lambda: [[[158], [158]], [[78], [78]], [[38, 39], [38, 39]], [[38, 39], [38, 39]],
                                                 [[38, 39], [38, 39]], [[38, 39], [38, 39]], [[38, 39], [38, 39]],
                                                 [[38, 39], [38, 39]], [[38, 39], [38, 39]]])
    output_path_known: str = "infs_known_cz.npz"
    output_path_opt: str = "infs_opt_cz.npz"
    plot_filename: str = "infs_GateTime_cz.pdf"

    # These will be loaded from the run files
    T: float = field(init=False)
    tau: float = field(init=False)
    mc: int = field(init=False)
    gamma: float = field(init=False)
    gamma12: float = field(init=False)

    # Calculated properties
    path: str = field(init=False)
    specs: dict = field(init=False)
    w_ideal: jnp.ndarray = field(init=False)
    wqns: jnp.ndarray = field(init=False)
    wkqns: jnp.ndarray = field(init=False)
    SMat: jnp.ndarray = field(init=False)
    SMat_ideal: jnp.ndarray = field(init=False)
    T2q1: float = field(init=False, default=jnp.inf)
    T2q2: float = field(init=False, default=jnp.inf)

    def __post_init__(self):
        self.path = os.path.join(self.parent_dir, self.run_name)
        params = np.load(os.path.join(self.path, "params.npz"))
        self.specs = np.load(os.path.join(self.path, "specs.npz"))

        self.T = params['T']
        self.tau = self.T / 80
        self.mc = int(params['truncate'])
        self.gamma = params['gamma']
        self.gamma12 = params['gamma_12']

        self.w_ideal = jnp.linspace(0.0001, 2 * jnp.pi * 2 * self.mc / self.T, 12000)
        self.wqns = jnp.linspace(0.0001, 2 * jnp.pi * self.mc / self.T, 4000)
        self.wkqns = jnp.array([2 * jnp.pi * n / self.T for n in range(self.mc + 1)])

        self.SMat = makeSMat_k(self.specs, self.wqns, self.wkqns, self.gamma)
        self.SMat_ideal = makeSMat_k_ideal(self.w_ideal, self.gamma, self.gamma12)
        self._calculate_T2()

    def _calculate_T2(self):
        taxis = jnp.linspace(1 * self.T, 20 * self.T, 20)
        vt_T2 = jnp.array([jnp.array([[0, taxis[i]], [0, taxis[i]]]) for i in range(taxis.shape[0])])

        q1T2 = jnp.array([T2(vt_T2[i], self.SMat_ideal, 1, self.w_ideal, 1) for i in range(vt_T2.shape[0])])
        q2T2 = jnp.array([T2(vt_T2[i], self.SMat_ideal, 1, self.w_ideal, 2) for i in range(vt_T2.shape[0])])

        for i in range(q1T2.size):
            if q1T2[i] < 0.5:
                self.T2q1 = float((taxis[i] + taxis[i - 1]) * 0.5)
                break
        for i in range(q2T2.size):
            if q2T2[i] < 0.5:
                self.T2q2 = float((taxis[i] + taxis[i - 1]) * 0.5)
                break
        print("###########################################################################################")
        print(f"T2 time for qubit 1 is {np.round(self.T2q1 / 1e-6, 2)} us")
        print(f"T2 time for qubit 2 is {np.round(self.T2q2 / 1e-6, 2)} us")
        print("###########################################################################################")


########################################################################################################################
################################## Utility and Physics Functions #####################################################
########################################################################################################################

@jax.jit
def Sc(w):
    tc = 4e-4
    S0 = 1e-6
    return S0 / (1 + (tc ** 2) * (w ** 2))


@jax.jit
def zzPTM():
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    U = jax.scipy.linalg.expm(-1j * jnp.kron(p1q[3], p1q[3]) * jnp.pi / 4)
    gamma = jnp.array([[(1 / 4) * jnp.trace(p2q[i] @ U @ p2q[j] @ U.conj().transpose()) for j in range(16)] for i in
                       range(16)])
    return jnp.real(gamma)


def pulse(mu, sig, t):
    return jnp.exp(-(t - mu) ** 2 / (2 * sig ** 2))


@jax.jit
def sgn(O, a, b):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(jnp.linalg.inv(O) @ z2q[a] @ z2q[b] @ O @ z2q[a] @ z2q[b]) / 4


@jax.jit
def ff(tk, w):
    return jnp.sum(jnp.array(
        [1j * ((-1) ** k) * (jnp.exp(-1j * w * tk[k + 1]) - jnp.exp(-1j * w * tk[k])) / w for k in
         range(jnp.size(tk) - 1)]), axis=0)


@jax.jit
def ft(f, t, w):
    return jax.scipy.integrate.trapezoid(f * jnp.exp(1j * w * t), t)


@jax.jit
def Gp_re(vti, vtj, w, M):
    return jnp.real(ff(vti, w) * ff(vtj, -w) * jnp.sin(w * M * vti[-1] * 0.5) ** 2 / jnp.sin(w * vti[-1] * 0.5) ** 2)


@jax.jit
def Gp_im(vti, vtj, w, M):
    return jnp.imag(ff(vti, w) * ff(vtj, -w) * jnp.sin(w * M * vti[-1] * 0.5) ** 2 / jnp.sin(w * vti[-1] * 0.5) ** 2)


@jax.jit
def y_t(t, tk):
    return jnp.sum(jnp.array(
        [((-1) ** i) * jnp.heaviside(t - tk[i], 1) * jnp.heaviside(tk[i + 1] - t, 1) for i in
         range(jnp.size(tk) - 1)]), axis=0)


def make_tk12(tk1, tk2):
    x = tk1[~jnp.isin(tk1, tk2)]
    y = tk2[~jnp.isin(tk2, tk1)]
    z = jnp.zeros(x.size + y.size + 2)
    z = z.at[0].set(0.)
    z = z.at[-1].set(tk1[-1])
    z = z.at[1:z.size - 1].set(jnp.sort(jnp.concatenate((x, y))))
    return z


def inf_CZ(params, i, j, SMat, M, T, w, Jmax, tau):
    vt1 = jnp.concatenate((jnp.array([0]), params[:i], jnp.array([T])))
    vt2 = jnp.concatenate((jnp.array([0]), params[i:i + j], jnp.array([T])))
    vt12 = make_tk12(vt1, vt2)
    vt = [vt1, vt2, vt12]
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i_ in range(3):
        for j_ in range(3):
            Gp = Gp.at[i_, j_].set(Gp_re_map(vt[i_], vt[j_], w, M) + 1j * Gp_im_map(vt[i_], vt[j_], w, M))
    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)),
                     in_axes=(0, None, None, None, None, None, None))
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i_], p1q[j_]) for i_ in range(4) for j_ in range(4)])
    tax = np.linspace(0, T, 1000)
    J = jnp.maximum(jnp.minimum(jnp.pi * 0.25 / (M * jax.scipy.integrate.trapezoid(y_t(tax, vt12), tax)), Jmax),
                    -Jmax)
    fid = jnp.trace(zzPTM().transpose() @ L_map(p2q, p2q, vt, SMat, M, w, J)) / 16.
    clustering = (1 / (vt[0].shape[0] + vt[1].shape[0] - 2)) * jnp.sum(
        jnp.array([jnp.tanh(vt[i_].shape[0] * (vt[i_][j_ + 1] - vt[i_][j_] / tau - 1)) for i_ in range(2) for j_ in
                   range(vt[i_].shape[0] - 1)]), axis=0)
    return -fid - clustering * 1e-3


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
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j * Gp_im_map(vt[i], vt[j], w, M))
    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)),
                     in_axes=(0, None, None, None, None, None, None))
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    return 1. - jnp.trace(zzPTM().transpose() @ L_map(p2q, p2q, vt, SMat, M, w, J)) / 16.


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
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, 1) + 1j * Gp_im_map(vt[i], vt[j], w, 1))
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


def hyperOpt(SMat, nPs, M, T, w, Jmax, tau, vtin=None):
    optimizer = jaxopt.ScipyBoundedMinimize(fun=inf_CZ, maxiter=80, jit=False, method='L-BFGS-B',
                                            options={'disp': False, 'gtol': 1e-9, 'ftol': 1e-8,
                                                     'maxfun': 1000, 'maxls': 20})
    opt_out = []
    if vtin is None:
        for i in nPs[0]:
            opt_out_temp = []
            for j in nPs[1]:
                vt = jnp.concatenate((jnp.sort(np.random.rand(i) * T), jnp.sort(np.random.rand(j) * T)))
                lower_bnd = jnp.zeros(vt.size)
                upper_bnd = jnp.ones(vt.size) * T
                bnds = (lower_bnd, upper_bnd)
                opt = optimizer.run(vt, bnds, i, j, SMat, M, T, w, Jmax, tau)
                opt_out_temp.append(opt)
                print("Optimized Cost: " + str(opt.state[0]) + ", No. of pulses on qubits:" + str([i, j]))
            opt_out.append(opt_out_temp)
        inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
        inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape)
        vt_min_arr = opt_out[inds_min[0]][inds_min[1]].params
        vt_opt_0 = params_to_tk(vt_min_arr, T, jnp.array([0, nPs[0][inds_min[0]]]))
        vt_opt_1 = params_to_tk(vt_min_arr, T, jnp.array(
            [nPs[0][inds_min[0]], nPs[0][inds_min[0]] + nPs[1][inds_min[1]]]))
        vt_opt = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
        tax = jnp.linspace(0, T, 1000)
        Jopt = jnp.pi * 0.25 / (M * jax.scipy.integrate.trapezoid(y_t(tax, vt_opt[2]), tax))
    else:
        for i, vt1 in enumerate(vtin):
            opt_out_temp = []
            for j, vt2 in enumerate(vtin):
                vt = jnp.concatenate((vt1, vt2))
                lower_bnd = jnp.zeros(vt.size)
                upper_bnd = jnp.ones(vt.size) * T
                bnds = (lower_bnd, upper_bnd)
                opt = optimizer.run(vt, bnds, vt1.size, vt2.size, SMat, M, T, w, Jmax, tau)
                opt_out_temp.append(opt)
                print("Optimized Cost: " + str(opt.state[0]) + ", No. of pulses on qubits:" + str(
                    [vt1.size, vt2.size]))
            opt_out.append(opt_out_temp)
        inf_ID_out = jnp.array([[opt_out[i][j].state[0] for j in range(len(opt_out[0]))] for i in range(len(opt_out))])
        inds_min = jnp.unravel_index(jnp.argmin(inf_ID_out), inf_ID_out.shape)
        vt_opt_0 = vtin[inds_min[0]]
        vt_opt_1 = vtin[inds_min[1]]
        vt_opt = [vt_opt_0, vt_opt_1, make_tk12(vt_opt_0, vt_opt_1)]
        tax = jnp.linspace(0, T, 10000)
        Jopt = np.maximum(np.minimum(jnp.pi * 0.25 / (M * jax.scipy.integrate.trapezoid(y_t(tax, vt_opt[2]), tax)),
                                     Jmax), -Jmax)
    return vt_opt, infidelity(vt_opt, SMat, M, w, Jopt), Jopt


def Lambda_CZ(Oi, Oj, vt, SMat, M, w, J):
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    Gp_re_map = jax.vmap(Gp_re, in_axes=(None, None, 0, None))
    Gp_im_map = jax.vmap(Gp_im, in_axes=(None, None, 0, None))
    Gp = jnp.zeros((3, 3, w.size), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            Gp = Gp.at[i, j].set(Gp_re_map(vt[i], vt[j], w, M) + 1j * Gp_im_map(vt[i], vt[j], w, M))
    tax = jnp.linspace(0, vt[2][-1], 1000)
    CO = 0
    for i in range(3):
        for j in range(3):
            CO += (-0.5 * (sgn(Oi, i + 1, j + 1) + 1) * z2q[i + 1] @ z2q[j + 1] * jax.scipy.integrate.trapezoid(
                SMat[i, j] * (sgn(Oi, i + 1, 0) - 1) * Gp[i, j], w) / jnp.pi)
    rot = (1. - sgn(Oi, 1, 2)) * M * jax.scipy.integrate.trapezoid(J * y_t(tax, vt[2]), tax) * z2q[3]
    return jnp.real(jnp.trace(Oi @ jax.scipy.linalg.expm(-1j * rot - CO) @ Oj) * 0.25)


def params_to_tk(params, T, shape: jnp.ndarray):
    vt = jnp.zeros(shape[1] - shape[0] + 2)
    vt = vt.at[0].set(0.)
    vt = vt.at[-1].set(T)
    vt = vt.at[1:shape[1] - shape[0] + 1].set(jnp.sort(params[shape[0]:shape[1]]))
    return vt


def cddn(T, n):
    if n == 0:
        return np.array([0., T])
    tk = [T * jnp.sin(jnp.pi * k / (2 * n + 2)) ** 2 for k in range(n + 1)]
    tk.append(T)
    tk.insert(0, 0)
    # remove duplicates
    #tk = list(dict.fromkeys(tk))
    return jnp.array(tk)


def opt_known_pulses(pLib, SMat, M, w, Jmax):
    infmin = jnp.inf
    inds_min = 0
    Jopt = 0
    for i in range(len(pLib)):
        for j in range(len(pLib)):
            for k in range(len(pLib[i])):
                for l in range(len(pLib[j])):
                    tk1 = pLib[i][k]
                    tk2 = pLib[j][l]
                    tk12 = make_tk12(tk1, tk2)
                    vt = [tk1, tk2, tk12]
                    tax = jnp.linspace(0, vt[2][-1], 1000)
                    J = np.maximum(
                        np.minimum(jnp.pi * 0.25 / (M * jax.scipy.integrate.trapezoid(y_t(tax, vt[2]), tax)),
                                   Jmax), -Jmax)
                    L_map = jax.vmap(jax.vmap(Lambda_CZ, in_axes=(None, 0, None, None, None, None, None)),
                                     in_axes=(0, None, None, None, None, None, None))
                    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
                    p2q = jnp.array([jnp.kron(p1q[i_], p1q[j_]) for i_ in range(4) for j_ in range(4)])
                    inf_ijkl = 1. - jnp.trace(zzPTM().transpose() @ L_map(p2q, p2q, vt, SMat, M, w, J)) / 16.
                    if inf_ijkl < infmin:
                        infmin = inf_ijkl
                        inds_min = (i, j, k, l)
                        Jopt = J
    tk1 = pLib[inds_min[0]][inds_min[2]]
    tk2 = pLib[inds_min[1]][inds_min[3]]
    tk12 = make_tk12(tk1, tk2)
    vt_opt = [tk1, tk2, tk12]
    return vt_opt, infmin, Jopt


def makeSMat_k(specs, wk, wkqns, gamma):
    SMat = jnp.zeros((3, 3, wk.size), dtype=jnp.complex128)
    SMat = SMat.at[0, 0].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_11(wk[0])]), specs["S11"]))))
    SMat = SMat.at[1, 1].set(jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_22(wk[0])]), specs["S22"]))))
    SMat = SMat.at[0, 1].set(
        jnp.interp(wk, wkqns, jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"]))))
    SMat = SMat.at[1, 0,].set(
        jnp.interp(wk, wkqns, jnp.conj(jnp.concatenate((jnp.array([S_1_2(wk[0], gamma)]), specs["S12"]))))
    )
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
    SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(wk, gamma12 - gamma))
    SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(wk, gamma12 - gamma)))
    return SMat_ideal


########################################################################################################################
####################################### Main Execution #################################################################
########################################################################################################################

def run_optimization(config_arg: CZOptConfig):
    """
    Runs the main optimization loop.
    """
    yaxis_base, xaxis_base = [], []
    yaxis_Tg, xaxis_Tg = [], []
    seqs = []
    best_seq = None

    # Known pulses loop
    for i in config_arg.gate_time_factors:
        Tg = config_arg.T / 2 ** (i - 1)
        base_gate = [[jnp.array([0, Tg])], [jnp.array([0, Tg])]]
        base_opt, base_inf, Jopt = opt_known_pulses(base_gate, config_arg.SMat_ideal, 1, config_arg.w_ideal, config_arg.Jmax)
        inf_base = infidelity(base_opt, config_arg.SMat_ideal, 1, config_arg.w_ideal, Jopt)
        yaxis_base.append(inf_base)
        xaxis_base.append(Tg)

        print("###########################################################################################")
        print(f"Gate Time: {np.round(Tg / config_arg.T2q1, 2)} T2q1 or {np.round(Tg / config_arg.T2q2, 2)} T2q2")
        print(f"Jmax: {config_arg.Jmax}")
        print(f'Infidelity of the uncorrected gate: {inf_base}')
        print(f'Number of pulses: {[p.shape[0] - 2 for p in base_opt[:2]]}')
        print(f"The coupling strength J: {Jopt}")
        print("###########################################################################################")

        best_inf = np.inf
        if Tg < config_arg.tau:
            yaxis_Tg.append(base_inf)
            xaxis_Tg.append(Tg)
            seqs.append(base_gate)
            continue

        for j in [1]:
            pLib = []
            cddLib = []
            Tknown = Tg/j
            Mknown = j
            if Tknown < config_arg.tau:
                continue
            
            cddOrd = 1
            while True:
                pul = cddn(Tknown, cddOrd)
                cddOrd += 1
                if any(pul[k + 1] - pul[k] < config_arg.tau for k in range(1, pul.size - 2)):
                    break
                cddLib.append(pul)
            
            pLib.append(cddLib)
            known_opt, known_inf, Jopt_known = opt_known_pulses(pLib, config_arg.SMat, Mknown, config_arg.wqns, config_arg.Jmax)
            
            if known_inf <= best_inf:
                inf_known = infidelity(known_opt, config_arg.SMat_ideal, Mknown, config_arg.w_ideal, Jopt_known)
                best_inf = inf_known
                best_seq = known_opt
                Jopt_best = Jopt_known
                print(f'Number of pulses: {[p.shape[0] - 2 for p in best_seq[:2]]}')
                print(f"The coupling strength J: {Jopt_best / config_arg.Jmax}")
                print(f'Best infidelity till now: {best_inf}, # of repetitions considered: {Mknown}')
                print("###########################################################################################")

        yaxis_Tg.append(best_inf)
        xaxis_Tg.append(Tg)
        seqs.append(best_seq)

    np.savez(os.path.join(config_arg.path, config_arg.output_path_known), infs_known=np.array(yaxis_Tg),
             infs_base=np.array(yaxis_base), taxis=np.array(xaxis_Tg))

    # Optimized pulses loop
    yaxis_opt, xaxis_opt = [], []
    seqs_opt = []
    count = 0
    for i in config_arg.gate_time_factors:
        Tg = config_arg.T / 2 ** (i - 1)
        if Tg < config_arg.tau:
            continue
        
        best_inf = np.inf
        for j in [1]:
            Topt = Tg/j
            Mopt = int(j)
            if Topt < config_arg.tau:
                continue
            
            nps = config_arg.nplist[count]
            #if 1 <= i <= 6:
            vt_opt, opt_inf, Jopt = hyperOpt(config_arg.SMat, nps, Mopt, Topt, config_arg.wqns, config_arg.Jmax, config_arg.tau,
                                                 [jnp.linspace(0, Topt, int(Topt / config_arg.tau)+2)[1:-1]])
            #else:
            #    vt_opt, opt_inf, Jopt = hyperOpt(config_arg.SMat, nps, Mopt, Topt, config_arg.wqns, config_arg.Jmax, config_arg.tau)
            #    count += 1
            
            inf_opt = infidelity(vt_opt, config_arg.SMat_ideal, Mopt, config_arg.w_ideal, Jopt)
            if inf_opt <= best_inf:
                best_inf = inf_opt
                best_seq = vt_opt
                Jopt_best = Jopt
                print(f'Number of pulses: {[p.shape[0] - 2 for p in best_seq[:2]]}')
                print(f"The coupling strength J: {Jopt_best / config_arg.Jmax}")
                print(f'Best infidelity till now: {best_inf}, # of repetitions considered: {Mopt}')
                print("###########################################################################################")

        yaxis_opt.append(best_inf)
        xaxis_opt.append(Tg)
        seqs_opt.append(best_seq)

    np.savez(os.path.join(config_arg.path, config_arg.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))

    return {
        "known_data": {"infs_known": yaxis_Tg, "infs_base": yaxis_base, "taxis": xaxis_Tg},
        "opt_data": {"infs_opt": yaxis_opt, "taxis": xaxis_opt}
    }


def plot_results(config_arg: CZOptConfig, results_arg: dict):
    """
    Plots the optimization results.
    """
    legendfont = 12
    labelfont = 16

    known_data = results_arg["known_data"]
    opt_data = results_arg["opt_data"]

    plt.figure(figsize=(16, 9))
    plt.plot(known_data["taxis"], known_data["infs_base"], "r^-")
    plt.plot(known_data["taxis"], known_data["infs_known"], "bs-")
    plt.plot(opt_data["taxis"], opt_data["infs_opt"], "ko-")
    plt.legend(["Uncorrected Gate", "DD Gate", "NT Gate"], fontsize=legendfont)
    plt.xlabel('Gate Time (s)', fontsize=labelfont)
    plt.ylabel('Gate Infidelity', fontsize=labelfont)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(config_arg.path, config_arg.plot_filename))
    plt.show()


if __name__ == '__main__':
    config = CZOptConfig()
    results = run_optimization(config)
    plot_results(config, results)

