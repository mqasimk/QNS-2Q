import numpy as np
import qutip as qt
from scipy.linalg import expm
from joblib import Parallel, delayed

def make_noise_mat(spec_vec, t_vec, **kwargs):
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    trunc_n = kwargs.get('trunc_n')
    gamma = kwargs.get('gamma')
    size_t = np.size(t_vec)
    size_w = int(trunc_n*w_grain)
    S = np.zeros((size_t, size_w))
    C = np.zeros((size_t, size_w))
    dw = wmax/w_grain
    for i in range(size_t):
        for j in range(size_w):
            S[i,j] = np.sqrt(dw*spec_vec(j*dw)/np.pi)*np.sin(j*dw*(t_vec[i] + gamma))
            C[i,j] = np.sqrt(dw*spec_vec(j*dw)/np.pi)*np.cos(j*dw*(t_vec[i] + gamma))
    return S, C

def make_noise_traj(S_list, C_list):
    S = S_list[0]
    C = C_list[0]
    Sg = S_list[1]
    Cg = C_list[1]
    A = np.random.normal(0, 1, (np.size(S,1), 1))
    B = np.random.normal(0, 1, (np.size(C,1), 1))
    traj = np.ndarray.flatten(np.matmul(S, A) + np.matmul(C, B))
    traj_g = np.ndarray.flatten(np.matmul(Sg, A) + np.matmul(Cg, B))
    return traj, traj_g

def make_init_state(a_sp, c, **kwargs):
    zp = qt.basis(2, 0)
    zm = qt.basis(2, 1)
    x_gates = [qt.tensor(qt.sigmax(), qt.identity(2)), qt.tensor(qt.identity(2), qt.sigmax())]
    asp_0 = a_sp[0]
    asp_1 = a_sp[1]
    c_0 = c[0]
    c_1 = c[1]
    basis2q = [qt.tensor(zp, zp), qt.tensor(zp, zm), qt.tensor(zm, zp), qt.tensor(zm, zm)]
    rho0_0 = 0.5*(1.+asp_0)*zp*zp.dag() + 0.5*(1.-asp_0)*zm*zm.dag() + 0.5*c_0*zp*zm.dag() + 0.5*c_0.conj()*zm*zp.dag()
    rho0_1 = 0.5*(1.+asp_1)*zp*zp.dag() + 0.5*(1.-asp_1)*zm*zm.dag() + 0.5*c_1*zp*zm.dag() + 0.5*c_1.conj()*zm*zp.dag()
    rho0 = qt.tensor(rho0_0, rho0_1)
    ry = [qt.tensor(np.cos(np.pi/4)*qt.identity(2) - 1j*np.sin(np.pi/4)*qt.sigmay(), qt.identity(2)),
          qt.tensor(qt.identity(2), np.cos(np.pi/4)*qt.identity(2) - 1j*np.sin(np.pi/4)*qt.sigmay())]
    if kwargs.get('state') == 'p0':
        return ry[0]*rho0*ry[0].dag()
    elif kwargs.get('state') == 'p1':
        return x_gates[1]*ry[0]*rho0*ry[0].dag()*x_gates[1].dag()
    elif kwargs.get('state') == '0p':
        return ry[1]*rho0*ry[1].dag()
    elif kwargs.get('state') == '1p':
        return x_gates[0]*ry[1]*rho0*ry[1].dag()*x_gates[0].dag()
    elif kwargs.get('state') == 'pp':
        return ry[1]*ry[0]*rho0*ry[0].dag()*ry[1].dag()
    else:
        raise Exception("Invalid state input")

def make_Hamiltonian(y_uv, b_t):
    b_t_1 = b_t[0]
    b_t_1_g = b_t[1]
    b_t_2 = b_t[0]
    b_t_2_g = b_t[1]
    b_t_12 = b_t[2]
    b_t_12_g = b_t[3]
    z_vec = [qt.tensor(qt.identity(2), qt.identity(2)),
             qt.tensor(qt.sigmaz(), qt.identity(2)),
             qt.tensor(qt.identity(2), qt.sigmaz()),
             qt.tensor(qt.sigmaz(), qt.sigmaz())]
    h_t = []
    B = [[qt.sigmax(), qt.sigmay()], [qt.sigmax(), qt.sigmay()], [qt.sigmax(), qt.sigmay()]]
    b_vec = [[b_t_1, b_t_1_g], [b_t_2, b_t_2_g], [b_t_12, b_t_12_g]]
    for i in range(3):
        for j in range(3):
            h_t.append([qt.tensor(z_vec[i], B[j][0]), y_uv[i,j]*b_vec[j][0]])
            h_t.append([qt.tensor(z_vec[i], B[j][1]), y_uv[i,j]*b_vec[j][1]])
    return h_t

def f(t, tk):
    return sum([((-1)**i)*np.heaviside(t-tk[i],1)*np.heaviside(tk[i+1]-t,1) for i in range(np.size(tk)-1)])

def cpmg(t, n):
    tk = [(k+0.50)*t[-1]/(2*n) for k in range(int(2*n))]
    tk.append(t[-1])
    tk.insert(0,0.)
    return f(t, tk)

def cdd1(t, n):
    return

def make_y(t_b, pulse, **kwargs):
    ctime = kwargs.get('ctime')
    M = kwargs.get('M')
    n = int(t_b[-1]/ctime)
    y = np.zeros((3,3,np.size(t_b)))
    for i in range(2):
        if pulse[i] == 'CPMG':
            y[i,i] = cpmg(t_b, n)
        elif pulse[i] == 'CDD1':
            pass
        elif pulse[i] == 'CDD3':
            pass
        elif pulse[i] == 'FID':
            y[i,i] = np.ones(np.size(t_b))
        else:
            raise Exception("Invalid pulse input")
    y[2,2] = np.multiply(y[1,1], y[0,0])
    return np.tile(y, M)


def solver(y_uv, noise_mats, t_vec, **kwargs):
    n_shots = kwargs.get('n_shots')
    state = kwargs.get('state')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    S, C = noise_mats[0,0], noise_mats[0,1]
    Sg, Cg = noise_mats[1,0], noise_mats[1,1]
    S_12, C_12 = noise_mats[2,0], noise_mats[2,1]
    Sg_12, Cg_12 = noise_mats[3,0], noise_mats[3,1]
    rho0 = make_init_state(a_sp, c, state = state)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    output = []
    for i in range(n_shots):
        b_t, b_t_g = make_noise_traj([S, Sg], [C, Cg])
        b_t_12, b_t_g_12 = make_noise_traj([S_12, Sg_12], [C_12, Cg_12])
        rho = qt.tensor(rho0, rho_B)
        H_t = make_Hamiltonian(y_uv, [b_t, b_t_g, b_t_12, b_t_g_12])
        sol = qt.mesolve(H_t, rho, t_vec)
        output.append(sol)
    rho_MT = []
    for i in range(n_shots):
        rho_MT.append(output[i].states[-1])
    return rho_MT

def make_propagator(H_t, t_vec):
    integrand = np.array([(-1j)*(np.array([H_t[i][1][j] * (H_t[i][0]).full() for j in range(H_t[i][1].shape[0])])) for i in range(len(H_t))])
    integrand = np.sum(integrand, axis=0)
    U = expm(np.trapz(integrand, t_vec, axis=0))
    return U

def solver_prop(y_uv, noise_mats, t_vec, **kwargs):
    n_shots = kwargs.get('n_shots')
    state = kwargs.get('state')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    S, C = noise_mats[0,0], noise_mats[0,1]
    Sg, Cg = noise_mats[1,0], noise_mats[1,1]
    S_12, C_12 = noise_mats[2,0], noise_mats[2,1]
    Sg_12, Cg_12 = noise_mats[3,0], noise_mats[3,1]
    rho0 = make_init_state(a_sp, c, state = state)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    output = []
    for i in range(n_shots):
        b_t, b_t_g = make_noise_traj([S, Sg], [C, Cg])
        b_t_12, b_t_g_12 = make_noise_traj([S_12, Sg_12], [C_12, Cg_12])
        rho = qt.tensor(rho0, rho_B)
        H_t = make_Hamiltonian(y_uv, [b_t, b_t_g, b_t_12, b_t_g_12])
        U = make_propagator(H_t, t_vec)
        rho_MT = U @ rho @ U.conjugate().transpose()
        output.append(qt.Qobj(rho_MT))
    return output