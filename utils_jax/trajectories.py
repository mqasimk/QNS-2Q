import numpy as np
import qutip as qt
from scipy.linalg import expm
from joblib import Parallel, delayed
import jax
import jax.numpy as jnp

def make_noise_mat_arr(act, **kwargs):
    spec_vec = kwargs.get('spec_vec')
    t_vec = kwargs.get('t_vec')
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    truncate = kwargs.get('truncate')
    gamma = kwargs.get('gamma')
    gamma_12 = kwargs.get('gamma_12')
    if act == 'load':
        return np.load('noise_mats.npy', allow_pickle=True)
    elif act == 'make':
        S, C = make_noise_mat(spec_vec[0], t_vec, w_grain = w_grain, wmax = wmax, trunc_n = truncate, gamma = 0)
        Sg, Cg = make_noise_mat(spec_vec[0], t_vec, w_grain = w_grain, wmax = wmax, trunc_n = truncate, gamma = gamma)
        S_12, C_12 = make_noise_mat(spec_vec[1], t_vec, w_grain = w_grain, wmax = wmax, trunc_n = truncate, gamma = 0)
        Sg_12, Cg_12 = make_noise_mat(spec_vec[1], t_vec, w_grain = w_grain, wmax = wmax, trunc_n = truncate,
                                      gamma = gamma_12)
        return np.array([[S, C], [Sg, Cg], [S_12, C_12], [Sg_12, Cg_12]])
    elif act == 'save':
        mats = make_noise_mat_arr('make', **kwargs)
        np.save('noise_mats.npy', mats)
        return mats
    else:
        raise Exception("Invalid action input")

def sinM(spec, w, t, dw, gamma):
    return jnp.sqrt(dw*spec(w)/np.pi)*jnp.sin(w*(t + gamma))


def cosM(spec, w, t, dw, gamma):
    return jnp.sqrt(dw*spec(w)/np.pi)*jnp.cos(w*(t + gamma))

def make_noise_mat(spec, t_vec, **kwargs):
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    #trunc_n = kwargs.get('trunc_n')
    gamma = kwargs.get('gamma')
    #size_t = np.size(t_vec)
    size_w = int(2*w_grain)
    w = jnp.linspace(0, 2*wmax, size_w)
    dw = wmax/w_grain
    Sf = jax.vmap(jax.vmap(sinM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    Cf = jax.vmap(jax.vmap(cosM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    return Sf(spec, w, t_vec, dw, gamma), Cf(spec, w, t_vec, dw, gamma)

@jax.jit
def make_noise_traj(S_arr, C_arr):
    # S = S_list[0]
    # C = C_list[0]
    # Sg = S_list[1]
    # Cg = C_list[1]
    A = jnp.array(np.random.normal(0, 1, (jnp.size(S_arr[0],1), 1)))
    B = jnp.array(np.random.normal(0, 1, (jnp.size(C_arr[0],1), 1)))
    traj = jnp.ravel(jnp.matmul(S_arr[0], A) + jnp.matmul(C_arr[0], B))
    traj_g = jnp.ravel(jnp.matmul(S_arr[1], A) + jnp.matmul(C_arr[1], B))
    return traj, traj_g


def make_init_state(a_sp, c, **kwargs):
    zp = qt.basis(2, 0)
    zm = qt.basis(2, 1)
    x_gates = [qt.tensor(qt.sigmax(), qt.identity(2)), qt.tensor(qt.identity(2), qt.sigmax())]
    asp_0 = a_sp[0]
    asp_1 = a_sp[1]
    c_0 = c[0]
    c_1 = c[1]
    #basis2q = [qt.tensor(zp, zp), qt.tensor(zp, zm), qt.tensor(zm, zp), qt.tensor(zm, zm)]
    rho0_0 = 0.5*(1.+asp_0)*zp*zp.dag() + 0.5*(1.-asp_0)*zm*zm.dag() + 0.5*c_0*zp*zm.dag() + 0.5*np.conj(c_0)*zm*zp.dag()
    rho0_1 = 0.5*(1.+asp_1)*zp*zp.dag() + 0.5*(1.-asp_1)*zm*zm.dag() + 0.5*c_1*zp*zm.dag() + 0.5*np.conj(c_1)*zm*zp.dag()
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

@jax.jit
def make_Hamiltonian(y_uv, b_t):
    paulis = jnp.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]], [[1., 0.], [0., -1.]]])
    z_vec = jnp.array([jnp.kron(paulis[0], paulis[0]), jnp.kron(paulis[3], paulis[0]), jnp.kron(paulis[0], paulis[3]),
                       jnp.kron(paulis[3], paulis[3])])
    B = jnp.array([[paulis[1], paulis[2]], [paulis[1], paulis[2]], [paulis[1], paulis[2]]])
    b_vec = jnp.array([[b_t[0], b_t[1]], [b_t[0], b_t[1]], [b_t[2], b_t[3]]])
    h_t_ops = jnp.array([[jnp.kron(z_vec[i+1], B[i, 0]), jnp.kron(z_vec[i+1], B[i, 1])] for i in range(3)])
    h_t = jnp.sum(jnp.array([jnp.tensordot(y_uv[i, j]*b_vec[i, 0], h_t_ops[i, 0], 0) +
                             jnp.tensordot(y_uv[i, j]*b_vec[i, 1], h_t_ops[i, 1], 0)
                             for i in range(3) for j in range(3)]), axis=0)
    return h_t

def f(t, tk):
    return sum([((-1)**i)*np.heaviside(t-tk[i],1)*np.heaviside(tk[i+1]-t,1) for i in range(np.size(tk)-1)])


def cpmg(t, n):
    tk = [(k+0.50)*t[-1]/(2*n) for k in range(int(2*n))]
    tk.append(t[-1])
    tk.insert(0,0.)
    return f(t, tk)


def cdd1(t, n):
    ct = np.linspace(0., t[-1]/n, int(t.shape[0]/n))
    n = int(t[-1]/ct[-1])
    tk = [(k+1)*(t[-1])/(2*n) for k in range(int(2*n-1))]
    tk.append(t[-1])
    tk.insert(0,0.)
    return f(t, tk)


def prim_cycle(ct):
    m = 1
    t = ct
    tk1 = [(k+0.5)*t[-1]/(4*m) for k in range(int(2))]
    tk1.insert(0,0.)
    tk1 = np.array(tk1)
    tk2 = tk1 + t[-1]/2
    tk2 = np.concatenate((tk2, [t[-1]]))
    tk = np.concatenate((tk1, tk2))
    return f(t,tk)


def cdd3(t, m):
    if m == 1:
        return prim_cycle(t)
    out = np.tile(prim_cycle(t[:int(t.shape[0]/m)]), m)
    if t.shape[0] > out.shape[0]:
        out = np.concatenate((out, -1*np.ones(t.shape[0]-out.shape[0])))
    return out


def make_y(t_b, pulse, **kwargs):
    ctime = kwargs.get('ctime')
    M = kwargs.get('M')
    n = int(t_b[-1]/ctime)
    y = np.zeros((3,3,np.size(t_b)))
    for i in range(2):
        if pulse[i] == 'CPMG':
            y[i, i] = cpmg(t_b, n)
            #y = y.at[i, i].set(cpmg(t_b, n))
        elif pulse[i] == 'CDD1':
            y[i, i] = cdd1(t_b, n)
            #y = y.at[i, i].set(cdd1(t_b, n))
        elif pulse[i] == 'CDD3':
            y[i, i] = cdd3(t_b, n)
            #y = y.at[i, i].set(cdd3(t_b, n))
        elif pulse[i] == 'FID':
            y[i, i] = np.ones(np.size(t_b))
            #y = y.at[i, i].set(np.ones(np.size(t_b)))
        else:
            raise Exception("Invalid pulse input")
    y[2,2] = np.multiply(y[1,1], y[0,0])
    #y = y.at[2, 2].set(np.multiply(y[1,1], y[0,0]))
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

@jax.jit
def make_propagator(H_t, t_vec):
    #integrand = jnp.array([(-1j)*(jnp.array([H_t[i, 1, j] * (H_t[i][0]).full() for j in range(H_t[i][1].shape[0])])) for i in range(len(H_t))])
    #integrand = jnp.sum(H_t, axis=0)
    U = jax.scipy.linalg.expm(jax.scipy.integrate.trapezoid(H_t, t_vec, axis=0))
    return U

@jax.jit
def single_shot_prop(noise_mats, t_vec, y_uv, rho0, n_shots):
    # S, C = noise_mats[0,0], noise_mats[0,1]
    # Sg, Cg = noise_mats[1,0], noise_mats[1,1]
    # S_12, C_12 = noise_mats[2,0], noise_mats[2,1]
    # Sg_12, Cg_12 = noise_mats[3,0], noise_mats[3,1]
    b_t, b_t_g = make_noise_traj(jnp.array([noise_mats[0,0], noise_mats[1,0]]), jnp.array([noise_mats[0,1], noise_mats[1,1]]))
    b_t_12, b_t_g_12 = make_noise_traj(jnp.array([noise_mats[2,0], noise_mats[3,0]]), jnp.array([noise_mats[2,1], noise_mats[3,1]]))
    H_t = make_Hamiltonian(y_uv, jnp.array([b_t, b_t_g, b_t_12, b_t_g_12]))
    U = make_propagator(H_t, t_vec)
    rho_MT = jnp.matmul(jnp.matmul(U, rho0), U.conjugate().transpose())
    return rho_MT

# def solver_prop(y_uv, noise_mats, t_vec, **kwargs):
#     n_shots = kwargs.get('n_shots')
#     state = kwargs.get('state')
#     a_sp = kwargs.get('a_sp')
#     c = kwargs.get('c')
#     S, C = noise_mats[0,0], noise_mats[0,1]
#     Sg, Cg = noise_mats[1,0], noise_mats[1,1]
#     S_12, C_12 = noise_mats[2,0], noise_mats[2,1]
#     Sg_12, Cg_12 = noise_mats[3,0], noise_mats[3,1]
#     rho0 = make_init_state(a_sp, c, state = state)
#     rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
#     output = []
#     for i in range(n_shots):
#         b_t, b_t_g = make_noise_traj([S, Sg], [C, Cg])
#         b_t_12, b_t_g_12 = make_noise_traj([S_12, Sg_12], [C_12, Cg_12])
#         rho = qt.tensor(rho0, rho_B)
#         H_t = make_Hamiltonian(y_uv, [b_t, b_t_g, b_t_12, b_t_g_12])
#         U = make_propagator(H_t, t_vec)
#         rho_MT = U @ rho.full() @ U.conjugate().transpose()
#         output.append(qt.Qobj(rho_MT, dims = [[2,2,2],[2,2,2]]))
#     return output



def solver_prop(y_uv, noise_mats, t_vec, rho, n_shots):
    #output = Parallel(n_jobs = 1, verbose=0)(delayed(single_shot_prop)(noise_mats, t_vec, y_uv, rho) for i in range(n_shots))
    #map = jax.vmap(single_shot_prop, in_axes=[None, None, None, None, 0])
    #n = jnp.arange(n_shots-1)
    #result = map(noise_mats, t_vec, y_uv, rho, n)
    result = jax.vmap(single_shot_prop, in_axes=[None, None, None, None, 0])(noise_mats, t_vec, y_uv,
                                                                                     rho, jnp.arange(n_shots))
    output = []
    for i in range(n_shots):
        output.append(qt.Qobj(np.array(result[i]), dims = [[2,2,2],[2,2,2]]))
    return output
