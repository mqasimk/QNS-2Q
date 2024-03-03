import numpy as np
import qutip as qt
from trajectories import make_y, make_init_state
import jax.numpy as jnp
from joblib import Parallel, delayed


def E_X(qubit, state, a_m, delta):
    rho_s = []
    if qubit == 1:
        for i in range(len(state)):
            rho_s.append(state[i].ptrace(0))
    else:
        for i in range(len(state)):
            rho_s.append(state[i].ptrace(1))
    return np.sum(np.array([(rho_s[i]*qt.sigmax()).tr() for i in range(len(state))]))/len(state)


def E_Y(qubit, state, a_m, delta):
    rho_s = []
    if qubit == 1:
        for i in range(len(state)):
            rho_s.append(state[i].ptrace(0))
    else:
        for i in range(len(state)):
            rho_s.append(state[i].ptrace(1))
    return np.sum(np.array([(rho_s[i]*qt.sigmay()).tr() for i in range(len(state))]))/len(state)


def E_XX(state, a_m, delta):
    rho_s = []
    for i in range(len(state)):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmax(),qt.sigmax())).tr() for i in range(len(state))]))/len(state)


def E_XY(state, a_m, delta):
    rho_s = []
    for i in range(len(state)):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmax(),qt.sigmay())).tr() for i in range(len(state))]))/len(state)


def E_YX(state, a_m, delta):
    rho_s = []
    for i in range(len(state)):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmay(),qt.sigmax())).tr() for i in range(len(state))]))/len(state)


def E_YY(state, a_m, delta):
    rho_s = []
    for i in range(len(state)):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmay(),qt.sigmay())).tr() for i in range(len(state))]))/len(state)


def A(expec):
    # send expec as (EX, EY) created with the appropriate $$\psi^{\pm}_l$$
    return -0.25*np.log(np.square(expec[0])+np.square(expec[1]))


def D(pm, expec):
    # send expec as (EX1X2, EY1Y2, EX1Y2, EY1X2) created with the state $$\psi_{12}$$
    if pm == '+':
        return -0.25*np.log(np.square(expec[2]+expec[3])+np.square(expec[0]-expec[1]))
    elif pm == '-':
        return -0.25*np.log(np.square(expec[2]-expec[3])+np.square(expec[0]+expec[1]))
    else:
        raise Exception("Invalid pm input")


def C_12_0_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    y_uv = np.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EX1X2 = E_XX(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EY1Y2 = E_YY(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EX1Y2 = E_XY(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EY1X2 = E_YX(sol, a_m, delta)
    return np.real(D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) + D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2)))

def make_C_12_0_MT(solver_ftn, pulse, noise_mats, t_vec, c_times, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    state = kwargs.get('state')
    C_12_0_MT = np.zeros(np.size(c_times))
    rho0 = make_init_state(a_sp, c, state = state)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    C_12_0_MT = Parallel(n_jobs=24)(delayed(C_12_0_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, rho,
                                                         c_times[i], n_shots=n_shots, M=M, a_m=a_m,
                                                         delta=delta) for i in range(np.size(c_times)))
    return C_12_0_MT

    # for i in range(np.size(c_times)):
    #     y_uv = np.array(make_y(t_b, pulse, ctime=c_times[i], M=M))
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EX1X2 = E_XX(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EY1Y2 = E_YY(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EX1Y2 = E_XY(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EY1X2 = E_YX(sol, a_m, delta)
    #     C_12_0_MT[i] = np.real(D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) + D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2)))
    # return C_12_0_MT


def C_12_12_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    y_uv = np.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EX1X2 = E_XX(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EY1Y2 = E_YY(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EX1Y2 = E_XY(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    EY1X2 = E_YX(sol, a_m, delta)
    return np.real(D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) - D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2)))


def make_C_12_12_MT(solver_ftn, pulse, noise_mats, t_vec, c_times, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    state = kwargs.get('state')
    rho0 = make_init_state(a_sp, c, state = state)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    return Parallel(n_jobs=24)(delayed(C_12_12_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, c_times[i],
                                                           n_shots=n_shots, M=M, a_m=a_m,
                                                           delta=delta) for i in range(np.size(c_times)))

    # for i in range(np.size(c_times)):
    #     y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EX1X2 = E_XX(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EY1Y2 = E_YY(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EX1Y2 = E_XY(sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    #     EY1X2 = E_YX(sol, a_m, delta)
    #     C_12_12_MT[i] = np.real(D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) - D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2)))
    # return C_12_12_MT


def C_a_b_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    rhop = rho[0]
    rhom = rho[1]
    y_uv = np.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    EXlp = E_X(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    EYlp = E_Y(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    Ap = A([EXlp, EYlp])
    EXlm = E_X(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    EYlm = E_Y(l, sol, a_m, delta)
    Am = A([EXlm, EYlm])
    return np.real(Ap + Am)

def make_C_a_b_MT(solver_ftn, pulse, noise_mats, t_vec, c_times, **kwargs):
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    n_shots = kwargs.get('n_shots')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    if l==1:
        state_p = 'p0'
        state_m = 'p1'
    elif l==2:
        state_p = '0p'
        state_m = '1p'
    else:
        raise Exception("Invalid state input")
    # C_a_b_MT = np.zeros(np.size(c_times))
    rho0p = make_init_state(a_sp, c, state = state_p)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rho0m = make_init_state(a_sp, c, state = state_m)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    return Parallel(n_jobs=24)(delayed(C_a_b_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, [rhop, rhom], c_times[i],
                                                   n_shots=n_shots, M=M, a_m=a_m, l=l,
                                                   delta=delta) for i in range(np.size(c_times)))

    # for i in range(np.size(c_times)):
    #     y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    #     state_vec = [sol[i] for i in range(n_shots)]
    #     EXlp = E_X(l, state_vec, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    #     state_vec = [sol[i] for i in range(n_shots)]
    #     EYlp = E_Y(l, state_vec, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    #     state_vec = [sol[i] for i in range(n_shots)]
    #     Ap = A([EXlp, EYlp])
    #     EXlm = E_X(l, state_vec, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    #     state_vec = [sol[i] for i in range(n_shots)]
    #     EYlm = E_Y(l, state_vec, a_m, delta)
    #     Am = A([EXlm, EYlm])
    #     C_a_b_MT[i] = np.real(Ap + Am)
    # return C_a_b_MT

def C_a_0_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    rhop = rho[0]
    rhom = rho[1]
    y_uv = np.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    EXlp = E_X(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    EYlp = E_Y(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    Ap = A([EXlp, EYlp])
    EXlm = E_X(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    EYlm = E_Y(l, sol, a_m, delta)
    Am = A([EXlm, EYlm])
    return np.real(Ap - Am)

def make_C_a_0_MT(solver_ftn, pulse, noise_mats, t_vec, c_times, **kwargs):
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    n_shots = kwargs.get('n_shots')
    a_sp = kwargs.get('a_sp')
    c = kwargs.get('c')
    if l==1:
        state_p = 'p0'
        state_m = 'p1'
    elif l==2:
        state_p = '0p'
        state_m = '1p'
    else:
        raise Exception("Invalid state input")
    C_a_0_MT = np.zeros(np.size(c_times))
    rho0p = make_init_state(a_sp, c, state = state_p)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rho0m = make_init_state(a_sp, c, state = state_m)
    rho_B = qt.basis(2, 0) * qt.basis(2, 0).dag()
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    return Parallel(n_jobs=24)(delayed(C_a_0_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, [rhop, rhom], c_times[i],
                                                   n_shots=n_shots, M=M, a_m=a_m, l=l,
                                                   delta=delta) for i in range(np.size(c_times)))

    # for i in range(np.size(c_times)):
    #     y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    #     EXlp = E_X(l, sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    #     EYlp = E_Y(l, sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    #     Ap = A([EXlp, EYlp])
    #     EXlm = E_X(l, sol, a_m, delta)
    #     sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    #     EYlm = E_Y(l, sol, a_m, delta)
    #     Am = A([EXlm, EYlm])
    #     C_a_0_MT[i] = np.real(Ap - Am)
    # return C_a_0_MT
