import numpy as np
import qutip as qt
from trajectories import make_y, make_init_state
import jax.numpy as jnp
from joblib import Parallel, delayed
import jax

# @jax.jit
def POVMs(a_m, delta):
    a1 = (a_m[0]+delta[0]+1)*0.5
    b1 = (a_m[0]-delta[0]+1)*0.5
    a2 = (a_m[1]+delta[1]+1)*0.5
    b2 = (a_m[1]-delta[1]+1)*0.5
    zp = qt.basis(2, 0)
    zm = qt.basis(2, 1)
    p1_0 = a1*zp*zp.dag() + (1-b1)*zm*zm.dag()
    p1_1 = (1-a1)*zp*zp.dag() + b1*zm*zm.dag()
    p2_0 = a2*zp*zp.dag() + (1-b2)*zm*zm.dag()
    p2_1 = (1-a2)*zp*zp.dag() + b2*zm*zm.dag()
    p1_0 = qt.tensor(p1_0, qt.identity(2), qt.identity(2))
    p1_1 = qt.tensor(p1_1, qt.identity(2), qt.identity(2))
    p2_0 = qt.tensor(qt.identity(2), p2_0, qt.identity(2))
    p2_1 = qt.tensor(qt.identity(2), p2_1, qt.identity(2))
    return [p1_0, p1_1, p2_0, p2_1]

def twoq_meas(probs):
    probs = np.real(probs)
    # set negative values to 0
    probs[probs < 0] = 0.
    # set values greater than 1 to 1
    probs[probs > 1] = 1.
    # normalize the array
    return np.random.multinomial(1, probs)

def gen_probs(gate, rho, povms):
    p00 = (povms[0]*povms[2]*gate*rho*gate.dag()).tr()
    p01 = (povms[0]*povms[3]*gate*rho*gate.dag()).tr()
    p10 = (povms[1]*povms[2]*gate*rho*gate.dag()).tr()
    p11 = (povms[1]*povms[3]*gate*rho*gate.dag()).tr()
    return np.array([p00, p01, p10, p11])

def E_X(qubit, state, a_m, delta):
    if qubit == 1:
        op = qt.tensor(qt.sigmax(), qt.identity(2), qt.identity(2))
    else:
        op = qt.tensor(qt.identity(2), qt.sigmax(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)

def E_X_hat(qubit, state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros(len(state))
    h = [np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1,1],[1,-1]])/np.sqrt(2)),
         np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.array([[1,1],[1,-1]])/np.sqrt(2))]
    for i in range(len(h)):
        h[i] = qt.Qobj(np.kron(h[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    p0 = np.zeros(len(state))
    for i in range(len(state)):
        if qubit == 1:
            p0[i] = np.real((povms[0]*h[0]*state[i]*h[0]).tr())
            if np.real(p0[i]) > 1:
                p0[i] = 1
            # counts[i] = np.random.binomial(1, np.real(p0))
        else:
            p0[i] = np.real((povms[2]*h[1]*state[i]*h[1]).tr())
            if np.real(p0[i]) > 1:
                p0[i] = 1
            # counts[i] = np.random.binomial(1, np.real(p0))
    # p = np.sum(counts)/counts.shape[0]
    p = np.sum(p0)/len(p0)
    return 2.*p-1.

def E_Y(qubit, state, a_m, delta):
    if qubit == 1:
        op = qt.tensor(qt.sigmay(), qt.identity(2), qt.identity(2))
    else:
        op = qt.tensor(qt.identity(2), qt.sigmay(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)


def E_Y_hat(qubit, state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros(len(state))
    rx = [np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)),
          np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2))]
    for i in range(len(rx)):
        rx[i] = qt.Qobj(np.kron(rx[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    p0 = np.zeros(len(state))
    for i in range(len(state)):
        if qubit == 1:
            p0[i] = np.real((povms[0]*rx[0]*state[i]*rx[0].dag()).tr())
            if np.real(p0[i]) > 1:
                p0[i] = 1
            # counts[i] = np.random.binomial(1, np.real(p0))
        else:
            p0[i] = np.real((povms[2]*rx[1]*state[i]*rx[1].dag()).tr())
            if np.real(p0[i]) > 1:
                p0[i] = 1
            # counts[i] = np.random.binomial(1, np.real(p0))
    # p = np.sum(counts)/len(counts)
    p = np.sum(p0)/len(p0)
    return 2.*p-1.

def E_XX(state, a_m, delta):
    op = qt.tensor(qt.sigmax(), qt.sigmax(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)

def E_XX_hat(state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros((len(state), 4))
    h = [np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1,1],[1,-1]])/np.sqrt(2)),
         np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.array([[1,1],[1,-1]])/np.sqrt(2))]
    for i in range(len(h)):
        h[i] = qt.Qobj(np.kron(h[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    probs = np.zeros((len(state), 4))
    for i in range(len(state)):
        probs[i, :] = np.real(gen_probs(h[2], state[i], povms))
    p00 = np.sum(probs[:, 0])/probs.shape[0]
    p01 = np.sum(probs[:, 1])/probs.shape[0]
    p10 = np.sum(probs[:, 2])/probs.shape[0]
    p11 = np.sum(probs[:, 3])/probs.shape[0]
    # for i in range(len(state)):
    #     counts[i, :] = twoq_meas(gen_probs(h[2], state[i], povms))
    # p00 = np.sum(counts[:, 0])/counts.shape[0]
    # p01 = np.sum(counts[:, 1])/counts.shape[0]
    # p10 = np.sum(counts[:, 2])/counts.shape[0]
    # p11 = np.sum(counts[:, 3])/counts.shape[0]
    return p00 + p11 - p01 - p10


def E_XY(state, a_m, delta):
    op = qt.tensor(qt.sigmax(), qt.sigmay(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)

def E_XY_hat(state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros((len(state), 4))
    h = [np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1,1],[1,-1]])/np.sqrt(2)),
         np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.array([[1,1],[1,-1]])/np.sqrt(2))]
    rx = [np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)),
          np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2))]
    for i in range(len(h)):
        h[i] = qt.Qobj(np.kron(h[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
        rx[i] = qt.Qobj(np.kron(rx[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    probs = np.zeros((len(state), 4))
    for i in range(len(state)):
        probs[i, :] = np.real(gen_probs(h[0]*rx[1], state[i], povms))
    p00 = np.sum(probs[:, 0])/probs.shape[0]
    p01 = np.sum(probs[:, 1])/probs.shape[0]
    p10 = np.sum(probs[:, 2])/probs.shape[0]
    p11 = np.sum(probs[:, 3])/probs.shape[0]
    # for i in range(len(state)):
    #     counts[i, :] = twoq_meas(gen_probs(h[0]*rx[1], state[i], povms))
    # p00 = np.sum(counts[:, 0])/counts.shape[0]
    # p01 = np.sum(counts[:, 1])/counts.shape[0]
    # p10 = np.sum(counts[:, 2])/counts.shape[0]
    # p11 = np.sum(counts[:, 3])/counts.shape[0]
    return  p00 + p11 - p01 - p10


def E_YX(state, a_m, delta):
    op = qt.tensor(qt.sigmay(), qt.sigmax(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)

def E_YX_hat(state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros((len(state), 4))
    h = [np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1,1],[1,-1]])/np.sqrt(2)),
         np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.array([[1,1],[1,-1]])/np.sqrt(2))]
    rx = [np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)),
          np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2))]
    for i in range(len(h)):
        h[i] = qt.Qobj(np.kron(h[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
        rx[i] = qt.Qobj(np.kron(rx[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    probs = np.zeros((len(state), 4))
    for i in range(len(state)):
        probs[i, :] = np.real(gen_probs(h[1]*rx[0], state[i], povms))
    p00 = np.sum(probs[:, 0])/probs.shape[0]
    p01 = np.sum(probs[:, 1])/probs.shape[0]
    p10 = np.sum(probs[:, 2])/probs.shape[0]
    p11 = np.sum(probs[:, 3])/probs.shape[0]
    # for i in range(len(state)):
    #     counts[i, :] = twoq_meas(gen_probs(h[1]*rx[0], state[i], povms))
    # p00 = np.sum(counts[:, 0])/counts.shape[0]
    # p01 = np.sum(counts[:, 1])/counts.shape[0]
    # p10 = np.sum(counts[:, 2])/counts.shape[0]
    # p11 = np.sum(counts[:, 3])/counts.shape[0]
    return p00 + p11 - p01 - p10


def E_YY(state, a_m, delta):
    op = qt.tensor(qt.sigmay(), qt.sigmay(), qt.identity(2))
    return np.sum(np.array([(state[i]*op).tr() for i in range(len(state))]))/len(state)

def E_YY_hat(state, a_m, delta):
    povms = POVMs(a_m, delta)
    # counts = np.zeros((len(state), 4))
    h = [np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1,1],[1,-1]])/np.sqrt(2)),
         np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), np.array([[1,1],[1,-1]])/np.sqrt(2))]
    rx = [np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.identity(2)), np.kron(np.identity(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)),
          np.kron(np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2), np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2))]
    for i in range(len(h)):
        h[i] = qt.Qobj(np.kron(h[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
        rx[i] = qt.Qobj(np.kron(rx[i], np.identity(2)), dims=[[2, 2, 2], [2, 2, 2]])
    probs = np.zeros((len(state), 4))
    for i in range(len(state)):
        probs[i, :] = np.real(gen_probs(rx[2], state[i], povms))
    p00 = np.sum(probs[:, 0])/probs.shape[0]
    p01 = np.sum(probs[:, 1])/probs.shape[0]
    p10 = np.sum(probs[:, 2])/probs.shape[0]
    p11 = np.sum(probs[:, 3])/probs.shape[0]
    # for i in range(len(state)):
    #     counts[i, :] = twoq_meas(gen_probs(rx[2], state[i], povms))
    # p00 = np.sum(counts[:, 0])/counts.shape[0]
    # p01 = np.sum(counts[:, 1])/counts.shape[0]
    # p10 = np.sum(counts[:, 2])/counts.shape[0]
    # p11 = np.sum(counts[:, 3])/counts.shape[0]
    return p00 + p11 - p01 - p10

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

def frame_correct(sol, pulse):
    x1 = qt.tensor(qt.sigmax(), qt.identity(2), qt.identity(2))
    x2 = qt.tensor(qt.identity(2), qt.sigmax(), qt.identity(2))
    if pulse[0] == 'CDD3' or pulse[0] == 'CDD1':
        for i in range(len(sol)):
            sol[i] = x1*sol[i]*x1
    if pulse[1] == 'CDD3' or pulse[1] == 'CDD1':
        for i in range(len(sol)):
            sol[i] = x2*sol[i]*x2
    return sol

def C_12_0_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))
    rho = jnp.array(rho)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EX1X2 = E_XX(sol, a_m, delta)
    EX1X2 = E_XX_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EY1Y2 = E_YY(sol, a_m, delta)
    EY1Y2 = E_YY_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EX1Y2 = E_XY(sol, a_m, delta)
    EX1Y2 = E_XY_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EY1X2 = E_YX(sol, a_m, delta)
    EY1X2 = E_YX_hat(sol, a_m, delta)
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
    rho0 = make_init_state(a_sp, c, state = state)
    rho_B = 0.5*qt.identity(2)#qt.basis(2, 0) * qt.basis(2, 0).dag()
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    C_12_0_MT = Parallel(n_jobs=2)(delayed(C_12_0_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, rho,
                                                         c_times[i], n_shots=n_shots, M=M, a_m=a_m,
                                                         delta=delta) for i in range(np.size(c_times)))
    return C_12_0_MT


def C_12_12_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EX1X2 = E_XX(sol, a_m, delta)
    EX1X2 = E_XX_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EY1Y2 = E_YY(sol, a_m, delta)
    EY1Y2 = E_YY_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EX1Y2 = E_XY(sol, a_m, delta)
    EX1Y2 = E_XY_hat(sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
    sol = frame_correct(sol, pulse)
    # EY1X2 = E_YX(sol, a_m, delta)
    EY1X2 = E_YX_hat(sol, a_m, delta)
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
    rho_B = 0.5*qt.identity(2) #qt.basis(2, 0) * qt.basis(2, 0).dag()
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    return Parallel(n_jobs=2)(delayed(C_12_12_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, c_times[i],
                                                           n_shots=n_shots, M=M, a_m=a_m,
                                                           delta=delta) for i in range(np.size(c_times)))


def C_a_b_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    rhop = rho[0]
    rhom = rho[1]
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    sol = frame_correct(sol, pulse)
    # EXlp = E_X(l, sol, a_m, delta)
    EXlp = E_X_hat(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    sol = frame_correct(sol, pulse)
    # EYlp = E_Y(l, sol, a_m, delta)
    EYlp = E_Y_hat(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    sol = frame_correct(sol, pulse)
    Ap = A([EXlp, EYlp])
    # EXlm = E_X(l, sol, a_m, delta)
    EXlm = E_X_hat(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    sol = frame_correct(sol, pulse)
    # EYlm = E_Y(l, sol, a_m, delta)
    EYlm = E_Y_hat(l, sol, a_m, delta)
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
    rho_B = 0.5*qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rho0m = make_init_state(a_sp, c, state = state_m)
    rho_B = 0.5*qt.identity(2)
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    return Parallel(n_jobs=2)(delayed(C_a_b_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, [rhop, rhom], c_times[i],
                                                   n_shots=n_shots, M=M, a_m=a_m, l=l,
                                                   delta=delta) for i in range(np.size(c_times)))

def C_a_0_MT_i(solver_ftn, t_b, pulse, noise_mats, t_vec, rho, ct, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    a_m = kwargs.get('a_m')
    l = kwargs.get('l')
    delta = kwargs.get('delta')
    rhop = rho[0]
    rhom = rho[1]
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    sol = frame_correct(sol, pulse)
    # EXlp = E_X(l, sol, a_m, delta)
    EXlp = E_X_hat(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhop, n_shots)
    sol = frame_correct(sol, pulse)
    # EYlp = E_Y(l, sol, a_m, delta)
    EYlp = E_Y_hat(l, sol, a_m, delta)
    Ap = A([EXlp, EYlp])
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    sol = frame_correct(sol, pulse)
    # EXlm = E_X(l, sol, a_m, delta)
    EXlm = E_X_hat(l, sol, a_m, delta)
    sol = solver_ftn(y_uv, noise_mats, t_vec, rhom, n_shots)
    sol = frame_correct(sol, pulse)
    # EYlm = E_Y(l, sol, a_m, delta)
    EYlm = E_Y_hat(l, sol, a_m, delta)
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
    rho0p = make_init_state(a_sp, c, state = state_p)
    rho_B = 0.5*qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rho0m = make_init_state(a_sp, c, state = state_m)
    rho_B = 0.5*qt.identity(2)
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    return Parallel(n_jobs=2)(delayed(C_a_0_MT_i)(solver_ftn, t_b, pulse, noise_mats, t_vec, [rhop, rhom], c_times[i],
                                                   n_shots=n_shots, M=M, a_m=a_m, l=l,
                                                   delta=delta) for i in range(np.size(c_times)))


# print(POVMs([0.9, 0.9], [0.01, 0.02]))
# print(np.array([[(i,j) for j in range(4)] for i in range(4)]))
