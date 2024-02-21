import numpy as np
import qutip as qt
from trajectories import solver, make_y

def E_X(qubit, state, a_m, delta):
    rho_s = []
    if qubit == 1:
        for i in range(state.shape[0]):
            rho_s.append(state[i].ptrace(0))
    else:
        for i in range(state.shape[0]):
            rho_s.append(state[i].ptrace(1))
    return np.sum(np.array([(rho_s[i]*qt.sigmax()).tr() for i in range(state.shape[0])]))/state.shape[0]

def E_Y(qubit, state, a_m, delta):
    rho_s = []
    if qubit == 1:
        for i in range(state.shape[0]):
            rho_s.append(state[i].ptrace(0))
    else:
        for i in range(state.shape[0]):
            rho_s.append(state[i].ptrace(1))
    return np.sum(np.array([(rho_s[i]*qt.sigmay()).tr() for i in range(state.shape[0])]))/state.shape[0]

def E_XX(state, a_m, delta):
    rho_s = []
    for i in range(state.shape[0]):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmax(),qt.sigmax())).tr() for i in range(state.shape[0])]))/state.shape[0]

def E_XY(state, a_m, delta):
    rho_s = []
    for i in range(state.shape[0]):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmax(),qt.sigmay())).tr() for i in range(state.shape[0])]))/state.shape[0]

def E_YX(state, a_m, delta):
    rho_s = []
    for i in range(state.shape[0]):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmay(),qt.sigmax())).tr() for i in range(state.shape[0])]))/state.shape[0]

def E_YY(state, a_m, delta):
    rho_s = []
    for i in range(state.shape[0]):
        rho_s.append(state[i].ptrace([0,1]))
    return np.sum(np.array([(rho_s[i]*qt.tensor(qt.sigmay(),qt.sigmay())).tr() for i in range(state.shape[0])]))/state.shape[0]

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

def make_C_12_0_MT(pulse, noise_mats, t_vec, c_times, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    C_12_0_MT = np.zeros(np.size(c_times))
    for i in range(np.size(c_times)):
        y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EX1X2 = E_XX(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EY1Y2 = E_YY(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EX1Y2 = E_XY(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EY1X2 = E_YX(state_vec, a_m, delta)
        C_12_0_MT[i] = D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) + D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2))
    return C_12_0_MT

def make_C_12_12_MT(pulse, noise_mats, t_vec, c_times, **kwargs):
    n_shots = kwargs.get('n_shots')
    M = kwargs.get('M')
    t_b = kwargs.get('t_b')
    a_m = kwargs.get('a_m')
    delta = kwargs.get('delta')
    C_12_12_MT = np.zeros(np.size(c_times))
    for i in range(np.size(c_times)):
        y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EX1X2 = E_XX(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EY1Y2 = E_YY(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EX1Y2 = E_XY(state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, **kwargs)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EY1X2 = E_YX(state_vec, a_m, delta)
        C_12_12_MT[i] = D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2)) - D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2))
    return C_12_12_MT

def make_C_a_b_MT(pulse, noise_mats, t_vec, c_times, **kwargs):
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
    C_a_b_MT = np.zeros(np.size(c_times))
    for i in range(np.size(c_times)):
        y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
        sol = solver(y_uv, noise_mats, t_vec, state=state_p, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EXlp = E_X(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_p, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EYlp = E_Y(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_m, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        Ap = A([EXlp, EYlp])
        EXlm = E_X(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_m, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EYlm = E_Y(l, state_vec, a_m, delta)
        Am = A([EXlm, EYlm])
        C_a_b_MT[i] = Ap + Am
    return C_a_b_MT

def make_C_a_0_MT(pulse, noise_mats, t_vec, c_times, **kwargs):
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
    for i in range(np.size(c_times)):
        y_uv = make_y(t_b, pulse, ctime=c_times[i], M=M)
        sol = solver(y_uv, noise_mats, t_vec, state=state_p, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EXlp = E_X(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_p, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EYlp = E_Y(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_m, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        Ap = A([EXlp, EYlp])
        EXlm = E_X(l, state_vec, a_m, delta)
        sol = solver(y_uv, noise_mats, t_vec, state=state_m, a_sp=a_sp, c=c, n_shots=n_shots)
        state_vec = np.array([sol[i].states[-1] for i in range(n_shots)])
        EYlm = E_Y(l, state_vec, a_m, delta)
        Am = A([EXlm, EYlm])
        C_a_0_MT[i] = Ap - Am
    return C_a_0_MT
