import jax.numpy as jnp
import numpy as np
from trajectories import solver_prop, make_init_state, make_noise_mat_arr, make_y
from observables import E_X_hat, E_XX_hat, frame_correct
from matplotlib import pyplot as plt
import qutip as qt

def S_11(w):
    tc=0.5/(1*10**6)
    S0 = 2e3
    w0=0*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))


def S_12(w):
    tc=0.5/(1*10**6)
    S0 = 3e3
    w0=2*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))

def state_evolution(y_uv, pulse, noise_mats, t_vec, rho0, n_shots, am, delta, l, obs):
    sol = solver_prop(y_uv, noise_mats, t_vec, rho0, n_shots)
    sol = frame_correct(sol, pulse)
    if obs == 'E_X':
        E = E_X_hat(sol, am, delta, l)
    elif obs == 'E_XX':
        E = E_XX_hat(sol, am, delta)
    else:
        raise Exception("Invalid observable")
    return E

T = 10**(-5)
M = 10
t_grain = int(1e3)
t_b = jnp.linspace(0, T, t_grain)
truncate = 12
wmax = 2*np.pi*truncate/T
w_grain = 1000
spec_vec = [S_11, S_12]
a_sp = np.array([1, 1])
c = np.array([np.array(0.+0.*1j), np.array(0.+0.*1j)])
a1 = 1
b1 = 1
a2 = 1
b2 = 1
a_m = np.array([a1+b1-1, a2+b2-1])
delta = np.array([a1-b1, a2-b2])
gamma = T/5
gamma_12 = 0.
t_vec = np.linspace(0, M*T, M*jnp.size(t_b))
c_times = jnp.array([T/n for n in range(1, truncate+1)])
n_shots = 1000
t_inds = np.linspace(0, np.size(t_vec)-1, 100).astype(int)

rho0 = make_init_state(a_sp, c, state='pp')
rho_B = 0.5*qt.identity(2)
rho = jnp.array((qt.tensor(rho0, rho_B)).full())
noise_mats = jnp.array(make_noise_mat_arr('save', spec_vec=spec_vec, t_vec=jnp.array(t_vec), w_grain=w_grain, wmax=wmax,
                                          truncate=truncate, gamma=gamma, gamma_12=gamma_12))
pulse = ['FID', 'FID']
E_FID = []
for i in t_inds[1:]:
    yuv = jnp.array(make_y(t_b, pulse, ctime=c_times[0], M=M))
    E_FID.append(state_evolution(yuv, pulse, noise_mats, jnp.array(t_vec[:i]), rho, n_shots, a_m, delta, 1, 'E_XX'))

pulse = ['CDD1', 'CPMG']
E_DD = []
for i in t_inds[1:]:
    yuv = jnp.array(make_y(t_b, pulse, ctime=c_times[4], M=M))
    E_DD.append(state_evolution(yuv, pulse, noise_mats, jnp.array(t_vec[:i]), rho, n_shots, a_m, delta, 1, 'E_XX'))

plt.plot(t_vec[t_inds[1:]], E_FID, 'r', label='FID')
plt.plot(t_vec[t_inds[1:]], E_DD, 'b', label='CDD1-CPMG')
plt.xlabel('Time (s)')
plt.ylabel(r'Expectation Value $E(X_1 X_2)$')
plt.title('Expectation Value vs Time under Dephasing noise')
plt.show()

