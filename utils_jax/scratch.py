import jax.scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from trajectories import make_y
import matplotlib.gridspec as gridspec
from reconstruction import recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12
from observables import E_X
import jax.numpy as jnp
import scipy
import os

def L(w, w0, tc):
    return 1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2)

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

def S_1212(w):
    tc=1/(1*10**6)
    S0 = 2e3
    w0=0*10**6
    return 0.5*S0*L(w, w0, tc*0.5)

def ff(y, t, w):
    return np.trapz(np.exp(1j*w*t)*y, t)

parent_dir = os.pardir
fname = "Run_jax_12_SPAM"
path = os.path.join(parent_dir, fname)
params = np.load(os.path.join(path, "params.npz"))
t_vec = params['t_vec']
w_grain = params['w_grain']
wmax = params['wmax']
truncate = params['truncate']
gamma = params['gamma']
gamma_12 = params['gamma_12']
t_b = params['t_b']
a_m = params['a_m']
delta = params['delta']
c_times = params['c_times']
n_shots = params['n_shots']
M = params['M']
a_sp = params['a_sp']
c = params['c']
T = params['T']
observables = np.load(os.path.join(path, "results.npz"))
C_12_0_MT_1 = observables['C_12_0_MT_1']
C_12_0_MT_2 = observables['C_12_0_MT_2']
C_12_0_MT_3 = observables['C_12_0_MT_3']
C_12_12_MT_1 = observables['C_12_12_MT_1']
C_12_12_MT_2 = observables['C_12_12_MT_2']
C_1_0_MT_1 = observables['C_1_0_MT_1']
C_2_0_MT_1 = observables['C_2_0_MT_1']
C_12_0_MT_4 = observables['C_12_0_MT_4']
trunc = 36
wmax = 2*np.pi*trunc/T
wk = np.array([2*np.pi*(n+1)/T for n in range(trunc)])

#
t_grain = int(1e4)
# tb = np.linspace(0, T, t_grain)
M=3
t_vec = np.linspace(0, M*T, M*np.size(t_b))
w_grain = int(1e3)
w = np.linspace(0, wk[-1], w_grain)
ct_arr = np.array([T/n for n in range(1, trunc+1)])
y_base = np.array([make_y(t_b, ['CPMG', 'CDD1'], ctime=ct, M=1) for ct in ct_arr])
y_test = np.array([make_y(t_b, ['CPMG', 'CDD1'], ctime=ct/2, M=1) for ct in ct_arr])

# wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
# U = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
# tb = np.linspace(0, T, 10**4)
# y_arr = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=c_times[i], M=1) for i in range(np.size(c_times))])
# for i in range(np.size(c_times)):
#     for j in range(np.size(c_times)):
#         U[i, j] = ((M/T)*np.abs(ff(y_arr[i][0, 0], tb, wk[j]))**2).round(10)


plt.plot(t_b, y_test[0][1, 1]*y_base[0][1, 1], '--')
plt.plot(t_b, y_base[0][0, 0]+3, '--')
plt.show()
print("End")


