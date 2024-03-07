from matplotlib import pyplot as plt
import numpy as np
import os
from reconstruction import recon_S_11, recon_S_22

def S_11(w):
    tc=0.5/(1*10**6)
    S0 = 1
    w0=0*10**6
    return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))

# load the variables
parent_dir = os.pardir
fname = "Run_jax_debug_1"
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

# load the observables array
observables = np.load(os.path.join(path, "results.npz"))
C_12_0_MT_1 = observables['C_12_0_MT_1']
C_12_0_MT_2 = observables['C_12_0_MT_2']
C_12_0_MT_3 = observables['C_12_0_MT_3']
C_12_12_MT_1 = observables['C_12_12_MT_1']
C_12_12_MT_2 = observables['C_12_12_MT_2']

w = np.linspace(0.01, wmax, w_grain)
S_id = S_11(w)
wk = np.array([2*np.pi*n/T for n in range(1, truncate+1)])
S_11 = recon_S_11([C_12_0_MT_1, C_12_0_MT_2], M=M, T=T, c_times=c_times)
plt.plot(wk, S_11)
# plt.plot(w, S_id)
plt.legend(['Reconstructed', 'Ideal'])
plt.show()



