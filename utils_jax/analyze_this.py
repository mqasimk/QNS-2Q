import numpy as np
import os
from reconstruction import recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12
from matplotlib import pyplot as plt
from matplotlib import gridspec
def S_11(w):
    tc=1/(1*10**6)
    S0 = 1e3
    w0=0*10**6
    return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))

def S_12(w):
    tc=0.5/(1*10**6)
    S0 = 1e3
    w0=3*10**6
    return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))

# load the variables
parent_dir = os.pardir
fname = "Run_jax_3"
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
C_1_0_MT_1 = observables['C_1_0_MT_1']
C_2_0_MT_1 = observables['C_2_0_MT_1']
# C_1_2_MT_1 = observables['C_1_2_MT_1']
# C_1_2_MT_2 = observables['C_1_2_MT_2']
# C_2_1_MT_1 = observables['C_2_1_MT_1']
# C_2_1_MT_2 = observables['C_2_1_MT_2']

w = np.linspace(0, wmax, w_grain)
wk = np.array([2*np.pi*(n+1)/T for n in range(truncate)])
S_11_k = recon_S_11([C_12_0_MT_1, C_12_0_MT_2], c_times=c_times, M=M, T=T)
S_22_k = recon_S_22([C_12_0_MT_1, C_12_0_MT_3], c_times=c_times, M=M, T=T)
S_1_2_k = recon_S_1_2([C_12_12_MT_1, C_12_12_MT_2], c_times=c_times, M=M, T=T)
S_12_12_k = recon_S_12_12([C_1_0_MT_1, C_2_0_MT_1, C_12_0_MT_2], c_times=c_times, M=M, T=T)

plt.plot(4*wk/1e6, -S_12_12_k/1e3, 'r.')
plt.plot(4*w/1e6, S_12(4*w)/1e3, 'r--')
plt.show()

plt.plot(wk/1e6, S_11_k/1e3, 'r.')
plt.plot(w/1e6, S_11(w)/1e3, 'r--')
plt.show()

plt.plot(wk/1e6, S_1_2_k/1e3, 'r.')
plt.show()

fig = gridspec.GridSpec(4, 1)
fig.update(hspace=1)
ax1 = plt.subplot(fig[0, 0])
ax2 = plt.subplot(fig[1, 0])
ax3 = plt.subplot(fig[2, 0])
ax4 = plt.subplot(fig[3, 0])

ax1.plot(wk/1e6, S_11_k/1e3, 'r.')
ax1.plot(w/1e6, S_11(w)/1e3, 'r--')
ax1.set_ylabel(r'$S^+_{1,1}(\omega_k)$')
# remove x ticks
ax1.set_xticks([])
ax2.plot(wk/1e6, S_22_k/1e3, 'r.')
ax2.plot(w/1e6, S_11(w)/1e3, 'r--')
ax2.set_ylabel(r'$S^+_{2,2}(\omega_k)$')
ax2.set_xticks([])
ax3.plot(wk/1e6, S_12_12_k/1e3, 'r.')
ax3.set_ylabel(r'$S^+_{12,12}(4\omega_k)$')
# ax3.set_ylim(0,1)
ax3.set_xticks([])
ax4.plot(wk/1e6, np.real(S_1_2_k)/1e3, 'r.')
ax4.plot(w/1e6, np.real(S_11(w)*np.exp(-1j*w*gamma))/1e3, 'r--')
ax4.plot(wk/1e6, np.imag(S_1_2_k)/1e3, 'b.')
ax4.plot(w/1e6, np.imag(S_11(w)*np.exp(-1j*w*gamma))/1e3, 'b--')
ax4.set_ylabel(r'$S^+_{1,2}(\omega_k)$')
ax4.set_xlabel(r'$\omega$(MHz)')
plt.show()


