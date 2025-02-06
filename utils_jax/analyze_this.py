import numpy as np
import os
from reconstruction import recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12, recon_S_1_12, recon_S_2_12
from matplotlib import pyplot as plt
from spectra import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12


# load the variables
parent_dir = os.pardir
fname = "DraftRun_NoSPAM_hat"
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
C_12_0_MT_4 = observables['C_12_0_MT_4']
C_1_2_MT_1 = observables['C_1_2_MT_1']
C_1_2_MT_2 = observables['C_1_2_MT_2']
C_2_1_MT_1 = observables['C_2_1_MT_1']
C_2_1_MT_2 = observables['C_2_1_MT_2']


w = np.linspace(0.1, wmax, w_grain)
wk = np.array([2*np.pi*(n+1)/T for n in range(truncate)])


# Reconstruct the spectra
S_11_k = recon_S_11([C_12_0_MT_1, C_12_0_MT_2], c_times=c_times, M=M, T=T)
S_22_k = recon_S_22([C_12_0_MT_1, C_12_0_MT_3], c_times=c_times, M=M, T=T)
S_1_2_k = recon_S_1_2([C_12_12_MT_1, C_12_12_MT_2], c_times=c_times, M=M, T=T)
S_12_12_k = recon_S_12_12([C_1_0_MT_1, C_2_0_MT_1, C_12_0_MT_4], c_times=c_times, M=M, T=T)
S_1_12_k = recon_S_1_12([C_1_2_MT_1, C_1_2_MT_2], c_times=c_times, M=M, T=T)
S_2_12_k = recon_S_2_12([C_2_1_MT_1, C_2_1_MT_2], c_times=c_times, M=M, T=T)


# Plot the reconstruction
fig, axs = plt.subplots(2, 3, figsize=(16,9))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[0, 2]
ax4 = axs[1, 0]
ax5 = axs[1, 1]
ax6 = axs[1, 2]
lw = 1
legendfont = 12
xlabelfont = 16
ylabelfont = 16
tickfont = 12


yunits = 1
xunits = 1e6


w=w[100:]
ax1.plot(wk/xunits, S_11_k/yunits, 'r^')
ax1.plot(w/xunits, S_11(w)/yunits, 'r--', lw=1.5*lw)
ax1.set_ylabel(r'$S^+_{a,b}(\omega)$ Hz', fontsize=ylabelfont)
ax1.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax1.tick_params(direction='in')
ax1.tick_params(axis='both', labelsize=tickfont)
ax1.legend([r'$\hat{S}_{1,1}^+(\omega_k)$', r'$S_{1,1}^+(\omega)$'], fontsize=legendfont)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
ax2.plot(wk/xunits, S_22_k/yunits, 'r^')
ax2.plot(w/xunits, S_22(w)/yunits, 'r--', lw=1.5*lw)
ax2.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax2.tick_params(direction='in')
ax2.tick_params(axis='both', labelsize=tickfont)
ax2.legend([r'$\hat{S}_{2,2}^+(\omega_k)$', r'$S_{2,2}^+(\omega)$'], fontsize=legendfont)
# ax2.set_yscale('log')
# ax2.set_xscale('log')
ax3.plot(wk/xunits, S_12_12_k/yunits, 'r^')
ax3.plot(w/xunits, S_1212(w)/yunits, 'r--', lw=1.5*lw)
ax3.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax3.tick_params(direction='in')
ax3.tick_params(axis='both', labelsize=tickfont)
ax3.legend([r'$\hat{S}_{12,12}^+(\omega_k)$', r'$S_{12,12}^+(\omega)$'], fontsize=legendfont)
# ax3.set_yscale('log')
# ax3.set_xscale('log')
ax4.plot(wk/xunits, np.real(S_1_2_k)/yunits, 'r^')
ax4.plot(w/xunits, np.real(S_1_2(w, gamma))/yunits, 'r--', lw=1.5*lw)
ax4.plot(wk/xunits, np.imag(S_1_2_k)/yunits, 'b^')
ax4.plot(w/xunits, np.imag(S_1_2(w, gamma))/yunits, 'b--', lw=1.5*lw)
ax4.set_ylabel(r'$S^+_{a,b}(\omega)$ Hz', fontsize=ylabelfont)
ax4.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax4.tick_params(direction='in')
ax4.tick_params(axis='both', labelsize=tickfont)
ax4.legend([r'Re[$\hat{S}_{1,2}^+(\omega_k)$]', r'Re[$S_{1,2}^+(\omega)$]', r'Im[$\hat{S}_{1,2}^+(\omega_k)$]',
            r'Im[$S_{1,2}^+(\omega)$]'], fontsize=legendfont)
ax5.plot(wk/xunits, np.real(S_1_12_k)/yunits, 'r^')
ax5.plot(w/xunits, np.real(S_1_12(w, gamma_12))/yunits, 'r--', lw=1.5*lw)
ax5.plot(wk/xunits, np.imag(S_1_12_k)/yunits, 'b^')
ax5.plot(w/xunits, np.imag(S_1_12(w, gamma_12))/yunits, 'b--', lw=1.5*lw)
ax5.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax5.tick_params(direction='in')
ax5.tick_params(axis='both', labelsize=tickfont)
ax5.legend([r'Re[$\hat{S}_{1,12}^+(\omega_k)$]', r'Re[$S_{1,12}^+(\omega)$]', r'Im[$\hat{S}_{1,12}^+(\omega_k)$]',
            r'Im[$S_{1,12}^+(\omega)$]'], fontsize=legendfont)
ax6.plot(wk/xunits, np.real(S_2_12_k)/yunits, 'r^')
ax6.plot(w/xunits, np.real(S_2_12(w, gamma_12-gamma))/yunits, 'r--', lw=1.5*lw)
ax6.plot(wk/xunits, np.imag(S_2_12_k)/yunits, 'b^')
ax6.plot(w/xunits, np.imag(S_2_12(w, gamma_12-gamma))/yunits, 'b--', lw=1.5*lw)
ax6.set_xlabel(r'$\omega$(MHz)', fontsize=xlabelfont)
ax6.tick_params(direction='in')
ax6.tick_params(axis='both', labelsize=tickfont)
ax6.legend([r'Re[$\hat{S}_{2,12}^+(\omega_k)$]', r'Re[$S_{2,12}^+(\omega)$]', r'Im[$\hat{S}_{2,12}^+(\omega_k)$]',
            r'Im[$S_{2,12}^+(\omega)$]'], fontsize=legendfont)
plt.savefig(os.path.join(path, 'reconstruct.png'), dpi = 1200)
plt.show()


# Save the reconstruction to be used in the optimization
np.savez(os.path.join(path, "specs.npz"), S11=S_11_k, S22=S_22_k, S12=S_1_2_k, S1212=S_12_12_k, S112=S_1_12_k,
         S212=S_2_12_k)


