import jax.scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from trajectories import make_y
import matplotlib.gridspec as gridspec
from trajectories import make_noise_mat_arr, make_noise_traj, make_Hamiltonian
from observables import E_X
import jax.numpy as jnp
import scipy

def S_11(w):
    tc=1/(1*10**6)
    S0 = 1e3
    w0=0*10**6
    return S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))

def S_12(w):
    tc=1/(1*10**6)
    S0 = 1
    w0=0*10**6
    return 0.#S0*(1/(1+(tc**2)*(jnp.abs(w)-w0)**2))

def ff(y, t, w):
    return np.trapz(np.exp(1j*w*t)*y, t)

T = 10**(-5)
ct = T
trunc = 8
M = 20

wmax = 2*np.pi*trunc/T
wk = np.array([2*np.pi*(n+1)/10**(-5) for n in range(trunc)])

#
t_grain = int(1e3)
tb = np.linspace(0, T, t_grain)
t_vec = np.linspace(0, M*T, M*np.size(tb))
w_grain = int(1e3)
w = np.linspace(0, wk[-1], w_grain)
ct_arr = np.array([T/n for n in range(1, trunc+1)])
y_base = np.array([make_y(tb, ['CPMG', 'CDD3'], ctime=ct, M=1) for ct in ct_arr])
#
S_k = np.array([S_11(w) for w in wk])
c_times = np.array([T/n for n in range(1, trunc+1)])
U = np.zeros((np.size(ct_arr), np.size(ct_arr)), dtype=np.complex128)
y_arr = [make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[n], M=1) for n in range(np.size(c_times))]
for i in range(np.size(c_times)):
    n = int(T/c_times[i])
    for j in range(np.size(c_times)):
        U[i, j] = ((2*M/T)*(np.absolute(ff(y_arr[i][0,0], tb, wk[j]))**2 - np.absolute(ff(y_arr[i][1,1], tb, wk[j]))**2)).round(10)

print(np.linalg.cond(U))
# print((np.linalg.inv(U)@U))

import os

pdir = os.pardir
fname = "Run_jax_debug_1"
observed = np.load(os.path.join(pdir, fname, "results.npz"))
C_12_0_MT_1 = observed['C_12_0_MT_1']
C_12_0_MT_2 = observed['C_12_0_MT_2']
C_12_0_MT_3 = observed['C_12_0_MT_3']

# plt.plot(wk, C_12_0_MT_1, 'r.')
plt.plot(wk, np.matmul(np.linalg.inv(U), (C_12_0_MT_1 - C_12_0_MT_2)), 'r.')
plt.plot(wk, np.matmul(np.linalg.inv(U), (C_12_0_MT_1 - C_12_0_MT_3)), 'bx')
plt.plot(w, S_11(w), 'g-')
plt.show()
# ax2.plot(wk, np.matmul(U, S_k), 'gx')
# ind = 0
# plt.plot(tb, y_base[ind][0, 0])
# plt.plot(tb/2, y_base[ind][0, 0])
# plt.show()
# yax = np.zeros(np.size(wk))
# plt.plot(w, [np.absolute(ff(y_base[0][0, 0], tb, wi))**2 for wi in w])
# plt.plot(w, [np.absolute(ff(y_base[0][0, 0], tb/2, wi))**2 for wi in w])
# plt.plot(w, [np.absolute(ff(y_base[0][0, 0], tb/3, wi))**2 for wi in w])
# plt.plot(wk, yax, 'r.')
# plt.show()
# print(U)
# plt.plot(wk, U@S_k)
# plt.show()
# S_11_k = np.linalg.inv(U)@np.reshape(C_12_0_MT_1-C_12_0_MT_2, (C_12_0_MT_1.shape[0], 1))
# plt.plot(wk, S_11_k)
# plt.plot(wk, S_k)
# plt.show()
# plt.plot(wk, [ff(y_arr[3][0,0], tb, wk[k]) for k in range(np.size(wk))], 'r.')
# plt.show()

# fig = gridspec.GridSpec(8, 1)
# ax1 = plt.subplot(fig[0,0])
# ax2 = plt.subplot(fig[1,0])
# ax3 = plt.subplot(fig[2,0])
# ax4 = plt.subplot(fig[3,0])
# ax5 = plt.subplot(fig[4,0])
# ax6 = plt.subplot(fig[5,0])
# ax7 = plt.subplot(fig[6,0])
# ax8 = plt.subplot(fig[7,0])
#
# ax1.plot(w, [ff(y_base[0][0, 0], tb, wi) for wi in w])
# ax2.plot(w, [ff(y_base[1][0, 0], tb, wi) for wi in w])
# ax3.plot(w, [ff(y_base[2][0, 0], tb, wi) for wi in w])
# ax4.plot(w, [ff(y_base[3][0, 0], tb, wi) for wi in w])
# ax5.plot(w, [ff(y_base[4][0, 0], tb, wi) for wi in w])
# ax6.plot(w, [ff(y_base[5][0, 0], tb, wi) for wi in w])
# ax7.plot(w, [ff(y_base[6][0, 0], tb, wi) for wi in w])
# ax8.plot(w, [ff(y_base[7][0, 0], tb, wi) for wi in w])
# # plt.show()
#
# print(np.linalg.inv(U)@U)

# y_arr = np.array([make_y(tb, ['CPMG', 'CDD3'], ctime=c_times[n], M=M)[0, 0] for n in range(np.size(c_times))])
# ff_arr = np.array([np.array([ff(y_arr[n], t_vec, w[k]) for k in range(np.size(w))]) for n in range(np.size(c_times))])
# S_w = np.array([S_11(wi) for wi in w])
# C12_0 = np.array([(1/(2*np.pi))*np.trapz(S_w*np.abs(ff_arr[n])**2, w) for n in range(np.size(c_times))])
# np.save(os.path.join(pdir, fname, "C12_0.npy"), C12_0)
# #
# C12_0 = np.load(os.path.join(pdir, fname, "C12_0.npy"))
# plt.plot(wk, np.linalg.inv(U)@C12_0, 'r.')
# plt.plot(wk, np.linalg.inv(U)@C_12_0_MT_1/5, 'b.')
# plt.plot(w, S_11(w), 'g-')
# plt.plot(wk, C12_0, 'r.')
# plt.plot(wk, C_12_0_MT_1, 'b.')
# plt.show()

# noise_mats = np.array(make_noise_mat_arr('make', spec_vec=[S_11, S_12], t_vec=t_vec, w_grain=w_grain, wmax=wmax, truncate=trunc, gamma=0., gamma_12=0.))
# dw = wmax/w_grain
# size_w = 2*w_grain
# size_t = np.size(t_vec)
# S = np.zeros((np.size(t_vec), size_w))
# C = np.zeros((np.size(t_vec), size_w))
# for i in range(size_t):
#     if i%10 == 0:
#         print(i)
#     for j in range(size_w):
#         S[i, j] = np.sqrt(dw*S_11(j*dw)/np.pi)*np.sin(j*dw*(t_vec[i]))
#         C[i, j] = np.sqrt(dw*S_11(j*dw)/np.pi)*np.cos(j*dw*(t_vec[i]))
# np.savez(os.path.join(pdir, fname, "noise_mats_og.npz"), S=S, C=C)

# S = np.load(os.path.join(pdir, fname, "noise_mats_og.npz"))['S']
# C = np.load(os.path.join(pdir, fname, "noise_mats_og.npz"))['C']
#
# plt.plot(t_vec, S[99, :])
# plt.plot(t_vec, noise_mats[0, 0, 99, :])
# plt.show()
# y = make_y(tb, ['CPMG', 'CDD3'], ctime=T/5, M=20)
# bt = make_noise_traj(noise_mats[0, 0], noise_mats[0, 1])
# h = make_Hamiltonian(y, jnp.array([bt, bt, bt, bt]))
# print(h.shape)
# int = jax.scipy.integrate.trapezoid(h, t_vec, axis=0)
# U=jax.scipy.linalg.expm(-1j*int)
#
# rho = jnp.kron(jnp.kron(0.5*(jnp.array([[1, 0], [0, 1]])+jnp.array([[0, 1], [1, 0]])), 0.5*(jnp.array([[1, 0], [0, 1]])+jnp.array([[0, 1], [1, 0]]))), 0.5*jnp.array([[1, 0], [0, 1]]))
# ex1 = jnp.kron(jnp.kron(jnp.array([[0, 1], [1, 0]]), jnp.array([[1, 0], [0, 1]])), jnp.array([[1, 0], [0, 1]]))
# res = (U@rho@U.conjugate().transpose()@ex1).trace().real
# print(res)