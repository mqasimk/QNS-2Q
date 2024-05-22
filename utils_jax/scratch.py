import jax.scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from trajectories import make_y
import matplotlib.gridspec as gridspec
from trajectories import make_noise_mat_arr, make_noise_traj, make_Hamiltonian
from reconstruction import recon_S_11, recon_S_22, recon_S_1_2
from observables import E_X
import jax.numpy as jnp
import scipy

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

def ff(y, t, w):
    return np.trapz(np.exp(1j*w*t)*y, t)

T = 10**(-5)
ct = T
trunc = 10
M = 10

wmax = 2*np.pi*trunc/T
wk = np.array([2*np.pi*(n+1)/10**(-5) for n in range(trunc)])

#
t_grain = int(1e4)
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
        U[i, j] = ((M/T)*(np.absolute(ff(y_arr[i][0,0], tb, wk[j]))**2 - np.absolute(ff(y_arr[i][1,1], tb, wk[j]))**2)).round(10)
print(np.linalg.cond(U))
print(U.shape)
# print((np.linalg.inv(U)@U))
y_arr = np.array([make_y(tb, ['CDD3', 'CDD1'], ctime=ct, M=1) for ct in ct_arr])
U_1 = np.zeros((np.size(ct_arr), np.size(ct_arr)), dtype=np.complex128)
for i in range(np.size(c_times)):
    for j in range(np.size(c_times)):
        U_1[i, j] = ((2*M/T)*(ff(y_arr[i][0, 0], tb, wk[j])*ff(y_arr[i][1, 1], tb, -wk[j]))).round(10)
print(np.linalg.cond(U_1))

y_arr = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=ct, M=1) for ct in ct_arr])
U_2 = np.zeros((np.size(ct_arr), np.size(ct_arr)), dtype=np.complex128)
for i in range(np.size(c_times)):
    for j in range(np.size(c_times)):
        U_2[i, j] = ((2*M/T)*(ff(y_arr[i][0, 0], tb, wk[j])*ff(y_arr[i][1, 1], tb, -wk[j]))).round(10)
print(np.linalg.cond(U_2))

y_arr = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=ct, M=1) for ct in ct_arr])
U_3 = np.zeros((np.size(ct_arr), np.size(ct_arr)), dtype=np.complex128)
for i in range(np.size(c_times)):
    for j in range(np.size(c_times)):
        U_3[i, j] = ((M/T)*(np.square(np.absolute(ff(y_arr[i][2, 2], tb, 4*wk[j]))))).round(25)
print(np.linalg.cond(U_3))
# print((np.linalg.inv(U_3)@U_3).round(10))

# import os
#
# pdir = os.pardir
# fname = "Run_jax_3"
# observed = np.load(os.path.join(pdir, fname, "results.npz"))
# C_12_0_MT_1 = observed['C_12_0_MT_1']
# C_12_0_MT_2 = observed['C_12_0_MT_2']
# C_12_0_MT_3 = observed['C_12_0_MT_3']
# C_12_12_MT_1 = observed['C_12_12_MT_1']
# C_12_12_MT_2 = observed['C_12_12_MT_2']
# C_1_0_MT_1 = observed['C_1_0_MT_1']
# C_2_0_MT_1 = observed['C_2_0_MT_1']
# print(C_1_0_MT_1+C_2_0_MT_1-C_12_0_MT_2)
# yax = np.zeros(np.size(wk))
# plt.plot(4*wk, -np.linalg.inv(U_3)@np.real(C_1_0_MT_1+C_2_0_MT_1-C_12_0_MT_2)/np.sqrt(8), 'r.')
# plt.plot(4*w, S_12(4*w), 'r--')
# plt.plot(4*wk, S_12(4*wk), 'k.')
# plt.xlim([0, 4*wk[-1]])
# plt.show()
# print(S_12(4*wk)-(-np.linalg.inv(U_3)@(C_1_0_MT_1+C_2_0_MT_1-C_12_0_MT_2)))
# print(recon_S_1_2([C_12_12_MT_1, C_12_12_MT_2], M=M, T=T, c_times=c_times))
# ax2.plot(wk, np.matmul(U, S_k), 'gx')
# ind = 0
# plt.plot(tb, y_base[ind][0, 0])
# plt.plot(tb/2, y_base[ind][0, 0])
# plt.show()
y_base = np.array([make_y(tb, ['CDD1', 'CPMG'], ctime=ct, M=1) for ct in ct_arr])
yax = np.zeros(np.size(wk))
plt.plot(w, [np.real(ff(y_base[4][0, 0], tb, wi)*ff(y_base[2][0, 0], tb, -wi)) for wi in w])
plt.plot(w, [np.imag(ff(y_base[4][0, 0], tb, wi)*ff(y_base[2][0, 0], tb, -wi)) for wi in w])
plt.plot(w, [np.real(ff(y_base[4][1, 1], tb, wi)*ff(y_base[2][1, 1], tb, -wi)) for wi in w])
plt.plot(w, [np.imag(ff(y_base[4][1, 1], tb, wi)*ff(y_base[2][1, 1], tb, -wi)) for wi in w])
plt.plot(w, [np.real(ff(y_base[4][2, 2], tb, wi)*ff(y_base[2][2, 2], tb, -wi)) for wi in w])
plt.plot(w, [np.imag(ff(y_base[4][2, 2], tb, wi)*ff(y_base[2][2, 2], tb, -wi)) for wi in w])
plt.plot(wk, yax, 'r.')
plt.show()
plt.plot(w, S_12(w))
plt.plot(w, S_11(w))
plt.show()
# y_base = np.array([make_y(tb, ['CDD3', 'CPMG'], ctime=ct, M=1) for ct in ct_arr])
# yax = np.zeros(np.size(wk))
# ctn = [2*T/n for n in range(1, (trunc+1))]
# y1_arr = np.array([make_y(tb, ['CPMG', 'CDD3'], ctime=ct, M=1) for ct in ct_arr])
# ct_arr1 = np.array([2*np.pi*(n+1)/T for n in range(32)])
# y2_arr = np.array([make_y(tb, ['CDD1', 'CDD3'], ctime=ct, M=1) for ct in ct_arr])
# g11_1212 = np.array([np.abs(ff(y1_arr[2][0, 0]*y1_arr[2][1, 1], tb, 4*wi))**2 for wi in w])
# g11_1212_1 = np.array([np.abs(ff(y1_arr[8][0, 0], tb, wi))**2 - np.abs(ff(y1_arr[8][1, 1], tb, wi))**2 for wi in w])
# plt.plot(4*w, np.real(g11_1212), 'r')
# plt.plot(4*w, np.imag(g11_1212), 'b')
# plt.plot(w, g11_1212_1, 'g')
# plt.plot(4*w, np.imag(g11_1212_1), 'y--')
# plt.plot(4*w, S_12(4*w)/1e13, 'g--')
# plt.plot(w, [np.imag(ff(y_arr[0][0, 0]*y_arr[0][1, 1], tb, wi)) for wi in w], 'b')
# plt.plot(tb, y1_arr[1][0, 0]*y1_arr[1][1, 1], 'r')
# plt.plot(tb, y2_arr[7][0, 0], 'b--')
# plt.plot(w, [np.abs(ff(y1_arr[1][0, 0]*y1_arr[1][1, 1], tb, wi))**2 for wi in w], 'b')
# plt.plot(w, [np.abs(ff(y1_arr[3][0, 0], tb, wi))**2 for wi in w], 'r')
# plt.plot(wk, yax, 'k.')
# plt.show()

print([ 9.99840479e-01 -3.72529046e-09 + 1.57486654e-04 + 2.03773388e-06])