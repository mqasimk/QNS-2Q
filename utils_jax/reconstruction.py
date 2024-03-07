import numpy as np
from trajectories import make_y
from matplotlib import pyplot as plt

def f1_cpmg(ct, T, w):
    y = make_y(np.linspace(0, T, 10**5), ['CPMG', 'CPMG'], ctime=ct, M=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapz(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def f1_fid(ct, T, w):
    t_vec = np.linspace(0, T, 10**5)
    return np.trapz(np.exp(1j*w*t_vec), t_vec)

def f1_cdd1(ct, T, w):
    y = make_y(np.linspace(0, T, 10**5), ['CDD1', 'CDD1'], ctime=ct, M=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapz(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def f1_cdd3(ct, T, w):
    if w == 0:
        return 0
    y = make_y(np.linspace(0, T, 10**5), ['CDD3', 'CDD3'], ctime=ct, M=1)
    t_vec = np.linspace(0, T, 10**5)
    return np.trapz(y[0, 0]*np.exp(1j*w*t_vec), t_vec)

def Gp(ffs, w, T, ct):
    return ffs[0](ct, T, w)*ffs[1](ct, T, -w)

def recon_S_11(coefs, **kwargs):
    c_times = kwargs.get('c_times')
    M = kwargs.get('M')
    T = kwargs.get('T')
    C_12_0_MT_1 = coefs[0]
    C_12_0_MT_2 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    U_1 = np.zeros((np.size(c_times), np.size(c_times)), dtype=np.complex128)
    for i in range(np.size(c_times)):
        n = int(T/c_times[i])
        for j in range(np.size(c_times)):
            U_1[i, j] = ((n*M/T**2)*(np.square(np.absolute(f1_cpmg(c_times[i], T, wk[j]))) - np.square(np.absolute(f1_cdd3(c_times[i], T, wk[j])))))
    # U = U.round(20)
    S_11_k = np.matmul(np.linalg.inv(U_1), np.reshape(C_12_0_MT_1-C_12_0_MT_2, (C_12_0_MT_1.shape[0], 1)))
    return S_11_k

def recon_S_22(coefs, **kwargs):
    c_times = kwargs.get('c_times')
    M = kwargs.get('M')
    T = kwargs.get('T')
    C_12_0_MT_1 = coefs[0]
    C_12_0_MT_3 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(1, np.size(c_times))])
    U = np.zeros((np.size(c_times), np.size(c_times)))
    for i in range(np.size(c_times)):
        n = int(T/c_times[i])
        for j in range(np.size(c_times)):
            U[i, j] = (n*M/T)*(Gp([f1_cpmg, f1_cpmg], wk[j], T, c_times[i]) - Gp([f1_cpmg, f1_cdd3], wk[j], T, c_times[i]))
    S_22_k = np.linalg.inv(U)@np.transpose(C_12_0_MT_1-C_12_0_MT_3)
    return S_22_k

def recon_S_1_2(coefs, **kwargs):
    c_times = kwargs.get('c_times')
    M = kwargs.get('M')
    T = kwargs.get('T')
    C_12_12_MT_1 = coefs[0]
    C_12_12_MT_2 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(1, np.size(c_times))])
    U_1 = np.zeros((np.size(c_times), np.size(c_times)))
    U_2 = np.zeros((np.size(c_times), np.size(c_times)))
    for i in range(np.size(c_times)):
        n = int(T/c_times[i])
        for j in range(np.size(c_times)):
            U_1[i, j] = (n*M/T)*(Gp([f1_cdd3, f1_cdd1], wk[j], T, c_times[i]))
            U_2[i, j] = (n*M/T)*(Gp([f1_cdd3, f1_cpmg], wk[j], T, c_times[i]))
    Re_S_1_2_k = np.linalg.inv(U_1)@np.transpose(C_12_12_MT_1)
    Im_S_1_2_k = np.linalg.inv(U_2)@np.transpose(C_12_12_MT_2)
    return Re_S_1_2_k, Im_S_1_2_k

def recon_S_12_12(coefs, **kwargs):
    c_times = kwargs.get('c_times')
    M = kwargs.get('M')
    T = kwargs.get('T')
    C_1_0_MT = coefs[0]
    C_2_0_MT = coefs[1]
    C_12_0_MT_2 = coefs[2]
    wk = np.array([2*np.pi*(n+1)/T for n in range(1, np.size(c_times))])
    U = np.zeros((np.size(c_times), np.size(c_times)))
    return


# T = 1e-5
# ct = T/16
# n = int(T/ct)
# M = 20
# t_b = np.linspace(0, T, 1000)
# t_vec = np.linspace(0, M*T, M*np.size(t_b))
# y = make_y(t_b, ['CDD3', 'CDD3'], ctime=ct, M=M)
# wk = np.array([2*np.pi*k/T for k in range(160)])
# plt.plot(wk, [int(T/ct)*(M/T)*(np.square(np.absolute(f1_cpmg(ct, T, wk[i]))) - np.square(np.absolute(f1_cdd3(ct, T, wk[i])))) for i in range(np.size(wk))])
# plt.show()
# plt.plot(wk, [int(T/ct)*(M/T)*np.square(np.absolute(f1_cdd3(ct, T, wk[i]))) for i in range(np.size(wk))])
# plt.plot(wk, [int(T/ct)*(M/T)*np.square(np.absolute(f1_cpmg(ct, T, wk[i]))) for i in range(np.size(wk))])
# plt.legend(['CDD3', 'CPMG'])
# plt.show()

# def S_11(w):
#     tc=1/(1*10**6)
#     S0 = 1
#     w0=4*10**6
#     return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))
#
# T = 1e-5
# wk = np.array([2*np.pi*(n+1)/T for n in range(16)])
# S_11_k = S_11(wk)
# w = np.linspace(0, 2*np.pi*16/T, 1000)
# plt.plot(wk, S_11_k, 'ro')
# plt.plot(w, S_11(w))
# plt.show()
# print(S_11_k[-1])
