import numpy as np
from trajectories import make_y
from matplotlib import pyplot as plt

def f1_cpmg(ct, T, w):
    n = int(T/ct)
    lims = []
    for i in range(n):
        lims.append([[i*T/n, i*T/n + T/(4*n)], [i*T/n + T/(4*n), i*T/n + 3*T/(4*n)], [i*T/n + 3*T/(4*n), i*T/n + T/n]])
    lims = np.array(lims)
    ft = 0. + 0.*1j
    for i in range(n):
        for j in range(3):
            ft += -1j*((-1)**j)*(np.exp(1j*w*lims[i, j, 1])-np.exp(1j*w*lims[i, j, 0]))/w
    return ft

def f1_fid(ct, T, w):
    return 2.*np.exp(1j*w*T/2.)*np.sin(w*T/2.)/w

def f1_cdd1(ct, T, w):
    n = int(T/ct)
    lims = []
    for i in range(n):
        lims.append([[i*T/n, i*T/n + T/(2*n)], [i*T/n + T/(2*n), i*T/n + T/n]])
    lims = np.array(lims)
    ft = 0. + 0.*1j
    for i in range(n):
        for j in range(2):
            ft += -1j*((-1)**j)*(np.exp(1j*w*lims[i, j, 1])-np.exp(1j*w*lims[i, j, 0]))/w
    return ft

def f1_cdd3(ct, T, w):
    n = int(T/ct)
    lims = []
    for i in range(n):
        lims.append([[i*T/n, i*T/n + T/(8*n)], [i*T/n + T/(8*n), i*T/n + 3*T/(8*n)], [i*T/n + 3*T/(8*n), i*T/n + T/(2*n)],
                     [i*T/n + T/(2*n), i*T/n + 5*T/(8*n)], [i*T/n + 5*T/(8*n), i*T/n + 7*T/(8*n)], [i*T/n + 7*T/(8*n), i*T/n + T/n]])
    lims = np.array(lims)
    ft = 0. + 0.*1j
    for i in range(n):
        for j in range(6):
            ft += -1j*((-1)**j)*(np.exp(1j*w*lims[i, j, 1])-np.exp(1j*w*lims[i, j, 0]))/w
    return ft

def Gp(ffs, w, T, ct):
    return ffs[0](ct, T, w)*ffs[1](ct, T, -w)

def recon_S_11(coefs, **kwargs):
    c_times = kwargs.get('c_times')
    M = kwargs.get('M')
    T = kwargs.get('T')
    C_12_0_MT_1 = coefs[0]
    C_12_0_MT_2 = coefs[1]
    wk = np.array([2*np.pi*(n+1)/T for n in range(np.size(c_times))])
    U = np.zeros((np.size(c_times), np.size(c_times)))
    for i in range(np.size(c_times)):
        n = int(T/c_times[i])
        for j in range(np.size(c_times)):
            U[i, j] = (n*M/T)*(Gp([f1_cpmg, f1_cpmg], wk[j], T, c_times[i]) - Gp([f1_cdd3, f1_cpmg], wk[j], T, c_times[i]))
    S_11_k = np.linalg.inv(U)@np.transpose(C_12_0_MT_1-C_12_0_MT_2)
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


# T = 1
# ct = 1/20
# n = int(T/ct)
# M = 20
# t_b = np.linspace(0, T, 1000)
# t_vec = np.linspace(0, M*T, M*np.size(t_b))
# y = make_y(t_b, ['CDD1', 'CPMG'], ctime=ct, M=M)
# wk = np.array([2*np.pi*k/T for k in range(160)])
# ft = []
# for w in wk:
#     ft.append(np.trapz(y[2,2]*np.exp(1j*w*t_vec), t_vec))
# ft = np.array(ft)
# plt.plot(wk, np.real(ft))
# plt.plot(wk, np.imag(ft))
# plt.show()