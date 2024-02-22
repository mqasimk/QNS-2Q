from trajectories import make_noise_mat_arr
import numpy as np
from observables import (make_C_12_0_MT,
                         make_C_12_12_MT,
                         make_C_a_b_MT,
                         make_C_a_0_MT)
from trajectories import solver_prop
import matplotlib.pyplot as plt
import qutip as qt

#time the code
import time
start_time = time.time()


def S_11(w):
    tc=0.5/(1*10**6)
    S0 = 10**3
    w0=2*10**6
    return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))

def S_12(w):
    tc=0.5/(1*10**6)
    S0 = 10**3
    w0=1*10**6
    return S0*(1/(1+(tc**2)*(np.abs(w)-w0)**2))

T = 10**(-5)
M = 10
t_b = np.linspace(0, T, 100)
truncate = 8
wmax = 2*np.pi*truncate/T
w_grain = 1000
spec_vec = [S_11, S_12]
a_sp = [1., 1.]
c = [np.array(0.+0.*1j), np.array(0.+0.*1j)]
a_m = [1., 1.]
delta = [0., 0.]
gamma = 0
gamma_12 = 0
t_vec = np.linspace(0, M*T, M*np.size(t_b))
c_times = [T/n for n in range(1, truncate+1)]
n_shots = 100

# make noise matrices
noise_mats = make_noise_mat_arr('load', spec_vec=spec_vec, t_vec=t_vec, w_grain=w_grain, wmax=wmax, truncate=truncate,
                                gamma=gamma, gamma_12=gamma_12)

# Step 1.1) CPMG on both qubits and evaluate $$C_{12,0}^{1,k}(MT)$$
pulse = ['CPMG', 'CPMG']
C_12_0_MT_1 = make_C_12_0_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c)
print("Experiment 1.1 complete")
# Step 1.2) CDD3 on qubit 1 and CPMG on qubit 2 and evaluate $$C_{12,0}^{2,k}(MT)$$
pulse = ['CDD3', 'CPMG']
C_12_0_MT_2 = make_C_12_0_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c)
print("Experiment 1.2 complete")
# Step 1.3) CPMG on qubit 1 and CDD3 on qubit 2 and evaluate $$C_{12,0}^{3,k}(MT)$$
pulse = ['CPMG', 'CDD3']
C_12_0_MT_3 = make_C_12_0_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c)
print("Experiment 1.3 complete")
# Step 2.1) CDD3 on qubit 1 and CDD1 on qubit 2 and evaluate $$C_{12,12}^{1,k}(MT)$$
pulse = ['CDD3', 'CDD1']
C_12_12_MT_1 = make_C_12_12_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                               delta=delta, state='pp', a_sp=a_sp, c=c)
print("Experiment 2.1 complete")
# Step 2.2) CDD3 on qubit 1 and CP on qubit 2 and evaluate $$C_{12,12}^{2,k}(MT)$$
pulse = ['CDD3', 'CPMG']
C_12_12_MT_2 = make_C_12_12_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                               delta=delta, state='pp', a_sp=a_sp, c=c)
print("Experiment 2.2 complete")
# Step 3.1) CDD3 on qubit 1 and CPMG on qubit 2 and evaluate $$C_{1,0}^{1,k}(MT)$$
pulse = ['CDD3', 'CPMG']
C_1_0_MT_1 = make_C_a_0_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c)
print("Experiment 3.1 complete")
# Step 3.2) CDD3 on qubit 1 and CPMG on qubit 2 and evaluate $$C_{2,0}^{2,k}(MT)$$
pulse = ['CDD3', 'CPMG']
C_2_0_MT_1 = make_C_a_0_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c)
print("Experiment 3.2 complete")
# Step 4.1) CPMG on both qubits and evaluate $$C_{1,2}^{1,k}(MT)$$
pulse = ['CPMG', 'CPMG']
C_1_2_MT_1 = make_C_a_b_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c)
print("Experiment 4.1 complete")
# Step 4.2) CDD3 on qubit 1 and CDD3 on qubit 2 and evaluate $$C_{1,2}^{2,k}(MT)$$
pulse = ['CDD3', 'CDD3']
C_1_2_MT_2 = make_C_a_b_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c)
print("Experiment 4.2 complete")
# Step 5.1) CPMG on both qubits and evaluate $$C_{2,1}^{1,k}(MT)$$
pulse = ['CPMG', 'CPMG']
C_2_1_MT_1 = make_C_a_b_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c)
print("Experiment 5.1 complete")
# Step 5.2) CDD3 on qubit 1 and CDD3 on qubit 2 and evaluate $$C_{2,1}^{2,k}(MT)$$
pulse = ['CDD3', 'CDD3']
C_2_1_MT_2 = make_C_a_b_MT(solver_prop, pulse, noise_mats, t_vec, c_times, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c)
print("Experiment 5.2 complete")

print("--- %s seconds ---" % (time.time() - start_time))