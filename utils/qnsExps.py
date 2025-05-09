from trajectories import make_noise_mat_arr
import numpy as np
from observables import (make_C_12_0_MT,
                         make_C_12_12_MT,
                         make_C_a_b_MT,
                         make_C_a_0_MT)
from trajectories import solver_prop#, solver_prop_noMats
import jax.numpy as jnp
import os
#time the code
from spectraIn import S_11, S_22, S_1212
import time


start_time = time.time()


T = 4.e-6
M = 12
t_grain = int(1e3)
t_b = jnp.linspace(0, T, t_grain)
truncate = 20
wmax = 2*np.pi*truncate/T
w_grain = 4000
w = jnp.linspace(0, wmax, w_grain)
spec_vec = [S_11, S_22, S_1212]


a_sp = np.array([0.97, 0.98])
c = np.array([np.array(0.+0.2*1j), np.array(0.-0.1*1j)])
a1 = 0.99
b1 = 0.97
a2 = 0.995
b2 = 0.98
a_m = np.array([a1+b1-1, a2+b2-1])
delta = np.array([a1-b1, a2-b2])
CM = jnp.kron(jnp.array([[0.5*(1+a_m[0]+delta[0]),0.5*(1-a_m[0]+delta[0])],[0.5*(1-a_m[0]-delta[0]),0.5*(1+a_m[0]-delta[0])]]),
                jnp.array([[0.5*(1+a_m[1]+delta[1]),0.5*(1-a_m[1]+delta[1])],[0.5*(1-a_m[1]-delta[1]),0.5*(1+a_m[1]-delta[1])]]))
# CM = jnp.eye(4)
spMit = False


gamma = T/7
gamma_12 = T/14
t_vec = jnp.linspace(0, M*T, M*jnp.size(t_b))
c_times = jnp.array([T/n for n in range(1, truncate+1)])
n_shots = 4000
# create a folder in the parent directory where the data will be stored
parent_dir = os.pardir
fname = "DraftRun_SPAM_hat"
if not os.path.exists(os.path.join(parent_dir, fname)):
    path = os.path.join(parent_dir, fname)
    os.mkdir(path)
else:
    path = os.path.join(parent_dir, fname)


# save all the variables in the folder
np.savez(os.path.join(path, "params.npz"), t_vec=t_vec, w_grain=w_grain, wmax=wmax,
         truncate=truncate, gamma=gamma, gamma_12=gamma_12, t_b=t_b, a_m=a_m, delta=delta, c_times=c_times,
         n_shots=n_shots, M=M, a_sp=a_sp, c=c, T=T)


# make noise matrices
noise_mats = jnp.array(make_noise_mat_arr('make', spec_vec=spec_vec, t_vec=t_vec, w_grain=w_grain, wmax=wmax,
                                         truncate=truncate, gamma=gamma, gamma_12=gamma_12))


print("Starting experiments")
# Step 1.1) CPMG on both qubits and evaluate $$C_{12,0}^{1,k}(MT)$$
pulse_1_1 = ['CPMG', 'CPMG']
C_12_0_MT_1 = make_C_12_0_MT(solver_prop, pulse_1_1, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 1.1 complete")


# Step 1.2) CDD3 on qubit 1 and CPMG on qubit 2 and evaluate $$C_{12,0}^{2,k}(MT)$$
pulse_1_2 = ['CDD3', 'CPMG']
C_12_0_MT_2 = make_C_12_0_MT(solver_prop, pulse_1_2, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 1.2 complete")


# Step 1.3) CPMG on qubit 1 and CDD3 on qubit 2 and evaluate $$C_{12,0}^{3,k}(MT)$$
pulse_1_3 = ['CPMG', 'CDD3']
C_12_0_MT_3 = make_C_12_0_MT(solver_prop, pulse_1_3, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 1.3 complete")


# Step 2.1) CDD3 on qubit 1 and CDD1 on qubit 2 and evaluate $$C_{12,12}^{1,k}(MT)$$
pulse_2_1 = ['CPMG', 'CPMG']
C_12_12_MT_1 = make_C_12_12_MT(solver_prop, pulse_2_1, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b,
                               a_m=a_m, delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 2.1 complete")


# Step 2.2) CDD3 on qubit 1 and CP on qubit 2 and evaluate $$C_{12,12}^{2,k}(MT)$$
pulse_2_2 = ['CDD3', 'CPMG']
C_12_12_MT_2 = make_C_12_12_MT(solver_prop, pulse_2_2, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b,
                               a_m=a_m, delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 2.2 complete")


# Step 3.1) CPMG on qubit 1 and CDD1/2 on qubit 2 and evaluate $$C_{1,0}^{1,k}(MT)$$
pulse_3_1 = ['CDD1', 'CDD1-1/2']
C_1_0_MT_1 = make_C_a_0_MT(solver_prop, pulse_3_1, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 3.1 complete")


# Step 3.2) CDD1/2 on qubit 1 and CPMG on qubit 2 and evaluate $$C_{2,0}^{2,k}(MT)$$
pulse_3_2 = ['CDD1-1/2', 'CDD1']
C_2_0_MT_1 = make_C_a_0_MT(solver_prop, pulse_3_2, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 3.2 complete")


pulse_3_3 = ['CDD1', 'CDD1']
C_12_0_MT_4 = make_C_12_0_MT(solver_prop, pulse_3_3, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                             delta=delta, state='pp', a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 3.3 complete")


# Step 4.1) CPMG on qubit 1 and CDD1 on qubit 2 and evaluate $$C_{1,2}^{1,k}(MT)$$
pulse_4_1 = ['CPMG', 'FID']
C_1_2_MT_1 = make_C_a_b_MT(solver_prop, pulse_4_1, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 4.1 complete")


# Step 4.2) CDD1 on qubit 1 and CDD1-1/2 on qubit 2 and evaluate $$C_{1,2}^{2,k}(MT)$$
pulse_4_2 = ['CPMG', 'CDD1-1/4']
C_1_2_MT_2 = make_C_a_b_MT(solver_prop, pulse_4_2, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=1, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 4.2 complete")


# Step 5.1) CPMG on both qubits and evaluate $$C_{2,1}^{1,k}(MT)$$
pulse_5_1 = ['FID', 'CPMG']
C_2_1_MT_1 = make_C_a_b_MT(solver_prop, pulse_5_1, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 5.1 complete")


# Step 5.2) CDD3 on qubit 1 and CDD3 on qubit 2 and evaluate $$C_{2,1}^{2,k}(MT)$$
pulse_5_2 = ['CDD1-1/4', 'CPMG']
C_2_1_MT_2 = make_C_a_b_MT(solver_prop, pulse_5_2, t_vec, c_times, CM, spMit, n_shots=n_shots, M=M, t_b=t_b, a_m=a_m,
                           delta=delta, l=2, a_sp=a_sp, c=c, noise_mats=noise_mats)
print("Experiment 5.2 complete")


# Print the time taken for the code to run
print("--- %s seconds ---" % (time.time() - start_time))


# save all the results in the folder created earlier
np.savez(os.path.join(path, "results.npz"), C_12_0_MT_1=C_12_0_MT_1, C_12_0_MT_2=C_12_0_MT_2, C_12_0_MT_3=C_12_0_MT_3,
         C_12_12_MT_1=C_12_12_MT_1, C_12_12_MT_2=C_12_12_MT_2, C_1_0_MT_1=C_1_0_MT_1, C_2_0_MT_1=C_2_0_MT_1,
         C_12_0_MT_4=C_12_0_MT_4, C_1_2_MT_1=C_1_2_MT_1, C_1_2_MT_2=C_1_2_MT_2, C_2_1_MT_1=C_2_1_MT_1,
         C_2_1_MT_2=C_2_1_MT_2)


