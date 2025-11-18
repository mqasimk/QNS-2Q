from matplotlib import pyplot as plt
import numpy as np
import os
from trajectories import f

# Use Matplotlib's built-in text rendering for a professional look
plt.rc('font', family='serif', serif='STIXGeneral')
plt.rc('mathtext', fontset='stix')

params = np.load(os.path.join(os.pardir, "DraftRun_NoSPAM_Boring", "optimizeLog.npz"))

print(params["best_inf"])
print(params["opt_inf"])

Tg = params["gtime"]
print("Gate Time: "+str(Tg)+"s")

M_best = params["best_M"]
M_opt = params["opt_M"]
opt_seq_1 = params["opt_seq_1"]
opt_seq_2 = params["opt_seq_2"]
opt_seq_12 = params["opt_seq_12"]
best_seq_1 = params["best_seq_1"]
best_seq_2 = params["best_seq_2"]
best_seq_12 = params["best_seq_12"]
taxis_opt = np.linspace(0, opt_seq_1[-1], 1000)
taxis_best = np.linspace(0, best_seq_1[-1], 1000)

fig, ax = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

# Main title
fig.suptitle('Optimized Control Pulses', fontsize=20, fontweight='bold')

# Plotting the optimized sequences
ax[0, 0].plot(taxis_opt, f(taxis_opt, opt_seq_1), color='blue')
ax[0, 0].set_title('Optimized Pulse 1', fontsize=14)
ax[0, 0].set_ylabel('Amplitude (a.u.)', fontsize=12)

ax[1, 0].plot(taxis_opt, f(taxis_opt, opt_seq_2), color='blue')
ax[1, 0].set_title('Optimized Pulse 2', fontsize=14)
ax[1, 0].set_ylabel('Amplitude (a.u.)', fontsize=12)

ax[2, 0].plot(taxis_opt, f(taxis_opt, opt_seq_12), color='blue')
ax[2, 0].set_title('Optimized Pulse 12', fontsize=14)
ax[2, 0].set_xlabel('Time (s)', fontsize=12)
ax[2, 0].set_ylabel('Amplitude (a.u.)', fontsize=12)

# Plotting the best sequences
ax[0, 1].plot(taxis_best, f(taxis_best, best_seq_1), color='red')
ax[0, 1].set_title('Best Pulse 1', fontsize=14)

ax[1, 1].plot(taxis_best, f(taxis_best, best_seq_2), color='red')
ax[1, 1].set_title('Best Pulse 2', fontsize=14)

ax[2, 1].plot(taxis_best, f(taxis_best, best_seq_12), color='red')
ax[2, 1].set_title('Best Pulse 12', fontsize=14)
ax[2, 1].set_xlabel('Time (s)', fontsize=12)

# Common settings for all subplots
for axis in ax.flat:
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout and display the plot
plt.show()
