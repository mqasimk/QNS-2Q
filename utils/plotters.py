from matplotlib import pyplot as plt
import numpy as np
import os
from trajectories import f


params = np.load("/home/mqasimk/IdeaProjects/QNS-2Q/DraftRun_NoSPAM/optimizeLog.npz")

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

fig, ax = plt.subplots(3,2)
ax[0, 0].plot(taxis_opt, f(taxis_opt, opt_seq_1))
ax[1, 0].plot(taxis_opt, f(taxis_opt, opt_seq_2))
ax[2, 0].plot(taxis_opt, f(taxis_opt, opt_seq_12))
ax[0, 1].plot(taxis_best, f(taxis_best, best_seq_1))
ax[1, 1].plot(taxis_best, f(taxis_best, best_seq_2))
ax[2, 1].plot(taxis_best, f(taxis_best, best_seq_12))
plt.show()