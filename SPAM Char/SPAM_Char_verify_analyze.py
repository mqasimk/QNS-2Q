import matplotlib.pyplot as plt
import numpy as np
import os

folder = "ibm_cusco_disjoint_verify_8q_SPAMjobs_1"

#load all the numpy arrays from the folder in the current directory
alpha_arr = []
alpha_m_arr = []
alpha_sp_arr = []
delta_arr = []
qubit_set = []
for i in range(5):
    alpha_arr.append(np.load(folder + "/alpha_arr_"+str(i)+".npy"))
    alpha_m_arr.append(np.load(folder + "/alpha_m_arr_"+str(i)+".npy"))
    alpha_sp_arr.append(np.load(folder + "/alpha_sp_arr_"+str(i)+".npy"))
    delta_arr.append(np.load(folder + "/delta_arr_"+str(i)+".npy"))
    qubit_set.append(np.load(folder + "/qubit_set_"+str(i)+".npy"))

fig, axs = plt.subplots(np.size(qubit_set[0]), 5)

for i in range(np.size(qubit_set[0])):
    for j in range(5):
        axs[i, j].boxplot([alpha_m_arr[j][:, i], alpha_sp_arr[j][:, i], alpha_arr[j][:, i]], meanline=True)
        axs[i, j].set_title("Qubit " + str(qubit_set[j][i]))
        axs[i, j].set_ylim(0.85, 1)
        axs[i, j].set_xticklabels(["M", "SP", "SPAM"])
fig.tight_layout()
# increase the size of the figure
fig.set_size_inches(10, 10)
#increase padding between subplots
plt.tight_layout()
plt.show()

alpha_arr = np.array(alpha_arr)
alpha_m_arr = np.array(alpha_m_arr)
alpha_sp_arr = np.array(alpha_sp_arr)
delta_arr = np.array(delta_arr)
fig, axs = plt.subplots(np.size(qubit_set[0]), 1)
# plt.tight_layout()
phi_list = np.linspace(0, np.pi*0.1, 5)
for i in range(np.size(qubit_set[0])):
    axs[i].plot(phi_list/np.pi, [np.mean(alpha_m_arr[j, :, i]) for j in range(5)], 'o--', label="M")
    axs[i].plot(phi_list/np.pi, [np.mean(alpha_sp_arr[j, :, i]) for j in range(5)], 's-', label="SP")
    axs[i].plot(phi_list/np.pi, [np.mean(alpha_arr[j, :, i]) for j in range(5)], '^:', label="SPAM")
    axs[i].set_title("Qubit " + str(qubit_set[0][i]))
    if i == np.size(qubit_set[0])-1:
        axs[i].set_xlabel(r"$\phi/\pi$ (rad)")
        axs[i].legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    else:
        axs[i].set_xlabel(" ")
        axs[i].set_xticklabels([])
    # axs[i].set_ylim(0.85, 1)
# increase the size of the figure
fig.set_size_inches(20, 20)
fig.subplots_adjust(hspace=0.5)
plt.show()

diff_arr = np.array([[np.mean(alpha_m_arr[0, :, i])-np.mean(alpha_m_arr[-1, :, i]),
                      np.mean(alpha_sp_arr[0, :, i])-np.mean(alpha_sp_arr[-1, :, i]),
                      np.mean(alpha_arr[0, :, i])-np.mean(alpha_arr[-1, :, i])] for i in range(np.size(qubit_set[0]))])
plt.plot(qubit_set[0], np.log(np.abs(diff_arr[:,0])), 'o', label=r"$\Delta$M")
plt.plot(qubit_set[0], np.log(np.abs(diff_arr[:,1])), 's', label=r"$\Delta$SP")
plt.plot(qubit_set[0], np.log(np.abs(diff_arr[:,2])), '^', label=r"$\Delta$SPAM")
plt.xlabel("Qubit Number")
plt.ylabel("Log of the Difference in Mean")
plt.legend()
plt.show()