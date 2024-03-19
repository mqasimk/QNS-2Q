from matplotlib import pyplot as plt
import numpy as np
# load data from the appropriate folder

import csv
p0to1_ibm = []
p1to0_ibm = []
with open('ibm_osaka_calibrations_2024-01-16T15_16_48Z.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'Qubit':
            p0to1_ibm.append(row[6])
            p1to0_ibm.append(row[7])

p0to1_ibm = np.array(p0to1_ibm, dtype=float)
p1to0_ibm = np.array(p1to0_ibm, dtype=float)
alpha_ibm = 1-(p0to1_ibm+p1to0_ibm)
delta_ibm = p1to0_ibm-p0to1_ibm

foldername = "ibm_osaka_joint_7q_SPAMjobs_1"
alpha_arr = np.load(foldername + "/alpha_arr.npy")
alpha_m_arr = np.load(foldername + "/alpha_m_arr.npy")
alpha_sp_arr = np.load(foldername + "/alpha_sp_arr.npy")
delta_arr = np.load(foldername + "/delta_arr.npy")

x_axis = [0,1,2,3,4,5,6]

# make a grid of boxplot of the data with the x axis being the qubit number
fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(alpha_arr, meanline=True)
axs[0, 0].plot(x_axis, [alpha_ibm[i] for i in x_axis], 's', label="IBM")
axs[0, 0].set_title("SPAM-Error")
axs[0, 0].set_xlabel("Qubit Number")
axs[0, 0].set_ylabel(r"$\alpha$")
axs[0, 0].set_ylim(0.85, 1)
axs[0, 0].set_xticklabels(x_axis)

axs[0, 1].boxplot(alpha_sp_arr, meanline=True)
axs[0, 1].set_title("SP-Error")
axs[0, 1].set_xlabel("Qubit Number")
axs[0, 1].set_ylabel(r"$\alpha_{SP}$")
axs[0, 1].set_ylim(0.85, 1)
axs[0, 1].set_xticklabels(x_axis)

axs[1, 0].boxplot(alpha_m_arr, meanline=True)
axs[1, 0].set_title("M-Error")
axs[1, 0].set_xlabel("Qubit Number")
axs[1, 0].set_ylabel(r"$\alpha_M$")
axs[1, 0].set_ylim(0.85, 1)
axs[1, 0].set_xticklabels(x_axis)

axs[1, 1].boxplot(delta_arr, meanline=True)
axs[1, 1].set_title("Delta")
axs[1, 1].set_xlabel("Qubit Number")
axs[1, 1].set_ylabel(r"$\delta$")
axs[1, 1].set_xticklabels(x_axis)

plt.tight_layout()
plt.savefig(foldername + "/SPAM-Error-Grid.png")

