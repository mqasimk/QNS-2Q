import numpy as np
import matplotlib.pyplot as plt
import os

parent_dir = os.pardir
fname = "DraftRun_NoSPAM"
path = os.path.join(parent_dir, fname)

out = np.load(os.path.join(path, "infs_known.npz"))
x = out["taxis"]
yb = out["infs_base"]
yk = out["infs_known"]
print(x)
print(yb)
print(yk)

legendfont = 12
labelfont = 16
tickfont = 12

fig = plt.figure(figsize=(16,9))
plt.plot(x, yb, "r^")
plt.plot(x, yk, "bs")
plt.legend(["Uncorrected Gate", "DD Gate"], fontsize=legendfont)
plt.xlabel('Gate Time (s)', fontsize=labelfont)
plt.ylabel('Gate Infidelity', fontsize=labelfont)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', labelsize=tickfont)
plt.show()