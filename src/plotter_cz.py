import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker

# Load data from the npz files
opt_data = np.load("/home/mqasimk/IdeaProjects/QNS-2Q/DraftRun_NoSPAM/infs_opt_cz.npz")
known_data = np.load("/home/mqasimk/IdeaProjects/QNS-2Q/DraftRun_NoSPAM/infs_known_cz.npz")

# Extract data for plotting
taxis_opt = opt_data['taxis']
infs_opt = opt_data['infs_opt']
taxis_known = known_data['taxis']
infs_known = known_data['infs_known']
infs_base = known_data['infs_base']

# Create the plot
legendfont = 12
labelfont = 16

plt.figure(figsize=(16, 9))
plt.plot(taxis_known/taxis_known[-1], infs_base, "r^-")
plt.plot(taxis_known/taxis_known[-1], infs_known, "bs-")
plt.plot(taxis_opt/taxis_known[-1], infs_opt, "ko-")
plt.legend(["Uncorrected Gate", "DD Gate", "NT Gate"], fontsize=legendfont)
plt.xlabel(r'Gate Time $(1/T_g)$', fontsize=labelfont)
plt.ylabel('Gate Infidelity', fontsize=labelfont)
plt.xscale('log', base=2)
plt.yscale('log')

# Set denser x-axis ticks
plt.gca().xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0, numticks=20))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=2.0, subs=(0.5, 1.0), numticks=20))

plt.grid(True, which="both", ls="--")
plt.savefig("/home/mqasimk/IdeaProjects/QNS-2Q/DraftRun_NoSPAM/infs_GateTime_cz.pdf")
plt.show()
