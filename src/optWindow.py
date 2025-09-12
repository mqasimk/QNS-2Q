
import matplotlib.pyplot as plt
import numpy as np
import os

# Use a style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')

# Set font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 14,
})

def lorentzian(x, x0, gamma, A):
    """Returns the value of a Lorentzian peak with height A."""
    return (A * gamma**2 / ((x - x0)**2 + gamma**2) + A * gamma**2 / ((x + x0)**2 + gamma**2) + 0.11*A * gamma**2 / ((x)**2 + 0.1*gamma**2))

# Define tau
tau = 1
T_values = [6 * tau, 12 * tau, 18 * tau]

# Create the figure and subplots
fig, axes = plt.subplots(len(T_values), 1, figsize=(8, 12))

# Common settings
xlim_val = 1.7 * np.pi
nyquist_freq = np.pi / tau

# --- Lorentzian peak parameters ---
lorentz_x0 = 2.5  # Peak center (non-zero)
lorentz_gamma = 0.2  # Peak width
lorentz_A = 0.8  # Peak height

# --- Calculate the specific sampling frequencies from the T=10*tau case ---
T_for_sampling = T_values[1]
omega_harmonic_for_sampling = 2 * np.pi / T_for_sampling
max_harmonic_for_sampling = int(np.floor(xlim_val / omega_harmonic_for_sampling))
sample_frequencies = []
for n in range(-max_harmonic_for_sampling, max_harmonic_for_sampling + 1):
    sample_frequencies.append(n * omega_harmonic_for_sampling)
sample_frequencies = np.array(sample_frequencies)
lorentzian_at_samples = lorentzian(sample_frequencies, lorentz_x0, lorentz_gamma, lorentz_A)


# Loop through the different T values and create a subplot for each
for i, T in enumerate(T_values):
    ax = axes[i]

    # Set plot limits
    ax.set_xlim(-xlim_val, xlim_val)
    ax.set_ylim(0, 1)

    # Plot solid lines for Nyquist frequency
    ax.axvline(nyquist_freq, color='black', linestyle='-', linewidth=2.5, label=r'$|\omega_\text{Ny}|$')
    ax.axvline(-nyquist_freq, color='black', linestyle='-', linewidth=2.5)

    # Plot dashed lines for harmonics for the current T
    omega_harmonic = 2 * np.pi / T
    max_harmonic = int(np.floor(xlim_val / omega_harmonic))
    
    harmonic_ticks = []
    harmonic_labels = []
    nyquist_harmonic_n = int(round(nyquist_freq / omega_harmonic))

    for n in range(-max_harmonic, max_harmonic + 1):
        pos = n * omega_harmonic
        label = None
        if n == 1:
            label = r'$|\omega_k|=2\pi k/T$'
        ax.axvline(pos, color='gray', linestyle='--', linewidth=1.5, label=label)

        harmonic_ticks.append(pos)
        harmonic_labels.append(str(n))

    # Superpose the pre-calculated sampled Lorentzian spectrum using a stem plot
    ax.plot(sample_frequencies, lorentzian_at_samples, 'ro-',
            label=r'$S^+_{a,b}(\omega_k)$' if i == 1 else "")

    # Set subplot title and legend
    ax.set_title(fr'$T = {int(T/tau)}\tau$')
    if i == 1:
        ax.legend(loc='upper right')
    ax.grid(False)

    # Set custom x-axis ticks and labels for each subplot
    ax.set_xticks(harmonic_ticks)
    ax.set_xticklabels(harmonic_labels)
    ax.set_yticks([])

# Set common y-label for the figure
# fig.supylabel(r'$G^{+}_{a,a;b,b}(\omega, MT)$')

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for supylabel

output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'optimization_windows.pdf'))
