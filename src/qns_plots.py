import matplotlib.pyplot as plt
import numpy as np
import os
from spectra_input import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12

# --- Configuration ---
# Directories for the two datasets to be compared
DIR1 = "DraftRun_NoSPAM"
DIR2 = "DraftRun_SPAM"

# Directory for saving plots
OUTPUT_DIR = "SPAM_Mitigation_Plots"

# The directory containing the params.npz file
PARAMS_DIR = DIR1

# Plotting parameters
FIG_SIZE = (16, 9)
LINE_WIDTH = 1
LEGEND_FONT_SIZE = 8
X_LABEL_FONT_SIZE = 16
Y_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 12
X_UNITS = 1e6
Y_UNITS = 1
# --- End Configuration ---

def load_data(project_dir):
    """Loads spectral data and parameters from the specified directories."""
    dir1_path = os.path.join(project_dir, DIR1)
    dir2_path = os.path.join(project_dir, DIR2)
    params_path = os.path.join(project_dir, PARAMS_DIR)

    specs1 = np.load(os.path.join(dir1_path, 'specs.npz'))
    specs2 = np.load(os.path.join(dir2_path, 'specs.npz'))
    params = np.load(os.path.join(params_path, "params.npz"))

    return specs1, specs2, params

def plot_real_spectrum(ax, w, wk, y_k, y_k_mit, y_theory, title, xlabel, ylabel, legend_labels):
    """Helper function to plot a real-valued spectrum."""
    ax.plot(wk / X_UNITS, y_k / Y_UNITS, 'r.--', lw=0.5 * LINE_WIDTH, label=legend_labels[0])
    ax.plot(wk / X_UNITS, y_k_mit / Y_UNITS, 'b^--', lw=0.5 * LINE_WIDTH, label=legend_labels[1])
    ax.plot(w / X_UNITS, y_theory / Y_UNITS, 'k-', lw=0.75 * LINE_WIDTH, label=legend_labels[2])

    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)
    ax.tick_params(direction='in', labelsize=TICK_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)

def plot_complex_spectrum(ax, w, wk, y_k, y_k_mit, y_theory, title, xlabel, ylabel, legend_labels):
    """Helper function to plot a complex-valued spectrum."""
    ax.plot(wk / X_UNITS, np.real(y_k) / Y_UNITS, 'r.--', lw=0.5 * LINE_WIDTH, label=legend_labels[0])
    ax.plot(wk / X_UNITS, np.real(y_k_mit) / Y_UNITS, 'b^--', lw=0.5 * LINE_WIDTH, label=legend_labels[1])
    ax.plot(w / X_UNITS, np.real(y_theory) / Y_UNITS, 'k-', lw=0.75 * LINE_WIDTH, label=legend_labels[2])

    ax.plot(wk / X_UNITS, np.imag(y_k) / Y_UNITS, 'g.--', lw=0.5 * LINE_WIDTH, label=legend_labels[3])
    ax.plot(wk / X_UNITS, np.imag(y_k_mit) / Y_UNITS, 'm^--', lw=0.5 * LINE_WIDTH, label=legend_labels[4])
    ax.plot(w / X_UNITS, np.imag(y_theory) / Y_UNITS, 'c-', lw=0.75 * LINE_WIDTH, label=legend_labels[5])

    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)
    ax.tick_params(direction='in', labelsize=TICK_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)

def plot_all_spectrums(axs, w_plot, wk, specs1, specs2, params):
    """Creates all the spectrum subplots."""
    # Extract spectral data
    s1 = {k: specs1[k] for k in specs1.files}
    s2 = {k: specs2[k] for k in specs2.files}
    gamma, gamma_12 = params['gamma'], params['gamma_12']

    plot_real_spectrum(axs[0, 0], w_plot, wk, s1['S11'], s2['S11'], S_11(w_plot), r'$S_{1,1}^+(\omega)$', r'$\omega$(MHz)', r'$S^+_{a,b}(\omega)$ Hz', [r'$\hat{S}_{1,1}^+(\omega_k)$', r'$\hat{S}_{1,1,mit}^+(\omega_k)$', r'$S_{1,1}^+(\omega)$'])
    plot_real_spectrum(axs[0, 1], w_plot, wk, s1['S22'], s2['S22'], S_22(w_plot), r'$S_{2,2}^+(\omega)$', r'$\omega$(MHz)', '', [r'$\hat{S}_{2,2}^+(\omega_k)$', r'$\hat{S}_{2,2,mit}^+(\omega_k)$', r'$S_{2,2}^+(\omega)$'])
    plot_real_spectrum(axs[0, 2], w_plot, wk, s1['S1212'], s2['S1212'], S_1212(w_plot), r'$S_{12,12}^+(\omega)$', r'$\omega$(MHz)', '', [r'$\hat{S}_{12,12}^+(\omega_k)$', r'$\hat{S}_{12,12,mit}^+(\omega_k)$', r'$S_{12,12}^+(\omega)$'])

    plot_complex_spectrum(axs[1, 0], w_plot, wk, s1['S12'], s2['S12'], S_1_2(w_plot, gamma), r'$S_{1,2}^+(\omega)$', r'$\omega$(MHz)', r'$S^+_{a,b}(\omega)$ Hz', [r'Re[$\hat{S}_{1,2}^+(\omega_k)$]', r'Re[$\hat{S}_{1,2,mit}^+(\omega_k)$]', r'Re[$S_{1,2}^+(\omega)$]', r'Im[$\hat{S}_{1,2}^+(\omega_k)$]', r'Im[$\hat{S}_{1,2,mit}^+(\omega_k)$]', r'Im[$S_{1,2}^+(\omega)$]'])
    plot_complex_spectrum(axs[1, 1], w_plot, wk, s1['S112'], s2['S112'], S_1_12(w_plot, gamma_12), r'$S_{1,12}^+(\omega)$', r'$\omega$(MHz)', '', [r'Re[$\hat{S}_{1,12}^+(\omega_k)$]', r'Re[$\hat{S}_{1,12,mit}^+(\omega_k)$]', r'Re[$S_{1,12}^+(\omega)$]', r'Im[$\hat{S}_{1,12}^+(\omega_k)$]', r'Im[$\hat{S}_{1,12,mit}^+(\omega_k)$]', r'Im[$S_{1,12}^+(\omega)$]'])
    plot_complex_spectrum(axs[1, 2], w_plot, wk, s1['S212'], s2['S212'], S_2_12(w_plot, gamma_12 - gamma), r'$S_{2,12}^+(\omega)$', r'$\omega$(MHz)', '', [r'Re[$\hat{S}_{2,12}^+(\omega_k)$]', r'Re[$\hat{S}_{2,12,mit}^+(\omega_k)$]', r'Re[$S_{2,12}^+(\omega)$]', r'Im[$\hat{S}_{2,12}^+(\omega_k)$]', r'Im[$\hat{S}_{2,12,mit}^+(\omega_k)$]', r'Im[$S_{2,12}^+(\omega)$]'])

def main():
    """Main function to load data, plot, and save the results."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_dir, OUTPUT_DIR)

    specs1, specs2, params = load_data(project_dir)

    # Prepare frequency data for plotting
    wmax = params['wmax']
    w_grain = params['w_grain']
    truncate = params['truncate']
    T = params['T']
    w = np.linspace(0.1, wmax, w_grain)
    wk = np.array([2 * np.pi * (n + 1) / T for n in range(truncate)])
    w_plot = w[100:]

    # Create and save the plot
    fig, axs = plt.subplots(2, 3, figsize=FIG_SIZE)
    plot_all_spectrums(axs, w_plot, wk, specs1, specs2, params)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'reconstruct_SPAMmit.png'), dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
