"""
Plotting utilities for pulse sequence optimization.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import jax.numpy as jnp

# --- Figure output directory layout ---
# All figures are saved under <data_folder>/figures/<category>/
FIGURES_DIR = "figures"
OPTIMIZATION_FIGURES_DIR = os.path.join(FIGURES_DIR, "optimization")
PUBLICATION_FIGURES_DIR = os.path.join(FIGURES_DIR, "publication")


def get_figures_dir(base_path, subdir=OPTIMIZATION_FIGURES_DIR):
    """Return (and create) a figure output subdirectory inside the data folder.

    Parameters
    ----------
    base_path : str
        Root data folder path (e.g., config.path).
    subdir : str
        Subfolder relative to base_path (default: figures/optimization).

    Returns
    -------
    str
        Absolute path to the created directory.
    """
    path = os.path.join(base_path, subdir)
    os.makedirs(path, exist_ok=True)
    return path

def make_tk12(tk1, tk2):
    """
    Combines two pulse sequences into a single sequence for the 12 interaction.
    Assumes inputs are [0, t1..., T] and [0, t2..., T].
    """
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

def get_spectral_amplitudes(pulse_times, T, w):
    """Computes the spectral amplitude Z(w) for a pulse sequence."""
    pulse_times = np.array(pulse_times)
    n_intervals = len(pulse_times) - 1
    signs = (-1.0)**np.arange(n_intervals)
    
    # exp_t: shape (len(pulse_times), len(w))
    exp_t = np.exp(1j * np.outer(pulse_times, w))
    
    # diffs: shape (n_intervals, len(w))
    diffs = exp_t[1:] - exp_t[:-1]
    
    # Z_w: shape (len(w),)
    Z_w = np.sum(signs[:, None] * diffs, axis=0)
    return Z_w

def plot_infidelity_vs_gatetime(xaxis_known, yaxis_known, xaxis_opt, yaxis_opt, yaxis_nopulse, tau, save_path, min_gate_time=None):
    """Plots Infidelity vs Gate Time with consistent formatting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert x-axis to units of tau
    # xaxis_known and xaxis_opt are in seconds. tau is in seconds.
    # The ratio should be > 1.
    xaxis_known_tau = np.array(xaxis_known) / tau
    xaxis_opt_tau = np.array(xaxis_opt) / tau
    
    ax.plot(xaxis_known_tau, yaxis_known, 'bs-', label='Known')
    ax.plot(xaxis_opt_tau, yaxis_opt, 'ko-', label='Optimized')
    if yaxis_nopulse is not None:
        # Ensure yaxis_nopulse matches length of xaxis_known if it was collected in the same loop
        # Assuming yaxis_nopulse corresponds to xaxis_known
        ax.plot(xaxis_known_tau, yaxis_nopulse, 'r^-', label='No Pulse')
    
    if min_gate_time is not None:
        min_gate_time_tau = min_gate_time / tau
        ax.axvline(x=min_gate_time_tau, color='gray', linestyle='--', label='Min Gate Time')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Gate Time ($\tau$)')
    ax.set_ylabel('Infidelity')
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    
    # Set ticks at data points
    # ax.set_xticks(xaxis_known_tau)
    # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

def plot_infidelity_vs_M(M_values, yaxis_known, yaxis_opt, yaxis_nopulse, save_path):
    """Plots Infidelity vs M (Repetitions) with consistent formatting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(M_values, yaxis_known, 'bs-', label='Known')
    ax.plot(M_values, yaxis_opt, 'ko-', label='Optimized')
    if yaxis_nopulse is not None:
        ax.plot(M_values, yaxis_nopulse, 'r^-', label='No Pulse')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Repetitions (M)')
    ax.set_ylabel('Infidelity')
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    
    # Set ticks at data points
    ax.set_xticks(M_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

def plot_infidelity_vs_M_labeled(M_values, known_infs, known_labels, opt_infs, opt_labels, nopulse_infs, save_path):
    """Plots Infidelity vs M with labels for sequences."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Known
    ax.plot(M_values, known_infs, 'bs-', label='Known')
    for m, inf, lab in zip(M_values, known_infs, known_labels):
        ax.annotate(lab, (m, inf), textcoords="offset points", xytext=(0, 5), ha='center', va='bottom', fontsize=8, rotation=45)

    # Opt
    ax.plot(M_values, opt_infs, 'ko-', label='Optimized')
    for m, inf, lab in zip(M_values, opt_infs, opt_labels):
        ax.annotate(lab, (m, inf), textcoords="offset points", xytext=(0, -5), ha='center', va='top', fontsize=8, rotation=45)

    # No Pulse
    if nopulse_infs is not None:
        ax.plot(M_values, nopulse_infs, 'r^-', label='No Pulse')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Repetitions (M)')
    ax.set_ylabel('Infidelity')
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    ax.set_xticks(M_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

def plot_infidelity_vs_gatetime_all_M(results_by_M, tau, save_path):
    """Plots Infidelity vs Gate Time for multiple M values with labels."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate colors
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results_by_M)))
    
    for idx, (M, data) in enumerate(results_by_M.items()):
        gate_times = np.array(data['gate_times'])
        gate_times_tau = gate_times / tau
        
        # Known
        known_infs = [x[0] for x in data['known']]
        known_labels = [x[1] for x in data['known']]
        
        color = colors[idx]
        ax.plot(gate_times_tau, known_infs, marker='s', linestyle='--', color=color, label=f'Known (M={M})', alpha=0.7)
        
        # Opt
        opt_infs = [x[0] for x in data['opt']]
        opt_labels = [x[1] for x in data['opt']]
        
        ax.plot(gate_times_tau, opt_infs, marker='o', linestyle='-', color=color, label=f'Opt (M={M})')
        
        # Labels
        for i, (x, y, lab) in enumerate(zip(gate_times_tau, known_infs, known_labels)):
             ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6, color=color, rotation=45)

        for i, (x, y, lab) in enumerate(zip(gate_times_tau, opt_infs, opt_labels)):
             ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=6, color=color, rotation=45)

    # No Pulse (from first M)
    first_M = list(results_by_M.keys())[0]
    gate_times = np.array(results_by_M[first_M]['gate_times'])
    gate_times_tau = gate_times / tau
    nopulse_infs = results_by_M[first_M]['nopulse']
    ax.plot(gate_times_tau, nopulse_infs, 'r^-', label='No Pulse', linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Gate Time ($\tau$)')
    ax.set_ylabel('Infidelity')
    ax.grid(True, which='both', linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

def plot_comparison(config, known_seq, opt_seq, T_seq, filename_suffix=""):
    """Plots the switching functions y(t) for comparison."""
    has_known = known_seq is not None
    has_opt = opt_seq is not None
    
    cols = 0
    if has_known: cols += 1
    if has_opt: cols += 1
    
    if cols == 0:
        return

    fig, axs = plt.subplots(3, cols, figsize=(6 * cols, 10), sharex=True, squeeze=False)
    
    def get_switching_function(pulse_times, T, num_points=1000):
        pulse_times = np.array(pulse_times) # Ensure numpy for plotting
        t_grid = np.linspace(0, T, num_points)
        y = np.ones_like(t_grid)
        if len(pulse_times) > 2:
            internal_pulses = pulse_times[1:-1]
            for t_pulse in internal_pulses:
                y[t_grid >= t_pulse] *= -1
        return t_grid, y

    def plot_col(col_idx, seq, title_prefix):
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        
        t, y1 = get_switching_function(pt1, T_seq)
        _, y2 = get_switching_function(pt2, T_seq)
        _, y12 = get_switching_function(pt12, T_seq)
        
        # Row 0: y1
        axs[0, col_idx].step(t*1e6, y1, 'k-', where='post')
        axs[0, col_idx].set_title(f"{title_prefix}\nQubit 1 Switching Function ($y_1$)")
        axs[0, col_idx].set_ylabel("$y_1(t)$")
        axs[0, col_idx].set_ylim(-1.2, 1.2)
        axs[0, col_idx].grid(True, alpha=0.3)
        
        # Row 1: y2
        axs[1, col_idx].step(t*1e6, y2, 'k-', where='post')
        axs[1, col_idx].set_title("Qubit 2 Switching Function ($y_2$)")
        axs[1, col_idx].set_ylabel("$y_2(t)$")
        axs[1, col_idx].set_ylim(-1.2, 1.2)
        axs[1, col_idx].grid(True, alpha=0.3)
        
        # Row 2: y12
        axs[2, col_idx].step(t*1e6, y12, 'k-', where='post')
        axs[2, col_idx].set_title("Interaction Switching Function ($y_{12}$)")
        axs[2, col_idx].set_ylabel("$y_{12}(t)$")
        axs[2, col_idx].set_xlabel(r"Time ($\mu$s)")
        axs[2, col_idx].set_ylim(-1.2, 1.2)
        axs[2, col_idx].grid(True, alpha=0.3)

    current_col = 0
    if has_known:
        plot_col(current_col, known_seq, "Best Known Sequence")
        current_col += 1
    
    if has_opt:
        plot_col(current_col, opt_seq, "Best Optimized Sequence")
        current_col += 1

    plt.tight_layout()
    
    save_path = os.path.join(get_figures_dir(config.path), f"sequence_comparison{filename_suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.close(fig)

def plot_filter_functions(config, known_seq, opt_seq, T_seq, filename_suffix=""):
    """Plots the filter functions F(omega) for comparison using asinh scaling."""
    has_known = known_seq is not None
    has_opt = opt_seq is not None
    
    cols = 0
    if has_known: cols += 1
    if has_opt: cols += 1
    
    if cols == 0:
        return

    fig, axs = plt.subplots(3, cols, figsize=(6 * cols, 10), sharex=True, squeeze=False)
    
    # Frequency grid for plotting (avoid 0 to prevent division by zero)
    w_plot = np.linspace(1e3, 2 * np.pi * 5e6, 1000) 
    
    def get_filter_function(pulse_times, T, w):
        Z_w = get_spectral_amplitudes(pulse_times, T, w)
        F_w = np.abs(Z_w)**2 / (w**2 * T)
        return w, F_w

    def plot_col(col_idx, seq, title_prefix):
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        
        w, F1 = get_filter_function(pt1, T_seq, w_plot)
        _, F2 = get_filter_function(pt2, T_seq, w_plot)
        _, F12 = get_filter_function(pt12, T_seq, w_plot)
        
        freqs_mhz = w / (2 * np.pi * 1e6)
        
        # Helper for asinh plotting
        def plot_asinh(ax, x, y, color, label):
            scale_factor = np.median(np.abs(y))
            if scale_factor == 0: scale_factor = 1e-6
            try:
                ax.set_yscale('asinh', linear_width=scale_factor)
                ax.plot(x, y, color=color)
            except ValueError:
                y_trans = np.arcsinh(y / scale_factor)
                ax.plot(x, y_trans, color=color)
                ax.set_ylabel(f"asinh(F/{scale_factor:.1e})")

        # Row 0: F1
        plot_asinh(axs[0, col_idx], freqs_mhz, F1, 'b', "$F_1$")
        axs[0, col_idx].set_title(f"{title_prefix}\nQubit 1 Filter Function ($F_1$)")
        if not axs[0, col_idx].get_ylabel(): axs[0, col_idx].set_ylabel(r"$F_1(\omega)$")
        axs[0, col_idx].grid(True, alpha=0.3)
        
        # Row 1: F2
        plot_asinh(axs[1, col_idx], freqs_mhz, F2, 'r', "$F_2$")
        axs[1, col_idx].set_title("Qubit 2 Filter Function ($F_2$)")
        if not axs[1, col_idx].get_ylabel(): axs[1, col_idx].set_ylabel(r"$F_2(\omega)$")
        axs[1, col_idx].grid(True, alpha=0.3)
        
        # Row 2: F12
        plot_asinh(axs[2, col_idx], freqs_mhz, F12, 'g', "$F_{12}$")
        axs[2, col_idx].set_title("Interaction Filter Function ($F_{12}$)")
        if not axs[2, col_idx].get_ylabel(): axs[2, col_idx].set_ylabel(r"$F_{12}(\omega)$")
        axs[2, col_idx].set_xlabel("Frequency (MHz)")
        axs[2, col_idx].grid(True, alpha=0.3)

    current_col = 0
    if has_known:
        plot_col(current_col, known_seq, "Best Known Sequence")
        current_col += 1
    
    if has_opt:
        plot_col(current_col, opt_seq, "Best Optimized Sequence")
        current_col += 1

    plt.tight_layout()
    
    save_path = os.path.join(get_figures_dir(config.path), f"filter_function_comparison{filename_suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved filter function comparison plot to {save_path}")
    plt.close(fig)

def plot_filter_functions_with_spectra(config, known_seq, opt_seq, T_seq, filename_suffix=""):
    """
    Plots filter functions overlaid with spectra.
    When M is large, highlights the filter function values at harmonic frequencies.
    """
    has_known = known_seq is not None
    has_opt = opt_seq is not None
    
    cols = 0
    if has_known: cols += 1
    if has_opt: cols += 1
    
    if cols == 0:
        return

    print("Plotting filter functions with spectra overlay...")

    fig, axs = plt.subplots(3, cols, figsize=(8 * cols, 12), sharex=True, squeeze=False)
    
    # Frequency grid for continuous plotting
    w_plot = np.linspace(1e3, config.w_max, 2000)
    freqs_mhz = w_plot / (2 * np.pi * 1e6)
    
    # Harmonics if M is large
    use_harmonics = (config.M > 10)
    if use_harmonics:
        w0 = 2 * np.pi / (config.Tg / config.M) # Base frequency of the sequence
        max_k = int(config.w_max / w0)
        k_vals = np.arange(1, max_k + 1)
        w_harmonics = k_vals * w0
        freqs_harmonics_mhz = w_harmonics / (2 * np.pi * 1e6)
    
    # Spectra (interpolated)
    # SMat indices: 1->S11, 2->S22, 3->S1212
    # SMat is on config.w
    # We can interpolate SMat to w_plot for plotting
    
    def get_spectrum_interp(idx):
        # idx: 1, 2, 3
        S_vals = config.SMat[idx, idx] # Diagonal elements are real
        return np.interp(w_plot, config.w, np.real(S_vals))

    S11_plot = get_spectrum_interp(1)
    S22_plot = get_spectrum_interp(2)
    S1212_plot = get_spectrum_interp(3)
    
    spectra_data = [S11_plot, S22_plot, S1212_plot]
    spectra_labels = ["$S_{11}$", "$S_{22}$", "$S_{1212}$"]
    
    def get_filter_function(pulse_times, T, w):
        Z_w = get_spectral_amplitudes(pulse_times, T, w)
        F_w = np.abs(Z_w)**2 / (w**2 * T)
        return F_w

    def plot_col(col_idx, seq, title_prefix):
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        
        # Calculate Filter Functions
        F1 = get_filter_function(pt1, T_seq, w_plot)
        F2 = get_filter_function(pt2, T_seq, w_plot)
        F12 = get_filter_function(pt12, T_seq, w_plot)
        
        Fs = [F1, F2, F12]
        F_labels = ["$F_1$", "$F_2$", "$F_{12}$"]
        
        # Harmonics
        F_harmonics = []
        if use_harmonics:
            F1_h = get_filter_function(pt1, T_seq, w_harmonics)
            F2_h = get_filter_function(pt2, T_seq, w_harmonics)
            F12_h = get_filter_function(pt12, T_seq, w_harmonics)
            F_harmonics = [F1_h, F2_h, F12_h]
        
        for row in range(3):
            ax1 = axs[row, col_idx]
            ax2 = ax1.twinx()
            
            # Plot Filter Function (Left Axis)
            color_F = 'tab:blue'
            ax1.set_ylabel(f"Filter Function {F_labels[row]}", color=color_F)
            ax1.tick_params(axis='y', labelcolor=color_F)
            
            # Continuous
            ax1.plot(freqs_mhz, Fs[row], color=color_F, alpha=0.6, label='Filter (Cont.)')
            ax1.set_yscale('log')
            
            if use_harmonics:
                ax1.scatter(freqs_harmonics_mhz, F_harmonics[row], color=color_F, marker='o', s=20, label='Filter (Harmonics)')
            
            # Plot Spectrum (Right Axis)
            color_S = 'tab:orange'
            ax2.set_ylabel(f"Spectrum {spectra_labels[row]}", color=color_S)
            ax2.tick_params(axis='y', labelcolor=color_S)
            
            ax2.plot(freqs_mhz, spectra_data[row], color=color_S, linestyle='--', alpha=0.6, label='Spectrum')
            ax2.set_yscale('log') # Spectra are usually log-scale friendly
            
            ax1.set_title(f"{title_prefix}\n{F_labels[row]} vs {spectra_labels[row]}")
            if row == 2:
                ax1.set_xlabel("Frequency (MHz)")
            
            ax1.grid(True, which='both', alpha=0.3)

    current_col = 0
    if has_known:
        plot_col(current_col, known_seq, "Best Known Sequence")
        current_col += 1
    
    if has_opt:
        plot_col(current_col, opt_seq, "Best Optimized Sequence")
        current_col += 1

    plt.tight_layout()
    save_path = os.path.join(get_figures_dir(config.path), f"filter_spectra_overlay{filename_suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved filter-spectra overlay plot to {save_path}")
    plt.close(fig)

def plot_generalized_filter_functions(config, seq, T_seq, label):
    """Plots the 3x3 generalized filter functions G_{a,b}(omega)."""
    if seq is None:
        return

    print(f"Plotting generalized filter functions for {label}...")
    
    w_plot = np.linspace(1e3, 2 * np.pi * 5e6, 1000)
    freqs_mhz = w_plot / (2 * np.pi * 1e6)
    
    pt1, pt2 = seq
    pt12 = make_tk12(pt1, pt2)
    
    Z1 = get_spectral_amplitudes(pt1, T_seq, w_plot)
    Z2 = get_spectral_amplitudes(pt2, T_seq, w_plot)
    Z12 = get_spectral_amplitudes(pt12, T_seq, w_plot)
    
    Zs = [Z1, Z2, Z12]
    labels = ['1', '2', '12']
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
    
    for i in range(3):
        for j in range(3):
            # G_{ab} = Z_a Z_b^* / (w^2 T)
            G_ab = (Zs[i] * np.conj(Zs[j])) / (w_plot**2 * T_seq)
            
            ax = axs[i, j]
            
            # Scale factor
            scale_factor = np.median(np.abs(G_ab))
            if scale_factor == 0: scale_factor = 1e-6
            
            try:
                ax.set_yscale('asinh', linear_width=scale_factor)
                ax.plot(freqs_mhz, np.real(G_ab), label='Real')
                ax.plot(freqs_mhz, np.imag(G_ab), label='Imag', alpha=0.7)
            except ValueError:
                y_real = np.arcsinh(np.real(G_ab) / scale_factor)
                y_imag = np.arcsinh(np.imag(G_ab) / scale_factor)
                ax.plot(freqs_mhz, y_real, label='asinh(Real)')
                ax.plot(freqs_mhz, y_imag, label='asinh(Imag)', alpha=0.7)
                ax.set_ylabel(f"asinh(G/{scale_factor:.1e})")
            
            ax.set_title(f"$G_{{{labels[i]},{labels[j]}}}(\\omega)$")
            if i == 2:
                ax.set_xlabel("Frequency (MHz)")
            if j == 0 and not ax.get_ylabel():
                ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
            
    plt.suptitle(f"Generalized Filter Functions - {label}")
    plt.tight_layout()
    filename = f"generalized_filter_functions_{label.replace(' ', '_')}.pdf"
    save_path = os.path.join(get_figures_dir(config.path), filename)
    plt.savefig(save_path)
    print(f"Saved generalized filter functions plot to {save_path}")
    plt.close(fig)

def plot_spectra_filter_overlay_6(config, seq, T_seq, label):
    """
    Plots 6 separate PDF files for S_ii/S_ij and corresponding Filter functions.
    (1,1), (2,2), (12,12), (1,2), (1,12), (2,12).
    Uses asinh scale for y-axis.
    """
    if seq is None:
        return

    print(f"Plotting 6-panel spectra/filter overlay for {label}...")
    
    # Use config.w_max for range
    w_plot = np.linspace(1e3, config.w_max, 1000)
    freqs_mhz = w_plot / (2 * np.pi * 1e6)
    
    pt1, pt2 = seq
    pt12 = make_tk12(pt1, pt2)
    
    Z1 = get_spectral_amplitudes(pt1, T_seq, w_plot)
    Z2 = get_spectral_amplitudes(pt2, T_seq, w_plot)
    Z12 = get_spectral_amplitudes(pt12, T_seq, w_plot)
    
    # Pairs: (S_idx1, S_idx2, Z_a, Z_b, S_label, G_label, Filename_Suffix)
    # SMat indices: 1->1, 2->2, 3->12
    pairs = [
        (1, 1, Z1, Z1, r"$S^+_{1,1}(\omega)$", r"$G^+_{1,1}(\omega, T)$", "S11"),
        (2, 2, Z2, Z2, r"$S^+_{2,2}(\omega)$", r"$G^+_{2,2}(\omega, T)$", "S22"),
        (3, 3, Z12, Z12, r"$S^+_{12,12}(\omega)$", r"$G^+_{12,12}(\omega, T)$", "S1212"),
        (1, 2, Z1, Z2, r"$S^+_{1,2}(\omega)$", r"$G^+_{1,2}(\omega, T)$", "S12"),
        (1, 3, Z1, Z12, r"$S^+_{1,12}(\omega)$", r"$G^+_{1,12}(\omega, T)$", "S112"),
        (2, 3, Z2, Z12, r"$S^+_{2,12}(\omega)$", r"$G^+_{2,12}(\omega, T)$", "S212")
    ]
    
    for idx, (s_i, s_j, Za, Zb, s_label, g_label, suffix) in enumerate(pairs):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        
        # Filter Function G = Za * conj(Zb) / (w^2 T)
        G = (Za * np.conj(Zb)) / (w_plot**2 * T_seq)
        
        # Spectrum S
        # Interpolate SMat[s_i, s_j]
        S_vals = config.SMat[s_i, s_j]
        # Use right=0. for safety
        S_interp = np.interp(w_plot, config.w, np.real(S_vals), right=0.) + 1j * np.interp(w_plot, config.w, np.imag(S_vals), right=0.)
        
        # Plot G on Left Axis (asinh)
        scale_G = np.median(np.abs(G))
        if scale_G == 0: scale_G = 1e-6
        
        color_G_real = 'tab:blue'
        color_G_imag = 'tab:cyan'
        
        try:
            ax1.set_yscale('asinh', linear_width=scale_G)
            ax1.plot(freqs_mhz, np.real(G), color=color_G_real, label='Re[$G^+$]', linewidth=2)
            if idx >= 3: # Off-diagonal
                ax1.plot(freqs_mhz, np.imag(G), color=color_G_imag, linestyle='--', label='Im[$G^+$]', linewidth=2)
        except ValueError:
            y_real = np.arcsinh(np.real(G) / scale_G)
            ax1.plot(freqs_mhz, y_real, color=color_G_real, label='Re[$G^+$]', linewidth=2)
            if idx >= 3:
                y_imag = np.arcsinh(np.imag(G) / scale_G)
                ax1.plot(freqs_mhz, y_imag, color=color_G_imag, linestyle='--', label='Im[$G^+$]', linewidth=2)
            ax1.set_ylabel(f"asinh($G^+$/{scale_G:.1e})", fontsize=16)
            
        ax1.set_ylabel(g_label, color=color_G_real, fontsize=16)
        ax1.tick_params(axis='y', labelcolor=color_G_real, labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        
        # Plot S on Right Axis (asinh)
        scale_S = np.median(np.abs(S_interp))
        if scale_S == 0: scale_S = 1e-6
        
        color_S_real = 'tab:red'
        color_S_imag = 'tab:orange'
        
        try:
            ax2.set_yscale('asinh', linear_width=scale_S)
            ax2.plot(freqs_mhz, np.real(S_interp), color=color_S_real, label='Re[$S^+$]', linewidth=2)
            if idx >= 3:
                ax2.plot(freqs_mhz, np.imag(S_interp), color=color_S_imag, linestyle='--', label='Im[$S^+$]', linewidth=2)
        except ValueError:
            y_real = np.arcsinh(np.real(S_interp) / scale_S)
            ax2.plot(freqs_mhz, y_real, color=color_S_real, label='Re[$S^+$]', linewidth=2)
            if idx >= 3:
                y_imag = np.arcsinh(np.imag(S_interp) / scale_S)
                ax2.plot(freqs_mhz, y_imag, color=color_S_imag, linestyle='--', label='Im[$S^+$]', linewidth=2)
            ax2.set_ylabel(f"asinh($S^+$/{scale_S:.1e})", fontsize=16)
            
        ax2.set_ylabel(s_label, color=color_S_real, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color_S_real, labelsize=14)
        
        # No title
        ax1.grid(True, alpha=0.3)
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=14)
        
        ax1.set_xlabel(r"$\omega(MHz)$", fontsize=16)

        plt.tight_layout()
        filename = f"spectra_filter_overlay_{suffix}_{label.replace(' ', '_')}.pdf"
        save_path = os.path.join(get_figures_dir(config.path), filename)
        plt.savefig(save_path)
        print(f"Saved spectra/filter overlay to {save_path}")
        plt.close(fig)

def plot_noise_correlations(config):
    """Plots the 9 noise correlation functions R(tau)."""
    print("Plotting noise correlation functions...")
    
    # Parameters
    N = config.w.shape[0]
    dw = config.w[1] - config.w[0]
    dt = 2 * np.pi / (N * dw)
    lags = (np.arange(N) - N//2) * dt
    
    # Indices map: 0->1(Q1), 1->2(Q2), 2->3(Q12)
    indices = [1, 2, 3]
    labels = ['1', '2', '12']
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
    
    for i in range(3):
        for j in range(3):
            idx_i = indices[i]
            idx_j = indices[j]
            
            # Get Spectrum
            S = config.SMat[idx_i, idx_j]
            
            # Compute R(tau)
            # Use numpy fft for plotting to avoid JAX overhead/device transfer if not needed, 
            # but S is jax array.
            R_vals = jnp.fft.ifft(S)
            R_scaled = R_vals / dt
            R_shifted = jnp.fft.fftshift(R_scaled)
            
            R_np = np.array(R_shifted)
            
            ax = axs[i, j]
            
            # Use asinh scaling
            # Scale factor for asinh: linear region width
            # Heuristic: use median absolute value or similar
            scale_factor = np.median(np.abs(R_np))
            if scale_factor == 0: scale_factor = 1e-6
            
            # Plot scaled values
            # We plot the raw values but set the scale to asinh
            # Matplotlib doesn't have built-in asinh scale until recent versions (3.6+)
            # If available, use it. Otherwise, manually transform.
            
            try:
                ax.set_yscale('asinh', linear_width=scale_factor)
                ax.plot(lags * 1e6, np.real(R_np), label='Real')
                ax.plot(lags * 1e6, np.imag(R_np), label='Imag', alpha=0.7)
            except ValueError:
                # Fallback if asinh not available or parameters wrong
                # Manual transformation for visualization
                y_real = np.arcsinh(np.real(R_np) / scale_factor)
                y_imag = np.arcsinh(np.imag(R_np) / scale_factor)
                ax.plot(lags * 1e6, y_real, label='asinh(Real)')
                ax.plot(lags * 1e6, y_imag, label='asinh(Imag)', alpha=0.7)
                ax.set_ylabel(f"asinh(Amp/{scale_factor:.1e})")
            
            ax.set_title(f"$R_{{{labels[i]},{labels[j]}}}(\\tau)$")
            if i == 2:
                ax.set_xlabel(r"Lag $\tau$ ($\mu$s)")
            if j == 0 and not ax.get_ylabel():
                ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
            
    plt.tight_layout()
    save_path = os.path.join(get_figures_dir(config.path), "noise_correlations.pdf")
    plt.savefig(save_path)
    print(f"Saved noise correlations plot to {save_path}")
    plt.close(fig)

def plot_control_correlations(config, seq, T_seq, M, label):
    """Plots the 9 control correlation functions C(tau)."""
    if seq is None:
        return

    print(f"Plotting control correlation functions for {label}...")
    
    # Parameters (matching optimization)
    N = config.w.shape[0]
    dw = config.w[1] - config.w[0]
    dt = 2 * np.pi / (N * dw)
    
    T_total = M * T_seq
    num_steps = int(np.ceil(T_total / dt)) + 1
    t_grid = np.arange(num_steps) * dt
    
    def get_y_samples(pt):
        # pt is [0, t1, ..., T_seq]
        # y is periodic with T_seq
        pt = np.array(pt) # Ensure numpy
        t_mod = np.mod(t_grid, T_seq)
        indices = np.searchsorted(pt, t_mod, side='right')
        return (-1.0) ** (indices - 1)

    pt1, pt2 = seq
    pt12 = make_tk12(pt1, pt2)
    
    y1 = get_y_samples(pt1)
    y2 = get_y_samples(pt2)
    y12 = get_y_samples(pt12)
    
    ys = [y1, y2, y12]
    labels_y = ['1', '2', '12']
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
    
    import jax.scipy.signal
    
    for i in range(3):
        for j in range(3):
            # Pad to match optimization
            y_i_pad = np.pad(ys[i], (0, num_steps), mode='constant')
            y_j_pad = np.pad(ys[j], (0, num_steps), mode='constant')
            
            # C_{a,b}
            # correlate(y_i, y_j)
            # mode='full'
            # Use JAX correlate to be consistent with optimization
            corr_jax = jax.scipy.signal.correlate(jnp.array(y_i_pad), jnp.array(y_j_pad), mode='full') * dt
            corr = np.array(corr_jax)
            
            lags = (np.arange(corr.shape[0]) - (2 * num_steps - 1)) * dt
            
            ax = axs[i, j]
            ax.plot(lags * 1e6, corr)
            
            ax.set_title(f"$C_{{{labels_y[i]},{labels_y[j]}}}(\\tau)$")
            if i == 2:
                ax.set_xlabel(r"Lag $\tau$ ($\mu$s)")
            if j == 0:
                ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            
    plt.suptitle(f"Control Correlation Functions - {label}")
    plt.tight_layout()
    filename = f"control_correlations_{label.replace(' ', '_')}.pdf"
    save_path = os.path.join(get_figures_dir(config.path), filename)
    plt.savefig(save_path)
    print(f"Saved control correlations plot to {save_path}")
    plt.close(fig)
