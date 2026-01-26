import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_publication_plot():
    """
    Generates a publication-quality 'Infidelity vs. Gate Time' plot.
    """
    
    # 1. Configuration
    # ----------------
    
    # Dimensions: Single-column width (3.375 inches)
    FIG_WIDTH = 3.375
    FIG_HEIGHT = 2.5
    
    # Typography & Aesthetics
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.2,
        "grid.color": "grey",
    })

    # Okabe-Ito Color Palette (Colorblind-friendly)
    colors = {
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
        "black": "#000000"
    }

    # 2. Load Data
    # ------------
    # Locate the data file relative to the project root
    # Project structure assumption:
    # QNS-2Q/
    #   src/
    #     CZopt_plots.py (this file)
    #   DraftRun_NoSPAM_Feature/
    #     plotting_data_cz_v2.npz
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, "DraftRun_NoSPAM_Feature", "plotting_data_cz_v2.npz")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file) as data:
            taxis = data['taxis']
            infs_known = data['infs_known']
            infs_opt = data['infs_opt']
            tau = float(data['tau'])
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # Convert Gate Time to units of tau
    x_data = taxis / tau

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots()

    # Plot Known Sequences
    ax.loglog(x_data, infs_known, 
              marker='o', linestyle='-', color=colors["blue"], 
              label='Known Sequences', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

    # Plot Optimized Sequences
    ax.loglog(x_data, infs_opt, 
              marker='s', linestyle='--', color=colors["vermillion"], 
              label='Optimized', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

    # 4. Axes & Labels
    # ----------------
    ax.set_xlabel(r"Gate Time ($\tau$)")
    ax.set_ylabel("Infidelity")
    
    # Calculate dynamic x-axis limits
    min_x = np.min(x_data)
    max_x = np.max(x_data)
    
    # Add a small buffer (e.g., 10% in log space)
    log_min = np.log10(min_x)
    log_max = np.log10(max_x)
    log_range = log_max - log_min
    
    # If range is 0 (single point), add arbitrary buffer
    if log_range == 0:
        buffer = 0.5
    else:
        buffer = log_range * 0.1

    ax.set_xlim(10**(log_min - buffer), 10**(log_max + buffer))
    ax.set_ylim(1e-3, 1e0)

    # 5. Grid & Legend
    # ----------------
    ax.grid(True, which='major', zorder=0)
    ax.grid(False, which='minor') # No minor grid
    
    ax.legend(frameon=False, loc='lower right', fontsize=9, handlelength=2.5)

    # 6. Save Output
    # --------------
    plt.tight_layout(pad=0.3)
    output_filename = "infidelity_vs_gatetime_pub.pdf"
    # Save where the data was obtained from
    output_dir = os.path.dirname(data_file)
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated publication plot: {output_path}")

if __name__ == "__main__":
    generate_publication_plot()
