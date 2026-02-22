import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def make_tk12(tk1, tk2):
    """
    Constructs the interaction frame switching times from single qubit switching times.
    Logic mirrors cz_optimize.py.
    """
    # tk1, tk2 are arrays of pulse times including 0 and T
    # If they are just [0, T], there are no internal pulses.
    
    int1 = tk1[1:-1] if len(tk1) > 2 else np.array([])
    int2 = tk2[1:-1] if len(tk2) > 2 else np.array([])
    
    combined_int = np.concatenate([int1, int2])
    combined_int = np.sort(combined_int)
    
    # Assuming T is the same for both, taken from tk1[-1]
    T = tk1[-1]
    return np.concatenate([np.array([0.]), combined_int, np.array([T])])

def get_switching_function(pulse_times, T, num_points=2000):
    """
    Generates the switching function y(t) (+1/-1) from pulse times.
    """
    t_grid = np.linspace(0, T, num_points)
    y = np.ones_like(t_grid)
    
    if len(pulse_times) > 2:
        internal_pulses = pulse_times[1:-1]
        for t_pulse in internal_pulses:
            # Toggle state at each pulse
            # Using logical indexing to flip the sign for all times after the pulse
            mask = t_grid >= t_pulse
            y[mask] *= -1.0
            
    return t_grid, y

def generate_pulse_plot():
    # 1. Configuration
    # ----------------
    FIG_WIDTH = 7.0
    FIG_HEIGHT = 4.0
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.6,
    })
    
    COLOR_KNOWN = '#0072B2'     # Blue
    COLOR_OPT = '#D55E00'       # Vermillion
    
    # 2. Load Data
    # ------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, "DraftRun_NoSPAM_Feature", "plotting_data_cz_v2.npz")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file) as data:
            # Check if keys exist
            if 'best_known_seq_pt1' not in data or 'best_opt_seq_pt1' not in data:
                print("Error: Best sequence data not found in npz file.")
                return
                
            known_pt1 = data['best_known_seq_pt1']
            known_pt2 = data['best_known_seq_pt2']
            T_known = float(data['T_seq_best_known'])
            
            opt_pt1 = data['best_opt_seq_pt1']
            opt_pt2 = data['best_opt_seq_pt2']
            T_opt = float(data['T_seq_best_opt'])
            
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # 3. Process Data
    # ---------------
    # Known Sequence
    known_pt12 = make_tk12(known_pt1, known_pt2)
    t_known_1, y_known_1 = get_switching_function(known_pt1, T_known)
    t_known_2, y_known_2 = get_switching_function(known_pt2, T_known)
    t_known_12, y_known_12 = get_switching_function(known_pt12, T_known)
    
    # Optimized Sequence
    opt_pt12 = make_tk12(opt_pt1, opt_pt2)
    t_opt_1, y_opt_1 = get_switching_function(opt_pt1, T_opt)
    t_opt_2, y_opt_2 = get_switching_function(opt_pt2, T_opt)
    t_opt_12, y_opt_12 = get_switching_function(opt_pt12, T_opt)

    # 4. Plotting
    # -----------
    fig, axs = plt.subplots(3, 2, sharex='col', sharey=True)
    # axs shape: (3, 2)
    # Col 0: Known, Col 1: Optimized
    
    # --- Column 0: Known ---
    # Row 0: Qubit 1
    axs[0, 0].step(t_known_1 * 1e6, y_known_1, where='post', color=COLOR_KNOWN)
    axs[0, 0].set_title("Best Known Sequence")
    
    # Row 1: Qubit 2
    axs[1, 0].step(t_known_2 * 1e6, y_known_2, where='post', color=COLOR_KNOWN)
    
    # Row 2: Interaction
    axs[2, 0].step(t_known_12 * 1e6, y_known_12, where='post', color=COLOR_KNOWN)
    axs[2, 0].set_xlabel(r"Time ($\mu$s)")

    # --- Column 1: Optimized ---
    # Row 0: Qubit 1
    axs[0, 1].step(t_opt_1 * 1e6, y_opt_1, where='post', color=COLOR_OPT)
    axs[0, 1].set_title("Best Optimized Sequence")
    
    # Row 1: Qubit 2
    axs[1, 1].step(t_opt_2 * 1e6, y_opt_2, where='post', color=COLOR_OPT)
    
    # Row 2: Interaction
    axs[2, 1].step(t_opt_12 * 1e6, y_opt_12, where='post', color=COLOR_OPT)
    axs[2, 1].set_xlabel(r"Time ($\mu$s)")

    # 5. Styling & Labels
    # -------------------
    row_labels = [r"$y_{1,1}(t)$", r"$y_{2,2}(t)$", r"$y_{12,12}(t)$"]
    
    for row in range(3):
        for col in range(2):
            ax = axs[row, col]
            
            # Y-axis ticks
            ax.set_yticks([-1, 1])
            ax.set_ylim(-1.2, 1.2)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Row Labels (only on left column)
            if col == 0:
                ax.set_ylabel(row_labels[row], rotation=0, labelpad=20, va='center')
            
            # Hide x-axis labels for top rows (handled by sharex, but ensure visibility logic)
            if row < 2:
                ax.tick_params(labelbottom=False)
                
    # 6. Save Output
    # --------------
    plt.tight_layout()
    
    output_filename = "pulse_sequence_comparison_pub.pdf"
    data_dir = os.path.dirname(os.path.dirname(data_file))  # up from plotting_data/
    output_dir = os.path.join(data_dir, "figures", "publication")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated pulse comparison plot: {output_path}")

if __name__ == "__main__":
    generate_pulse_plot()
