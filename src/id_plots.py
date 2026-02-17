import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import jax.numpy as jnp
from spectra_input import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12

def format_label_for_latex(label):
    """Formats a label string for LaTeX rendering."""
    if label is None:
        return ""
        
    # Ensure string and handle potential bytes string representation
    s = str(label)
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    
    # Regex to parse "Name(args)^Exp"
    # Example: CDD(1, 2)^64 -> Name=CDD, Args=1, 2, Exp=64
    # Example: NT(10,12)^64 -> Name=NT, Args=10,12, Exp=64
    match = re.match(r"([A-Za-z]+)\(([^)]+)\)\^(\d+)", s)
    if match:
        name = match.group(1)
        args = match.group(2)
        exp = match.group(3)
        
        # Clean up args for subscript
        # Remove spaces
        args_clean = args.replace(" ", "")
        
        # Clean up parameter names for mqCDD to make it more readable/compact (e.g. n=1,m=2 -> 1,2)
        args_clean = args_clean.replace("n=", "").replace("m=", "")
        
        return r"$\textrm{" + name + r"}_{" + args_clean + r"}^{" + exp + r"}$"

    # Fallback for other formats
    if "^" in s:
        parts = s.split("^")
        base = parts[0]
        exponent = parts[1] if len(parts) > 1 else ""
        base = base.replace("_", r"\_")
        return r"$\textrm{" + base + r"}^{" + exponent + r"}$"
    
    return s.replace("_", r"\_")

def get_data_paths():
    """
    Returns the paths to the data files used in this script.
    Consolidates path logic to a single location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Base directory for this run
    base_dir = os.path.join(project_root, "DraftRun_NoSPAM_Boring")
    
    paths = {
        "plotting_data": os.path.join(base_dir, "plotting_data", "plotting_data_id_v4.npz"),
        "optimization_data": os.path.join(base_dir, "optimization_data_all_M.npz"),
        "params": os.path.join(base_dir, "params.npz"),
        "output_dir": os.path.join(base_dir, "plotting_data") # Save plots here
    }
    
    # Ensure output directory exists
    if not os.path.exists(paths["output_dir"]):
        os.makedirs(paths["output_dir"])
        
    return paths

def generate_publication_plot():
    """
    Generates a publication-quality 'Infidelity vs. Gate Time' plot for ID gate.
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
    paths = get_data_paths()
    data_file = paths["plotting_data"]
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file) as data:
            taxis = data['taxis']
            infs_known = data['infs_known']
            infs_opt = data['infs_opt']
            infs_nopulse = data['infs_nopulse']
            tau = float(data['tau'])
            # min_gate_time might not be relevant for ID gate in the same way, but we load it if present
            min_gate_time = float(data['min_gate_time']) if 'min_gate_time' in data else None
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # Convert Gate Time to units of tau
    x_data = taxis / tau
    min_gate_time_tau = min_gate_time / tau if min_gate_time is not None else None

    # Print Data to Terminal
    print("-" * 60)
    print(f"{'Gate Time (tau)':<20} {'No Pulse':<15} {'Known':<15} {'Optimized':<15}")
    print("-" * 60)
    for i in range(len(x_data)):
        print(f"{x_data[i]:<20.2f} {infs_nopulse[i]:<15.2e} {infs_known[i]:<15.2e} {infs_opt[i]:<15.2e}")
    print("-" * 60)
    if min_gate_time_tau is not None:
        print(f"Minimum Gate Time (tau): {min_gate_time_tau:.2f}")
    print("-" * 60)

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots()

    # Plot No Pulse
    ax.loglog(x_data, infs_nopulse, 
              marker='^', linestyle=':', color=colors["black"], 
              label='No Pulse', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

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
              
    # Plot Minimum Gate Time if available
    # if min_gate_time_tau is not None:
    #     ax.axvline(x=min_gate_time_tau, color='grey', linestyle='--', linewidth=1.0, zorder=5)

    # 4. Axes & Labels
    # ----------------
    ax.set_xlabel(r"Gate Time ($\tau$)")
    ax.set_ylabel("Infidelity")
    
    # Calculate dynamic x-axis limits
    # Include min_gate_time_tau in the range calculation if it exists
    # if min_gate_time_tau is not None:
    #     all_x_points = np.append(x_data, min_gate_time_tau)
    # else:
    all_x_points = x_data
        
    min_x = np.min(all_x_points)
    max_x = np.max(all_x_points)
    
    # Add a small buffer (e.g., 10% in log space)
    log_min_x = np.log10(min_x)
    log_max_x = np.log10(max_x)
    log_range_x = log_max_x - log_min_x
    
    if log_range_x == 0:
        buffer_x = 0.5
    else:
        buffer_x = log_range_x * 0.1

    ax.set_xlim(10**(log_min_x - buffer_x), 10**(log_max_x + buffer_x))

    # Calculate dynamic y-axis limits
    all_y_points = np.concatenate([infs_nopulse, infs_known, infs_opt])
    min_y = np.min(all_y_points)
    max_y = np.max(all_y_points)
    
    log_min_y = np.log10(min_y)
    log_max_y = np.log10(max_y)
    log_range_y = log_max_y - log_min_y
    
    if log_range_y == 0:
        buffer_y = 0.5
    else:
        buffer_y = log_range_y * 0.1
        
    ax.set_ylim(10**(log_min_y - buffer_y), 10**(log_max_y + buffer_y))

    # 5. Grid & Legend
    # ----------------
    ax.grid(True, which='major', zorder=0)
    ax.grid(False, which='minor') # No minor grid
    
    # ax.legend(frameon=False, loc='lower right', fontsize=9, handlelength=2.5)

    # 6. Save Output
    # --------------
    plt.tight_layout(pad=0.3)
    output_filename = "infidelity_vs_gatetime_id_pub.pdf"
    output_path = os.path.join(paths["output_dir"], output_filename)
    
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated publication plot: {output_path}")

def generate_all_M_plot():
    """
    Generates a publication-quality 'Infidelity vs. Gate Time' plot for ID gate
    showing results for all M values.
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
    paths = get_data_paths()
    data_file = paths["optimization_data"]
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file, allow_pickle=True) as data:
            M_values = data['M_values']
            
            # We need tau. It's not explicitly in optimization_data_all_M.npz, 
            # but we can try to load it from one of the single M files or assume it's consistent.
            # Let's try loading from plotting_data_id_v4.npz if available, or infer/hardcode.
            # Better yet, let's just read one of the M files if they exist.
            # Or, we can just assume tau is consistent with previous runs.
            # Let's try to find tau from plotting_data_id_v4.npz
            tau_file = paths["plotting_data"]
            if os.path.exists(tau_file):
                with np.load(tau_file) as tau_data:
                    tau = float(tau_data['tau'])
            else:
                print("Warning: Could not find tau source. Using default or failing.")
                return

            results_by_M = {}
            for m in M_values:
                prefix = f"M{m}_"
                gate_times = data[prefix + 'gate_times']
                infs_known = data[prefix + 'infs_known']
                infs_opt = data[prefix + 'infs_opt']
                infs_nopulse = data[prefix + 'infs_nopulse']
                
                results_by_M[m] = {
                    'gate_times': gate_times,
                    'infs_known': infs_known,
                    'infs_opt': infs_opt,
                    'infs_nopulse': infs_nopulse
                }
                
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots()
    
    # Generate colors for different M values
    # We can use a colormap
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(M_values)-1)
    
    # Plot No Pulse (only once, from first M)
    first_M = M_values[0]
    gate_times_np = results_by_M[first_M]['gate_times'] / tau
    infs_np = results_by_M[first_M]['infs_nopulse']
    
    ax.loglog(gate_times_np, infs_np, 
              marker='^', linestyle=':', color=colors["black"], 
              label='No Pulse', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

    for i, m in enumerate(M_values):
        res = results_by_M[m]
        x_data = res['gate_times'] / tau
        infs_k = res['infs_known']
        infs_o = res['infs_opt']
        
        color = cmap(norm(i))
        
        # Plot Known
        ax.loglog(x_data, infs_k, 
                  marker='o', linestyle='--', color=color, 
                  label=f'Known (M={m})', clip_on=False,
                  linewidth=1.0, markersize=3, alpha=0.7)
        
        # Plot Opt
        ax.loglog(x_data, infs_o, 
                  marker='s', linestyle='-', color=color, 
                  label=f'Opt (M={m})', clip_on=False,
                  linewidth=1.0, markersize=3)

    # 4. Axes & Labels
    # ----------------
    ax.set_xlabel(r"Gate Time ($\tau$)")
    ax.set_ylabel("Infidelity")
    
    # Set limits based on all data
    # ... (similar logic to single plot)

    # 5. Grid & Legend
    # ----------------
    ax.grid(True, which='major', zorder=0)
    
    # Legend might be too crowded, maybe put outside
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

    # 6. Save Output
    # --------------
    plt.tight_layout(pad=0.3)
    output_filename = "infidelity_vs_gatetime_id_all_M_pub.pdf"
    output_path = os.path.join(paths["output_dir"], output_filename)
    
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated all M publication plot: {output_path}")

def generate_best_M_plot():
    """
    Generates a plot showing the best infidelity across all M values for each gate time,
    labeled with the sequence name.
    """
    
    # 1. Configuration
    # ----------------
    FIG_WIDTH = 3.375
    FIG_HEIGHT = 2.5
    
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
    paths = get_data_paths()
    data_file = paths["optimization_data"]
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file, allow_pickle=True) as data:
            M_values = data['M_values']
            
            tau_file = paths["plotting_data"]
            if os.path.exists(tau_file):
                with np.load(tau_file) as tau_data:
                    tau = float(tau_data['tau'])
            else:
                print("Warning: Could not find tau source. Using default or failing.")
                return

            # Collect all unique gate times
            all_gate_times = set()
            for m in M_values:
                prefix = f"M{m}_"
                gts = data[prefix + 'gate_times']
                for gt in gts:
                    all_gate_times.add(round(gt, 10)) # Round to avoid float issues
            
            sorted_gts = sorted(list(all_gate_times))
            
            best_known_infs = []
            best_known_labels = []
            best_opt_infs = []
            best_opt_labels = []
            nopulse_infs = []
            
            # Get No Pulse from first M (should be same for all M at same gate time)
            # We need to interpolate or find matching
            first_M = M_values[0]
            prefix_first = f"M{first_M}_"
            gts_first = data[prefix_first + 'gate_times']
            infs_np_first = data[prefix_first + 'infs_nopulse']
            
            for gt in sorted_gts:
                # Find best known
                curr_best_inf = 1.0
                curr_best_lbl = ""
                
                for m in M_values:
                    prefix = f"M{m}_"
                    m_gts = data[prefix + 'gate_times']
                    m_infs = data[prefix + 'infs_known']
                    m_lbls = data[prefix + 'labels_known']
                    
                    indices = np.where(np.abs(m_gts - gt) < 1e-9)[0]
                    if len(indices) > 0:
                        idx = indices[0]
                        inf = m_infs[idx]
                        if inf < curr_best_inf:
                            curr_best_inf = inf
                            curr_best_lbl = str(m_lbls[idx])
                
                best_known_infs.append(curr_best_inf)
                best_known_labels.append(curr_best_lbl)
                
                # Find best opt
                curr_best_inf = 1.0
                curr_best_lbl = ""
                
                for m in M_values:
                    prefix = f"M{m}_"
                    m_gts = data[prefix + 'gate_times']
                    m_infs = data[prefix + 'infs_opt']
                    m_lbls = data[prefix + 'labels_opt']
                    
                    indices = np.where(np.abs(m_gts - gt) < 1e-9)[0]
                    if len(indices) > 0:
                        idx = indices[0]
                        inf = m_infs[idx]
                        if inf < curr_best_inf:
                            curr_best_inf = inf
                            curr_best_lbl = str(m_lbls[idx])
                            
                best_opt_infs.append(curr_best_inf)
                best_opt_labels.append(curr_best_lbl)
                
                # No Pulse
                indices = np.where(np.abs(gts_first - gt) < 1e-9)[0]
                if len(indices) > 0:
                    nopulse_infs.append(infs_np_first[indices[0]])
                else:
                    nopulse_infs.append(np.nan)

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots()
    
    x_data = np.array(sorted_gts) / tau
    
    # Plot No Pulse
    ax.loglog(x_data, nopulse_infs, 
              marker='^', linestyle=':', color=colors["black"], 
              label='No Pulse', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

    # Plot Best Known
    ax.loglog(x_data, best_known_infs, 
              marker='o', linestyle='-', color=colors["blue"], 
              label='Best Known', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)
              
    # Annotate Known
    for x, y, lbl in zip(x_data, best_known_infs, best_known_labels):
        if lbl != "N/A":
            formatted_lbl = format_label_for_latex(lbl)
            ax.annotate(formatted_lbl, (x, y), textcoords="offset points", xytext=(0, 5), 
                        ha='center', va='bottom', fontsize=6, color=colors["blue"], rotation=45)

    # Plot Best Opt
    ax.loglog(x_data, best_opt_infs, 
              marker='s', linestyle='--', color=colors["vermillion"], 
              label='Best Optimized', clip_on=False,
              linewidth=2.0, markersize=5, zorder=10)

    # Annotate Opt
    for x, y, lbl in zip(x_data, best_opt_infs, best_opt_labels):
        if lbl != "N/A":
            formatted_lbl = format_label_for_latex(lbl)
            ax.annotate(formatted_lbl, (x, y), textcoords="offset points", xytext=(0, -10), 
                        ha='center', va='top', fontsize=6, color=colors["vermillion"], rotation=45)

    # 4. Axes & Labels
    # ----------------
    ax.set_xlabel(r"Gate Time ($\tau$)")
    ax.set_ylabel("Infidelity")
    
    # 5. Grid & Legend
    # ----------------
    ax.grid(True, which='major', zorder=0)
    ax.legend(frameon=False, loc='best', fontsize=6)

    # 6. Save Output
    # --------------
    plt.tight_layout(pad=0.3)
    output_filename = "infidelity_vs_gatetime_id_best_M_pub.pdf"
    output_path = os.path.join(paths["output_dir"], output_filename)
    
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated best M publication plot: {output_path}")

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

def generate_spectra_overlay_plot():
    """
    Generates publication-quality overlay plots of ideal spectra and pulse sequence filter functions.
    Two vertical panels: Top for Known, Bottom for Optimized.
    Left Axis: Noise Spectrum (S) and Filter Function (G) labels.
    Right Axis: Filter Function (G) scale (no label).
    """
    
    # 1. Configuration
    # ----------------
    FIG_WIDTH = 3.375
    FIG_HEIGHT = 2.5 * 2 # Double height for 2 panels
    
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
        "lines.linewidth": 1.5, # Thicker lines
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.2,
        "grid.color": "grey",
    })

    # 2. Load Data
    # ------------
    paths = get_data_paths()
    data_file = paths["optimization_data"]
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file, allow_pickle=True) as data:
            M_values = data['M_values']
            
            # Find longest gate time
            all_gate_times = set()
            for m in M_values:
                prefix = f"M{m}_"
                gts = data[prefix + 'gate_times']
                for gt in gts:
                    all_gate_times.add(round(gt, 10))
            
            longest_gt = max(all_gate_times)
            print(f"Longest Gate Time: {longest_gt:.2e} s")
            
            # Find best known and opt for this gate time
            best_seq_known = None
            best_seq_opt = None
            best_M_known = -1
            best_M_opt = -1
            
            curr_best_inf_known = 1.0
            curr_best_inf_opt = 1.0
            
            for m in M_values:
                prefix = f"M{m}_"
                m_gts = data[prefix + 'gate_times']
                m_infs_known = data[prefix + 'infs_known']
                m_seqs_known = data[prefix + 'sequences_known']
                
                m_infs_opt = data[prefix + 'infs_opt']
                m_seqs_opt = data[prefix + 'sequences_opt']
                
                indices = np.where(np.abs(m_gts - longest_gt) < 1e-9)[0]
                if len(indices) > 0:
                    idx = indices[0]
                    
                    # Known
                    if m_infs_known[idx] < curr_best_inf_known:
                        curr_best_inf_known = m_infs_known[idx]
                        best_seq_known = m_seqs_known[idx]
                        best_M_known = m
                        
                    # Opt
                    if m_infs_opt[idx] < curr_best_inf_opt:
                        curr_best_inf_opt = m_infs_opt[idx]
                        best_seq_opt = m_seqs_opt[idx]
                        best_M_opt = m
            
            # Load params
            params_file = paths["params"]
            if os.path.exists(params_file):
                params = np.load(params_file)
                gamma = float(params['gamma'])
                gamma12 = float(params['gamma_12'])
                T_sys = float(params['T'])
                truncate = int(params['truncate'])
            else:
                tau_sys = 2.5e-8
                T_sys = 160 * tau_sys
                truncate = 20
                gamma = T_sys / 14
                gamma12 = T_sys / 28
                print("Warning: params.npz not found, using defaults.")

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Generate Spectra
    # -------------------
    w_max = 2 * np.pi * truncate / T_sys
    w = jnp.linspace(-w_max, w_max, 5001)
    
    spectra = {
        "S_11": S_11(w),
        "S_22": S_22(w),
        "S_1212": S_1212(w),
        "S_1_2": S_1_2(w, gamma),
        "S_1_12": S_1_12(w, gamma12),
        "S_2_12": S_2_12(w, gamma12)
    }

    # 4. Pre-compute Filter Functions
    # -----------------------------
    sequences_data = {}
    
    def prepare_sequence_data(seq, M, label):
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        
        data = {'M': M, 'label': label}
        
        if M > 10:
            w0 = 2 * np.pi / (longest_gt / M)
            max_k = int(w_max / w0)
            k_vals_full = np.arange(-max_k, max_k + 1)
            k_vals_full = k_vals_full[k_vals_full != 0]
            w_eval = k_vals_full * w0
            
            sort_idx = np.argsort(w_eval)
            w_eval_sorted = w_eval[sort_idx]
            data['w_eval'] = w_eval_sorted
            data['is_harmonic'] = True
            data['w0'] = w0
            
            Z1 = get_spectral_amplitudes(pt1, longest_gt, w_eval_sorted)
            Z2 = get_spectral_amplitudes(pt2, longest_gt, w_eval_sorted)
            Z12 = get_spectral_amplitudes(pt12, longest_gt, w_eval_sorted)
        else:
            w_eval = w
            data['w_eval'] = w_eval
            data['is_harmonic'] = False
            
            Z1 = get_spectral_amplitudes(pt1, longest_gt, w_eval)
            Z2 = get_spectral_amplitudes(pt2, longest_gt, w_eval)
            Z12 = get_spectral_amplitudes(pt12, longest_gt, w_eval)
            
        data['Zs'] = [Z1, Z2, Z12]
        return data

    if best_seq_known is not None:
        sequences_data['Known'] = prepare_sequence_data(best_seq_known, best_M_known, 'Known')
    if best_seq_opt is not None:
        sequences_data['Optimized'] = prepare_sequence_data(best_seq_opt, best_M_opt, 'Optimized')

    # 5. Plotting Loop
    # ----------------
    components = [
        ("S11", "S_11", 0, 0, 'real', r"$S_{1,1}(\omega)$", r"$G_{1,1}(\omega, T_G)$"),
        ("S22", "S_22", 1, 1, 'real', r"$S_{2,2}(\omega)$", r"$G_{2,2}(\omega, T_G)$"),
        ("S1212", "S_1212", 2, 2, 'real', r"$S_{12,12}(\omega)$", r"$G_{12,12}(\omega, T_G)$"),
        ("Re_S12", "S_1_2", 0, 1, 'real', r"Re[$S_{1,2}(\omega)$]", r"Re[$G_{1,2}(\omega, T_G)$]"),
        ("Im_S12", "S_1_2", 0, 1, 'imag', r"Im[$S_{1,2}(\omega)$]", r"Im[$G_{1,2}(\omega, T_G)$]"),
        ("Re_S112", "S_1_12", 0, 2, 'real', r"Re[$S_{1,12}(\omega)$]", r"Re[$G_{1,12}(\omega, T_G)$]"),
        ("Im_S112", "S_1_12", 0, 2, 'imag', r"Im[$S_{1,12}(\omega)$]", r"Im[$G_{1,12}(\omega, T_G)$]"),
        ("Re_S212", "S_2_12", 1, 2, 'real', r"Re[$S_{2,12}(\omega)$]", r"Re[$G_{2,12}(\omega, T_G)$]"),
        ("Im_S212", "S_2_12", 1, 2, 'imag', r"Im[$S_{2,12}(\omega)$]", r"Im[$G_{2,12}(\omega, T_G)$]")
    ]
    
    xunits = 1e6 # MHz
    
    output_dir = paths["output_dir"]

    for suffix, s_key, idx_a, idx_b, part, lbl_s, lbl_g in components:
        # Create 2 vertical subplots sharing x-axis
        fig, axs = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
        
        # S value
        S_val = spectra[s_key]
        if part == 'real':
            S_plot = np.real(S_val)
        else:
            S_plot = np.imag(S_val)
            
        # Styles
        styles = {
            'Known': {'color': '#0072B2', 'ls': '-', 'label': 'Known'},     # Blue, Solid
            'Optimized': {'color': '#E69F00', 'ls': '--', 'label': 'Optimized'} # Orange, Dashed
        }
        
        # Loop over subplots
        for i, seq_type in enumerate(['Known', 'Optimized']):
            ax = axs[i]
            
            # Plot Noise (Background) on Left Axis
            ax.fill_between(w / xunits, 0, S_plot, color='#E0E0E0', alpha=1.0, label='Spectra', zorder=0)
            
            # Combined Label on Left Axis
            s_clean = lbl_s.replace('$', '')
            g_clean = lbl_g.replace('$', '')
            combined_label = r"$" + s_clean + r", " + g_clean + r"$"
            
            ax.set_ylabel(combined_label, color='k')
            ax.tick_params(axis='y', labelcolor='k')
            ax.set_yscale('log')
            ax.set_xlim(min(w)/xunits, max(w)/xunits)
            
            # Plot Filter Function (Foreground) on Right Axis
            ax2 = ax.twinx()
            
            if seq_type in sequences_data:
                data = sequences_data[seq_type]
                Zs = data['Zs']
                w_eval = data['w_eval']
                is_harmonic = data['is_harmonic']
                
                Za = Zs[idx_a]
                Zb = Zs[idx_b]
                G = (Za * np.conj(Zb)) / (w_eval**2 * longest_gt)
                
                if part == 'real':
                    G_vals = np.real(G)
                else:
                    G_vals = np.imag(G)
                    
                st = styles[seq_type]
                
                if is_harmonic:
                    # Spike plot
                    w0 = data['w0']
                    epsilon = w0 * 0.01
                    
                    w_spikes = []
                    G_spikes = []
                    
                    if w_eval[0] > -w_max:
                        w_spikes.append(-w_max)
                        G_spikes.append(0)
                        
                    for k in range(len(w_eval)):
                        wk = w_eval[k]
                        gk = G_vals[k]
                        w_spikes.extend([wk - epsilon, wk, wk + epsilon])
                        G_spikes.extend([0, gk, 0])
                        
                    if w_eval[-1] < w_max:
                        w_spikes.append(w_max)
                        G_spikes.append(0)
                    
                    ax2.plot(np.array(w_spikes)/xunits, np.array(G_spikes), 
                             color=st['color'], linestyle=st['ls'], label=st['label'], linewidth=1.5)
                else:
                    ax2.plot(w_eval/xunits, G_vals, 
                             color=st['color'], linestyle=st['ls'], label=st['label'], linewidth=1.5)
            
            # No label on Right Axis, but keep ticks
            ax2.tick_params(axis='y', labelcolor='k')
            #ax2.set_yscale('log')
            
            # Legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False, fontsize=8)
            
        # X-label only on bottom
        axs[1].set_xlabel(r'$\omega$ (MHz)')
        
        plt.tight_layout()
        filename = f"spectra_overlay_{suffix}_pub.pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"Saved {filename}")
        plt.close(fig)

if __name__ == "__main__":
    generate_publication_plot()
    generate_all_M_plot()
    generate_best_M_plot()
    generate_spectra_overlay_plot()
