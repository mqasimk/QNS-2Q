"""
Standalone plotting script for optimization results.

Loads saved NPZ data from optimization runs and generates all figures
without re-running the optimization.

Usage:
    python plot_optimization.py --data-dir ../DraftRun_NoSPAM_Feature --gate-type cz
    python plot_optimization.py --data-dir ../DraftRun_NoSPAM_Boring --gate-type id
    python plot_optimization.py --data-dir ../DraftRun_NoSPAM_Boring --gate-type id --all-m
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np

import plot_utils


def load_plot_config(npz_path, base_path):
    """Load NPZ and return a PlotConfig namespace compatible with plot_utils functions.

    Returns
    -------
    config : SimpleNamespace
        Duck-typed config with attributes: path, w, w_max, SMat, M, Tg.
    data : NpzFile
        The raw loaded NPZ data.
    """
    data = np.load(npz_path, allow_pickle=True)

    config = SimpleNamespace()
    config.path = base_path
    config.w = data['w']
    config.w_max = float(data['w_max'])
    config.SMat = data['SMat_real'] + 1j * data['SMat_imag']
    config.M = int(data['M'])
    config.Tg = float(data['Tg'])

    return config, data


def plot_single_run(data_dir, gate_type):
    """Generate all plots for a single optimization run."""
    if gate_type == 'cz':
        filename = "plotting_data_cz_v2.npz"
    else:
        filename = "plotting_data_id_v4.npz"

    npz_path = os.path.join(data_dir, "plotting_data", filename)
    if not os.path.exists(npz_path):
        print(f"Error: Data file not found at {npz_path}")
        print("Run the optimization first, or check the --data-dir path.")
        return

    config, data = load_plot_config(npz_path, data_dir)

    taxis = data['taxis']
    infs_known = data['infs_known']
    infs_opt = data['infs_opt']
    infs_nopulse = data['infs_nopulse']
    tau = float(data['tau'])
    min_gate_time = float(data['min_gate_time']) if 'min_gate_time' in data else None

    # 1. Infidelity vs gate time
    save_path = os.path.join(
        plot_utils.get_figures_dir(config.path),
        f"infs_GateTime_{gate_type}.pdf"
    )
    plot_utils.plot_infidelity_vs_gatetime(
        taxis, infs_known, taxis, infs_opt, infs_nopulse,
        tau, save_path, min_gate_time=min_gate_time
    )

    # 2. Reconstruct sequences
    best_known_seq = None
    best_opt_seq = None
    T_seq_best_known = None
    T_seq_best_opt = None

    if 'best_known_seq_pt1' in data:
        best_known_seq = (data['best_known_seq_pt1'], data['best_known_seq_pt2'])
        T_seq_best_known = float(data['T_seq_best_known'])
    if 'best_opt_seq_pt1' in data:
        best_opt_seq = (data['best_opt_seq_pt1'], data['best_opt_seq_pt2'])
        T_seq_best_opt = float(data['T_seq_best_opt'])

    # 3. Detailed sequence plots
    suffix = f"_{gate_type}"
    M = config.M
    label_k = f"Best Known Sequence ({gate_type.upper()})"
    label_o = f"Best Optimized Sequence ({gate_type.upper()})"

    if best_known_seq or best_opt_seq:
        if best_known_seq and best_opt_seq and T_seq_best_known == T_seq_best_opt:
            T_seq = T_seq_best_known
            plot_utils.plot_comparison(config, best_known_seq, best_opt_seq, T_seq, filename_suffix=suffix)
            plot_utils.plot_filter_functions(config, best_known_seq, best_opt_seq, T_seq, filename_suffix=suffix)
            plot_utils.plot_filter_functions_with_spectra(config, best_known_seq, best_opt_seq, T_seq, filename_suffix=suffix)
            plot_utils.plot_spectra_filter_overlay_6(config, best_known_seq, T_seq, label_k)
            plot_utils.plot_spectra_filter_overlay_6(config, best_opt_seq, T_seq, label_o)
        else:
            if best_known_seq:
                plot_utils.plot_comparison(config, best_known_seq, None, T_seq_best_known, filename_suffix=suffix + "_known")
                plot_utils.plot_filter_functions(config, best_known_seq, None, T_seq_best_known, filename_suffix=suffix + "_known")
                plot_utils.plot_filter_functions_with_spectra(config, best_known_seq, None, T_seq_best_known, filename_suffix=suffix + "_known")
                plot_utils.plot_spectra_filter_overlay_6(config, best_known_seq, T_seq_best_known, label_k)
            if best_opt_seq:
                plot_utils.plot_comparison(config, None, best_opt_seq, T_seq_best_opt, filename_suffix=suffix + "_opt")
                plot_utils.plot_filter_functions(config, None, best_opt_seq, T_seq_best_opt, filename_suffix=suffix + "_opt")
                plot_utils.plot_filter_functions_with_spectra(config, None, best_opt_seq, T_seq_best_opt, filename_suffix=suffix + "_opt")
                plot_utils.plot_spectra_filter_overlay_6(config, best_opt_seq, T_seq_best_opt, label_o)

        plot_utils.plot_noise_correlations(config)

        if best_known_seq:
            plot_utils.plot_control_correlations(config, best_known_seq, T_seq_best_known, M, label_k)
            plot_utils.plot_generalized_filter_functions(config, best_known_seq, T_seq_best_known, label_k)
        if best_opt_seq:
            plot_utils.plot_control_correlations(config, best_opt_seq, T_seq_best_opt, M, label_o)
            plot_utils.plot_generalized_filter_functions(config, best_opt_seq, T_seq_best_opt, label_o)

    print(f"\nAll plots saved to {plot_utils.get_figures_dir(config.path)}")


def plot_all_m(data_dir):
    """Generate aggregate multi-M plots from optimization_data_all_M.npz."""
    npz_path = os.path.join(data_dir, "optimization_data_all_M.npz")
    if not os.path.exists(npz_path):
        print(f"Error: Multi-M data file not found at {npz_path}")
        print("Run the ID optimization with multiple M values first.")
        return

    data = np.load(npz_path, allow_pickle=True)

    M_values = list(data['M_values'])
    tau = float(data['tau'])

    # Reconstruct results_by_M dict expected by plot_infidelity_vs_gatetime_all_M
    results_by_M = {}
    for m in M_values:
        prefix = f"M{m}_"
        gate_times = data[prefix + 'gate_times']
        infs_known = data[prefix + 'infs_known']
        labels_known = data[prefix + 'labels_known']
        infs_opt = data[prefix + 'infs_opt']
        labels_opt = data[prefix + 'labels_opt']
        infs_nopulse = data[prefix + 'infs_nopulse']

        results_by_M[m] = {
            'gate_times': gate_times,
            'known': list(zip(infs_known, labels_known)),
            'opt': list(zip(infs_opt, labels_opt)),
            'nopulse': list(infs_nopulse),
        }

    # Use data_dir as base_path for figure output
    figures_dir = plot_utils.get_figures_dir(data_dir)

    # Extract longest gate time results for infidelity vs M plot
    known_infs = []
    known_labels = []
    opt_infs = []
    opt_labels = []
    nopulse_infs = []

    for m in M_values:
        res = results_by_M[m]
        k_inf, k_label = res['known'][-1]
        o_inf, o_label = res['opt'][-1]
        known_infs.append(k_inf)
        known_labels.append(k_label)
        opt_infs.append(o_inf)
        opt_labels.append(o_label)
        nopulse_infs.append(res['nopulse'][-1])

    plot_utils.plot_infidelity_vs_M_labeled(
        M_values, known_infs, known_labels, opt_infs, opt_labels, nopulse_infs,
        os.path.join(figures_dir, "infs_vs_M_id_v4.pdf")
    )

    plot_utils.plot_infidelity_vs_gatetime_all_M(
        results_by_M, tau,
        os.path.join(figures_dir, "infs_GateTime_id_v4_all_M.pdf")
    )

    print(f"\nAll multi-M plots saved to {figures_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate optimization plots from saved data.'
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to the data directory (e.g., ../DraftRun_NoSPAM_Feature)'
    )
    parser.add_argument(
        '--gate-type', choices=['cz', 'id'], required=True,
        help='Gate type: cz (CZ gate) or id (identity gate)'
    )
    parser.add_argument(
        '--all-m', action='store_true',
        help='Also generate multi-M aggregate plots (ID gate only)'
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    plot_single_run(data_dir, args.gate_type)

    if args.all_m:
        if args.gate_type != 'id':
            print("Warning: --all-m is only applicable to ID gate optimization. Skipping.")
        else:
            plot_all_m(data_dir)
