"""Plotting stage: draws the paper's "best-known vs best-NT CZ pulse sequence"
comparison figure (`showcase_pulse_sequences.pdf`).

Where this sits in the pipeline
--------------------------------
This module is pure visualization (`qns2q.viz`) and belongs to the **control**
arm of the two-arm pipeline described in the repo's CLAUDE.md (characterize ->
reconstruct -> control -> viz). It is Stage 4 (figure generation): it does no
physics computation itself, it only re-draws pulse timings that
`qns2q.control.cz` (Stage 3a, the CZ-gate optimizer) already computed and saved.

Input: `<run folder>/plotting_data/plotting_data_cz_v2<tag>.npz`, written by
`control/cz.py`'s CZ optimization loop. That file carries, for the best gate
time scanned, the pulse-timing arrays of two competing CZ sequences:
  * the "best known" sequence -- the best performer from the library of
    standard dynamical-decoupling sequences (CDD / mqCDD permutations;
    `control/cz.py::construct_pulse_library`) evaluated against the
    characterized noise, i.e. what an experimenter would use *without* the
    noise-tailoring machinery in this repo;
  * the "best NT" (noise-tailored) sequence -- the pulse timing found by this
    repo's optimizer, tailored to the reconstructed noise spectra.
Each sequence is stored as `pt1`/`pt2`: qubit-1 and qubit-2 pulse times
(arrays that always include the endpoints 0 and the gate duration T), plus the
scalar T_seq for that sequence.

Output: a 3-row x 2-column PDF of the pulse sequences' "switching functions"
(the +-1 toggling-frame square wave standard in dynamical-decoupling / filter-
function theory: +1 between an even number of applied pi pulses, -1 between an
odd number) for qubit 1, qubit 2, and the *effective interaction channel*
(the two-qubit ZZ/Ising coupling, which toggles sign at every pulse applied to
either qubit -- see `make_tk12` below). Saved under
`<run folder>/figures/publication/pulse_sequence_comparison_pub.pdf`; per
FIGURE_PROVENANCE.md this file is the one manually copied/renamed to the
paper's `showcase_pulse_sequences.pdf`.

Call graph: only entered via `scripts/run_cz_pulse_plot.py`, a thin wrapper
that `runpy`-executes this module's `__main__` block (so this file's own
`--folder`/`--tag` argparse flags, added in CLEANUP-0616, are the CLI contract
-- see FIGURE_PROVENANCE.md's `showcase_pulse_sequences.pdf` row and CLAUDE.md's
Stage 4 command list). It reads data produced by `qns2q.control.cz` and uses
`qns2q.paths` for the regime-aware run-folder path; nothing else in the repo
imports this module.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: this script only saves a PDF, it never opens a GUI window (needed so it also works over SSH / in batch jobs with no display).
import matplotlib.pyplot as plt
import numpy as np
import os
from qns2q.paths import run_path, project_root

def make_tk12(tk1, tk2):
    """
    Constructs the interaction (two-qubit coupling) frame's switching times
    from the two single-qubit pulse-time arrays.

    Physics: a pi pulse on *either* qubit flips the sign of the effective ZZ
    (Ising) coupling term in the toggling frame used for the CZ gate's filter-
    function calculation, so the coupling channel's switching times are simply
    the union of both qubits' internal pulse times, sorted into time order.
    This is a plain-NumPy reimplementation of the JAX-jitted
    `control/cz.py::make_tk12` -- duplicated here (rather than imported) so
    this plotting-only script stays lightweight and does not need to import
    JAX or the heavier `control.cz` module just to draw a figure; the two
    must be kept logically identical (see also the JAX version in
    `control/idle.py`).

    Parameters
    ----------
    tk1, tk2 : array_like
        Pulse-time arrays for qubit 1 and qubit 2, each including the two
        endpoints 0 and T (so `tk1[1:-1]` are the *internal* pulse times).

    Returns
    -------
    ndarray
        Combined, sorted switching-time array for the interaction channel,
        again bookended by 0 and T (T taken from `tk1[-1]`; both qubits share
        the same total sequence duration by construction).
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
    Samples the +-1 toggling-frame "switching function" y(t) for a single
    channel (qubit 1, qubit 2, or the combined interaction channel from
    `make_tk12`) on a uniform time grid, for plotting as a step function.

    y(t) starts at +1 and flips sign at each *internal* pulse time (i.e. each
    entry of `pulse_times` other than the 0/T endpoints) -- this models
    instantaneous (hard) pi pulses, the usual idealization in dynamical-
    decoupling filter-function plots.

    Parameters
    ----------
    pulse_times : array_like
        Pulse times for this channel, including the endpoints 0 and T.
    T : float
        Total sequence duration (defines the plotted time window).
    num_points : int, default 2000
        Resolution of the time grid; purely a plotting-smoothness knob (higher
        gives visually sharper step edges), it does not affect the physics.

    Returns
    -------
    t_grid, y : ndarray, ndarray
        The time grid and the corresponding +-1 switching-function values.
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

def generate_pulse_plot(folder=None, tag=""):
    """
    Build and save the "best-known vs best-NT CZ pulse sequence" comparison
    figure (the paper's `showcase_pulse_sequences.pdf`).

    Loads the winning pulse timings that `control/cz.py`'s CZ optimizer
    already found and saved (see the module docstring for what "best known"
    and "best NT" mean physically), converts each into its three switching
    functions (qubit 1, qubit 2, effective interaction channel), and draws
    them as a 3-row x 2-column grid of step plots: left column = best known
    (library) sequence, right column = best noise-tailored (NT/optimized)
    sequence.

    Parameters
    ----------
    folder : str or None, default None
        Run folder to read `plotting_data/plotting_data_cz_v2<tag>.npz` from.
        May be an absolute path or a path relative to the repo root
        (`qns2q.paths.project_root()`). `None` (the default) falls back to
        the active regime's canonical NoSPAM folder via `qns2q.paths.run_path()`
        (regime selected by the `QNS2Q_REGIME` env var, per CLAUDE.md).
        Mirrors the `--folder` CLI flag below.
    tag : str, default ""
        Filename tag appended to `plotting_data_cz_v2` (e.g. `_cap` for the
        showcase run's `plotting_data_cz_v2_cap.npz` -- CLAUDE.md's
        CLEANUP-0616 note: the showcase data carries this vestigial `_cap`
        filename tag from an earlier naming scheme). Mirrors the `--tag` CLI
        flag below.

    Side effects
    ------------
    Writes `<folder>/figures/publication/pulse_sequence_comparison_pub.pdf`
    (created if missing). Prints a message and returns early (no exception)
    if the data file is missing or missing the expected keys, so a bad
    `--folder`/`--tag` combination fails loudly in the console rather than
    crashing with a traceback.
    """
    # 1. Configuration
    # ----------------
    FIG_WIDTH = 7.0
    FIG_HEIGHT = 4.0

    plt.rcParams.update({
        "text.usetex": True,             # render all text (labels, ticks) through a real LaTeX install so math like $y_{12}(t)$ typesets identically to the paper -- requires a working `latex` binary on PATH.
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

    # Okabe-Ito colorblind-safe palette (blue / vermillion): chosen so the two
    # sequences stay distinguishable for colorblind readers and in grayscale print.
    COLOR_KNOWN = '#0072B2'     # Blue
    COLOR_OPT = '#D55E00'       # Vermillion

    # 2. Load Data
    # ------------
    if folder is None:
        base = run_path()  # active-regime canonical NoSPAM folder (QNS2Q_REGIME env var; see qns2q.paths)
    else:
        # Accept either an absolute path or one given relative to the repo root,
        # so `--folder DraftRun_NoSPAM_showcase_cap` works from any CWD.
        base = folder if os.path.isabs(folder) else os.path.join(project_root(), folder)
    data_file = os.path.join(base, "plotting_data", f"plotting_data_cz_v2{tag}.npz")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        with np.load(data_file) as data:
            # Check if keys exist
            if 'best_known_seq_pt1' not in data or 'best_opt_seq_pt1' not in data:
                print("Error: Best sequence data not found in npz file.")
                return

            # pt1/pt2: pulse-time arrays (endpoints 0, T included) for qubit 1/2;
            # "known" = best library (CDD/mqCDD) sequence, "opt" = best noise-
            # tailored (NT) sequence found by control/cz.py's optimizer -- see
            # this module's docstring for the physical meaning of each.
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
    known_pt12 = make_tk12(known_pt1, known_pt2)  # interaction-channel switching times: derived from both qubits' pulses, not stored directly in the npz
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
    # `step(..., where='post')` draws the +-1 value as held constant from each
    # sample point until the *next* one (a "staircase"), which is the correct
    # rendering of a piecewise-constant switching function sampled on a grid.
    fig, axs = plt.subplots(3, 2, sharex='col', sharey=True)
    # axs shape: (3, 2)
    # Col 0: Known, Col 1: Optimized

    # --- Column 0: Known ---
    # Row 0: Qubit 1
    axs[0, 0].step(t_known_1, y_known_1, where='post', color=COLOR_KNOWN)
    axs[0, 0].set_title("(a) Best CDD Sequence")

    # Row 1: Qubit 2
    axs[1, 0].step(t_known_2, y_known_2, where='post', color=COLOR_KNOWN)

    # Row 2: Interaction
    axs[2, 0].step(t_known_12, y_known_12, where='post', color=COLOR_KNOWN)
    axs[2, 0].set_xlabel(r"$t/\tau$")

    # --- Column 1: Optimized ---
    # Row 0: Qubit 1
    axs[0, 1].step(t_opt_1, y_opt_1, where='post', color=COLOR_OPT)
    axs[0, 1].set_title("(b) Best NT Sequence")

    # Row 1: Qubit 2
    axs[1, 1].step(t_opt_2, y_opt_2, where='post', color=COLOR_OPT)

    # Row 2: Interaction
    axs[2, 1].step(t_opt_12, y_opt_12, where='post', color=COLOR_OPT)
    axs[2, 1].set_xlabel(r"$t/\tau$")

    # 5. Styling & Labels
    # -------------------
    # y_1, y_2: single-qubit toggling frames; y_12: the derived interaction-
    # channel toggling frame (see make_tk12) -- these three row labels match
    # the physics-notation used in the manuscript for the CZ filter functions.
    row_labels = [r"$y_1(t)$", r"$y_2(t)$", r"$y_{12}(t)$"]

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
    # Mirrors the <run folder>/figures/... convention used by the rest of the
    # viz/plotting code; "publication" separates this print-ready copy from
    # any exploratory/diagnostic plots a run folder might also hold.
    output_dir = os.path.join(data_dir, "figures", "publication")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Successfully generated pulse comparison plot: {output_path}")

if __name__ == "__main__":
    # This block only runs when the module is executed as a script, not on a
    # plain `import`. `scripts/run_cz_pulse_plot.py` triggers it via
    # `runpy.run_module(..., run_name="__main__")`, which fakes exactly that
    # "executed as __main__" condition while forwarding this process's
    # command-line arguments (`sys.argv`) through untouched -- so `--folder`/
    # `--tag` below are this file's real CLI contract even though users
    # normally invoke the thin wrapper script instead of this file directly.
    import argparse
    ap = argparse.ArgumentParser(
        description="Plot the best-known vs best-NT CZ pulse-sequence comparison "
                    "(the paper's showcase_pulse_sequences figure).")
    ap.add_argument("--folder", default=None,
                    help="run folder holding plotting_data/, relative to the repo root "
                         "or absolute (e.g. DraftRun_NoSPAM_showcase_cap). "
                         "Default: the active QNS2Q_REGIME's canonical NoSPAM folder.")
    ap.add_argument("--tag", default="",
                    help="filename tag on the plotting-data npz, e.g. _cap for "
                         "plotting_data_cz_v2_cap.npz.")
    args = ap.parse_args()
    generate_pulse_plot(folder=args.folder, tag=args.tag)
