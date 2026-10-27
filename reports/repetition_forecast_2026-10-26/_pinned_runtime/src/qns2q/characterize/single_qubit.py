"""
Single-qubit variant of the two-qubit QNS experiment runner -- used ONLY to
produce one standalone paper figure, not part of the main Stage 1/2 pipeline.

Physics purpose. The published figure ``C_1_0_MT_vs_M.pdf`` demonstrates that the
QNS reconstruction coefficient ``C_{1,0}(MT)`` (built here from qubit 1's raw,
SPAM-corrupted measurement statistics, WITHOUT dividing out the residual
state-prep error -- ``spMit=False`` below) grows LINEARLY in the number of
pulse-sequence repetitions ``M``. A linear fit's slope recovers the SPAM-free
noise estimate, while its y-intercept isolates exactly the leftover SPAM
contamination as a closed-form combination of the readout visibility alpha_M
and the state-prep error alpha_SP (see the companion paper's Eq.
C_approx_concise and the figure caption for the exact intercept formula). This
is the mathematical fact that motivates the package's SPAM-mitigated/robust
QNS protocols (CLAUDE.md's "SPAM pipeline" section): this module runs the
single-qubit toy experiment sweeping ``M`` that demonstrates it, and makes the
plot.

Where this sits in the pipeline. It belongs conceptually to the "characterize"
arm (QNS experiment simulation -> spectral reconstruction) alongside
``qns2q.characterize.experiments`` + ``qns2q.characterize.reconstruct``, but it
is a SEPARATE, self-contained script: it does not feed the two-qubit Stage 1/2
run folders (``DraftRun_NoSPAM_*``/``DraftRun_SPAM_*``) and nothing in the rest
of the package imports from it. Despite the "single-qubit" name, the simulator
underneath is the SAME two-qubit-plus-bath 8-dimensional Hilbert space used by
the full pipeline (see CLAUDE.md, "3-Qubit Hilbert Space Convention"): the
correlation-function builders imported below (``make_c_a_0_mt`` etc.) prepare
qubit 2 in a fixed reference state and only report the single-qubit coefficient
for the measured qubit (``l_index`` in ``main()``) -- i.e. this file exercises
the shared two-qubit machinery restricted to a single-qubit observable, rather
than a genuinely separate one-qubit simulator.

Inputs / outputs. There is no upstream ``.npz`` to read: every experiment
parameter is a default on ``QNSExperimentConfig``, and the noise spectra come
from ``qns2q.noise.spectra`` at import time (whichever regime ``QNS2Q_REGIME``
selects -- see CLAUDE.md). Running ``main()`` (only via the ``__main__`` guard
at the bottom of this file) writes ``C_1_0_MT_vs_M.pdf`` and
``C_1_0_MT_vs_M.npz`` directly into the current working directory (NOT a
regime run-folder -- see FIGURE_PROVENANCE.md), so it must be launched from the
repo root; ``scripts/run_single_qubit.py`` does exactly that by executing this
file's ``__main__`` block via ``runpy``.

The main components are:
- QNSExperimentConfig: A dataclass for storing all the experiment parameters.
- ExperimentRunner: A class that takes a configuration object and runs the
  experiments.
- A main execution block that demonstrates how to use these components.
"""


import os
import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from qns2q.model.observables import (make_c_12_0_mt, make_c_12_12_mt, make_c_a_0_mt,
                         make_c_a_b_mt)
from qns2q.noise.spectra import S_11, S_22, S_1212
from qns2q.model.trajectories import make_noise_mat_arr, solver_prop
from qns2q.paths import project_root

# Fixed RNG seed. solver_prop() draws its per-shot noise keys from np.random,
# so seeding makes the published C_1_0_MT_vs_M figure reproducible.
RANDOM_SEED = 20260608


@dataclass
class QNSExperimentConfig:
    """
    Configuration for the single-qubit QNS toy experiment (mirrors the
    two-qubit ``qns2q.characterize.experiments.QNSExperimentConfig``, but with
    only the single-qubit essentials -- no SPAM-protocol switch, no run-folder
    resolution via ``qns2q.paths.run_folder``).

    Constructing this object does real work: ``__post_init__`` derives the
    time/frequency grids and the readout confusion matrix from the raw fields
    below (this is the "config dataclass with a __post_init__" pattern used
    throughout the package -- building the object is cheap here, unlike the
    two-qubit configs, which also load prior-stage .npz files in __post_init__).

    Attributes:
        tau: Time unit (tau = 1 everywhere in this codebase; SI legacy anchor
            25 ns). All other times below are already expressed in units of tau.
        T: Total duration of one pulse-sequence block.
        M: Number of block repetitions (the x-axis of the published figure).
        t_grain: Number of time samples per block (propagator time grid).
        truncate: Number of measurement "harmonics" -- i.e. the number of
            distinct control times ``c_times = T/k`` for k=1..truncate at which
            the coefficient is estimated (see ``__post_init__``). Also plotted
            as one curve each in ``main()``.
        w_grain: Number of frequency bins used to synthesize the noise
            trajectories (see ``qns2q.model.trajectories.make_noise_mat_arr``).
        spec_vec: The noise self-spectra (S_11, S_22, S_1212, i.e. qubit-1,
            qubit-2, and Ising "12" spectra of the active QNS2Q_REGIME) this
            run is labeled with. NOTE: these functions are NOT actually fed
            into the noise-trajectory synthesis below (that uses its own
            fixed component spectra, S_el_A/S_el_B/S_nuc_1/S_nuc_2, internal
            to ``make_noise_mat_arr``); `spec_vec` only supplies human-readable
            names (`spec_vec_names`) recorded for provenance in `params.npz`.
        a_sp: State-preparation (SP) error along Z, per qubit -- the paper's
            alpha_SP^z; a_sp=1 means perfect prep. Complex-valued because it
            multiplies population terms only (kept as jnp for dtype parity
            with `c` below).
        c: State-preparation error in the X/Y (transverse/coherence) plane,
            per qubit -- complex Bloch-vector components entering the prepared
            density matrix off-diagonal (see `make_init_state`).
        a1, b1, a2, b2: Raw single-qubit readout (measurement, "M") confusion
            probabilities per qubit (qubit-1: a1/b1, qubit-2: a2/b2); combined
            below into the paper's visibility alpha_M and asymmetry delta,
            which parameterize the two-qubit POVM confusion matrix `CM`.
        spMit: SPAM-mitigation flag passed through to the correlation-function
            estimators as `sp_mit`; when True, the estimator divides by an
            estimate of alpha_SP^z (oracle: the true `a_sp`, since this file
            has no calibration step) -- see `_resolve_a_sp_div` in
            `qns2q.model.observables` for the exact behavior.
        gamma, gamma_12: Vestigial decay-rate fields from an earlier version of
            the noise model. The current noise model (qns2q.noise.spectra,
            see NOISE_MODEL_SPEC.md) no longer takes gamma/gamma_12 as inputs
            -- `make_noise_mat_arr` actively rejects them if passed -- and
            `_make_noise_mats` below does not pass them. They are kept only
            because `save_results` records the whole config for provenance;
            they have no effect on the simulated physics.
        n_shots: Number of Monte-Carlo noise realizations averaged per data
            point (statistical/shot noise, not readout/projective noise).
        fname: Output subfolder name under the repo root (only used if
            `save_results` is called; `main()` below does not call it).
        parent_dir: Unused by any path in this file (kept for parity with the
            two-qubit config); do not rely on it.
    """
    tau: jnp.float32 = 1.0   # tau units (legacy SI anchor: 25 ns)
    M: jnp.int32 = 18
    t_grain: jnp.int32 = 1500
    truncate: jnp.int32 = 5
    w_grain: jnp.int32 = 1000
    spec_vec: list = field(default_factory=lambda: [S_11, S_22, S_1212])
    a_sp: np.ndarray = field(default_factory=lambda: jnp.array([0.99, 0.98]))
    c: np.ndarray = field(
        default_factory=lambda: np.array(
            [jnp.array(0. + 0.01 * 1j),
             jnp.array(0. - 0.02 * 1j)]))
    a1: jnp.float32 = 0.990
    b1: jnp.float32 = 0.980
    a2: jnp.float32 = 0.985
    b2: jnp.float32 = 0.970
    spMit: bool = False
    # NB: this `tau` is the class-body name bound by the `tau: ... = 1.0` field
    # above (evaluated once, at class-definition time) -- NOT `self.tau` on a
    # particular instance. So passing QNSExperimentConfig(tau=2.0) changes that
    # instance's self.tau but does NOT rescale T; T's default stays 160.0. This
    # is a general dataclass-field gotcha, not specific to this line.
    T: jnp.float32 = 160*tau
    gamma: jnp.float32 = T / 14
    gamma_12: jnp.float32 = T / 28
    n_shots: jnp.int32 = 2000
    fname: str = "DraftRun_MScaling"
    parent_dir: str = os.pardir

    def __post_init__(self):
        """Derive the time/frequency grids and the readout confusion matrix
        from the raw fields above. Runs automatically right after the
        generated ``__init__`` (dataclass machinery: any code here executes
        once, right after all the fields above are assigned)."""
        # wmax: angular-frequency cutoff for the noise-synthesis grid, set so
        # `truncate` harmonics of the base sequence period T fit inside it.
        self.wmax = 2 * np.pi * self.truncate / self.T
        self.t_b = jnp.linspace(0, self.T, self.t_grain)  # time grid for one block
        self.t_vec = jnp.linspace(0, self.M * self.T,
                                  self.M * jnp.size(self.t_b))  # time grid for the full M-block sequence
        # The `truncate` distinct control times at which the coefficient is
        # sampled: c_times[k-1] = T/k for k = 1..truncate. These are the
        # "harmonic index l" curves plotted in main() (l = k here).
        self.c_times = jnp.array([self.T / n for n in range(1, self.truncate + 1)])
        # Store names of spectra functions for saving, as functions aren't picklable
        self.spec_vec_names = [f.__name__ for f in self.spec_vec]
        # a_m: measurement visibility alpha_M per qubit (1 = perfect readout).
        # delta: measurement asymmetry (0/1-outcome imbalance) per qubit.
        # Both are derived from the raw per-qubit confusion probabilities
        # (a1,b1,a2,b2) and used below to build the two-qubit POVM confusion
        # matrix CM that the estimators invert to correct for readout error.
        self.a_m = np.array([self.a1 + self.b1 - 1, self.a2 + self.b2 - 1])
        self.delta = np.array([self.a1 - self.b1, self.a2 - self.b2])
        # CM: two-qubit readout confusion matrix, built as the Kronecker
        # (tensor) product of the two single-qubit 2x2 confusion matrices
        # (each row/column pair parameterized by that qubit's alpha_M/delta).
        # This is the matrix the estimators in qns2q.model.observables invert
        # to correct measured populations for readout error.
        self.CM = jnp.kron(
            jnp.array([[
                0.5 * (1 + self.a_m[0] + self.delta[0]),
                0.5 * (1 - self.a_m[0] + self.delta[0])
            ],
                [
                    0.5 * (1 - self.a_m[0] - self.delta[0]),
                    0.5 * (1 + self.a_m[0] - self.delta[0])
                ]]),
            jnp.array([[
                0.5 * (1 + self.a_m[1] + self.delta[1]),
                0.5 * (1 - self.a_m[1] + self.delta[1])
            ],
                [
                    0.5 * (1 - self.a_m[1] - self.delta[1]),
                    0.5 * (1 + self.a_m[1] - self.delta[1])
                ]]))


class ExperimentRunner:
    """
    Runs a series of single-qubit QNS toy experiments for one
    ``QNSExperimentConfig`` (i.e. one fixed M): synthesizes the noise
    trajectories once, then lets the caller run one or more named
    correlation-function estimates (`run_experiment`) against them.
    """

    def __init__(self, config: QNSExperimentConfig):
        """
        Initializes the ExperimentRunner: stashes the config, ensures the
        (per-config) output folder exists, and eagerly synthesizes the noise
        trajectories for this config's time/frequency grid. This means
        constructing an ExperimentRunner does real (if cheap) work and
        creates a directory as a side effect -- see `_setup_output_directory`.

        Args:
            config: The configuration for the experiments.
        """
        self.config = config
        self.path = self._setup_output_directory()
        self.noise_mats = self._make_noise_mats()
        self.results = {}

    def _setup_output_directory(self):
        """
        Creates `<repo root>/<config.fname>` if it doesn't already exist and
        returns its path. NOTE: this always creates the folder on disk, even
        though `main()` below never calls `save_results()` -- so re-running
        `main()` will leave a (harmless, empty) `DraftRun_MScaling/` folder
        at the repo root; that is expected, not a bug.
        """
        path = os.path.join(project_root(), self.config.fname)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _make_noise_mats(self):
        """
        Precomputes this config's noise-synthesis matrices (sine/cosine basis
        for the classical noise trajectories), via
        `qns2q.model.trajectories.make_noise_mat_arr`. Shared across every
        `run_experiment` call on this runner so the (expensive) synthesis
        setup happens only once per M value.
        """
        return jnp.array(
            make_noise_mat_arr(
                'make',
                t_vec=self.config.t_vec,
                w_grain=self.config.w_grain,
                wmax=self.config.wmax,
                truncate=self.config.truncate))

    def run_experiment(self, exp_name: str, pulse_sequence: list,
                       exp_type: str, **kwargs):
        """
        Runs one named correlation-function estimate and stores its mean and
        standard error in `self.results[exp_name]` / `self.results[exp_name +
        '_err']`, one value per entry of `self.config.c_times`.

        Args:
            exp_name: Key under which to store the result in `self.results`
                (caller-chosen label, e.g. 'C_1_0_MT_1').
            pulse_sequence: The pulse sequence to use.
            exp_type: Which correlation-function estimator to run -- one of
                'C_12_0', 'C_12_12', 'C_a_0', 'C_a_b' (see `exp_map` below;
                these correspond to specific two-qubit reconstruction
                coefficients in `qns2q.model.observables`).
            **kwargs: Forwarded to the estimator (e.g. `l` = which physical
                qubit index is being measured, for the 'C_a_0'/'C_a_b' types).
        """
        print(f"Running experiment: {exp_name}")
        start_time = time.time()

        # Dict-based dispatch: map the string `exp_type` to the actual
        # estimator function to call, instead of an if/elif chain. Keeping
        # this map local (rebuilt each call) matches the original code; it
        # is cheap (four dict entries) and avoids any module-level state.
        exp_map = {
            'C_12_0': make_c_12_0_mt,
            'C_12_12': make_c_12_12_mt,
            'C_a_0': make_c_a_0_mt,
            'C_a_b': make_c_a_b_mt,
        }
        if exp_type not in exp_map:
            raise ValueError(f"Invalid experiment type: {exp_type}")

        exp_func = exp_map[exp_type]
        means, stderrs = exp_func(
            solver_prop,
            pulse_sequence,
            self.config.t_vec,
            self.config.c_times,
            self.config.CM,
            self.config.spMit,
            n_shots=self.config.n_shots,
            m=self.config.M,
            t_b=self.config.t_b,
            a_m=self.config.a_m,
            delta=self.config.delta,
            a_sp=self.config.a_sp,
            c=self.config.c,
            noise_mats=self.noise_mats,
            **kwargs)
        self.results[exp_name] = means
        self.results[exp_name + '_err'] = stderrs

        print(
            f"--- {exp_name} completed in {time.time() - start_time:.2f}s ---")

    def save_results(self):
        """
        Saves the config (as `params.npz`) and the accumulated
        `self.results` (as `results.npz`) into `self.path`. NOT called by
        `main()` below (see the commented-out call there) -- this method
        exists for interactive/standalone use of `ExperimentRunner`, e.g. if
        a student wants to inspect a single M's raw estimator output.
        """
        # Create a copy of the config dict to avoid modifying the original object.
        # The 'spec_vec' attribute contains function objects, which cannot be
        # pickled by `np.savez`. We pop it from the dictionary before saving.
        # The names of the spectra are saved in 'spec_vec_names' for reference.
        params_to_save = self.config.__dict__.copy()
        params_to_save.pop('spec_vec', None)
        params_to_save['tau'] = self.config.tau  # re-assert tau explicitly (already present via __dict__, but kept for clarity/safety)
        np.savez(
            os.path.join(self.path, "params.npz"),
            **params_to_save)
        np.savez(os.path.join(self.path, "results.npz"), **self.results)
        print(f"Results saved to {self.path}")

def main():
    """
    Main function to run the QNS experiments.

    Produces C_1_0_MT_vs_M.pdf: for a fixed measured qubit (``l_index``), the
    reconstruction observable C_{1,0}(MT) is evaluated at each of the
    ``truncate`` control-time harmonics (c_times = T/k, k = 1..truncate) and
    plotted against M. The plotted curves are indexed by that harmonic index --
    the superscript "(l)" in C_{1,0}^{(l)} -- which is distinct from
    ``l_index`` (the measured qubit) and from the CDD order; the symbol
    collision is noted in the figure caption.
    """
    # Pin the RNG: solver_prop draws its per-shot noise keys from np.random,
    # so seeding here makes the published figure reproducible.
    np.random.seed(RANDOM_SEED)

    m_values = list(range(5, 20))
    c_1_0_mt_1_results = []
    l_index = 1  # measured qubit (subscript in C_{1,0}); NOT the harmonic index

    # One full ExperimentRunner (its own noise-mat synthesis + Monte Carlo) per
    # M value, because the total sequence duration M*T and hence the time/
    # frequency grids in QNSExperimentConfig depend on M -- there is no way to
    # share a single runner across different M values.
    for m in m_values:
        print(f"Running experiment for M = {m}")
        config = QNSExperimentConfig(M=m)
        runner = ExperimentRunner(config)

        # 'C_1_0_MT_1' with pulse pair ['CDD1', 'CDD1-1/2'] and exp_type
        # 'C_a_0' is the SAME single-qubit-1 estimator used in the full
        # two-qubit pipeline (qns2q.characterize.experiments); it is run here
        # in isolation, swept over M, to produce this one figure. The pulse
        # pair (`pulse_sequence[0]` acts on qubit 1, `pulse_sequence[1]` on
        # qubit 2, per `qns2q.model.trajectories.make_y`) -- a CDD1 toggle on
        # the measured qubit plus the same toggle at half the period on the
        # (unmeasured, reference) qubit -- is what makes the estimated
        # coefficient a single-qubit-1 observable rather than a joint one.
        experiments = [
            ('C_1_0_MT_1', ['CDD1', 'CDD1-1/2'], 'C_a_0', {'l': l_index}),
        ]

        for exp_name, pulse_sequence, exp_type, kwargs in experiments:
            runner.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)
            c_1_0_mt_1_results.append(runner.results[exp_name])
        # runner.save_results() # Optional: decide if you want to save results for each M

    c_1_0_mt_1_results = np.array(c_1_0_mt_1_results)

    # Publication-quality plot settings
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif', size=12)
    plt.rc('axes', titlesize=14, labelsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)

    fig, ax = plt.subplots(figsize=(6, 4)) # Standard size for a single-column figure

    # One curve per control-time harmonic l = 1..truncate (the columns of the
    # results array, one per c_times entry). Build the legend from the actual
    # number of curves rather than a hardcoded 5-entry list, so it stays correct
    # if `truncate` changes.
    n_harmonics = c_1_0_mt_1_results.shape[1]
    for l in range(n_harmonics):
        ax.plot(m_values, c_1_0_mt_1_results[:, l], linestyle='-', marker='o',
                label=fr'$l={l + 1}$')

    ax.set_xlabel(r'$M$')
    ax.set_ylabel(fr'$C_{{1,0}}^{{(l)}}(MT)$')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Persist the exact figure-source data for reproducibility, and save under
    # the literal filename "C_1_0_MT_vs_M" that the manuscript's LaTeX
    # \includegraphics call and FIGURE_PROVENANCE.md both expect -- renaming
    # it here would silently break the figure link in the paper build.
    np.savez("C_1_0_MT_vs_M.npz",
             M_values=np.array(m_values),
             c_1_0_mt=c_1_0_mt_1_results,
             c_times=np.array(config.c_times),
             l_qubit=l_index,
             seed=RANDOM_SEED)
    plt.savefig("C_1_0_MT_vs_M.pdf", format='pdf', bbox_inches='tight')
    print("Plot saved to C_1_0_MT_vs_M.pdf")
    plt.show()


if __name__ == "__main__":
    main()
