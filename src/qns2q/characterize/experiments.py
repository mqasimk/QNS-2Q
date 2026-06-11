"""
This script provides a framework for running quantum noise spectroscopy (QNS)
experiments. It is designed to be modular and easy to configure, allowing
users to define and run a series of experiments with different pulse sequences
and parameters.

The main components are:
- QNSExperimentConfig: A dataclass for storing all the experiment parameters.
- ExperimentRunner: A class that takes a configuration object and runs the
  experiments.
- A main execution block that demonstrates how to use these components.

Author: [Q]
Date: [01/18/2026]
"""


import os
import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from qns2q.model.observables import (make_c_12_0_mt, make_c_12_12_mt, make_c_a_0_mt,
                         make_c_a_b_mt)
from qns2q.characterize.spam import (estimate_spam, make_c_12_0_mt_robust,
                                     make_c_12_12_mt_robust, make_c_a_0_mt_robust)
from qns2q.noise.spectra import S_11, S_22, S_1212
from qns2q.model.trajectories import (make_noise_mat_arr, solver_prop,
                                      solver_phase_coeffs_fast, phased_state)
from qns2q.model import observables as _observables
from qns2q.paths import run_folder, project_root

# Seed for reproducible QNS data. The per-shot noise keys in solver_prop are drawn
# with np.random (trajectories.solver_prop), so seeding here makes the whole
# experiment stage reproducible -- matching the optimizer/single-qubit scripts,
# which all use the same constant.
RANDOM_SEED = 20260608

# Correlation-function builders keyed by experiment type (shared by run_experiment
# and the DC time sweep).
_EXP_MAP = {
    'C_12_0': make_c_12_0_mt,
    'C_12_12': make_c_12_12_mt,
    'C_a_0': make_c_a_0_mt,
    'C_a_b': make_c_a_b_mt,
}

# SPAM-robust builders (raw + twisting + wringing; see characterize.spam). The
# C_a_b coefficients have no SPAM-robust estimator (paper Sec. SPAM-Robust QNS),
# so S_1_12 / S_2_12 are not accessible under the robust protocol.
_EXP_MAP_ROBUST = {
    'C_12_0': make_c_12_0_mt_robust,
    'C_12_12': make_c_12_12_mt_robust,
    'C_a_0': make_c_a_0_mt_robust,
}


@dataclass
class QNSExperimentConfig:
    """
    Configuration for QNS experiments.

    Attributes:
        T: The total time for the experiment.
        M: The number of blocks.
        t_grain: The number of time points in each block.
        truncate: The truncation order for the cumulant expansion.
        w_grain: The number of frequency points.
        spec_vec: A list of spectra to use.
        a_sp: The SPAM error parameters.
        c: The SPAM error parameters.
        a1, b1, a2, b2: The measurement operators.
        spMit: A flag for SPAM mitigation.
        spam_protocol: How the experiments handle the injected SPAM errors:
            'none'      -- legacy oracle behavior: invert the TRUE confusion matrix
                           and (if spMit) divide by the TRUE a_sp. With the default
                           identity SPAM parameters this is the NoSPAM pipeline.
            'raw'       -- no mitigation at all (identity CM, no SP division);
                           quantifies the SPAM-corrupted reconstruction.
            'mitigated' -- estimated-parameter mitigation (companion paper,
                           SPAM-Mitigated QNS): calibrate (delta_hat, products),
                           invert the ESTIMATED confusion matrix, divide by the
                           ESTIMATED a_sp; transverse SP components are removed
                           exactly by the prep-level phase twirl.
            'robust'    -- SPAM-robust estimators (twisting + wringing, raw
                           readout); harmonic experiments swept over M for the
                           downstream SPAM-intercept regression.
        spam_split_error: Relative error of the externally-supplied alpha_M vs
            alpha_SP^z gauge split used by the 'mitigated' protocol (khan2025);
            0 = faithful split. The measured products are always exact.
        m_sweep_robust: Repetition numbers for the robust M-regression; defaults
            to (M-4, M-2, M) when spam_protocol='robust'.
        gamma: The decay rate.
        gamma_12: The cross-decay rate.
        n_shots: The number of shots for the experiment.
        fname: The name of the folder to save the results in.
        parent_dir: The parent directory to save the results in.
    """
    # Time unit: the minimum pulse separation tau (tau = 1; all times in units of
    # tau, all frequencies in 1/tau, spectra dimensionless S*tau -- see
    # noise/spectra.py). The legacy SI anchor was tau = 25 ns.
    tau: jnp.float64 = 1.0
    M: jnp.int64 = 10
    t_grain: jnp.int64 = 3000
    truncate: jnp.int64 = 20
    w_grain: jnp.int64 = 500
    spec_vec: list = field(default_factory=lambda: [S_11, S_22, S_1212])
    a_sp: np.ndarray = field(default_factory=lambda: jnp.array([1., 1.]))
    c: np.ndarray = field(
        default_factory=lambda: np.array(
            [jnp.array(0. + 0. * 1j),
             jnp.array(0. + 0. * 1j)]))
    a1: jnp.float64 = 1.
    b1: jnp.float64 = 1.
    a2: jnp.float64 = 1.
    b2: jnp.float64 = 1.
    spMit: bool = False
    spam_protocol: str = 'none'
    # Projection (readout-sampling) noise: number of projective measurements per
    # noise realization. 0 = exact expectation values (the historical idealized
    # behavior: bars quote noise-ensemble sampling error only). Finite n_meas
    # adds multinomial readout statistics per shot on top.
    n_meas: int = 0
    spam_split_error: float = 0.0
    m_sweep_robust: tuple = ()
    # Bin-midpoint noise-synthesis grid: excludes the spurious w=0 static tone that
    # otherwise biases DC-sensitive observables by O(dw). Default True for new runs.
    midpoint: bool = True
    # Half-band override for the noise synthesis: the simulated world spans
    # [0, 2*synth_wmax] at resolution dw = synth_wmax/w_grain (make_noise_mat
    # convention). 0.0 = legacy behavior, half-band = wmax = 2*pi*truncate/T.
    # Set pi/2 to give a run a full control-Nyquist world [0, pi/tau] regardless
    # of its own comb reach (the two-scale Nyquist protocol needs the fine- and
    # high-band runs to live in the SAME world).
    synth_wmax: float = 0.0
    T: jnp.float64 = 160*tau
    gamma: jnp.float64 = T / 14
    gamma_12: jnp.float64 = T / 28
    n_shots: jnp.int64 = 10000
    fname: str = field(default_factory=run_folder)
    parent_dir: str = os.pardir

    def __post_init__(self):
        self.wmax = 2 * np.pi * self.truncate / self.T
        self.t_b = jnp.linspace(0, self.T, self.t_grain)
        self.t_vec = jnp.linspace(0, self.M * self.T,
                                 self.M * jnp.size(self.t_b))
        self.c_times = jnp.array([self.T / n for n in range(1, self.truncate + 1)])
        # Store names of spectra functions for saving, as functions aren't picklable
        self.spec_vec_names = [f.__name__ for f in self.spec_vec]
        self.a_m = np.array([self.a1 + self.b1 - 1, self.a2 + self.b2 - 1])
        self.delta = np.array([self.a1 - self.b1, self.a2 - self.b2])
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

        # --- SPAM-protocol resolution -------------------------------------------
        # The TRUE injected parameters are (a_sp, c) for SP and (a_m, delta) for M.
        # What varies per protocol is what the ESTIMATORS use:
        #   cm_use   -- confusion matrix inverted by the estimators (None = raw),
        #   a_sp_div -- SP divisor (None = legacy: true a_sp when spMit),
        #   c_prep   -- transverse SP component actually prepared (the mitigated
        #               protocol's phase twirl removes it exactly: every measured
        #               quantity is linear in rho_in, so averaging the twirl is
        #               identical to preparing the twirled, c=0 state).
        if self.spam_protocol not in ('none', 'raw', 'mitigated', 'robust'):
            raise ValueError(f"Invalid spam_protocol: {self.spam_protocol!r}")
        self.spam_estimate = None
        self.cm_use = self.CM
        self.a_sp_div = None
        self.c_prep = self.c
        if self.spam_protocol == 'raw':
            self.cm_use = None
            self.spMit = False
        elif self.spam_protocol == 'mitigated':
            est = estimate_spam(self.a_m, self.delta,
                                np.asarray(self.a_sp, dtype=float), self.c,
                                self.spam_split_error)
            self.spam_estimate = est
            self.cm_use = jnp.array(est.cm)
            self.a_sp_div = est.a_sp
            self.spMit = True
            self.c_prep = np.array([0. + 0.j, 0. + 0.j])
        elif self.spam_protocol == 'robust':
            self.cm_use = None
            self.spMit = False
            if not self.m_sweep_robust:
                self.m_sweep_robust = tuple(
                    m for m in (self.M - 4, self.M - 2, self.M) if m >= 2)
            if len(self.m_sweep_robust) < 2:
                raise ValueError("spam_protocol='robust' needs >= 2 M values "
                                 f"(got m_sweep_robust={self.m_sweep_robust})")


class ExperimentRunner:
    """
    Runs a series of QNS experiments based on a given configuration.
    """

    def __init__(self, config: QNSExperimentConfig, solver=None,
                 make_mats: bool = True):
        """
        Initializes the ExperimentRunner.

        Args:
            config: The configuration for the experiments.
            solver: Solver callable with `solver_prop`'s signature; defaults to
                the real trajectory solver. The record/replay SPAM pipeline
                passes a `PhaseRecorder` / `PhaseReplayer` here.
            make_mats: Skip the (expensive) noise-matrix build when False --
                a replaying runner never touches them.
        """
        self.config = config
        self.solver = solver_prop if solver is None else solver
        self.path = self._setup_output_directory()
        self.noise_mats = self._make_noise_mats() if make_mats else None
        self.results = {}

    def _setup_output_directory(self):
        """
        Creates the output directory if it doesn't exist.
        """
        path = os.path.join(project_root(), self.config.fname)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _make_noise_mats(self):
        """
        Generates the noise matrices.
        """
        return jnp.array(
            make_noise_mat_arr(
                'make',
                t_vec=self.config.t_vec,
                w_grain=self.config.w_grain,
                wmax=self.config.synth_wmax or self.config.wmax,
                truncate=self.config.truncate,
                midpoint=self.config.midpoint))

    def run_experiment(self, exp_name: str, pulse_sequence: list,
                       exp_type: str, c_times=None, **kwargs):
        """
        Runs a single experiment.

        Args:
            exp_name: The name of the experiment.
            pulse_sequence: The pulse sequence to use.
            exp_type: The type of experiment to run.
            c_times: Optional control-time vector override (default: config.c_times).
                The DC characterization experiments pass a short list of fast
                control times so they stay light.
            **kwargs: Additional arguments for the experiment function.
        """
        print(f"Running experiment: {exp_name}")
        start_time = time.time()

        ctimes = self.config.c_times if c_times is None else c_times
        means, stderrs = self._call_exp_func(
            exp_type, pulse_sequence, self.config.t_vec, ctimes,
            m=self.config.M, **kwargs)
        self.results[exp_name] = means
        self.results[exp_name + '_err'] = stderrs

        print(
            f"--- {exp_name} completed in {time.time() - start_time:.2f}s ---")

    def _call_exp_func(self, exp_type: str, pulse_sequence: list, t_vec, c_times,
                       m: int, **kwargs):
        """Dispatch one coefficient measurement through the active SPAM protocol.

        The estimator builder is selected from the robust map when
        spam_protocol='robust'; the confusion matrix / SP divisor are the
        protocol-resolved ``cm_use`` / ``a_sp_div`` (NOT necessarily the truth).
        State prep always injects the TRUE (a_sp, c_prep) parameters.
        """
        robust = self.config.spam_protocol == 'robust'
        exp_map = _EXP_MAP_ROBUST if robust else _EXP_MAP
        if exp_type not in exp_map:
            raise ValueError(
                f"Invalid experiment type for spam_protocol="
                f"{self.config.spam_protocol!r}: {exp_type}")
        exp_func = exp_map[exp_type]
        return exp_func(
            self.solver,
            pulse_sequence,
            t_vec,
            c_times,
            self.config.cm_use,
            self.config.spMit,
            n_shots=self.config.n_shots,
            m=m,
            t_b=self.config.t_b,
            a_m=self.config.a_m,
            delta=self.config.delta,
            a_sp=self.config.a_sp,
            c=self.config.c_prep,
            a_sp_div=self.config.a_sp_div,
            noise_mats=self.noise_mats,
            **kwargs)

    def run_experiment_msweep(self, exp_name: str, pulse_sequence: list,
                              exp_type: str, m_values, **kwargs):
        """Measure a harmonic observable at several repetition numbers M.

        Used by the SPAM-robust protocol: the comb coefficient scales linearly in
        M while the SPAM term is an M-independent intercept, so a linear
        regression over ``m_values`` (inversion.regress_observables_over_M)
        recovers the SPAM-free coefficient. Results are stored under
        ``{exp_name}_Mrep{m}``.
        """
        for m in m_values:
            m = int(m)
            print(f"Running experiment: {exp_name} [M={m}]")
            start_time = time.time()
            tvec_m = self.config.t_vec[:m * self.config.t_grain]
            means, stderrs = self._call_exp_func(
                exp_type, pulse_sequence, tvec_m, self.config.c_times,
                m=m, **kwargs)
            self.results[f'{exp_name}_Mrep{m}'] = means
            self.results[f'{exp_name}_Mrep{m}_err'] = stderrs
            print(f"--- {exp_name} [M={m}] completed in "
                  f"{time.time() - start_time:.2f}s ---")

    def run_dc_sweep(self, exp_name: str, pulse_sequence: list, exp_type: str,
                     m_sweep, dc_ct: float, **kwargs):
        """Measure a FID DC observable over a sweep of total evolution times t = m*T.

        Evolves m periods (t_vec[:m*t_grain]) for each m in ``m_sweep`` and records the
        decay exponent C(t) and its error as length-len(m_sweep) arrays. The adaptive
        DC slope fit (``inversion._ramsey_fit_dc``) then picks the measurable+linear
        window, so the effective measurement time auto-tracks the noise strength -- no
        per-spectrum tuning. ``dc_ct`` is the (fast) partner control time used by the
        FID/CDD3 self observables (irrelevant for the pulse-free FID/FID cross ones).
        """
        print(f"Running DC sweep: {exp_name} over m={list(int(m) for m in m_sweep)}")
        start_time = time.time()
        c_vals, c_errs = [], []
        for m in m_sweep:
            m = int(m)
            tvec_m = self.config.t_vec[:m * self.config.t_grain]
            means, stderrs = self._call_exp_func(
                exp_type, pulse_sequence, tvec_m, [dc_ct], m=m, **kwargs)
            c_vals.append(means[0])
            c_errs.append(stderrs[0])
        self.results[exp_name] = np.array(c_vals)
        self.results[exp_name + '_err'] = np.array(c_errs)
        # Total evolution times for the sweep (shared by all DC observables).
        self.results['dc_t_sweep'] = np.array([int(m) for m in m_sweep]) * self.config.T
        print(f"--- {exp_name} sweep done in {time.time() - start_time:.2f}s ---")

    def save_results(self):
        """
        Saves the parameters and results to .npz files.
        """
        # Create a copy of the config dict to avoid modifying the original object.
        # The 'spec_vec' attribute contains function objects, which cannot be
        # pickled by `np.savez`. We pop it from the dictionary before saving.
        # The names of the spectra are saved in 'spec_vec_names' for reference.
        params_to_save = self.config.__dict__.copy()
        params_to_save.pop('spec_vec', None)
        # SPAM-protocol fields: drop objects/None values np.savez can't store
        # plainly; persist the estimate as flat arrays for the reconstruction
        # stage and the record.
        params_to_save.pop('spam_estimate', None)
        params_to_save.pop('cm_use', None)
        if params_to_save.get('a_sp_div') is None:
            params_to_save.pop('a_sp_div', None)
        params_to_save['m_sweep_robust'] = np.array(self.config.m_sweep_robust,
                                                    dtype=int)
        est = self.config.spam_estimate
        if est is not None:
            params_to_save['est_a_m'] = est.a_m
            params_to_save['est_delta'] = est.delta
            params_to_save['est_a_sp'] = est.a_sp
            params_to_save['est_products'] = est.products
            params_to_save['est_cm'] = est.cm
        params_to_save['tau'] = self.config.tau
        # OPT-PROVENANCE: stamp the noise-model version the world was
        # synthesized under; propagated into specs.npz and the gate outputs so
        # downstream consumers can detect mixed-model states.
        from qns2q.noise.spectra import MODEL_VERSION
        params_to_save['model_version'] = MODEL_VERSION
        np.savez(
            os.path.join(self.path, "params.npz"),
            **params_to_save)
        np.savez(os.path.join(self.path, "results.npz"), **self.results)
        print(f"Results saved to {self.path}")


class PhaseRecorder:
    """Solver wrapper that records each call's per-shot phase coefficients.

    The dephasing propagators are diagonal, so three numbers per shot
    (C_a = int 0.5 y_a b_a dt, a in {1, 2, 12}) determine the evolution of ANY
    initial state. Recording them once makes every further SPAM-protocol arm a
    cheap replay: the arms differ only in the prepared state and in estimator
    post-processing, never in the noise. The recorder consumes np.random
    exactly like `solver_prop`, so the recording run doubles as a normal
    (typically SPAM-free reference) arm.
    """

    def __init__(self):
        self.calls = []
        self.t_lens = []

    def __call__(self, y_uv, noise_mats, t_vec, rho, n_shots):
        coeffs = solver_phase_coeffs_fast(y_uv, noise_mats, t_vec, n_shots)
        self.calls.append(np.asarray(coeffs))
        self.t_lens.append(int(np.size(t_vec)))
        return phased_state(coeffs, rho)


class PhaseReplayer:
    """Solver wrapper that replays a recorded phase dataset call-by-call.

    The experiment suite is a deterministic sequence of solver calls (identical
    across the non-robust protocols), so a simple call-order FIFO suffices;
    each call's shot count and time-grid length are checked against the
    recording. Replay arms skip noise-matrix construction and trajectory
    synthesis entirely (~minutes instead of ~40 per arm at the tuned config).
    """

    def __init__(self, calls, t_lens):
        self.calls = list(calls)
        self.t_lens = list(t_lens)
        self.i = 0

    def __call__(self, y_uv, noise_mats, t_vec, rho, n_shots):
        if self.i >= len(self.calls):
            raise RuntimeError("phase dataset exhausted: the replayed suite "
                               "issued more solver calls than were recorded")
        coeffs = self.calls[self.i]
        if coeffs.shape[0] != n_shots or self.t_lens[self.i] != int(np.size(t_vec)):
            raise RuntimeError(
                f"phase-dataset call {self.i} mismatch: recorded "
                f"(n_shots={coeffs.shape[0]}, n_t={self.t_lens[self.i]}) vs "
                f"requested (n_shots={n_shots}, n_t={int(np.size(t_vec))}) -- "
                "the replay config must match the recording config")
        self.i += 1
        return phased_state(coeffs, rho)


# Config fields that must match between a recording and a replay for the
# stored phases to describe the same experiment suite.
_DATASET_META_FIELDS = ('T', 'M', 't_grain', 'truncate', 'w_grain', 'n_shots',
                        'midpoint', 'synth_wmax')


def save_phase_dataset(path, recorder, config):
    """Persist a PhaseRecorder's calls (+ the grid metadata) to ``path``."""
    payload = {f'call_{i:04d}': c for i, c in enumerate(recorder.calls)}
    payload['n_calls'] = np.array(len(recorder.calls))
    payload['t_lens'] = np.array(recorder.t_lens, dtype=int)
    for f in _DATASET_META_FIELDS:
        payload[f'meta_{f}'] = np.array(getattr(config, f))
    payload['meta_seed'] = np.array(RANDOM_SEED)
    np.savez(path, **payload)
    print(f"Phase dataset saved to {path} ({len(recorder.calls)} calls)")


def load_phase_dataset(path, config):
    """Load a phase dataset and validate it against ``config``'s grid."""
    d = np.load(path)
    for f in _DATASET_META_FIELDS:
        if f'meta_{f}' not in d.files:
            continue  # dataset predates this meta field (defaults applied)
        rec, cur = d[f'meta_{f}'], np.array(getattr(config, f))
        if not np.array_equal(rec, cur):
            raise ValueError(f"phase dataset {path} was recorded with {f}={rec} "
                             f"but the replay config has {f}={cur}")
    n = int(d['n_calls'])
    calls = [d[f'call_{i:04d}'] for i in range(n)]
    return PhaseReplayer(calls, d['t_lens'])


def main(config=None, record_to=None, replay_from=None):
    """
    Main function to run the QNS experiments.

    Parameters
    ----------
    config : QNSExperimentConfig, optional
        Experiment configuration. Defaults to ``QNSExperimentConfig()``; pass a custom
        instance (e.g. a reduced t_grain/n_shots) to regenerate a run without editing
        this module.
    record_to : str, optional
        Path to write a phase dataset (run the suite with a PhaseRecorder).
    replay_from : str, optional
        Path of a phase dataset to replay instead of synthesizing noise.
    """
    np.random.seed(RANDOM_SEED)
    print(f"Running QNS experiments [seed={RANDOM_SEED}]...")
    config = QNSExperimentConfig() if config is None else config
    print(f"SPAM protocol: {config.spam_protocol}")
    _observables.set_projection_sampling(config.n_meas, seed=RANDOM_SEED + 7919)
    if config.n_meas:
        print(f"Projection noise: n_meas = {config.n_meas} measurements/realization")
    if (record_to or replay_from) and config.spam_protocol == 'robust':
        raise ValueError("record/replay covers the non-robust suite only (the "
                         "robust protocol runs different experiments)")
    solver = None
    if record_to is not None:
        solver = PhaseRecorder()
        print(f"[dataset] recording phase coefficients -> {record_to}")
    elif replay_from is not None:
        solver = load_phase_dataset(replay_from, config)
        print(f"[dataset] replaying phase coefficients <- {replay_from} "
              f"({len(solver.calls)} calls)")
    runner = ExperimentRunner(config, solver=solver,
                              make_mats=replay_from is None)

    if config.spam_protocol == 'robust':
        # SPAM-robust protocol: two-qubit (twisted+wrung) and single-qubit
        # (twisted) coefficients, swept over M so the reconstruction stage can
        # regress out the M-independent SPAM intercept. C_12_12 is exactly
        # SPAM-free (the intercept cancels in D+ - D-), so a single M suffices.
        # The C_a_b coefficients (-> S_1_12, S_2_12) have no robust estimator.
        msweep_experiments = [
            ('C_12_0_MT_1', ['CPMG', 'CPMG'], 'C_12_0', {'state': 'pp'}),
            ('C_12_0_MT_2', ['CDD3', 'CPMG'], 'C_12_0', {'state': 'pp'}),
            ('C_12_0_MT_3', ['CPMG', 'CDD3'], 'C_12_0', {'state': 'pp'}),
            ('C_12_0_MT_4', ['CDD1', 'CDD1'], 'C_12_0', {'state': 'pp'}),
            ('C_1_0_MT_1', ['CDD1', 'CDD1-1/2'], 'C_a_0', {'l': 1}),
            ('C_2_0_MT_1', ['CDD1-1/2', 'CDD1'], 'C_a_0', {'l': 2}),
        ]
        for exp_name, pulse_sequence, exp_type, kwargs in msweep_experiments:
            runner.run_experiment_msweep(exp_name, pulse_sequence, exp_type,
                                         config.m_sweep_robust, **kwargs)
        single_m_experiments = [
            ('C_12_12_MT_1', ['CPMG', 'CPMG'], 'C_12_12', {'state': 'pp'}),
            ('C_12_12_MT_2', ['CDD3', 'CPMG'], 'C_12_12', {'state': 'pp'}),
        ]
        for exp_name, pulse_sequence, exp_type, kwargs in single_m_experiments:
            runner.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)
    else:
        # Harmonic (comb) experiments -- reconstruct S(omega_k), k=1..truncate.
        harmonic_experiments = [
            ('C_12_0_MT_1', ['CPMG', 'CPMG'], 'C_12_0', {'state': 'pp'}),
            ('C_12_0_MT_2', ['CDD3', 'CPMG'], 'C_12_0', {'state': 'pp'}),
            ('C_12_0_MT_3', ['CPMG', 'CDD3'], 'C_12_0', {'state': 'pp'}),
            ('C_12_12_MT_1', ['CPMG', 'CPMG'], 'C_12_12', {'state': 'pp'}),
            ('C_12_12_MT_2', ['CDD3', 'CPMG'], 'C_12_12', {'state': 'pp'}),
            ('C_1_0_MT_1', ['CDD1', 'CDD1-1/2'], 'C_a_0', {'l': 1}),
            ('C_2_0_MT_1', ['CDD1-1/2', 'CDD1'], 'C_a_0', {'l': 2}),
            ('C_12_0_MT_4', ['CDD1', 'CDD1'], 'C_12_0', {'state': 'pp'}),
            ('C_1_2_MT_1', ['CPMG', 'FID'], 'C_a_b', {'l': 1}),
            ('C_1_2_MT_2', ['CPMG', 'CDD1-1/4'], 'C_a_b', {'l': 1}),
            ('C_2_1_MT_1', ['FID', 'CPMG'], 'C_a_b', {'l': 2}),
            ('C_2_1_MT_2', ['CDD1-1/4', 'CPMG'], 'C_a_b', {'l': 2}),
        ]
        for exp_name, pulse_sequence, exp_type, kwargs in harmonic_experiments:
            runner.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)

    # --- DC (zero-frequency) characterization: multi-time FID decay-slope fit --------
    # Each FID observable is measured over a sweep of total evolution times t = m*T so
    # the adaptive estimator (inversion._ramsey_fit_dc) fits the slope in the window
    # where the decay is measurable AND linear. This recovers S(0) for strong noise
    # (the old single full-MT point is fully decayed there) and auto-adapts the
    # measurement time if the spectra change -- no per-spectrum tuning. The self
    # observables decouple the Ising with a fast partner CDD3 (dc_ct); for the
    # pulse-free FID/FID cross observables dc_ct is immaterial.
    # Floor at 8 tau: the CDD3 partner toggles at ct/8, so any shorter cycle
    # violates the minimal pulse spacing tau (bites for short-T runs, e.g. the
    # T=40 high-band comb of the two-scale Nyquist protocol).
    dc_ct = max(config.T / 8, 8.0)
    dc_m_sweep = range(1, config.M + 1)
    dc_experiments = [
        # Self-DC: partner CDD3 nulls the Ising -> qubit-a FID governed by S_aa alone.
        ('C_1_0_FIDCDD3', ['FID', 'CDD3'], 'C_a_0',  {'l': 1}),
        ('C_2_0_CDD3FID', ['CDD3', 'FID'], 'C_a_0',  {'l': 2}),
        # Ising self-DC: C_1_0_FF + C_2_0_FF - C_12_0_FF = Var(Phi_12) (partner FID keeps
        # the Ising in the single-qubit coefficients; the ZZ one subtracts the selfs).
        ('C_1_0_FIDFID',  ['FID', 'FID'], 'C_a_0',  {'l': 1}),
        ('C_2_0_FIDFID',  ['FID', 'FID'], 'C_a_0',  {'l': 2}),
        ('C_12_0_FID_FID', ['FID', 'FID'], 'C_12_0',  {'state': 'pp'}),
        # Cross-spectra DC from the FID/FID cross coefficients (slope = S_xy(0)).
        ('C_12_12_FID',    ['FID', 'FID'], 'C_12_12', {'state': 'pp'}),
        ('C_1_12_FID',     ['FID', 'FID'], 'C_a_b',   {'l': 1}),
        ('C_2_12_FID',     ['FID', 'FID'], 'C_a_b',   {'l': 2}),
    ]
    if config.spam_protocol == 'robust':
        # The C_a_b DC observables have no SPAM-robust estimator; their DC points
        # (S_1_12(0), S_2_12(0)) are not accessible under the robust protocol.
        # Note: the DC slope fit is itself insensitive to the M-independent SPAM
        # intercept, so no M-regression beyond the existing time sweep is needed.
        dc_experiments = [e for e in dc_experiments if e[2] != 'C_a_b']
    for exp_name, pulse_sequence, exp_type, kwargs in dc_experiments:
        runner.run_dc_sweep(exp_name, pulse_sequence, exp_type, dc_m_sweep, dc_ct, **kwargs)

    # Ising self-DC, direct (double echo): simultaneous CDD1 on both qubits echoes
    # Phi_1 (y_1 balanced) while y_12 = y_1*y_2 = +1 keeps FULL DC weight; the
    # CDD1/CPMG reference has the SAME y_1 filter (Var Phi_1 cancels exactly in the
    # difference) but a fast-toggled y_12 with no DC weight. The difference therefore
    # isolates Var Phi_12 at FIRST order -- unlike the FF combination above, which
    # extracts it as a ~25x-smaller difference of single-qubit variances and is
    # statistically swamped at realistic shots. The fast cycle (T/64 = 2.5 tau;
    # per-qubit pulse spacing >= 1.25 tau) parks the echo passband (w ~ pi/ct) above
    # the QNS band where the S_1212 weight is small; the residual mixed-filter
    # pickup is a deterministic systematic mirrored by the DC forward model.
    # Floor at 2 tau: CDD1/CPMG toggle at ct/2, the minimal legal pulse spacing.
    dc_echo_ct = max(config.T / 64, 2.0)
    for exp_name, pulse_sequence in (('C_1_0_CDD1CDD1', ['CDD1', 'CDD1']),
                                     ('C_1_0_CDD1CPMG', ['CDD1', 'CPMG'])):
        runner.run_dc_sweep(exp_name, pulse_sequence, 'C_a_0', dc_m_sweep,
                            dc_echo_ct, l=1)
    runner.results['dc_echo_ct'] = dc_echo_ct

    runner.save_results()
    if record_to is not None:
        save_phase_dataset(record_to, solver, config)
    if replay_from is not None and solver.i != len(solver.calls):
        print(f"[dataset] WARNING: replay consumed {solver.i} of "
              f"{len(solver.calls)} recorded calls")


if __name__ == "__main__":
    main()
