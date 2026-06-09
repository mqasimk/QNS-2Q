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
from qns2q.noise.spectra import S_11, S_22, S_1212
from qns2q.model.trajectories import make_noise_mat_arr, solver_prop
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
        gamma: The decay rate.
        gamma_12: The cross-decay rate.
        n_shots: The number of shots for the experiment.
        fname: The name of the folder to save the results in.
        parent_dir: The parent directory to save the results in.
    """
    tau: jnp.float64 = 2.5e-8
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
    # Bin-midpoint noise-synthesis grid: excludes the spurious w=0 static tone that
    # otherwise biases DC-sensitive observables by O(dw). Default True for new runs.
    midpoint: bool = True
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


class ExperimentRunner:
    """
    Runs a series of QNS experiments based on a given configuration.
    """

    def __init__(self, config: QNSExperimentConfig):
        """
        Initializes the ExperimentRunner.

        Args:
            config: The configuration for the experiments.
        """
        self.config = config
        self.path = self._setup_output_directory()
        self.noise_mats = self._make_noise_mats()
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
                spec_vec=self.config.spec_vec,
                t_vec=self.config.t_vec,
                w_grain=self.config.w_grain,
                wmax=self.config.wmax,
                truncate=self.config.truncate,
                gamma=self.config.gamma,
                gamma_12=self.config.gamma_12,
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

        if exp_type not in _EXP_MAP:
            raise ValueError(f"Invalid experiment type: {exp_type}")

        ctimes = self.config.c_times if c_times is None else c_times
        exp_func = _EXP_MAP[exp_type]
        means, stderrs = exp_func(
            solver_prop,
            pulse_sequence,
            self.config.t_vec,
            ctimes,
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
        if exp_type not in _EXP_MAP:
            raise ValueError(f"Invalid experiment type: {exp_type}")
        exp_func = _EXP_MAP[exp_type]
        c_vals, c_errs = [], []
        for m in m_sweep:
            m = int(m)
            tvec_m = self.config.t_vec[:m * self.config.t_grain]
            means, stderrs = exp_func(
                solver_prop, pulse_sequence, tvec_m, [dc_ct], self.config.CM,
                self.config.spMit, n_shots=self.config.n_shots, m=m,
                t_b=self.config.t_b, a_m=self.config.a_m, delta=self.config.delta,
                a_sp=self.config.a_sp, c=self.config.c, noise_mats=self.noise_mats,
                **kwargs)
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
        params_to_save['tau'] = self.config.tau
        np.savez(
            os.path.join(self.path, "params.npz"),
            **params_to_save)
        np.savez(os.path.join(self.path, "results.npz"), **self.results)
        print(f"Results saved to {self.path}")


def main(config=None):
    """
    Main function to run the QNS experiments.

    Parameters
    ----------
    config : QNSExperimentConfig, optional
        Experiment configuration. Defaults to ``QNSExperimentConfig()``; pass a custom
        instance (e.g. a reduced t_grain/n_shots) to regenerate a run without editing
        this module.
    """
    np.random.seed(RANDOM_SEED)
    print(f"Running QNS experiments [seed={RANDOM_SEED}]...")
    config = QNSExperimentConfig() if config is None else config
    runner = ExperimentRunner(config)

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
    dc_ct = config.T / 8
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
    for exp_name, pulse_sequence, exp_type, kwargs in dc_experiments:
        runner.run_dc_sweep(exp_name, pulse_sequence, exp_type, dc_m_sweep, dc_ct, **kwargs)

    runner.save_results()


if __name__ == "__main__":
    main()
