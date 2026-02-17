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

from observables import (make_c_12_0_mt, make_c_12_12_mt, make_c_a_0_mt,
                         make_c_a_b_mt)
from spectra_input import S_11, S_22, S_1212
from trajectories import make_noise_mat_arr, solver_prop


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
    t_grain: jnp.int64 = 1600
    truncate: jnp.int64 = 20
    w_grain: jnp.int64 = 1000
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
    T: jnp.float64 = 160*tau
    gamma: jnp.float64 = T / 14
    gamma_12: jnp.float64 = T / 28
    n_shots: jnp.int64 = 4000
    fname: str = "DraftRun_NoSPAM_Feature"
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
        path = os.path.join(self.config.parent_dir, self.config.fname)
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
                gamma_12=self.config.gamma_12))

    def run_experiment(self, exp_name: str, pulse_sequence: list,
                       exp_type: str, **kwargs):
        """
        Runs a single experiment.

        Args:
            exp_name: The name of the experiment.
            pulse_sequence: The pulse sequence to use.
            exp_type: The type of experiment to run.
            **kwargs: Additional arguments for the experiment function.
        """
        print(f"Running experiment: {exp_name}")
        start_time = time.time()

        exp_map = {
            'C_12_0': make_c_12_0_mt,
            'C_12_12': make_c_12_12_mt,
            'C_a_0': make_c_a_0_mt,
            'C_a_b': make_c_a_b_mt,
        }
        if exp_type not in exp_map:
            raise ValueError(f"Invalid experiment type: {exp_type}")

        exp_func = exp_map[exp_type]
        result = exp_func(
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
        self.results[exp_name] = result

        print(
            f"--- {exp_name} completed in {time.time() - start_time:.2f}s ---")

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


def main():
    """
    Main function to run the QNS experiments.
    """
    config = QNSExperimentConfig()
    runner = ExperimentRunner(config)

    experiments = [
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

    for exp_name, pulse_sequence, exp_type, kwargs in experiments:
        runner.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)

    runner.save_results()


if __name__ == "__main__":
    main()
