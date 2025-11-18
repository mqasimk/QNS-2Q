"""
This script runs QNS experiments for the S_1212 spectrum with and without
SPAM errors, and then performs reconstruction of the spectrum.
"""

import os
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

# Imports from project files
from qnsExps import QNSExperimentConfig, ExperimentRunner
from reconSpectra import SpectraReconConfig
from spectraIn import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
from fixedAS import recon_S_12_12


def run_spam_experiment():
    """
    Run QNS experiments for S_1212 with and without SPAM errors.
    """
    all_spectra = [S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12]
    
    # Shared simulation parameters from qnsExps.py
    tau = 2.5e-8
    M = 10
    t_grain = 3500
    truncate = 20
    w_grain = 1000
    T = 160 * tau
    gamma = T / 14
    gamma_12 = T / 28
    n_shots = 4000

    # --- No SPAM Configuration ---
    config_no_spam = QNSExperimentConfig(
        tau=tau,
        M=M,
        t_grain=t_grain,
        truncate=truncate,
        w_grain=w_grain,
        spec_vec=all_spectra,
        T=T,
        gamma=gamma,
        gamma_12=gamma_12,
        n_shots=n_shots,
        fname="S1212_NoSPAM",
    )
    runner_no_spam = ExperimentRunner(config_no_spam)

    # --- SPAM Configuration ---
    config_spam = QNSExperimentConfig(
        tau=tau,
        M=M,
        t_grain=t_grain,
        truncate=truncate,
        w_grain=w_grain,
        spec_vec=all_spectra,
        T=T,
        gamma=gamma,
        gamma_12=gamma_12,
        n_shots=n_shots,
        fname="S1212_SPAM",
        a_sp=np.array([0.95, 0.95]),
        c=np.array([0.+0.3j, 0.-0.3j]),
        a1=0.99,    # Measurement error parameter
        b1=0.95,   # Measurement error parameter
        a2=0.98,   # Measurement error parameter
        b2=0.95,   # Measurement error parameter
        spMit=False
    )
    runner_spam = ExperimentRunner(config_spam)

    # Experiments needed for S_1212 reconstruction
    experiments = [
        ('C_1_0_MT_1', ['CDD1', 'CDD1-1/2'], 'C_a_0', {'l': 1}),
        ('C_2_0_MT_1', ['CDD1-1/2', 'CDD1'], 'C_a_0', {'l': 2}),
        ('C_12_0_MT_4', ['CDD1', 'CDD1'], 'C_12_0', {'state': 'pp'}),
    ]

    print("--- Running No SPAM experiments ---")
    for exp_name, pulse_sequence, exp_type, kwargs in experiments:
        runner_no_spam.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)
    runner_no_spam.save_results()

    print("\n--- Running SPAM experiments ---")
    for exp_name, pulse_sequence, exp_type, kwargs in experiments:
        runner_spam.run_experiment(exp_name, pulse_sequence, exp_type, **kwargs)
    runner_spam.save_results()


def reconstruct_s1212_spectrum(data_folder: str, label: str, axis):
    """
    Reconstructs and plots the S_1212 spectrum from experiment results.
    """
    try:
        config = SpectraReconConfig(data_folder=data_folder)
        
        path = os.path.join(os.pardir, config.data_folder, "results.npz")
        observables = np.load(path)

        c = config
        obs = observables
        wk = np.array([2 * np.pi * (n + 1) / c.T for n in range(c.truncate)])

        # Reconstruct only S_12_12
        s1212_recon = recon_S_12_12(
            [obs['C_1_0_MT_1'], obs['C_2_0_MT_1'], obs['C_12_0_MT_4']],
            c_times=c.c_times, m=c.M, T=c.T
        )

        # Plotting
        w = np.linspace(0.1, c.wmax, c.w_grain)
        w = w[40:]
        xunits = 1e6

        axis.plot(wk / xunits, s1212_recon, '^--', label=f'$\hat{{S}}_{{12,12}}^{{+}}(\omega_k)$ - {label}')
        axis.plot(w / xunits, S_1212(w), 'r-', lw=1.5, label=r'$S_{12,12}^+(\omega)$' if label == "No SPAM" else None)
        axis.set_xlabel(r'$\omega$ (MHz)', fontsize=16)
        axis.set_ylabel(r'$S_{12,12}^+(\omega)$', fontsize=16)
        axis.legend(fontsize=12)
        axis.tick_params(direction='in', labelsize=12)
        
        # Save plot
        parent_dir = os.pardir
        path = os.path.join(parent_dir, data_folder)
        
        print(f"Reconstruction plot for {label} will be saved in recon_S1212.pdf")

    except FileNotFoundError as e:
        print(f"Error for {data_folder}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {data_folder}: {e}")


def main():
    """
    Main function to run experiments and reconstruction for S_1212.
    """
    # --- User Configuration ---
    run_new_simulation = True  # Set to False to reuse existing data
    # ------------------------

    if run_new_simulation:
        run_spam_experiment()

    fig, ax = plt.subplots(figsize=(10, 6))

    print("\n--- Reconstructing No SPAM data ---")
    reconstruct_s1212_spectrum("S1212_NoSPAM", "No SPAM", ax)

    print("\n--- Reconstructing SPAM data ---")
    reconstruct_s1212_spectrum("S1212_SPAM", "SPAM", ax)
    
    plt.savefig('recon_S1212.pdf')
    plt.show()


if __name__ == "__main__":
    main()