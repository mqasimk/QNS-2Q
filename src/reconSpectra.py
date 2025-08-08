"""
This script reconstructs and plots spectra from pre-calculated observables.

It defines a configuration class to load parameters, a reconstructor class to manage
the reconstruction process, and a main execution block to run the analysis for a
specified data folder.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from fixedAS import recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12, recon_S_1_12, recon_S_2_12
from spectraIn import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12


@dataclass
class SpectraReconConfig:
    """Configuration and parameters for spectra reconstruction."""
    data_folder: str
    params: Dict[str, Any] = field(init=False)
    t_vec: np.ndarray = field(init=False)
    w_grain: int = field(init=False)
    wmax: float = field(init=False)
    truncate: int = field(init=False)
    gamma: float = field(init=False)
    gamma_12: float = field(init=False)
    c_times: np.ndarray = field(init=False)
    M: int = field(init=False)
    T: float = field(init=False)

    def __post_init__(self):
        """Load parameters from the data folder after initialization."""
        parent_dir = os.pardir
        path = os.path.join(parent_dir, self.data_folder)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Data folder not found at: {path}")

        params_path = os.path.join(path, "params.npz")
        self.params = np.load(params_path)

        # Assign parameters to attributes
        for key, value in self.params.items():
            setattr(self, key, value)


class SpectraReconstructor:
    """Handles the reconstruction of spectra from observables."""

    def __init__(self, config: SpectraReconConfig):
        """Initializes the reconstructor with a given configuration."""
        self.config = config
        self.observables: Dict[str, np.ndarray] = {}
        self.reconstructed_spectra: Dict[str, np.ndarray] = {}
        self.wk: np.ndarray = np.array([])

    def load_observables(self):
        """Loads the observables array from the data folder."""
        path = os.path.join(os.pardir, self.config.data_folder, "results.npz")
        self.observables = np.load(path)

    def reconstruct(self):
        """Reconstructs the spectra from the loaded observables."""
        c = self.config
        obs = self.observables
        self.wk = np.array([2 * np.pi * (n + 1) / c.T for n in range(c.truncate)])

        self.reconstructed_spectra = {
            "S_11_k": recon_S_11([obs['C_12_0_MT_1'], obs['C_12_0_MT_2']], c_times=c.c_times, M=c.M, T=c.T),
            "S_22_k": recon_S_22([obs['C_12_0_MT_1'], obs['C_12_0_MT_3']], c_times=c.c_times, M=c.M, T=c.T),
            "S_1_2_k": recon_S_1_2([obs['C_12_12_MT_1'], obs['C_12_12_MT_2']], c_times=c.c_times, M=c.M, T=c.T),
            "S_12_12_k": recon_S_12_12([obs['C_1_0_MT_1'], obs['C_2_0_MT_1'], obs['C_12_0_MT_4']], c_times=c.c_times, M=c.M, T=c.T),
            "S_1_12_k": recon_S_1_12([obs['C_1_2_MT_1'], obs['C_1_2_MT_2']], c_times=c.c_times, M=c.M, T=c.T),
            "S_2_12_k": recon_S_2_12([obs['C_2_1_MT_1'], obs['C_2_1_MT_2']], c_times=c.c_times, M=c.M, T=c.T),
        }

    def plot_reconstruction(self):
        """Plots the reconstructed spectra against the original spectra."""
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        w = np.linspace(0.1, self.config.wmax, self.config.w_grain)
        w = w[100:]
        xunits = 1e6

        plot_params = {
            'lw': 1,
            'legendfont': 12,
            'xlabelfont': 16,
            'ylabelfont': 16,
            'tickfont': 12,
        }

        # Plot S_11
        axs[0, 0].plot(self.wk / xunits, self.reconstructed_spectra['S_11_k'], 'r^')
        axs[0, 0].plot(w / xunits, S_11(w), 'r--', lw=1.5 * plot_params['lw'])
        axs[0, 0].legend([r'$\hat{S}_{1,1}^+(\omega_k)$', r'$S_{1,1}^+(\omega)$'], fontsize=plot_params['legendfont'])

        # Plot S_22
        axs[0, 1].plot(self.wk / xunits, self.reconstructed_spectra['S_22_k'], 'r^')
        axs[0, 1].plot(w / xunits, S_22(w), 'r--', lw=1.5 * plot_params['lw'])
        axs[0, 1].legend([r'$\hat{S}_{2,2}^+(\omega_k)$', r'$S_{2,2}^+(\omega)$'], fontsize=plot_params['legendfont'])

        # Plot S_12_12
        axs[0, 2].plot(self.wk / xunits, self.reconstructed_spectra['S_12_12_k'], 'r^')
        axs[0, 2].plot(w / xunits, S_1212(w), 'r--', lw=1.5 * plot_params['lw'])
        axs[0, 2].legend([r'$\hat{S}_{12,12}^+(\omega_k)$', r'$S_{12,12}^+(\omega)$'], fontsize=plot_params['legendfont'])

        # Plot S_1_2
        axs[1, 0].plot(self.wk / xunits, np.real(self.reconstructed_spectra['S_1_2_k']), 'r^')
        axs[1, 0].plot(w / xunits, np.real(S_1_2(w, self.config.gamma)), 'r--', lw=1.5 * plot_params['lw'])
        axs[1, 0].plot(self.wk / xunits, np.imag(self.reconstructed_spectra['S_1_2_k']), 'b^')
        axs[1, 0].plot(w / xunits, np.imag(S_1_2(w, self.config.gamma)), 'b--', lw=1.5 * plot_params['lw'])
        axs[1, 0].legend([r'Re[$\hat{S}_{1,2}^+(\omega_k)$]', r'Re[$S_{1,2}^+(\omega)$]', r'Im[$\hat{S}_{1,2}^+(\omega_k)$]', r'Im[$S_{1,2}^+(\omega)$]'], fontsize=plot_params['legendfont'])

        # Plot S_1_12
        axs[1, 1].plot(self.wk / xunits, np.real(self.reconstructed_spectra['S_1_12_k']), 'r^')
        axs[1, 1].plot(w / xunits, np.real(S_1_12(w, self.config.gamma_12)), 'r--', lw=1.5 * plot_params['lw'])
        axs[1, 1].plot(self.wk / xunits, np.imag(self.reconstructed_spectra['S_1_12_k']), 'b^')
        axs[1, 1].plot(w / xunits, np.imag(S_1_12(w, self.config.gamma_12)), 'b--', lw=1.5 * plot_params['lw'])
        axs[1, 1].legend([r'Re[$\hat{S}_{1,12}^+(\omega_k)$]', r'Re[$S_{1,12}^+(\omega)$]', r'Im[$\hat{S}_{1,12}^+(\omega_k)$]', r'Im[$S_{1,12}^+(\omega)$]'], fontsize=plot_params['legendfont'])

        # Plot S_2_12
        axs[1, 2].plot(self.wk / xunits, np.real(self.reconstructed_spectra['S_2_12_k']), 'r^')
        axs[1, 2].plot(w / xunits, np.real(S_2_12(w, self.config.gamma_12 - self.config.gamma)), 'r--', lw=1.5 * plot_params['lw'])
        axs[1, 2].plot(self.wk / xunits, np.imag(self.reconstructed_spectra['S_2_12_k']), 'b^')
        axs[1, 2].plot(w / xunits, np.imag(S_2_12(w, self.config.gamma_12 - self.config.gamma)), 'b--', lw=1.5 * plot_params['lw'])
        axs[1, 2].legend([r'Re[$\hat{S}_{2,12}^+(\omega_k)$]', r'Re[$S_{2,12}^+(\omega)$]', r'Im[$\hat{S}_{2,12}^+(\omega_k)$]', r'Im[$S_{2,12}^+(\omega)$]'], fontsize=plot_params['legendfont'])

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel(r'$\omega$(MHz)', fontsize=plot_params['xlabelfont'])
                ax.tick_params(direction='in', labelsize=plot_params['tickfont'])

        path = os.path.join(os.pardir, self.config.data_folder, 'reconstruct.pdf')
        plt.savefig(path)
        plt.show()

    def save_reconstructed_spectra(self):
        """Saves the reconstructed spectra to a .npz file."""
        path = os.path.join(os.pardir, self.config.data_folder, "specs.npz")
        np.savez(path, S11=self.reconstructed_spectra['S_11_k'], S22=self.reconstructed_spectra['S_22_k'],
                 S12=self.reconstructed_spectra['S_1_2_k'], S1212=self.reconstructed_spectra['S_12_12_k'],
                 S112=self.reconstructed_spectra['S_1_12_k'], S212=self.reconstructed_spectra['S_2_12_k'])

    def run(self):
        """Runs the full reconstruction pipeline."""
        self.load_observables()
        self.reconstruct()
        self.plot_reconstruction()
        self.save_reconstructed_spectra()


def main():
    """Main function to run the spectra reconstruction."""
    # --- User Configuration ---
    data_folder = "DraftRun_NoSPAM"
    # ------------------------

    try:
        config = SpectraReconConfig(data_folder=data_folder)
        reconstructor = SpectraReconstructor(config)
        reconstructor.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
