"""
This script reconstructs and plots spectra from pre-calculated observables.

It defines a configuration class to load parameters, a reconstructor class to manage
the reconstruction process, and a main execution block to run the analysis for a
specified data folder.
"""

import matplotlib
matplotlib.use('Agg')

import os
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from spectral_inversion import (recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12, recon_S_1_12, recon_S_2_12,
                                recon_S_11_dc, recon_S_22_dc, recon_S_1212_dc,
                                recon_S_1_2_dc, recon_S_1_12_dc, recon_S_2_12_dc)
from spectra_input import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12


# --- Publication figure constants ---

FIG_WIDTH = 7.0    # Two-column width (inches)
FIG_HEIGHT = 4.5   # 2-row panel height

COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
    "grey_fill": "#E0E0E0",
}

# Subfolder inside each data folder for figures
FIGURES_SUBDIR = "figures"
RECONSTRUCTION_SUBDIR = os.path.join(FIGURES_SUBDIR, "reconstruction")


def setup_pub_rcparams():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.2,
        "grid.color": "grey",
    })


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

        required_params = {
            't_vec': np.ndarray,
            'w_grain': int,
            'wmax': float,
            'truncate': int,
            'gamma': float,
            'gamma_12': float,
            'c_times': np.ndarray,
            'M': int,
            'T': float
        }

        for param_name, expected_type in required_params.items():
            if param_name not in self.params:
                raise KeyError(f"Missing required parameter: '{param_name}' in params.npz")

            value = self.params[param_name]

            # For numerical types, extract scalar from 0-d numpy array
            if expected_type in [int, float] and hasattr(value, 'item'):
                value = value.item()

            if value is None:
                raise ValueError(f"Parameter '{param_name}' cannot be None.")

            try:
                if expected_type in [int, float]:
                    setattr(self, param_name, expected_type(value))
                else:  # For np.ndarray and other types
                    setattr(self, param_name, value)
            except (ValueError, TypeError):
                raise TypeError(
                    f"Parameter '{param_name}' has an invalid value '{value}' for expected type {expected_type.__name__}.")


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

        # Reconstruct spectra at comb harmonics wk = 2*pi*(k+1)/T
        wk_harmonics = np.array([2 * np.pi * (n + 1) / c.T for n in range(c.truncate)])

        S_11_k = recon_S_11([obs['C_12_0_MT_1'], obs['C_12_0_MT_2']], c_times=c.c_times, m=c.M, T=c.T)
        S_22_k = recon_S_22([obs['C_12_0_MT_1'], obs['C_12_0_MT_3']], c_times=c.c_times, m=c.M, T=c.T)
        S_12_12_k = recon_S_12_12([obs['C_1_0_MT_1'], obs['C_2_0_MT_1'], obs['C_12_0_MT_4']], c_times=c.c_times, m=c.M, T=c.T)
        S_1_2_k = recon_S_1_2([obs['C_12_12_MT_1'], obs['C_12_12_MT_2']], c_times=c.c_times, m=c.M, T=c.T)
        S_1_12_k = recon_S_1_12([obs['C_1_2_MT_1'], obs['C_1_2_MT_2']], c_times=c.c_times, m=c.M, T=c.T)
        S_2_12_k = recon_S_2_12([obs['C_2_1_MT_1'], obs['C_2_1_MT_2']], c_times=c.c_times, m=c.M, T=c.T)

        # Reconstruct DC (w=0) values from FID experiments
        S_11_dc = recon_S_11_dc([obs['C_12_0_FID_CPMG']], c_times=c.c_times, m=c.M, T=c.T,
                                S_22_k=S_22_k, S_1212_k=S_12_12_k)
        S_22_dc = recon_S_22_dc([obs['C_12_0_CPMG_FID']], c_times=c.c_times, m=c.M, T=c.T,
                                S_11_k=S_11_k, S_1212_k=S_12_12_k)
        S_1212_dc = recon_S_1212_dc([obs['C_12_0_FID_FID']], m=c.M, T=c.T,
                                    S_11_dc=S_11_dc, S_22_dc=S_22_dc)
        S_1_2_dc = recon_S_1_2_dc([obs['C_12_12_FID']], m=c.M, T=c.T)
        S_1_12_dc = recon_S_1_12_dc([obs['C_1_12_FID']], m=c.M, T=c.T)
        S_2_12_dc = recon_S_2_12_dc([obs['C_2_12_FID']], m=c.M, T=c.T)

        print(f"DC values: S_11(0)={S_11_dc:.4f}, S_22(0)={S_22_dc:.4f}, "
              f"S_1212(0)={S_1212_dc:.4f}, S_1_2(0)={S_1_2_dc:.4f}, "
              f"S_1_12(0)={S_1_12_dc:.4f}, S_2_12(0)={S_2_12_dc:.4f}")

        # Prepend DC values (x2 for one-sided PSD convention) to spectrum arrays so wk[0] = 0
        self.wk = np.concatenate(([0.0], wk_harmonics))
        self.reconstructed_spectra = {
            "S_11_k": np.concatenate(([2 * S_11_dc], S_11_k)),
            "S_22_k": np.concatenate(([2 * S_22_dc], S_22_k)),
            "S_12_12_k": np.concatenate(([2 * S_1212_dc], S_12_12_k)),
            "S_1_2_k": np.concatenate(([2 * S_1_2_dc + 0j], S_1_2_k)),
            "S_1_12_k": np.concatenate(([2 * S_1_12_dc + 0j], S_1_12_k)),
            "S_2_12_k": np.concatenate(([2 * S_2_12_dc + 0j], S_2_12_k)),
        }

    def _get_output_dir(self, subdir):
        """Returns (and creates) a subfolder inside the data folder for figures."""
        path = os.path.join(os.pardir, self.config.data_folder, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def plot_reconstruction(self):
        """Plots publication-quality reconstructed spectra against analytical curves."""
        setup_pub_rcparams()

        w = np.linspace(0, self.config.wmax, self.config.w_grain)
        xunits = 1e6

        # Marker/line styling (Okabe-Ito palette)
        eb_re = dict(fmt='^', color=COLORS["vermillion"],
                     markersize=3.5, linewidth=0.8, zorder=10, label='Reconstructed')
        eb_im = dict(fmt='s', color=COLORS["blue"],
                     markersize=3.0, linewidth=0.8, zorder=10, label='Reconstructed (Im)')
        theory_re_kw = dict(color=COLORS["vermillion"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory')
        theory_im_kw = dict(color=COLORS["blue"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory (Im)')

        # --- Real-valued spectra: S_11, S_22, S_1212 ---
        real_spectra = [
            ('S_11_k', S_11, None,
             r'$S_{1,1}^+(\omega)$'),
            ('S_22_k', S_22, None,
             r'$S_{2,2}^+(\omega)$'),
            ('S_12_12_k', S_1212, None,
             r'$S_{12,12}^+(\omega)$'),
        ]

        # --- Complex-valued spectra: S_1_2, S_1_12, S_2_12 ---
        complex_spectra = [
            ('S_1_2_k',
             lambda w_: S_1_2(w_, self.config.gamma),
             r'$S_{1,2}^+(\omega)$'),
            ('S_1_12_k',
             lambda w_: S_1_12(w_, self.config.gamma_12),
             r'$S_{1,12}^+(\omega)$'),
            ('S_2_12_k',
             lambda w_: S_2_12(w_, self.config.gamma_12 - self.config.gamma),
             r'$S_{2,12}^+(\omega)$'),
        ]

        fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

        # Top row: real-valued self-spectra
        for col, (s_key, theory_fn, _, ylabel) in enumerate(real_spectra):
            ax = axs[0, col]
            ax.fill_between(w / xunits, 0, theory_fn(w),
                            color=COLORS["grey_fill"], alpha=1.0, zorder=0)
            ax.plot(w / xunits, theory_fn(w), **theory_re_kw)
            ax.errorbar(self.wk / xunits, self.reconstructed_spectra[s_key],
                        **eb_re)
            ax.set_ylabel(ylabel)

        # Bottom row: complex-valued cross-spectra
        for col, (s_key, theory_fn, ylabel) in enumerate(complex_spectra):
            ax = axs[1, col]
            S_theory = theory_fn(w)

            # Real part
            ax.plot(w / xunits, np.real(S_theory), **theory_re_kw)
            ax.errorbar(self.wk / xunits, np.real(self.reconstructed_spectra[s_key]),
                        **{**eb_re, 'label': r'Re (recon.)'})

            # Imaginary part
            ax.plot(w / xunits, np.imag(S_theory), **theory_im_kw)
            ax.errorbar(self.wk / xunits, np.imag(self.reconstructed_spectra[s_key]),
                        **{**eb_im, 'label': r'Im (recon.)'})
            ax.set_ylabel(ylabel)

        # Common formatting
        for ax_row in axs:
            for ax in ax_row:
                ax.set_yscale('asinh')
                ax.grid(True, which='major', zorder=0)
                ax.grid(False, which='minor')

        # X-labels only on bottom row
        for ax in axs[1, :]:
            ax.set_xlabel(r'$\omega$ (MHz)')

        # Legends
        for ax in axs[0, :]:
            ax.legend(frameon=False, loc='upper right')
        for ax in axs[1, :]:
            ax.legend(frameon=False, loc='upper right', ncol=2)

        plt.tight_layout(pad=0.3)
        output_dir = self._get_output_dir(RECONSTRUCTION_SUBDIR)
        output_path = os.path.join(output_dir, "spectral_reconstruction_pub.pdf")
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"Saved reconstruction plot to {output_path}")
        plt.close(fig)

    def save_reconstructed_spectra(self):
        """Saves the reconstructed spectra (including DC at w=0) to a .npz file."""
        # Save specs.npz at the data folder root (consumed by downstream scripts)
        path = os.path.join(os.pardir, self.config.data_folder, "specs.npz")
        save_dict = dict(
            wk=self.wk,
            S11=self.reconstructed_spectra['S_11_k'], S22=self.reconstructed_spectra['S_22_k'],
            S12=self.reconstructed_spectra['S_1_2_k'], S1212=self.reconstructed_spectra['S_12_12_k'],
            S112=self.reconstructed_spectra['S_1_12_k'], S212=self.reconstructed_spectra['S_2_12_k'],
        )
        np.savez(path, **save_dict)

    def run(self):
        """Runs the full reconstruction pipeline."""
        self.load_observables()
        self.reconstruct()
        self.plot_reconstruction()
        self.save_reconstructed_spectra()


def main():
    """Main function to run the spectra reconstruction."""
    # --- User Configuration ---
    data_folder = "DraftRun_NoSPAM_Featureless"
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
