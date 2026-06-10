"""
Experimentally-anchored two-qubit dephasing noise model (PSD definitions).

**Units: the minimum pulse separation tau is the unit of time (tau = 1).**
Frequencies are dimensionless angular frequencies w~ = w*tau, spectra are
dimensionless S~ = tau*S. ``_TAU_SI = 25 ns`` is a conversion constant only.

Construction (see ``NOISE_MODEL_SPEC.md`` for the full provenance ledger; every
parameter is tagged there [M]easured / [E]xtrapolated / [C]hoice):

Two partially-correlated local electric-field processes e_A, e_B plus two
qubit-local nuclear processes n_1, n_2 build the three dephasing channels

    zeta_1  = e_A + n_1
    zeta_2  = e_B + n_2
    zeta_12 = A_J * e_A - B_J * e_B        # J couples to the field *difference*

with cross(e_A, e_B) = C2_SHARE * sqrt(S_el_A S_el_B) * exp(-i w DT_SHIFT).
This mirrors the susceptibility model validated in Yoneda et al., Nat. Phys. 19,
1793 (2023), which measured all six auto-/cross-PSDs of (nu_A, nu_B, J) in a
Si/SiGe qubit pair: the (+, +, -) sign pattern of the three coherences is
impossible with a single fully-shared source. Achieved in-band values here
(scripts/calibrate_noise_model.py): |c_12| = 0.67, c_1,12 = 0.28+0.50j,
c_2,12 = -0.43+0.44j at w~ = 0.35 [measured at <~ Hz: 0.7 / +0.8 / -(0.3-0.6)].

Spectral shapes (broken-power-law exponents measured in-band, 10 kHz-1 MHz):
  S_el_A ~ w^-0.7, S_el_B ~ w^-0.4   [Rojas-Arias et al., npj QI (2025)]
  S_nuc  ~ w^-1.2 (local, uncorrelated; heavier on qubit 2)  [Yoneda 2023]
  S_1212: in-band ratio S_1212/sqrt(S_11 S_22) = 0.10 at w~ = 0.35 -- gate-
  operating-point exchange noise [Dial et al., PRL 110, 146804 (2013)]
  IR cutoff W_IR regularizes S(0) (finite for the FID-slope DC protocol).
Amplitudes solve T2*(FID) = 260 tau per qubit (purified-28Si target, 6.5 us at
the 25 ns anchor; Rojas-Arias et al., PRApplied 20, 054024 (2023)).

Two regimes, selected at import by ``QNS2Q_REGIME`` (see ``qns2q.paths``):

* ``bland``    -- Class M (monotonic): the construction above, no lines.
* ``featured`` -- Class F: Class M + the nuclear-Larmor-difference line triplet
  (two near-degenerate lines + one at twice the frequency, positions ~ B_eff =
  600 mT; Malinowski et al., Nat. Nanotech. 12, 16 (2017)) on the *qubit-local*
  nuclear components only -- J-noise is electrical [Yoneda 2023], so S_1212
  carries no lines and the coherence c_12(w) dips at the line frequencies.

The public API (S_11, S_22, S_1212 self-spectra and S_1_2, S_1_12, S_2_12
cross-spectra) is shared by the synthesis (model.trajectories), the QNS forward
model, the reconstruction overlays, and the gate optimizers. Cross-spectra take
NO lag argument: their phases are internal to the model (DT_SHIFT). The
trajectory synthesis (model.trajectories.make_channel_trajs) uses the component
spectra and mixing constants exported here, so synthesized trajectories and the
analytic spectra agree by construction.
"""

import jax
import jax.numpy as jnp

from qns2q.paths import current_regime, run_folder, project_root

# Legacy SI anchor (tau = 25 ns): conversion constant only.
_TAU_SI = 2.5e-8

MODEL_VERSION = "anchored-tarucha-20260610"

_REGIME = current_regime()


@jax.jit
def L(w, w0, tc):
    """Symmetric Lorentzian: peak at +/-w0, width set by correlation time tc."""
    return 0.5*(1/(1+(tc**2)*(w-w0)**2)+1/(1+(tc**2)*(w+w0)**2))


@jax.jit
def Gauss(w, w0, sig):
    """Symmetric (non-normalized) Gaussian: peak at +/-w0, std dev sig."""
    return 0.5*(jnp.exp(-(w-w0)**2/(2*sig**2))+jnp.exp(-(w+w0)**2/(2*sig**2)))


# --- Calibrated constants (solved by scripts/calibrate_noise_model.py) -----------
# Targets and provenance: NOISE_MODEL_SPEC.md sections 3-5.

W_IR = 0.02                      # IR cutoff [C, constrained by the DC protocol]
_G_EL_1, _G_EL_2 = 0.7, 0.4      # in-band charge-noise exponents [M, npj 2025]
_G_NUC = 1.2                     # local nuclear slope [M, sub-Hz hyperfine]

A_EL_1 = 3.560544e-04            # amplitudes: T2*(FID) = 260 tau per qubit with
A_EL_2 = 5.243307e-04            # electrical fractions 0.88 / 0.80 at w~ = 0.35
A_NUC_1 = 2.874769e-05
A_NUC_2 = 5.667167e-05

C2_SHARE = 0.8                   # shared fraction of the electrical noise power
A_J = 4.269373e-01               # J difference-coupling weights; B_J/A_J = 1.05
B_J = 4.482842e-01               # (B_J > A_J*C2_SHARE makes c_{2,12} anti-phase);
                                 # overall scale sets S_1212/sqrt(S11 S22) = 0.10
DT_SHIFT = 1.5                   # causal lag of e_B's shared part [C, Im parts]

# Class-F lines: GaAs nuclear-difference triplet at B_eff = 600 mT
# [Malinowski 2017]. Absolute amplitudes (the x8 / x20 peak-over-smooth-total
# factors are already folded in), qubit-local only.
_LINE_CENTERS = jnp.array([0.261, 0.273, 0.534])
_LINE_SIGMA = 0.02
_LINE_AMP_Q1 = jnp.array([8.428e-03, 8.143e-03, 4.905e-03])
_LINE_AMP_Q2 = jnp.array([2.3587e-02, 2.2973e-02, 1.5878e-02])

_LINES_ON = (_REGIME != "bland")


def _plaw(w, g):
    """IR-regularized power law (w^2 + W_IR^2)^(-g/2); finite at w = 0."""
    return (w**2 + W_IR**2)**(-g/2)


def _lines(w, amps):
    out = jnp.zeros_like(w)
    for i in range(3):
        out = out + amps[i]*Gauss(w, _LINE_CENTERS[i], _LINE_SIGMA)
    return out


# --- Component spectra (consumed by the trajectory synthesis) ---------------------

@jax.jit
def S_el_A(w):
    """Electrical (charge-noise) PSD seen at qubit 1. w, S in tau units."""
    return A_EL_1*_plaw(w, _G_EL_1)


@jax.jit
def S_el_B(w):
    """Electrical (charge-noise) PSD seen at qubit 2."""
    return A_EL_2*_plaw(w, _G_EL_2)


if _LINES_ON:
    @jax.jit
    def S_nuc_1(w):
        """Local nuclear PSD at qubit 1 (Class F: smooth + reduced line triplet)."""
        return A_NUC_1*_plaw(w, _G_NUC) + _lines(w, _LINE_AMP_Q1)

    @jax.jit
    def S_nuc_2(w):
        """Local nuclear PSD at qubit 2 (Class F: smooth + full line triplet)."""
        return A_NUC_2*_plaw(w, _G_NUC) + _lines(w, _LINE_AMP_Q2)
else:
    @jax.jit
    def S_nuc_1(w):
        """Local nuclear PSD at qubit 1 (Class M: smooth)."""
        return A_NUC_1*_plaw(w, _G_NUC)

    @jax.jit
    def S_nuc_2(w):
        """Local nuclear PSD at qubit 2 (Class M: smooth)."""
        return A_NUC_2*_plaw(w, _G_NUC)


# --- Public six spectra ------------------------------------------------------------

@jax.jit
def S_11(w):
    """Self-spectrum of qubit 1: electrical + local nuclear."""
    return S_el_A(w) + S_nuc_1(w)


@jax.jit
def S_22(w):
    """Self-spectrum of qubit 2: electrical + local nuclear."""
    return S_el_B(w) + S_nuc_2(w)


@jax.jit
def _cross_el(w):
    """Cross-PSD of the two electrical fields (shared fraction, lag DT_SHIFT)."""
    return C2_SHARE*jnp.sqrt(S_el_A(w)*S_el_B(w))*jnp.exp(-1j*w*DT_SHIFT)


@jax.jit
def S_1212(w):
    """Self-spectrum of the ZZ (Ising) channel zeta_12 = A_J e_A - B_J e_B."""
    return (A_J**2*S_el_A(w) + B_J**2*S_el_B(w)
            - 2*A_J*B_J*jnp.real(_cross_el(w)))


@jax.jit
def S_1_2(w):
    """Cross-spectrum of qubits 1 and 2 (the shared electrical part only)."""
    return _cross_el(w)


@jax.jit
def S_1_12(w):
    """Cross-spectrum of qubit 1 and the ZZ channel."""
    return A_J*S_el_A(w) - B_J*_cross_el(w)


@jax.jit
def S_2_12(w):
    """Cross-spectrum of qubit 2 and the ZZ channel."""
    return A_J*jnp.conj(_cross_el(w)) - B_J*S_el_B(w)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Comb parameters (QNSExperimentConfig defaults; tau units)
    T = 160.0
    truncate = 20

    w = jnp.linspace(-2*np.pi*truncate/T, 2*np.pi*truncate/T, 5001)
    wk = jnp.linspace(-2*np.pi*truncate/T, 2*np.pi*truncate/T, 2*truncate + 1)

    spectra = {"S_11": S_11(w), "S_22": S_22(w), "S_1212": S_1212(w),
               "S_1_2": S_1_2(w), "S_1_12": S_1_12(w), "S_2_12": S_2_12(w)}
    spectra_k = {"S_11": S_11(wk), "S_22": S_22(wk), "S_1212": S_1212(wk),
                 "S_1_2": S_1_2(wk), "S_1_12": S_1_12(wk), "S_2_12": S_2_12(wk)}

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, vals) in zip(axs.ravel(), spectra.items()):
        vals_np = np.array(vals)
        ax.plot(np.array(w), np.real(vals_np), 'r--', lw=1)
        if np.iscomplexobj(vals_np):
            ax.plot(np.array(w), np.imag(vals_np), 'b--', lw=1)
        ax.plot(np.array(wk), np.real(np.array(spectra_k[name])), 'r^', ms=4)
        ax.set_title(name)
        ax.set_xlabel(r'$\omega\tau$')
        ax.set_yscale('asinh')
        ax.tick_params(direction='in')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = run_folder()
    path = os.path.join(project_root(), fname)
    if not os.path.exists(path):
        os.mkdir(path)

    key_map = {"S_11": "S11", "S_22": "S22", "S_1212": "S1212",
               "S_1_2": "S12", "S_1_12": "S112", "S_2_12": "S212"}
    save_data = {key_map[k]: np.array(v) for k, v in spectra_k.items()}
    save_data['wk'] = np.array(wk)
    save_data['T'] = T
    save_data['truncate'] = truncate
    save_data['dt_shift'] = DT_SHIFT
    save_data['model_version'] = MODEL_VERSION

    np.savez(os.path.join(path, 'simulated_spectra.npz'), **save_data)
    fig.savefig(os.path.join(path, 'input_spectra.pdf'))
    print(f"Simulated spectra saved to {os.path.join(path, 'simulated_spectra.npz')}")
