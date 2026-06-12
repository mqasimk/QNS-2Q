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
Amplitudes solve T2*(FID) = 800 tau per qubit (purified-28Si target, 20 us at
the 25 ns anchor; Yoneda et al., Nat. Nanotech. 13, 102 (2018)). Retargeted
from the initial 260 tau (PRApplied 20, 054024 devices) after the 2026-06-10
acceptance-gate run: at 260 tau the 320-640 tau gate times sit at 1.4-2.8 T2*
(bare gate unrescuable, DC fit below the shot floor); see NOISE_MODEL_SPEC.md.

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

_REGIME = current_regime()

# SI anchor: conversion constant only. The anchored classes display at the
# legacy tau = 25 ns; the showcase regime displays at tau = 5 ns (Ge-hole /
# fast-Rabi anchor, where T2* = 3500 tau = 17.6 us matches Hendrickx 2024).
_TAU_SI = 5.0e-9 if _REGIME == "showcase" else 2.5e-8

MODEL_VERSION = ("showcase-trap-20260612" if _REGIME == "showcase"
                 else "anchored-tarucha-20260610")


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

A_EL_1 = 1.067936e-04            # amplitudes: T2*(FID) = 800 tau per qubit with
A_EL_2 = 1.565736e-04            # electrical fractions 0.88 / 0.80 at w~ = 0.35
A_NUC_1 = 8.622470e-06
A_NUC_2 = 1.692308e-05

C2_SHARE = 0.8                   # shared fraction of the electrical noise power
A_J = 4.270645e-01               # J difference-coupling weights; B_J/A_J = 1.05
B_J = 4.484177e-01               # (B_J > A_J*C2_SHARE makes c_{2,12} anti-phase);
                                 # overall scale sets S_1212/sqrt(S11 S22) = 0.10
DT_SHIFT = 1.5                   # causal lag of e_B's shared part [C, Im parts]

# Class-F lines: GaAs nuclear-difference triplet at B_eff = 600 mT
# [Malinowski 2017]. Absolute amplitudes (the x3.2 / x8 peak-over-smooth-total
# factors are already folded in), qubit-local only. 2026-06-10: factors
# reduced from x8/x20 -- at x20 the comb harmonic ON the line decays to
# coherence ~1e-4 (unmeasurable); at x8 the at-line coefficient is C ~ 1.8
# (coherence ~3e-2, measurable at 64k shots). Reserved knob #2 of
# NOISE_MODEL_SPEC.md; gate-side NT margin to be re-checked.
_LINE_CENTERS = jnp.array([0.261, 0.273, 0.534])
_LINE_SIGMA = 0.02
_LINE_AMP_Q1 = jnp.array([1.011e-03, 9.770e-04, 5.880e-04])
_LINE_AMP_Q2 = jnp.array([2.817e-03, 2.744e-03, 1.897e-03])

_LINES_ON = (_REGIME != "bland")

# --- SHOWCASE regime constants (SHOWCASE-0612) -------------------------------------
# Engineered trap landscape, solved by scripts/calibrate_showcase.py (see its
# header for the full design rationale). Composition per qubit: quasistatic
# hyperfine (carries T2* = 3500 tau), a Connors-type local TLF knee (catches
# CDD1-2, whose passbands at the featured Tg = 320 tau sit below comb tooth 1),
# a defect-harmonic line family w0..4*w0 (catches CDD3-5 + the CPMG-16 gap)
# plus a top-window line at 0.365 (catches the densest trains AND any
# line-blind design fleeing up the falling floor under the min-sep = 8 tau
# control-bandwidth scenario), all over a quiet 1/f^0.9 electrical floor that
# sets the noise-tailored target ~2e-4. The ZZ channel additionally carries an
# independent coupler TLF resonance + knee (j(t) in zeta_12) that ONLY the
# two-qubit spectra can reveal -- the ablation-ladder rung-(c) channel.
_SC_G_QS, _SC_W_QS = 2.0, 2.5e-3   # slow-bath exponent / IR cutoff (2.5e-3 =
                                   # 400 tau correlation: keeps the FID-slope
                                   # DC protocol inside its linear window)
_SC_G_FL = 0.9                     # electrical-floor exponent (W_IR shared)
_SC_W_TLF = 0.025                  # TLF knee position [Connors 2022 shape]
_SC_A_FL_1 = 1.356238e-08          # floor amplitudes: the stylized "quiet
_SC_A_FL_2 = 1.763109e-08          # electrical environment" contrast knob
_SC_A_QS_1 = 4.028474e-09          # quasistatic amplitudes: T2* = 3500 tau.
_SC_A_QS_2 = 4.027177e-09          # NOTE: with W_QS = 2.5e-3 the QS in-band
                                   # tail alone punishes CDD1/2 (~2e-2/6e-3)
                                   # -- no self-spectra TLF knee needed; its
                                   # w^-2 tail was the dominant NT-window toll
# NT parking window between the 4*w0 line's +3 sigma and the top line's
# -3 sigma: [0.258, 0.312] (probe-iterated 2026-06-12; wider flanks left the
# NT winner on a 2.5-sigma shoulder paying ~2x floor).
_SC_LINE_CENTERS = jnp.array([0.051, 0.102, 0.153, 0.204, 0.372])
# Flanking mines (4 w0, top) are CONSTANT-AREA narrowed (probe iteration 7):
# CDD's wide filter lobes feel the line AREA, NT's parking feels the TAILS --
# narrower-but-taller keeps CDD trapped while the window shoulders collapse.
_SC_LINE_SIGMAS = jnp.array([0.016, 0.020, 0.018, 0.014, 0.015])
_SC_LINE_AMP_Q1 = jnp.array([1.45e-05, 1.75e-05, 1.60e-05, 8.20e-05, 6.50e-05])
_SC_LINE_AMP_Q2 = jnp.array([1.95e-05, 2.25e-05, 2.05e-05, 1.03e-04, 8.10e-05])
_SC_ZZ_W0, _SC_ZZ_SIG = 0.2356, 0.020
_SC_H_ZZ_LINE = 1.0e-05            # coupler TLF resonance (2Q-only structure)
_SC_H_ZZ_KNEE = 0.5e-06            # 3e-6 cost full-NT ~1.1e-4 via the FORCED
                                   # low-w ZZ exposure (dc_12 >= pi/(4 Jmax))

HAS_ZZ_EXTRA = (_REGIME == "showcase")


def line_priors():
    """(centers, sigma) of the known qubit-local line set, or None (bland).

    The experimentally-known part of the line fingerprint only: the centers are
    set by the defect/nuclear species (and B field) and the width is the comb's
    resolution limit. Heights are deliberately NOT exposed -- a line-aware
    reconstruction model must fit them from its own reconstructed data (see
    ``characterize.systematics``). For per-line widths and the per-channel
    structure (showcase: the ZZ channel carries its own line) use
    ``line_priors_per_channel``; this legacy form quotes the smallest width.
    """
    import numpy as np
    if _REGIME == "showcase":
        return (np.asarray(_SC_LINE_CENTERS, dtype=float),
                float(np.min(np.asarray(_SC_LINE_SIGMAS))))
    if not _LINES_ON:
        return None
    return np.asarray(_LINE_CENTERS, dtype=float), float(_LINE_SIGMA)


def line_priors_per_channel():
    """{spectrum key: (centers, sigmas)} of known line content, or None (bland).

    Per-channel, per-line generalization of ``line_priors`` (same height-blind
    contract). The anchored featured class carries the nuclear-difference
    triplet on the qubit-local channels only; the showcase class adds its
    defect-harmonic family on S11/S22 and the coupler TLF resonance on S1212.
    """
    import numpy as np
    if _REGIME == "showcase":
        c = np.asarray(_SC_LINE_CENTERS, dtype=float)
        s = np.asarray(_SC_LINE_SIGMAS, dtype=float)
        return {'S11': (c, s), 'S22': (c, s),
                'S1212': (np.array([_SC_ZZ_W0]), np.array([_SC_ZZ_SIG]))}
    if not _LINES_ON:
        return None
    c = np.asarray(_LINE_CENTERS, dtype=float)
    s = np.full(c.size, float(_LINE_SIGMA))
    return {'S11': (c, s), 'S22': (c, s)}


def _plaw(w, g, wir=W_IR):
    """IR-regularized power law (w^2 + wir^2)^(-g/2); finite at w = 0."""
    return (w**2 + wir**2)**(-g/2)


def _knee(w, h, wc):
    """Lorentzian TLF knee: plateau h below wc, falling w^-2 above."""
    return h/(1 + (w/wc)**2)


def _lines(w, amps):
    out = jnp.zeros_like(w)
    for i in range(3):
        out = out + amps[i]*Gauss(w, _LINE_CENTERS[i], _LINE_SIGMA)
    return out


def _sc_lines(w, amps):
    out = jnp.zeros_like(w)
    for i in range(5):
        out = out + amps[i]*Gauss(w, _SC_LINE_CENTERS[i], _SC_LINE_SIGMAS[i])
    return out


# --- Component spectra (consumed by the trajectory synthesis) ---------------------

if _REGIME == "showcase":
    @jax.jit
    def S_el_A(w):
        """Quiet electrical floor at qubit 1 (showcase contrast knob)."""
        return _SC_A_FL_1*_plaw(w, _SC_G_FL)

    @jax.jit
    def S_el_B(w):
        """Quiet electrical floor at qubit 2."""
        return _SC_A_FL_2*_plaw(w, _SC_G_FL)

    @jax.jit
    def S_nuc_1(w):
        """Qubit-1 local low-frequency + featured component (showcase):
        quasistatic slow bath (T2* carrier, whose in-band w^-2 tail also
        punishes CDD1-2) + trap-line family."""
        return (_SC_A_QS_1*_plaw(w, _SC_G_QS, _SC_W_QS)
                + _sc_lines(w, _SC_LINE_AMP_Q1))

    @jax.jit
    def S_nuc_2(w):
        """Qubit-2 local low-frequency + featured component (showcase)."""
        return (_SC_A_QS_2*_plaw(w, _SC_G_QS, _SC_W_QS)
                + _sc_lines(w, _SC_LINE_AMP_Q2))

    @jax.jit
    def S_zz_extra(w):
        """Independent coupler-defect PSD j(t) on the ZZ channel ONLY
        (zeta_12 = A_J e_A - B_J e_B + j): TLF resonance + knee. Structure
        invisible to single-qubit QNS -- the rung-(c) ablation channel."""
        return (_SC_H_ZZ_LINE*Gauss(w, _SC_ZZ_W0, _SC_ZZ_SIG)
                + _knee(w, _SC_H_ZZ_KNEE, _SC_W_TLF))
else:
    @jax.jit
    def S_el_A(w):
        """Electrical (charge-noise) PSD seen at qubit 1. w, S in tau units."""
        return A_EL_1*_plaw(w, _G_EL_1)

    @jax.jit
    def S_el_B(w):
        """Electrical (charge-noise) PSD seen at qubit 2."""
        return A_EL_2*_plaw(w, _G_EL_2)

    S_zz_extra = None

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
    """Self-spectrum of the ZZ (Ising) channel.

    zeta_12 = A_J e_A - B_J e_B (+ the independent coupler defect j(t) in the
    showcase regime, whose PSD S_zz_extra appears here and ONLY here -- j is
    independent of e_A/e_B, so the cross-spectra are untouched)."""
    base = (A_J**2*S_el_A(w) + B_J**2*S_el_B(w)
            - 2*A_J*B_J*jnp.real(_cross_el(w)))
    if HAS_ZZ_EXTRA:
        base = base + S_zz_extra(w)
    return base


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
    # float64 for the saved file: this block runs in its own process, so
    # enabling x64 here cannot perturb the record/replay pipeline numerics.
    # Layout note: the grid is two-sided [-wmax, wmax] (41 points, w=0 at the
    # CENTER, index 0 = -wmax) -- consumers that look for a DC sample at wk[0]
    # correctly treat this file as DC-less (OPT-MISC-0611).
    jax.config.update("jax_enable_x64", True)
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
