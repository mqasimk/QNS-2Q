"""
Stage 2 of the QNS pipeline ("characterize" arm): turn the raw correlator
observables measured in Stage 1 into the six power spectral densities (PSDs)
of the two-qubit dephasing noise, complete with error bars, and save/plot them.

Physics picture. A two-qubit QNS experiment does NOT measure a noise spectrum
directly -- it measures qubit/coupler *coherence decays* (expectation values
like C_12_0_MT_1) under a family of control-pulse sequences (CPMG/CDD combs,
Ramsey/FID). Each such observable is a linear functional of the underlying
noise PSD, weighted by that sequence's filter function. This module solves the
resulting linear inverse problem -- "given these decay curves, what spectrum
produced them?" -- at a discrete comb of harmonic frequencies (via
``characterize/inversion.py``) plus one extra DC (omega=0) point fit from a
separate multi-time Ramsey-slope sweep. It then folds in the deterministic
reconstruction bias inherent to that inversion (via
``characterize/systematics.py``, "the comb-inversion systematic") so the
quoted error bars are statistical+systematic, not just statistical.

Pipeline position. This is the second of two stages in the "characterize" arm
(the other arm is "control": ``control/cz.py`` / ``control/idle.py``, which
optimize gates against the spectra this module produces):

    characterize/experiments.py (Stage 1: simulate QNS shots)
            -> results.npz, params.npz  (per-run-folder; see qns2q.paths)
            -> THIS MODULE (Stage 2: invert observables -> spectra)
            -> specs.npz
            -> control/cz.py, control/idle.py (Stage 3: gate optimization)

Inputs (read from a run folder resolved by ``qns2q.paths.run_folder()``, e.g.
``DraftRun_NoSPAM_<regime>/``): ``results.npz`` (the measured observables and
their statistical errors) and ``params.npz`` (the experiment configuration:
pulse-comb timings, repetition count, noise-model stamp, etc.).

Outputs (written back into the same run folder): ``specs.npz`` (the six
reconstructed spectra on the frequency comb, each with statistical, systematic,
and combined error arrays -- consumed by ``control/cz.py``/``control/idle.py``
and by ``scripts/report_showcase_figs.py``) plus two publication-quality PDF
figures under ``<data_folder>/figures/reconstruction/``.

Callers: ``scripts/run_reconstruct.py`` and ``scripts/run_capture_arm.py`` run
this module's ``main()`` for the NoSPAM pipeline; ``scripts/run_spam_reconstruct.py``
runs it for each SPAM-protocol arm; ``scripts/report_showcase_figs.py`` imports
``setup_pub_rcparams`` from here to keep every paper figure's matplotlib style
consistent. See ``qns2q/characterize/inversion.py`` (the linear-algebra
inversion + DC fitting) and ``qns2q/characterize/systematics.py`` (the bias
model) for the machinery this file orchestrates.
"""

import matplotlib
matplotlib.use('Agg')

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qns2q.characterize.inversion import (recon_S_11, recon_S_22, recon_S_1_2, recon_S_12_12, recon_S_1_12, recon_S_2_12,
                                recon_S_11_dc, recon_S_22_dc, recon_S_1212_dc,
                                recon_S_1212_dc_echo,
                                recon_S_1_2_dc, recon_S_1_12_dc, recon_S_2_12_dc,
                                regress_observables_over_M,
                                truncation_bias_estimate)
from qns2q.characterize.systematics import (forward_model_systematic, analytic_spectra,
                                            dc_fit_systematic, selfconsistent_spectra)
from qns2q.noise.spectra import S_11, S_22, S_1_2, S_1212, S_1_12, S_2_12
from qns2q.paths import run_folder, project_root


# The same six spectra get three different short-name spellings across the
# codebase: the terse 'S11'/'S22'/... keys used by characterize/systematics.py,
# the '_k'-suffixed keys used internally below (self.reconstructed_spectra),
# and the plain 'S11'/'S22'/... keys written to specs.npz (see
# save_reconstructed_spectra). These three dicts are the lookup tables that
# translate between the systematics-module spelling and this file's internal
# spelling in both directions, so the same physical spectrum can be looked up
# by whichever name a given piece of code below already has in hand.
_SYS_TO_SPEC = {'S11': 'S_11_k', 'S22': 'S_22_k', 'S1212': 'S_12_12_k',
                'S12': 'S_1_2_k', 'S112': 'S_1_12_k', 'S212': 'S_2_12_k'}
_SYS_TO_ERR = {'S11': 'S_11_err', 'S22': 'S_22_err', 'S1212': 'S_12_12_err',
               'S12': 'S_1_2_err', 'S112': 'S_1_12_err', 'S212': 'S_2_12_err'}
_SPEC_TO_SYS = {v: k for k, v in _SYS_TO_SPEC.items()}


def _quad_combine(stat, sys):
    """Quadrature-combine statistical and systematic error arrays, per component for
    complex (cross-spectra) arrays: sqrt(Re_stat^2+Re_sys^2) + i sqrt(Im_stat^2+Im_sys^2).

    Statistical error (shot noise from a finite number of experiment repeats)
    and systematic error (the deterministic comb-inversion bias computed in
    ``add_systematic_errors``/``unfold_comb_bias`` below) come from unrelated
    mechanisms, so the standard error-propagation rule for independent sources
    applies: combined = sqrt(stat^2 + sys^2), taken separately on the real and
    imaginary parts for the complex-valued cross-spectra.
    """
    if np.iscomplexobj(stat) or np.iscomplexobj(sys):
        return (np.sqrt(np.real(stat) ** 2 + np.real(sys) ** 2)
                + 1j * np.sqrt(np.imag(stat) ** 2 + np.imag(sys) ** 2))
    return np.sqrt(np.asarray(stat) ** 2 + np.asarray(sys) ** 2)


# --- Publication figure constants ---
# Sizes (inches) and colors shared by plot_all_spectra/plot_cross_spectra below,
# so every reconstruction figure in the paper uses one consistent look.

FIG_WIDTH = 7.0    # Two-column width (inches)
FIG_HEIGHT = 4.5   # 2-row panel height (legacy, kept for reference)
FIG_HEIGHT_1ROW = 2.5  # Single-row panel height
FIG_HEIGHT_3ROW = 7.0  # Three-row panel height
FIG_HEIGHT_3x2 = 5.5  # Three-row, two-column panel height

# Okabe-Ito colorblind-safe palette (standard choice for physics figures);
# used consistently for "reconstructed" (vermillion) vs "theory" (also
# vermillion, dashed) vs the imaginary part of a cross-spectrum (blue).
COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
    "grey_fill": "#E0E0E0",
}

# Confidence level (in standard deviations) for the plotted reconstruction error bars.
# The saved specs.npz keeps 1-sigma semantics (_err/_sys/_errtot); only the *plotted*
# bars are scaled. 2-sigma bars are the QNS-figure convention: ~95% of points'
# statistically-consistent bars then cover the theory curve (state "2 sigma" in the
# paper caption). Set to 1 for 1-sigma bars.
ERRORBAR_SIGMA = 2

# Subfolder inside each data folder for figures
FIGURES_SUBDIR = "figures"
RECONSTRUCTION_SUBDIR = os.path.join(FIGURES_SUBDIR, "reconstruction")


def _set_asinh_scale(ax, scale_y, ylim_data=None):
    """asinh y-scale with a data-driven linear width, ignoring non-finite entries
    (e.g. spectra not accessible under the SPAM-robust protocol).

    Why asinh and not a plain log scale: these spectra span many decades in
    magnitude (self-spectra) but the cross-spectra also cross zero and can go
    negative, which a log axis cannot display at all. ``asinh(y/scale)``
    behaves like a log scale far from zero (so the huge dynamic range is still
    compressed nicely) but is linear near zero, so small/negative points stay
    visible instead of being dropped or throwing a domain error.

    The linear width is the median |finite nonzero| of ``scale_y``. For the
    SPAM-arm figures pass the (arm-independent) theory curve so the asinh warp
    matches across arms, plus ``ylim_data`` -- the cross-arm envelope -- to pin
    identical explicit y-limits (padded in asinh space) instead of per-arm
    autoscaling."""
    scale_y = np.asarray(scale_y)
    finite = scale_y[np.isfinite(scale_y) & (scale_y != 0)]
    scale = float(np.median(np.abs(finite))) if finite.size else 1.0
    ax.set_yscale('asinh', linear_width=scale)
    linthresh = 10 ** np.ceil(np.log10(scale))
    ax.yaxis.set_major_locator(ticker.SymmetricalLogLocator(linthresh=linthresh, base=10))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if ylim_data is not None:
        yd = np.asarray(np.real(ylim_data), dtype=float)
        yd = yd[np.isfinite(yd)]
        if yd.size:
            g = np.arcsinh(yd / scale)
            pad = 0.05 * max(float(g.max() - g.min()), 1.0)
            ax.set_ylim(scale * np.sinh(g.min() - pad), scale * np.sinh(g.max() + pad))


# Arms sharing per-panel figure axes. The robust arm is deliberately excluded:
# its current data is from a different run (weaker SPAM, 4k shots vs 64k), so
# its blown-up error bars would stretch every panel; it keeps per-run
# autoscaling until it is regenerated on the same replay as the other arms.
_SHARED_AXIS_ARMS = ('reference', 'raw', 'mitigated')


def _sibling_spam_envelopes(data_folder):
    """Per-spectrum y-envelopes (points +/- the plotted ERRORBAR_SIGMA bars) of
    the _SHARED_AXIS_ARMS' saved reconstructions, keyed like _SYS_TO_SPEC.

    Those arms are compared side by side, so their figures must share per-panel
    axes. Returns {} unless ``data_folder`` is itself one of the shared arms
    (``DraftRun_SPAM_<regime>_<protocol>``) -- NoSPAM and robust runs keep the
    legacy per-run autoscaling. Arms whose specs.npz does not exist yet are
    skipped, so after generating fresh data rerun the reconstruct stage once per
    arm after all arms exist to converge the shared limits."""
    name = os.path.basename(os.path.normpath(data_folder))
    m = re.match(r'^(DraftRun_SPAM_\w+?)_(raw|mitigated|robust|reference)$', name)
    if not m or m.group(2) not in _SHARED_AXIS_ARMS:
        return {}
    parent = os.path.dirname(os.path.normpath(data_folder))
    env = {k: [] for k in _SYS_TO_SPEC}
    for protocol in _SHARED_AXIS_ARMS:
        spec_path = os.path.join(project_root(), parent,
                                 f'{m.group(1)}_{protocol}', 'specs.npz')
        if not os.path.exists(spec_path):
            continue
        d = np.load(spec_path)
        for key in env:
            if key not in d.files:
                continue
            val = np.asarray(d[key])
            if not np.any(np.isfinite(val)) or np.all(val == 0):
                continue  # arm cannot access this spectrum (e.g. robust S_l,12)
            err_key = next((k for k in (f'{key}_errtot', f'{key}_err')
                            if k in d.files), None)
            bars = (ERRORBAR_SIGMA * np.asarray(d[err_key]) if err_key
                    else np.zeros_like(val))
            # A flagged DC point carries a deliberately inflated bar (the spectrum's
            # own scale); letting it set the shared limits would squash the harmonic
            # structure in every arm's panel.
            if f'{key}_dc_ok' in d.files and not bool(np.asarray(d[f'{key}_dc_ok'])):
                val, bars = val[1:], bars[1:]
            parts = (np.real, np.imag) if np.iscomplexobj(val) else (np.real,)
            for part in parts:
                v, b = part(val), np.abs(part(bars))
                env[key] += [v - b, v + b]
    return {k: np.concatenate(v) for k, v in env.items() if v}


def setup_pub_rcparams(font_scale='compact'):
    """Configure matplotlib rcParams for publication-quality figures.

    This is a *global* matplotlib style switch (it mutates ``plt.rcParams`` in
    place, so it affects every figure drawn afterward in the same process) --
    call it once before drawing, not per-axes. It is called both by this
    module's own plotting methods below and, imported directly, by
    ``scripts/report_showcase_figs.py``, so that every reconstruction-style
    panel across the whole paper (not just the ones made in this file) shares
    the same fonts/line widths/tick style.

    Parameters
    ----------
    font_scale : str
        'large' for standalone single-panel figures,
        'compact' for multi-panel combined figures (APS style).
    """
    sizes = {
        'large': {"font.size": 18, "axes.labelsize": 16, "axes.titlesize": 16,
                  "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16},
        'compact': {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8},
    }
    base = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
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
    }
    base.update(sizes[font_scale])
    plt.rcParams.update(base)


# A `@dataclass` auto-writes the __init__ from the type-annotated attributes
# below, so you construct this with SpectraReconConfig(data_folder=...) and the
# rest fill in from defaults / __post_init__. `field(init=False)` marks an
# attribute that is NOT a constructor argument -- it has no default value here
# and is instead populated by __post_init__ (below) after construction. Unlike
# a typical plain data container, __post_init__ here does real file I/O (it
# loads params.npz from disk), so simply *constructing* a SpectraReconConfig
# already reads a file and can raise FileNotFoundError/KeyError -- it is not a
# no-op the way building a plain dataclass usually is.
@dataclass
class SpectraReconConfig:
    """Configuration and parameters for spectra reconstruction.

    Reconstruction-method options (all default to the legacy behavior):

    inversion_method : {'direct','lstsq','tikhonov'}
        Linear-solve backend for the harmonic inversion. 'direct' is the legacy
        square inverse; 'lstsq'/'tikhonov' are robust to ill-conditioning and to
        overdetermined (tall-U) probe sets.
    reg_lambda : float
        Tikhonov ridge parameter (only used when inversion_method='tikhonov').
    enforce_nonneg : bool
        Project the three real self-spectra onto S>=0 via NNLS (physical prior).
    diagnostics : bool
        Print cond(U) / singular values and a high-frequency truncation-bias
        estimate for each reconstruction.
    """
    data_folder: str
    inversion_method: str = 'direct'
    reg_lambda: float = 0.0
    enforce_nonneg: bool = False
    diagnostics: bool = False
    # Fold the deterministic comb-inversion systematic into the quoted error bars
    # (see characterize.systematics). True for honest, n_shots-independent error bars.
    compute_systematic: bool = True
    # Bias-corrected unfolding (ON): subtract the SELF-CONSISTENT comb-inversion
    # bias (forward model built from the reconstructed spectra alone -- no
    # ground-truth knowledge) from the markers, then quote the HONEST residual
    # systematic in add_systematic_errors -- the EXACT forward-model comb bias of
    # the model minus what the unfold actually subtracted. This corrects the
    # in-band-correctable bias (self-spectra, the S_1_2 line) so those channels
    # carry tight ~statistical bars on near-truth markers, while the out-of-band-
    # aliasing-dominated S_1_12/S_2_12 high harmonics -- which no blind
    # extrapolation reaches ([w_max, 2 w_max] is unsampled) -- keep the full comb
    # bias in their bars. Correct coverage (max pull ~2 sigma across all six
    # spectra) AND the lower cross-spectra rel-dev (V10-QNS-BARFIX-0624: the
    # 2026-06-24 fix described in the two comment blocks above/below -- this
    # tag is a self-reference to that fix, not an external doc, kept here so
    # the change is easy to find again by searching the git history/paper
    # notes for the same string).
    # NB: the earlier default quoted the fixed-point ITERATION INCREMENT as the
    # residual, which under-covered the uncorrected out-of-band bias ~10x ->
    # spurious 4-sigma teeth. The exact-minus-applied residual fixed that.
    unfold_bias: bool = True
    # Fraction of the APPLIED correction conservatively retained as residual
    # systematic (standard unfolding practice: the self-consistent model is
    # built from noisy points, so the fixed-point increment alone under-quotes
    # its error). With the line+tail-aware model (2026-06-11) the retained
    # fraction covers the fitted line-height/tail-exponent error; 0.5 was
    # validated against ground truth -- reference-arm pulls ~1.
    unfold_residual_frac: float = 0.5
    # Everything below is loaded FROM params.npz by __post_init__, not passed
    # in by the caller -- these mirror the exact experiment configuration
    # Stage 1 (characterize/experiments.py) used, so the inversion math here
    # matches the filter functions that actually produced the data:
    params: Dict[str, Any] = field(init=False)  # the raw params.npz archive (all fields, for provenance stamps etc.)
    t_vec: np.ndarray = field(init=False)  # Stage-1 time-domain integration grid (0 .. M*T); re-used by the systematics forward model
    w_grain: int = field(init=False)  # number of points in the continuous frequency grid used for plotting/synthesis (not the harmonic comb itself)
    wmax: float = field(init=False)  # half-bandwidth (max angular frequency, tau-units) of that continuous grid
    truncate: int = field(init=False)  # number of comb harmonics to reconstruct (== len(c_times))
    gamma: float = field(init=False)  # single-qubit dephasing decay rate the Stage-1 experiment was configured with (validated here, not otherwise used in the inversion below)
    gamma_12: float = field(init=False)  # qubit-qubit (Ising/ZZ) cross-decay rate, same role as gamma
    c_times: np.ndarray = field(init=False)  # control (probe pulse) timing parameter, one entry per harmonic reconstructed
    M: int = field(init=False)  # number of pulse-sequence repetitions per experiment shot (the comb's harmonic spacing is set by T, its height by M)
    T: float = field(init=False)  # total duration (tau-units) of one pulse-sequence repetition; comb harmonics sit at 2*pi*(k+1)/T

    def __post_init__(self):
        """Load parameters from the data folder after initialization.

        This is the dataclass "constructor does I/O" pattern flagged in the
        class comment above: building a SpectraReconConfig actually opens and
        reads params.npz right now, and fails fast (raises) if Stage 1 hasn't
        been run yet or wrote something malformed.
        """
        path = os.path.join(project_root(), self.data_folder)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Data folder not found at: {path}")

        params_path = os.path.join(path, "params.npz")
        self.params = np.load(params_path)  # np.load on an .npz returns a lazy dict-like archive, not the arrays themselves

        # Table-driven validation: rather than one hand-written check per
        # field, loop over (name, expected type) and apply the same
        # presence/None/type checks to each -- keeps the nine checks below in
        # sync and in one place instead of nine near-duplicated blocks.
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

        # Units guard: the pipeline now works in tau units (tau = 1; see
        # noise/spectra.py). A run folder generated by the SI-era code stores
        # tau = 2.5e-8 s -- its observables are still self-consistent, but the
        # analytic theory overlays and systematics computed against the tau-unit
        # spectra would be wrong. Regenerate Stage 1 for such folders.
        if 'tau' in self.params:
            tau_run = float(np.asarray(self.params['tau']))
            if not np.isclose(tau_run, 1.0):
                print(f"[reconstruct] WARNING: run folder '{self.data_folder}' was "
                      f"generated with tau={tau_run:g} (SI-era units). Theory "
                      f"overlays/systematics assume tau-unit data -- regenerate "
                      f"Stage 1 (scripts/run_capture_arm.py) for this folder.")

        # Noise-synthesis half-band override (the simulated world spans twice
        # this; 0.0 = legacy formula wmax). Needed by the echo-DC mirror so it
        # integrates over the band the run's world actually populates.
        self.synth_wmax = (float(np.asarray(self.params['synth_wmax']))
                           if 'synth_wmax' in self.params else 0.0)
        # Noise-synthesis grid convention (midpoint tones exclude w=0; legacy
        # endpoint grids include it). Together with synth_wmax/wmax + w_grain
        # this defines the world's exact tone grid for the DC mirrors.
        self.midpoint = (bool(np.asarray(self.params['midpoint']))
                         if 'midpoint' in self.params else False)

        # Optional SPAM-protocol metadata (absent in legacy / NoSPAM runs).
        self.spam_protocol = (str(self.params['spam_protocol'])
                              if 'spam_protocol' in self.params else 'none')
        self.m_sweep_robust = (np.asarray(self.params['m_sweep_robust'], dtype=int)
                               if 'm_sweep_robust' in self.params
                               else np.array([], dtype=int))


class SpectraReconstructor:
    """Orchestrates one full Stage-2 run: load observables -> invert to spectra
    -> (optionally) bias-correct and add systematic error bars -> plot -> save.

    One instance is built per run folder (see ``SpectraReconConfig``) and its
    methods are meant to be called in the order ``run()`` calls them (loosely:
    ``load_observables`` -> ``reconstruct`` -> ``unfold_comb_bias`` ->
    ``add_systematic_errors`` -> the two ``plot_*`` methods ->
    ``save_reconstructed_spectra``); most of them mutate ``self`` (e.g.
    ``self.reconstructed_spectra``) rather than returning values, so later
    methods depend on earlier ones having already run. Use ``run()`` unless you
    specifically need to intervene between stages.
    """

    def __init__(self, config: SpectraReconConfig):
        """Initializes the reconstructor with a given configuration."""
        self.config = config
        self.observables: Dict[str, np.ndarray] = {}
        self.reconstructed_spectra: Dict[str, np.ndarray] = {}
        self.wk: np.ndarray = np.array([])  # frequency comb (angular, tau-units) the reconstructed spectra live on; wk[0]=0 is the DC point

    def load_observables(self):
        """Loads the Stage-1 observables (results.npz: the measured coherence-decay
        correlators, e.g. C_12_0_MT_1, and their statistical errors) from the data folder."""
        path = os.path.join(project_root(), self.config.data_folder, "results.npz")
        self.observables = np.load(path)

    def _world_grid(self):
        """(half_band, w_grain, midpoint) of the run's noise-synthesis tone grid.

        The DC mirrors must quadrature over the world's actual tones (see
        ``systematics._world_w_grid``); a continuous-trapezoid mirror truncates
        the (0, w_min) DC cusp and integrates band the world never populated.
        """
        c = self.config
        return (float(c.synth_wmax or c.wmax), int(c.w_grain), bool(c.midpoint))

    def reconstruct(self):
        """Reconstructs the six spectra at the comb harmonics + DC from the
        loaded Stage-1 observables.

        This is the core Stage-2 step: for each of the six spectra, look up
        the handful of measured correlators (e.g. C_12_0_MT_1/2 for S_11) that
        its filter functions couple to, hand them to the matching
        ``characterize.inversion.recon_S_*`` function (which builds and solves
        the linear system relating "observable at each probe-pulse timing" to
        "spectrum at each comb harmonic"), and stitch the harmonic result
        together with a separately-fit DC (omega=0) point. Populates
        ``self.reconstructed_spectra``/``self.reconstructed_spectra_err`` and
        ``self.wk`` (the frequency grid those spectra live on, with wk[0]=0).
        """
        c = self.config
        obs = self.observables

        # Check if error data is available
        has_errors = any(k.endswith('_err') for k in obs.keys())

        # Under the SPAM-robust protocol, a single harmonic observable can't be
        # made SPAM-free by itself; instead the experiment repeats each probe
        # sequence at several different repetition counts m ("M-sweep") and
        # stores each rep's result under a distinct key {key}_Mrep{m}. Plotting
        # the observable vs m and extrapolating (linear regression) to what it
        # would be with NO SPAM contribution isolates the physical (SPAM-free)
        # coefficient as the fitted slope evaluated at the reference repetition
        # count c.M -- the fitted intercept is where the SPAM-induced offset
        # gets absorbed instead of contaminating the spectrum. C_12_12 keys
        # already have an exactly SPAM-free estimator on their own, so they
        # skip this and are read directly (no regression needed).
        robust = getattr(c, 'spam_protocol', 'none') == 'robust'
        msweep = [int(m) for m in getattr(c, 'm_sweep_robust', [])]

        def get_coef(key):
            """(coefficient vector, stat error) for one experiment key."""
            if robust and msweep and f'{key}_Mrep{msweep[-1]}' in obs:
                obs_by_M = {m: obs[f'{key}_Mrep{m}'] for m in msweep}
                return regress_observables_over_M(obs_by_M, msweep, c.M)
            err = obs[key + '_err'] if (has_errors and key + '_err' in obs) else None
            return obs[key], err

        def get_errs(keys):
            if not has_errors: return None
            return [obs[k + '_err'] for k in keys]

        # Reconstruct spectra at comb harmonics wk = 2*pi*(k+1)/T
        wk_harmonics = np.array([2 * np.pi * (n + 1) / c.T for n in range(c.truncate)])

        # Helper to unpack result: each recon_S_* function takes a LIST of
        # observable arrays (one per probe-pulse timing) and, if errors were
        # supplied for every one of them, returns (spectrum, error); with no
        # errors it returns just the spectrum, so a same-shaped zero array
        # stands in for "error unavailable" to keep downstream code uniform.
        def call_recon(func, keys, **kwargs):
            vals, errs = zip(*(get_coef(k) for k in keys))
            errs = None if any(e is None for e in errs) else [np.asarray(e) for e in errs]
            res = func(list(vals), obs_err=errs, **kwargs)
            if errs is not None:
                return res
            return res, np.zeros(c.truncate) # Dummy error

        # Inversion-backend options (selectable via SpectraReconConfig).
        inv_opts = dict(inversion_method=c.inversion_method, reg_lambda=c.reg_lambda,
                        diagnostics=c.diagnostics)
        self_opts = dict(inv_opts, enforce_nonneg=c.enforce_nonneg)
        if c.diagnostics:
            print(f"[reconstruct] inversion_method={c.inversion_method} "
                  f"reg_lambda={c.reg_lambda} enforce_nonneg={c.enforce_nonneg}")

        S_11_k, S_11_err = call_recon(recon_S_11, ['C_12_0_MT_1', 'C_12_0_MT_2'], c_times=c.c_times, m=c.M, T=c.T, **self_opts)
        S_22_k, S_22_err = call_recon(recon_S_22, ['C_12_0_MT_1', 'C_12_0_MT_3'], c_times=c.c_times, m=c.M, T=c.T, **self_opts)
        S_12_12_k, S_12_12_err = call_recon(recon_S_12_12, ['C_1_0_MT_1', 'C_2_0_MT_1', 'C_12_0_MT_4'], c_times=c.c_times, m=c.M, T=c.T, **self_opts)
        S_1_2_k, S_1_2_err = call_recon(recon_S_1_2, ['C_12_12_MT_1', 'C_12_12_MT_2'], c_times=c.c_times, m=c.M, T=c.T, **inv_opts)
        if robust:
            # The C_a_b coefficients have no SPAM-robust estimator (companion
            # paper, Sec. SPAM-Robust QNS): S_1_12 and S_2_12 are not accessible
            # under this protocol. Quote NaN so downstream consumers see the gap.
            nanv = np.full(c.truncate, np.nan, dtype=complex)
            S_1_12_k, S_1_12_err = nanv.copy(), nanv.copy()
            S_2_12_k, S_2_12_err = nanv.copy(), nanv.copy()
            print("[robust] S_1_12 / S_2_12 are not accessible under the "
                  "SPAM-robust protocol; quoting NaN.")
        else:
            S_1_12_k, S_1_12_err = call_recon(recon_S_1_12, ['C_1_2_MT_1', 'C_1_2_MT_2'], c_times=c.c_times, m=c.M, T=c.T, **inv_opts)
            S_2_12_k, S_2_12_err = call_recon(recon_S_2_12, ['C_2_1_MT_1', 'C_2_1_MT_2'], c_times=c.c_times, m=c.M, T=c.T, **inv_opts)

        if c.diagnostics:
            for name, fn, args in [('S_11', S_11, ()), ('S_22', S_22, ()), ('S_1212', S_1212, ())]:
                frac = truncation_bias_estimate(fn, c.T, c.truncate, args=args)
                print(f"    [diag] {name}: spectral weight above comb cutoff "
                      f"omega_kmax = {100*frac:.2f}% (truncation bias)")

        # Reconstruct DC (w=0) values from the multi-time FID decay sweep. (FID =
        # "free induction decay", the bare Ramsey experiment with no decoupling
        # pulses; CPMG/CDD3 below name specific refocusing-pulse sequences --
        # CPMG = Carr-Purcell-Meiboom-Gill, CDD3 = 3rd-order concatenated
        # dynamical decoupling. Combining two of them into one observable name
        # like C_1_0_FIDCDD3 means "qubit 1, no coupling correction, FID then
        # CDD3 halves of the sequence".) The comb-harmonic inversion above
        # cannot see omega=0 (it only samples the harmonics 2*pi*k/T), so the
        # single DC point is measured separately from how coherence decays
        # over a SWEEP of increasing wait times -- its initial slope is
        # proportional to S(0).
        # Each recon_*_dc fits S(0) from the slope of C(t) over the adaptively-selected
        # measurable+linear window (inversion._ramsey_fit_dc); returns (val, err,
        # reliable). reliable=False flags quasi-static / sub-comb-cusp noise whose DC is
        # only a lower bound -- the figure then quotes an inflated (honest) bar.
        t_sweep = obs['dc_t_sweep']
        self.dc_t_sweep = t_sweep
        self.dc_reliable = {}

        # Per-time errors of the DC-sweep observables, for the DC forward-model
        # mirror (systematics.dc_fit_systematic): with errors the data fit is a
        # statistically-windowed, 1/sigma^2-weighted estimator, and the mirror
        # must run the SAME estimator on its exact curves to quote its bias.
        self.dc_fid_obs_err = {}
        if has_errors:
            for sk, key in (('S11', 'C_1_0_FIDCDD3'), ('S22', 'C_2_0_CDD3FID'),
                            ('S12', 'C_12_12_FID'), ('S112', 'C_1_12_FID'),
                            ('S212', 'C_2_12_FID')):
                if key + '_err' in obs:
                    self.dc_fid_obs_err[sk] = np.asarray(obs[key + '_err'])
            legacy_keys = ('C_1_0_FIDFID', 'C_2_0_FIDFID', 'C_12_0_FID_FID')
            if all(k + '_err' in obs for k in legacy_keys):
                self.dc_fid_obs_err['S1212'] = [np.asarray(obs[k + '_err'])
                                                for k in legacy_keys]

        def call_recon_dc(func, keys, sk, **kwargs):
            if any(k not in obs for k in keys):
                # SPAM-robust runs omit the C_a_b DC observables (no robust
                # estimator exists); their DC points are not accessible.
                self.dc_reliable[sk] = False
                return np.nan, np.nan
            val, err, reliable = func([obs[k] for k in keys], obs_err=get_errs(keys),
                                      t_sweep=t_sweep, **kwargs)
            self.dc_reliable[sk] = reliable
            return val, err

        S_11_dc, S_11_dc_err = call_recon_dc(recon_S_11_dc, ['C_1_0_FIDCDD3'], 'S11')
        S_22_dc, S_22_dc_err = call_recon_dc(recon_S_22_dc, ['C_2_0_CDD3FID'], 'S22')
        if all(k in obs for k in ('C_1_0_CDD1CDD1', 'C_1_0_CDD1CPMG')):
            # Double-echo estimator: direct first-order Var Phi_12 (see
            # inversion.recon_S_1212_dc_echo). The echo cycle time AND the data's
            # per-point errors are needed by the DC forward-model mirror in the
            # systematics/unfold stage (the mirror must window the exact curves
            # the same SNR-based way the data fit does).
            self.dc_echo_ct = float(np.asarray(obs['dc_echo_ct']))
            self.dc_echo_obs_err = get_errs(['C_1_0_CDD1CDD1', 'C_1_0_CDD1CPMG'])
            S_1212_dc, S_1212_dc_err = call_recon_dc(
                recon_S_1212_dc_echo, ['C_1_0_CDD1CDD1', 'C_1_0_CDD1CPMG'], 'S1212')
        else:
            # Legacy FF combination for runs predating the double-echo observables.
            self.dc_echo_ct = None
            self.dc_echo_obs_err = None
            S_1212_dc, S_1212_dc_err = call_recon_dc(recon_S_1212_dc,
                                    ['C_1_0_FIDFID', 'C_2_0_FIDFID', 'C_12_0_FID_FID'], 'S1212')
        S_1_2_dc, S_1_2_dc_err = call_recon_dc(recon_S_1_2_dc, ['C_12_12_FID'], 'S12')
        S_1_12_dc, S_1_12_dc_err = call_recon_dc(recon_S_1_12_dc, ['C_1_12_FID'], 'S112')
        S_2_12_dc, S_2_12_dc_err = call_recon_dc(recon_S_2_12_dc, ['C_2_12_FID'], 'S212')

        # SELF spectra: a flagged (not-determined) DC can fit to an unphysical negative
        # S(0) when the signal is swamped; clamp to a non-negative floor (the first
        # harmonic value) so downstream consumers see a sane spectrum. The dc_reliable
        # flag + inflated bar carry the (large) uncertainty.
        def _floor_dc(val, reliable, harm0):
            val, harm0 = float(np.real(val)), float(np.real(harm0))
            return val if (reliable or val >= harm0) else harm0
        # CROSS spectra: S(0) may be legitimately negative / insignificant -- keep the
        # fitted value (flag + bar carry the uncertainty; flooring would fabricate the
        # plotted point and mask e.g. the raw-arm SPAM bias). Substitute the first
        # harmonic only when there is no fit at all (the robust protocol omits the DC
        # observables) so downstream interpolation stays finite.
        def _fallback_dc(val, harm0):
            val = float(np.real(val))
            return val if np.isfinite(val) else float(np.real(harm0))
        S_11_dc = _floor_dc(S_11_dc, self.dc_reliable['S11'], S_11_k[0])
        S_22_dc = _floor_dc(S_22_dc, self.dc_reliable['S22'], S_22_k[0])
        S_1212_dc = _floor_dc(S_1212_dc, self.dc_reliable['S1212'], S_12_12_k[0])
        S_1_2_dc = _fallback_dc(S_1_2_dc, S_1_2_k[0])
        S_1_12_dc = _fallback_dc(S_1_12_dc, S_1_12_k[0])
        S_2_12_dc = _fallback_dc(S_2_12_dc, S_2_12_k[0])

        print(f"DC values (reliable?): S_11(0)={S_11_dc:.3e}({self.dc_reliable['S11']}), "
              f"S_22(0)={S_22_dc:.3e}({self.dc_reliable['S22']}), "
              f"S_1212(0)={S_1212_dc:.3e}({self.dc_reliable['S1212']})")

        # Prepend the DC (w=0) point so wk[0] = 0. Each recon_*_dc already returns
        # the full one-sided S(0) on the same footing as the harmonic samples
        # (self-spectra: 2<C>/MT; cross-spectra: <C>/MT), so no extra factor is
        # applied here -- the former blanket x2 double-counted the self-spectra DC.
        self.wk = np.concatenate(([0.0], wk_harmonics))

        self.reconstructed_spectra = {
            "S_11_k": np.concatenate(([S_11_dc], S_11_k)),
            "S_22_k": np.concatenate(([S_22_dc], S_22_k)),
            "S_12_12_k": np.concatenate(([S_1212_dc], S_12_12_k)),
            "S_1_2_k": np.concatenate(([S_1_2_dc + 0j], S_1_2_k)),
            "S_1_12_k": np.concatenate(([S_1_12_dc + 0j], S_1_12_k)),
            "S_2_12_k": np.concatenate(([S_2_12_dc + 0j], S_2_12_k)),
        }

        self.reconstructed_spectra_err = {
            "S_11_err": np.concatenate(([S_11_dc_err], S_11_err)),
            "S_22_err": np.concatenate(([S_22_dc_err], S_22_err)),
            "S_12_12_err": np.concatenate(([S_1212_dc_err], S_12_12_err)),
            "S_1_2_err": np.concatenate(([S_1_2_dc_err + 0j], S_1_2_err)),
            "S_1_12_err": np.concatenate(([S_1_12_dc_err + 0j], S_1_12_err)),
            "S_2_12_err": np.concatenate(([S_2_12_dc_err + 0j], S_2_12_err)),
        }

    def unfold_comb_bias(self):
        """Self-consistent comb-bias correction ("unfolding" the markers).

        The comb-delta inversion carries a deterministic bias (truncation +
        finite-M tooth width + neighbor leakage) -- i.e. even with zero shot
        noise, this inversion algorithm alone would not return exactly the
        true spectrum. ``forward_model_systematic`` computes that bias exactly
        for ANY input spectra; feeding it the RECONSTRUCTED spectra
        (``selfconsistent_spectra`` -- de-lined piecewise-linear background +
        Gaussian lines at the experimentally-known nuclear-difference centers
        + fitted power-law tails; no ground-truth knowledge) predicts the bias
        of THIS OWN reconstruction, which is then subtracted from the plotted
        markers (``self.reconstructed_spectra``). One refinement iteration
        follows (bias computed again from the once-corrected spectrum) so the
        subtraction is self-consistent rather than a single noisy estimate.

        NOTE what is actually quoted as the residual systematic error bar is
        NOT computed here: this method only performs and stores the
        correction (``self._applied_bias``, ``self._applied_bias_sigma``,
        both DC+harmonics). The honest post-correction error bar --
        "exact forward-model bias of the truth minus what this method
        actually subtracted", in quadrature with the correction's own
        statistical scatter -- is computed afterward in
        ``add_systematic_errors`` (see that method's docstring; this replaced
        an earlier, less accurate choice that quoted only the fixed-point
        iteration increment |b2-b1|, which under-covered the bias left over
        in channels the correction cannot fully reach -- V10-QNS-BARFIX-0624).
        """
        from qns2q.noise.spectra import line_priors_per_channel
        c = self.config
        # Per-channel prior form: identical to the legacy (centers, sigma) on
        # the anchored classes; under the showcase regime it adds the coupler
        # line on S1212 (which the piecewise-linear background would otherwise
        # miss). This repeats a lesson learned earlier (tagged UNFOLD-RESIDUAL
        # in characterize/systematics.py, where it was first diagnosed): if a
        # narrow spectral line is fit only implicitly by a smooth background
        # model, the self-consistent bias correction under-corrects right at
        # the line and leaves a spuriously large "residual" there -- so any
        # new sharp feature (like this coupler line) needs its own explicit
        # entry in ``lines`` rather than being left to the smooth background.
        lines = line_priors_per_channel()
        inv_opts = dict(inversion_method=c.inversion_method, reg_lambda=c.reg_lambda,
                        enforce_nonneg=c.enforce_nonneg)
        n_h = len(c.c_times)
        wk_full = np.concatenate(([0.0],
                                  [2*np.pi*(j + 1)/c.T for j in range(n_h)]))
        raw0, nan_mask = {}, {}
        for sk, rk in _SYS_TO_SPEC.items():
            arr = np.asarray(self.reconstructed_spectra[rk])
            nan_mask[sk] = ~np.isfinite(arr)
            # Robust-protocol-inaccessible spectra are NaN; the robust protocol
            # ASSUMES S_l,12 = 0, so its self-consistent forward model uses 0.
            raw0[sk] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        def bias_of(recon):
            sc = selfconsistent_spectra(wk_full, recon, lines=lines)
            b = forward_model_systematic(sc, c.c_times, c.M, c.T, c.t_vec,
                                         c.w_grain, c.wmax, inv_opts=inv_opts)
            dcb = dc_fit_systematic(sc, self.dc_t_sweep,
                                    s1212_echo_ct=getattr(self, 'dc_echo_ct', None),
                                    s1212_echo_obs_err=getattr(self, 'dc_echo_obs_err', None),
                                    s1212_echo_wmax=2 * (self.config.synth_wmax or self.config.wmax),
                                    fid_obs_err=getattr(self, 'dc_fid_obs_err', None),
                                    world_grid=self._world_grid())
            out = {}
            for sk in _SYS_TO_SPEC:
                full = np.concatenate(([dcb[sk]], np.asarray(b[sk])))
                if not getattr(self, 'dc_reliable', {}).get(sk, True):
                    full[0] = 0.0          # flagged DC points are not corrected
                out[sk] = full
            return out

        # Two-step fixed-point iteration, not a single correction: b1 is the
        # bias predicted from the AS-MEASURED (still-biased) spectrum; recon1
        # is that spectrum with b1 subtracted (a better, but not perfect,
        # estimate of the truth); b2 is the bias predicted from recon1 -- a
        # more accurate bias estimate because it is evaluated closer to the
        # true spectrum. b2 (not b1) is what actually gets subtracted below.
        b1 = bias_of(raw0)
        recon1 = {sk: raw0[sk] - b1[sk] for sk in raw0}
        b2 = bias_of(recon1)

        # Statistical uncertainty OF the correction: the bias functional is
        # evaluated on the noisy reconstructed points, so its error inherits the
        # reconstruction's statistical scatter. Propagate by recomputing the
        # bias with every input point shifted by +1 sigma_stat.
        recon1_pert = {}
        for sk, rk in _SYS_TO_SPEC.items():
            err = np.nan_to_num(
                np.asarray(self.reconstructed_spectra_err[_SYS_TO_ERR[sk]]), nan=0.0)
            recon1_pert[sk] = recon1[sk] + err
        b2_pert = bias_of(recon1_pert)
        sigma_b = {sk: b2_pert[sk] - b2[sk] for sk in b2}

        # Store the applied correction and its statistical scatter so the honest
        # residual bar (forward-model comb systematic minus what was actually
        # subtracted) can be formed in add_systematic_errors. Each array carries
        # the DC point at [0] and the harmonics at [1:].
        self._applied_bias = dict(b2)
        self._applied_bias_sigma = dict(sigma_b)

        self._unfold_residual = {}
        print("[unfold] self-consistent comb-bias correction "
              "(applied bias RMS -> residual RMS, harmonics):")
        for sk, rk in _SYS_TO_SPEC.items():
            corrected = raw0[sk] - b2[sk]
            corrected[nan_mask[sk]] = np.nan
            self.reconstructed_spectra[rk] = corrected
            # A cheap, ground-truth-free estimate of what bias is LEFT after
            # subtracting b2: |b2-b1| (how much the correction changed on its
            # last iteration -- large means it hasn't converged, so trust it
            # less) plus a fixed conservative fraction of b2 itself (standard
            # unfolding practice: never claim to have removed 100% of a bias
            # estimated from noisy data) plus the correction's own
            # statistical scatter, in quadrature. This is what
            # add_systematic_errors falls back to for the DC point (which has
            # no exact ground-truth comparison available); for the harmonics,
            # in the simulation, add_systematic_errors instead uses the exact
            # forward-model bias of the known true spectrum minus what was
            # actually subtracted, which is the more accurate of the two.
            db = b2[sk] - b1[sk]
            sb = sigma_b[sk]
            frac = c.unfold_residual_frac
            if np.iscomplexobj(db) or np.iscomplexobj(b2[sk]):
                resid = (np.sqrt(np.real(db)**2 + (frac*np.real(b2[sk]))**2
                                 + np.real(sb)**2)
                         + 1j*np.sqrt(np.imag(db)**2 + (frac*np.imag(b2[sk]))**2
                                      + np.imag(sb)**2))
            else:
                resid = np.sqrt(db**2 + (frac*b2[sk])**2 + sb**2)
            self._unfold_residual[sk] = resid
            rms_b = np.sqrt(np.nanmean(np.abs(b2[sk][1:])**2))
            rms_r = np.sqrt(np.nanmean(np.abs(resid[1:])**2))
            print(f"    {sk:>6}: {rms_b:10.3e} -> {rms_r:10.3e}")
        self._unfolded = True

    def add_systematic_errors(self):
        """Fold the deterministic forward-model (comb-inversion) systematic into the bars.

        The simulator's propagator is exact pure dephasing, so the only non-statistical
        reconstruction error is the harmonic-comb inversion: the single-period comb
        kernel approximates the full finite-M filter response integrated over the noise
        synthesis grid. ``forward_model_systematic`` quantifies it with zero Monte Carlo
        -- it reconstructs the EXACT forward observables of the analytic ground-truth
        spectra (validated to reproduce the simulated observables within shot noise) and
        takes the residual vs truth. This supersedes the single-period
        ``comb_inversion_systematic``, which under-quoted the antisymmetric (CDD3)
        cross-channel bias ~2x and so left the Im S_1_2 points outside the bars. The
        w=0 DC point keeps its own Ramsey-slope bias from ``dc_systematic``. Sets
        ``reconstructed_spectra_sys`` (signed per-point bias) and
        ``reconstructed_spectra_err_total`` (sqrt(stat^2 + sys^2)).
        """
        c = self.config
        self.reconstructed_spectra_sys = {}
        self.reconstructed_spectra_err_total = dict(self.reconstructed_spectra_err)
        try:
            if getattr(self, '_unfolded', False):
                # Unfolded reconstruction: the self-consistent correction has been
                # subtracted from the markers. Quote the HONEST residual systematic
                # = the exact forward-model comb bias of the model MINUS what the
                # unfold actually subtracted (in quadrature with the correction's own
                # statistical scatter). Where the unfold corrects well (self-spectra,
                # the in-band S_1_2 line) the residual is small -> tight bars on
                # bias-corrected markers; where it cannot -- the out-of-band-aliasing-
                # dominated S_1_12/S_2_12 high harmonics, which no blind extrapolation
                # reaches -- the full bias survives and is quoted, so every tooth stays
                # within its bar. (Simulation: uses the known model for the comb bias,
                # exactly as the unfold-off path below does.)
                inv_opts = dict(inversion_method=c.inversion_method, reg_lambda=c.reg_lambda,
                                enforce_nonneg=c.enforce_nonneg)
                b_true = forward_model_systematic(analytic_spectra(), c.c_times, c.M,
                                                  c.T, c.t_vec, c.w_grain, c.wmax,
                                                  inv_opts=inv_opts)
                sys = {}
                for sk in _SYS_TO_SPEC:
                    diff = np.asarray(b_true[sk]) - np.asarray(self._applied_bias[sk][1:])
                    sb = np.asarray(self._applied_bias_sigma[sk][1:])
                    if np.iscomplexobj(diff) or np.iscomplexobj(sb):
                        sys[sk] = (np.sqrt(np.real(diff)**2 + np.real(sb)**2)
                                   + 1j*np.sqrt(np.imag(diff)**2 + np.imag(sb)**2))
                    else:
                        sys[sk] = np.sqrt(diff**2 + sb**2)
                # DC point: there is no analogous "exact true-model bias" to
                # compare against here (dc_fit_systematic uses the Ramsey-slope
                # sweep, a different estimator from the harmonic comb), so the
                # DC bar reuses the self-consistent residual estimate already
                # computed in unfold_comb_bias's [0]-th (DC) entry rather than
                # the truth-vs-applied comparison used for the harmonics above.
                dc_bias = {sk: float(np.real(self._unfold_residual[sk][0]))
                           for sk in _SYS_TO_SPEC}
            else:
                spectra = analytic_spectra()
                inv_opts = dict(inversion_method=c.inversion_method, reg_lambda=c.reg_lambda,
                                enforce_nonneg=c.enforce_nonneg)
                sys = forward_model_systematic(spectra, c.c_times, c.M, c.T, c.t_vec,
                                               c.w_grain, c.wmax, inv_opts=inv_opts)

                # DC (w=0) point: deterministic bias of the multi-time slope fit (tiny
                # where the noise reaches motional narrowing; large = honest inflated bar
                # where it is quasi-static / sub-comb-cusp -- see dc_reliable).
                dc_bias = dc_fit_systematic(spectra, self.dc_t_sweep,
                                            s1212_echo_ct=getattr(self, 'dc_echo_ct', None),
                                            s1212_echo_obs_err=getattr(self, 'dc_echo_obs_err', None),
                                            s1212_echo_wmax=2 * (self.config.synth_wmax or self.config.wmax),
                                            fid_obs_err=getattr(self, 'dc_fid_obs_err', None),
                                            world_grid=self._world_grid())

            print("[systematic] folded sigma_sys per spectrum (forward-model comb bias; "
                  "RMS over harmonics, |DC bias|):")
            for sk, rk in _SYS_TO_SPEC.items():
                ek = _SYS_TO_ERR[sk]
                sysk = np.concatenate(([0.0], sys[sk]))   # complex for cross, real for self
                sysk[0] = abs(dc_bias[sk])                # DC bias is real for every spectrum
                if not getattr(self, 'dc_reliable', {}).get(sk, True):
                    # DC not determined from the data (strong-noise cross channel whose
                    # self-decay swamps the signal, or quasi-static / sub-comb-cusp noise):
                    # quote an honest bar at the spectrum's own scale and flag it.
                    # (NaN-guarded: robust-protocol-inaccessible spectra stay NaN.)
                    harm_vals = np.abs(self.reconstructed_spectra[rk][1:])
                    harm_vals = harm_vals[np.isfinite(harm_vals)]
                    harm_scale = float(np.max(harm_vals)) if harm_vals.size else np.nan
                    sysk[0] = np.fmax(abs(dc_bias[sk]), harm_scale)
                self.reconstructed_spectra_sys[ek] = sysk
                self.reconstructed_spectra_err_total[ek] = _quad_combine(
                    self.reconstructed_spectra_err[ek], sysk)
                rms_a = np.sqrt(np.mean(np.abs(sys[sk]) ** 2))
                flag = '' if getattr(self, 'dc_reliable', {}).get(sk, True) else '  [DC flagged: not determined]'
                print(f"    {sk:>6}: harmonic RMS = {rms_a:10.3e}  |DC bias| = {abs(dc_bias[sk]):10.3e}{flag}")
        except Exception as e:
            print(f"[systematic] WARNING: systematic-error computation failed ({e}); "
                  f"falling back to statistical-only bars.")
            for ek in _SYS_TO_ERR.values():
                self.reconstructed_spectra_sys[ek] = np.zeros_like(self.reconstructed_spectra_err[ek])

    def _get_output_dir(self, subdir):
        """Returns (and creates) a subfolder inside the data folder for figures."""
        path = os.path.join(project_root(), self.config.data_folder, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def plot_all_spectra(self):
        """Plots all 6 spectra in a single 3x2 publication-quality figure.

        Left column: real-valued self-spectra (S_11, S_22, S_1212).
        Right column: complex-valued cross-spectra (S_1_2, S_1_12, S_2_12).
        """
        setup_pub_rcparams('compact')
        # Quote the honest combined bars (statistical (+) systematic) when available.
        err_dict = getattr(self, 'reconstructed_spectra_err_total',
                           getattr(self, 'reconstructed_spectra_err', {}))
        # Shared per-panel axes across the SPAM-protocol sibling arms ({} for
        # NoSPAM runs, which keep the legacy per-run autoscaling).
        shared_env = _sibling_spam_envelopes(self.config.data_folder)

        w = np.linspace(0, self.config.wmax, self.config.w_grain)
        # tau units: w is already the dimensionless angular frequency w*tau and
        # the spectra are dimensionless S*tau, so the axes plot raw values.
        xunits = 1.0

        # --- Style definitions ---
        eb_self = dict(fmt='^', color=COLORS["vermillion"],
                       markersize=3.5, linewidth=0.8, zorder=10, label='Reconstructed')
        theory_self_kw = dict(color=COLORS["vermillion"], linestyle='--', linewidth=1.2,
                              alpha=0.7, zorder=5, label='Theory')

        eb_re = dict(fmt='^', color=COLORS["vermillion"],
                     markersize=3.5, linewidth=0.8, zorder=10, label=r'Re (recon.)')
        eb_im = dict(fmt='s', color=COLORS["blue"],
                     markersize=3.0, linewidth=0.8, zorder=10, label=r'Im (recon.)')
        theory_re_kw = dict(color=COLORS["vermillion"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory (Re)')
        theory_im_kw = dict(color=COLORS["blue"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory (Im)')

        # --- Data definitions ---
        real_spectra = [
            ('S_11_k', 'S_11_err', S_11, r'$S_{1,1}(\omega)\,\tau$'),
            ('S_22_k', 'S_22_err', S_22, r'$S_{2,2}(\omega)\,\tau$'),
            ('S_12_12_k', 'S_12_12_err', S_1212, r'$S_{12,12}(\omega)\,\tau$'),
        ]

        complex_spectra = [
            ('S_1_2_k', 'S_1_2_err',
             lambda w_: S_1_2(w_),
             r'$S_{1,2}(\omega)\,\tau$'),
            ('S_1_12_k', 'S_1_12_err',
             lambda w_: S_1_12(w_),
             r'$S_{1,12}(\omega)\,\tau$'),
            ('S_2_12_k', 'S_2_12_err',
             lambda w_: S_2_12(w_),
             r'$S_{2,12}(\omega)\,\tau$'),
        ]

        panel_labels = [['(a)', '(d)'], ['(b)', '(e)'], ['(c)', '(f)']]

        fig, axs = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT_3x2))

        # --- Left column: self-spectra ---
        for row, (s_key, err_key, theory_fn, ylabel) in enumerate(real_spectra):
            ax = axs[row, 0]
            ax.fill_between(w / xunits, 0, theory_fn(w),
                            color=COLORS["grey_fill"], alpha=1.0, zorder=0)
            ax.plot(w / xunits, theory_fn(w), **theory_self_kw)

            yerr = None
            if err_key in err_dict:
                yerr = ERRORBAR_SIGMA * err_dict[err_key]

            own = self.reconstructed_spectra[s_key]
            # dc_ok/i0 pattern (repeated for every panel below): dc_reliable
            # flags whether the w=0 point was an actual fit or just a
            # placeholder/bound (see reconstruct() above). When it's not
            # reliable, i0=1 skips index 0 (the DC point) for the SOLID
            # markers/axis-limit data, and a second, hollow-marker errorbar
            # call just below draws that one flagged point separately so its
            # inflated bar is visibly distinct and does not stretch the axes.
            dc_ok = getattr(self, 'dc_reliable', {}).get(_SPEC_TO_SYS[s_key], True)
            i0 = 0 if dc_ok else 1
            ax.errorbar(self.wk[i0:] / xunits, own[i0:],
                        yerr=None if yerr is None else yerr[i0:], **eb_self)
            if not dc_ok:
                # Hollow marker: the S(0) slope fit is flagged not-determined; the
                # inflated bar (excluded from the axis limits) carries the uncertainty.
                ax.errorbar(self.wk[:1] / xunits, own[:1],
                            yerr=None if yerr is None else np.abs(yerr[:1]),
                            **dict(eb_self, mfc='white', label='_nolegend_'))
            ax.set_ylabel(ylabel)

            if _SPEC_TO_SYS[s_key] in shared_env:
                bars = np.abs(yerr) if yerr is not None else 0.0
                env = np.concatenate([theory_fn(w), [0.0], (own - bars)[i0:],
                                      (own + bars)[i0:],
                                      shared_env[_SPEC_TO_SYS[s_key]]])
                _set_asinh_scale(ax, theory_fn(w), ylim_data=env)
            else:
                _set_asinh_scale(ax, np.concatenate([theory_fn(w), own[i0:]]))
            ax.grid(True, which='major', zorder=0)
            ax.grid(False, which='minor')
            ax.text(0.03, 0.92, panel_labels[row][0], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

        # --- Right column: cross-spectra ---
        for row, (s_key, err_key, theory_fn, ylabel) in enumerate(complex_spectra):
            ax = axs[row, 1]
            S_theory = theory_fn(w)
            own = self.reconstructed_spectra[s_key]
            # All-NaN marks a spectrum the protocol cannot access; suppress the
            # point series entirely (np.imag of a real NaN array is 0, which
            # would otherwise draw a spurious marker row at y=0).
            accessible = bool(np.any(np.isfinite(own)))

            yerr_re = None
            yerr_im = None
            if err_key in err_dict:
                err_complex = err_dict[err_key]
                yerr_re = ERRORBAR_SIGMA * np.real(err_complex)
                yerr_im = ERRORBAR_SIGMA * np.imag(err_complex)

            ax.plot(w / xunits, np.real(S_theory), **theory_re_kw)
            ax.plot(w / xunits, np.imag(S_theory), **theory_im_kw)
            dc_ok = getattr(self, 'dc_reliable', {}).get(_SPEC_TO_SYS[s_key], True)
            i0 = 0 if dc_ok else 1
            if accessible:
                ax.errorbar(self.wk[i0:] / xunits, np.real(own)[i0:],
                            yerr=None if yerr_re is None else np.abs(yerr_re)[i0:],
                            **eb_re)
                ax.errorbar(self.wk[i0:] / xunits, np.imag(own)[i0:],
                            yerr=None if yerr_im is None else np.abs(yerr_im)[i0:],
                            **eb_im)
                if not dc_ok:
                    # Hollow markers: the S(0) slope fit is flagged not-determined;
                    # the inflated bar (excluded from the limits) carries it.
                    for vals, ye, eb in ((np.real(own), yerr_re, eb_re),
                                         (np.imag(own), yerr_im, eb_im)):
                        ax.errorbar(self.wk[:1] / xunits, vals[:1],
                                    yerr=None if ye is None else np.abs(ye)[:1],
                                    **dict(eb, mfc='white', label='_nolegend_'))

            ax.set_ylabel(ylabel)

            th_parts = np.concatenate([np.real(S_theory), np.imag(S_theory)])
            if _SPEC_TO_SYS[s_key] in shared_env:
                bre = np.abs(yerr_re) if yerr_re is not None else 0.0
                bim = np.abs(yerr_im) if yerr_im is not None else 0.0
                env = np.concatenate([th_parts,
                                      (np.real(own) - bre)[i0:], (np.real(own) + bre)[i0:],
                                      (np.imag(own) - bim)[i0:], (np.imag(own) + bim)[i0:],
                                      shared_env[_SPEC_TO_SYS[s_key]]])
                _set_asinh_scale(ax, th_parts, ylim_data=env)
            else:
                _set_asinh_scale(ax, np.concatenate([th_parts, np.real(own)[i0:],
                                                     np.imag(own)[i0:]]))
            ax.grid(True, which='major', zorder=0)
            ax.grid(False, which='minor')
            ax.text(0.03, 0.92, panel_labels[row][1], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
            if not np.any(np.isfinite(self.reconstructed_spectra[s_key])):
                ax.text(0.5, 0.5, 'not accessible\n(SPAM-robust protocol)',
                        transform=ax.transAxes, fontsize=8, color='0.35',
                        ha='center', va='center', style='italic')

        # --- Axis labels (bottom row only) ---
        for row in range(3):
            for col in range(2):
                if row == 2:
                    axs[row, col].set_xlabel(r'$\omega\tau$')
                else:
                    axs[row, col].tick_params(labelbottom=False)

        # --- Per-column legends in top panels ---
        axs[0, 0].legend(frameon=False, loc='upper right')
        axs[0, 1].legend(frameon=False, loc='upper right', ncol=2)

        if ERRORBAR_SIGMA != 1:
            fig.text(0.995, 0.005, rf'error bars: ${ERRORBAR_SIGMA}\sigma$',
                     ha='right', va='bottom', fontsize=7, color='0.4')

        plt.tight_layout(pad=0.3)
        output_dir = self._get_output_dir(RECONSTRUCTION_SUBDIR)
        output_path = os.path.join(output_dir, "spectral_reconstruction_all_pub.pdf")
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved combined spectra plot to {output_path}")
        plt.close(fig)

    def plot_cross_spectra(self):
        """Plots the three complex cross-spectra as a standalone single-column
        3x1 figure for the appendix (S_1_2, S_1_12, S_2_12).

        Companion to plot_all_spectra(): identical data and styling, laid out as
        a narrow single-column figure with a single compact in-panel legend. The
        asinh y-scale with a SymmetricalLogLocator keeps the y-ticks legible at
        column width, and the legend sits inside the top panel rather than
        underneath the figure.
        """
        setup_pub_rcparams('compact')
        # Quote the honest combined bars (statistical (+) systematic) when available.
        err_dict = getattr(self, 'reconstructed_spectra_err_total',
                           getattr(self, 'reconstructed_spectra_err', {}))
        # Shared per-panel axes across the SPAM-protocol sibling arms ({} for
        # NoSPAM runs, which keep the legacy per-run autoscaling).
        shared_env = _sibling_spam_envelopes(self.config.data_folder)

        w = np.linspace(0, self.config.wmax, self.config.w_grain)
        # tau units: w is already the dimensionless angular frequency w*tau and
        # the spectra are dimensionless S*tau, so the axes plot raw values.
        xunits = 1.0

        eb_re = dict(fmt='^', color=COLORS["vermillion"],
                     markersize=3.5, linewidth=0.8, zorder=10, label=r'Re (recon.)')
        eb_im = dict(fmt='s', color=COLORS["blue"],
                     markersize=3.0, linewidth=0.8, zorder=10, label=r'Im (recon.)')
        theory_re_kw = dict(color=COLORS["vermillion"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory (Re)')
        theory_im_kw = dict(color=COLORS["blue"], linestyle='--', linewidth=1.2,
                            alpha=0.7, zorder=5, label='Theory (Im)')

        complex_spectra = [
            ('S_1_2_k', 'S_1_2_err',
             lambda w_: S_1_2(w_),
             r'$S_{1,2}(\omega)\,\tau$'),
            ('S_1_12_k', 'S_1_12_err',
             lambda w_: S_1_12(w_),
             r'$S_{1,12}(\omega)\,\tau$'),
            ('S_2_12_k', 'S_2_12_err',
             lambda w_: S_2_12(w_),
             r'$S_{2,12}(\omega)\,\tau$'),
        ]

        panel_labels = ['(a)', '(b)', '(c)']

        fig, axs = plt.subplots(3, 1, figsize=(3.4, 5.4))

        for row, (s_key, err_key, theory_fn, ylabel) in enumerate(complex_spectra):
            ax = axs[row]
            S_theory = theory_fn(w)
            own = self.reconstructed_spectra[s_key]
            # All-NaN marks a spectrum the protocol cannot access; suppress the
            # point series entirely (np.imag of a real NaN array is 0, which
            # would otherwise draw a spurious marker row at y=0).
            accessible = bool(np.any(np.isfinite(own)))

            yerr_re = None
            yerr_im = None
            if err_key in err_dict:
                err_complex = err_dict[err_key]
                yerr_re = ERRORBAR_SIGMA * np.real(err_complex)
                yerr_im = ERRORBAR_SIGMA * np.imag(err_complex)

            ax.plot(w / xunits, np.real(S_theory), **theory_re_kw)
            ax.plot(w / xunits, np.imag(S_theory), **theory_im_kw)
            dc_ok = getattr(self, 'dc_reliable', {}).get(_SPEC_TO_SYS[s_key], True)
            i0 = 0 if dc_ok else 1
            if accessible:
                ax.errorbar(self.wk[i0:] / xunits, np.real(own)[i0:],
                            yerr=None if yerr_re is None else np.abs(yerr_re)[i0:],
                            **eb_re)
                ax.errorbar(self.wk[i0:] / xunits, np.imag(own)[i0:],
                            yerr=None if yerr_im is None else np.abs(yerr_im)[i0:],
                            **eb_im)
                if not dc_ok:
                    # Hollow markers: the S(0) slope fit is flagged not-determined;
                    # the inflated bar (excluded from the limits) carries it.
                    for vals, ye, eb in ((np.real(own), yerr_re, eb_re),
                                         (np.imag(own), yerr_im, eb_im)):
                        ax.errorbar(self.wk[:1] / xunits, vals[:1],
                                    yerr=None if ye is None else np.abs(ye)[:1],
                                    **dict(eb, mfc='white', label='_nolegend_'))

            ax.set_ylabel(ylabel)

            th_parts = np.concatenate([np.real(S_theory), np.imag(S_theory)])
            if _SPEC_TO_SYS[s_key] in shared_env:
                bre = np.abs(yerr_re) if yerr_re is not None else 0.0
                bim = np.abs(yerr_im) if yerr_im is not None else 0.0
                env = np.concatenate([th_parts,
                                      (np.real(own) - bre)[i0:], (np.real(own) + bre)[i0:],
                                      (np.imag(own) - bim)[i0:], (np.imag(own) + bim)[i0:],
                                      shared_env[_SPEC_TO_SYS[s_key]]])
                _set_asinh_scale(ax, th_parts, ylim_data=env)
            else:
                _set_asinh_scale(ax, np.concatenate([th_parts, np.real(own)[i0:],
                                                     np.imag(own)[i0:]]))
            ax.grid(True, which='major', zorder=0)
            ax.grid(False, which='minor')
            ax.text(0.03, 0.92, panel_labels[row], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
            if not np.any(np.isfinite(self.reconstructed_spectra[s_key])):
                ax.text(0.5, 0.5, 'not accessible\n(SPAM-robust protocol)',
                        transform=ax.transAxes, fontsize=8, color='0.35',
                        ha='center', va='center', style='italic')

            if row == 2:
                ax.set_xlabel(r'$\omega\tau$')
            else:
                ax.tick_params(labelbottom=False)

        # Single compact legend inside the top panel (not underneath the figure)
        axs[0].legend(frameon=False, loc='upper right', ncol=2, fontsize=7,
                      handlelength=1.5, columnspacing=1.0, handletextpad=0.4)

        if ERRORBAR_SIGMA != 1:
            fig.text(0.995, 0.005, rf'error bars: ${ERRORBAR_SIGMA}\sigma$',
                     ha='right', va='bottom', fontsize=7, color='0.4')

        plt.tight_layout(pad=0.3)
        output_dir = self._get_output_dir(RECONSTRUCTION_SUBDIR)
        output_path = os.path.join(output_dir, "spectral_reconstruction_cross_pub.pdf")
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved cross-spectra plot to {output_path}")
        plt.close(fig)

    def save_reconstructed_spectra(self):
        """Saves the reconstructed spectra (including DC at w=0) to specs.npz --
        the hand-off file to Stage 3 (``control/cz.py``, ``control/idle.py``)
        and to the figure scripts.

        On-disk schema (all arrays indexed like ``self.wk``, DC at index 0):
        ``wk`` (the frequency comb), ``S11``/``S22``/``S1212``/``S12``/``S112``/
        ``S212`` (the six spectra; complex for the three cross-spectra), each
        with matching ``*_err`` (statistical only), ``*_sys`` (systematic bias
        estimate), ``*_errtot`` (statistical+systematic in quadrature -- what
        the figures actually plot) and ``*_dc_ok`` (bool: was the w=0 point an
        actual fit or just a placeholder), plus the scalar ``spam_protocol``
        and ``model_version`` metadata below.
        """
        # Save specs.npz at the data folder root (consumed by downstream scripts)
        path = os.path.join(project_root(), self.config.data_folder, "specs.npz")
        save_dict = dict(
            wk=self.wk,
            spam_protocol=getattr(self.config, 'spam_protocol', 'none'),
            # OPT-PROVENANCE: stamp which noise-model version (see
            # noise/spectra.py's MODEL_VERSION) generated the data this
            # reconstruction came from, so a downstream gate-optimization run
            # that loads specs.npz from a folder built under an OLD/DIFFERENT
            # noise model can detect the mismatch and warn instead of silently
            # optimizing against stale physics ('unknown' for run folders
            # predating this stamp).
            model_version=(str(self.config.params['model_version'])
                           if 'model_version' in self.config.params
                           else 'unknown'),
            S11=self.reconstructed_spectra['S_11_k'], S22=self.reconstructed_spectra['S_22_k'],
            S12=self.reconstructed_spectra['S_1_2_k'], S1212=self.reconstructed_spectra['S_12_12_k'],
            S112=self.reconstructed_spectra['S_1_12_k'], S212=self.reconstructed_spectra['S_2_12_k'],
        )
        # Per-spectrum DC determination flags (False = the w=0 slope fit is flagged
        # not-determined; consumers should treat wk[0] as a bound, not a measurement).
        save_dict.update({f'{sk}_dc_ok': bool(getattr(self, 'dc_reliable', {}).get(sk, True))
                          for sk in _SYS_TO_SPEC})
        # S*_err = STATISTICAL error (unchanged semantics, backward-compatible).
        if hasattr(self, 'reconstructed_spectra_err'):
            save_dict.update(dict(
                S11_err=self.reconstructed_spectra_err['S_11_err'],
                S22_err=self.reconstructed_spectra_err['S_22_err'],
                S1212_err=self.reconstructed_spectra_err['S_12_12_err'],
                S12_err=self.reconstructed_spectra_err['S_1_2_err'],
                S112_err=self.reconstructed_spectra_err['S_1_12_err'],
                S212_err=self.reconstructed_spectra_err['S_2_12_err'],
            ))
        # S*_sys = comb-inversion systematic bias; S*_errtot = sqrt(stat^2 + sys^2),
        # the honest combined bar plotted in the figures.
        if getattr(self, 'reconstructed_spectra_sys', None):
            save_dict.update(dict(
                S11_sys=self.reconstructed_spectra_sys['S_11_err'],
                S22_sys=self.reconstructed_spectra_sys['S_22_err'],
                S1212_sys=self.reconstructed_spectra_sys['S_12_12_err'],
                S12_sys=self.reconstructed_spectra_sys['S_1_2_err'],
                S112_sys=self.reconstructed_spectra_sys['S_1_12_err'],
                S212_sys=self.reconstructed_spectra_sys['S_2_12_err'],
            ))
        if getattr(self, 'reconstructed_spectra_err_total', None):
            save_dict.update(dict(
                S11_errtot=self.reconstructed_spectra_err_total['S_11_err'],
                S22_errtot=self.reconstructed_spectra_err_total['S_22_err'],
                S1212_errtot=self.reconstructed_spectra_err_total['S_12_12_err'],
                S12_errtot=self.reconstructed_spectra_err_total['S_1_2_err'],
                S112_errtot=self.reconstructed_spectra_err_total['S_1_12_err'],
                S212_errtot=self.reconstructed_spectra_err_total['S_2_12_err'],
            ))
        np.savez(path, **save_dict)

    def run(self):
        """Runs the full reconstruction pipeline end to end, in the order the
        later stages depend on: load the Stage-1 data, invert it to spectra,
        (optionally) self-consistently remove the comb-inversion bias, add
        error bars, draw both figures, and write specs.npz. This is the one
        method external callers (``main`` below) actually need to call.
        """
        self.load_observables()
        self.reconstruct()
        if self.config.compute_systematic and self.config.unfold_bias:
            self.unfold_comb_bias()
        if self.config.compute_systematic:
            self.add_systematic_errors()
        else:
            # No systematic requested: quoted bars are statistical only.
            self.reconstructed_spectra_sys = {ek: np.zeros_like(v)
                                              for ek, v in self.reconstructed_spectra_err.items()}
            self.reconstructed_spectra_err_total = dict(self.reconstructed_spectra_err)
        self.plot_all_spectra()
        self.plot_cross_spectra()
        self.save_reconstructed_spectra()


def main(data_folder=None):
    """Main function to run the spectra reconstruction. This is this module's
    public entry point -- it is what ``scripts/run_capture_arm.py``,
    ``scripts/run_spam_reconstruct.py``, and this file's own ``__main__``
    block below all actually call; do not change its name or the
    ``data_folder`` parameter (other files import and call it by name).

    Parameters
    ----------
    data_folder : str, optional
        Run folder to reconstruct (relative to the project root). Defaults to the
        active regime's canonical NoSPAM folder; the SPAM pipeline scripts pass
        their protocol-suffixed folders here.

    Note: errors are caught and printed here rather than raised, so a failed
    reconstruction (e.g. a missing run folder) prints a message and returns
    normally instead of crashing a caller that runs this as one step of a
    longer pipeline script.
    """
    # --- User Configuration ---
    data_folder = run_folder() if data_folder is None else data_folder
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
    # Command-line entry point: `python -m qns2q.characterize.reconstruct
    # --folder <run_folder>` (also reachable via scripts/run_reconstruct.py,
    # which just re-execs this __main__ block). --folder is the only CLI flag
    # and is part of the documented interface (see CLAUDE.md) -- keep its name.
    import argparse
    ap = argparse.ArgumentParser(
        description="Reconstruct two-qubit noise spectra (incl. the cross-spectra "
                    "publication figure) from a run folder.")
    ap.add_argument("--folder", default=None,
                    help="run folder to reconstruct, relative to the repo root "
                         "(e.g. DraftRun_NoSPAM_showcase_cap). Default: the active "
                         "QNS2Q_REGIME's canonical NoSPAM folder.")
    args = ap.parse_args()
    main(data_folder=args.folder)
