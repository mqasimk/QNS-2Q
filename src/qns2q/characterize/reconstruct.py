"""
This script reconstructs and plots spectra from pre-calculated observables.

It defines a configuration class to load parameters, a reconstructor class to manage
the reconstruction process, and a main execution block to run the analysis for a
specified data folder.
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


# Maps systematics-module spectrum keys <-> reconstructed-spectra / error dict keys.
_SYS_TO_SPEC = {'S11': 'S_11_k', 'S22': 'S_22_k', 'S1212': 'S_12_12_k',
                'S12': 'S_1_2_k', 'S112': 'S_1_12_k', 'S212': 'S_2_12_k'}
_SYS_TO_ERR = {'S11': 'S_11_err', 'S22': 'S_22_err', 'S1212': 'S_12_12_err',
               'S12': 'S_1_2_err', 'S112': 'S_1_12_err', 'S212': 'S_2_12_err'}
_SPEC_TO_SYS = {v: k for k, v in _SYS_TO_SPEC.items()}


def _quad_combine(stat, sys):
    """Quadrature-combine statistical and systematic error arrays, per component for
    complex (cross-spectra) arrays: sqrt(Re_stat^2+Re_sys^2) + i sqrt(Im_stat^2+Im_sys^2)."""
    if np.iscomplexobj(stat) or np.iscomplexobj(sys):
        return (np.sqrt(np.real(stat) ** 2 + np.real(sys) ** 2)
                + 1j * np.sqrt(np.imag(stat) ** 2 + np.imag(sys) ** 2))
    return np.sqrt(np.asarray(stat) ** 2 + np.asarray(sys) ** 2)


# --- Publication figure constants ---

FIG_WIDTH = 7.0    # Two-column width (inches)
FIG_HEIGHT = 4.5   # 2-row panel height (legacy, kept for reference)
FIG_HEIGHT_1ROW = 2.5  # Single-row panel height
FIG_HEIGHT_3ROW = 7.0  # Three-row panel height
FIG_HEIGHT_3x2 = 5.5  # Three-row, two-column panel height

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
    # Bias-corrected unfolding: subtract the SELF-CONSISTENT comb-inversion bias
    # (forward model built from the reconstructed spectra alone -- no ground-truth
    # knowledge, so it is experimentally legitimate) and quote the iteration
    # residual as the remaining systematic. Two fixed-point iterations.
    unfold_bias: bool = True
    # Fraction of the APPLIED correction conservatively retained as residual
    # systematic (standard unfolding practice: the self-consistent model is
    # piecewise-linear through noisy points and cannot represent the narrow
    # lines between comb teeth, so the fixed-point increment alone under-quotes;
    # 0.5 was validated against ground truth -- reference-arm pulls ~1).
    unfold_residual_frac: float = 0.5
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
        path = os.path.join(project_root(), self.data_folder)
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
                      f"Stage 1 (scripts/run_experiments.py) for this folder.")

        # Optional SPAM-protocol metadata (absent in legacy / NoSPAM runs).
        self.spam_protocol = (str(self.params['spam_protocol'])
                              if 'spam_protocol' in self.params else 'none')
        self.m_sweep_robust = (np.asarray(self.params['m_sweep_robust'], dtype=int)
                               if 'm_sweep_robust' in self.params
                               else np.array([], dtype=int))


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
        path = os.path.join(project_root(), self.config.data_folder, "results.npz")
        self.observables = np.load(path)

    def reconstruct(self):
        """Reconstructs the spectra from the loaded observables."""
        c = self.config
        obs = self.observables
        
        # Check if error data is available
        has_errors = any(k.endswith('_err') for k in obs.keys())

        # SPAM-robust runs store the M-swept harmonic observables under
        # {key}_Mrep{m}; the SPAM-free coefficient is the slope of the linear
        # M-regression evaluated at the reference repetition c.M (the intercept
        # absorbs the SPAM term). C_12_12 keys are stored plainly (exactly
        # SPAM-free, no regression needed).
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

        # Helper to unpack result
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

        # Reconstruct DC (w=0) values from the multi-time FID decay sweep.
        # Each recon_*_dc fits S(0) from the slope of C(t) over the adaptively-selected
        # measurable+linear window (inversion._ramsey_fit_dc); returns (val, err,
        # reliable). reliable=False flags quasi-static / sub-comb-cusp noise whose DC is
        # only a lower bound -- the figure then quotes an inflated (honest) bar.
        t_sweep = obs['dc_t_sweep']
        self.dc_t_sweep = t_sweep
        self.dc_reliable = {}

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
        """Self-consistent comb-bias correction (unfolding).

        The comb-delta inversion carries a deterministic bias (truncation +
        finite-M tooth width + neighbor leakage). ``forward_model_systematic``
        computes that bias exactly for ANY spectra; feeding it the
        RECONSTRUCTED spectra (``selfconsistent_spectra`` -- piecewise-linear,
        no truth knowledge) predicts the bias of this very reconstruction,
        which is then subtracted. One refinement iteration follows; the quoted
        systematic becomes the iteration increment |b2 - b1| (the fixed-point
        convergence scale) instead of the full bias.
        """
        c = self.config
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
            sc = selfconsistent_spectra(wk_full, recon)
            b = forward_model_systematic(sc, c.c_times, c.M, c.T, c.t_vec,
                                         c.w_grain, c.wmax, inv_opts=inv_opts)
            dcb = dc_fit_systematic(sc, self.dc_t_sweep,
                                    s1212_echo_ct=getattr(self, 'dc_echo_ct', None),
                                    s1212_echo_obs_err=getattr(self, 'dc_echo_obs_err', None),
                                    s1212_echo_wmax=self.config.wmax)
            out = {}
            for sk in _SYS_TO_SPEC:
                full = np.concatenate(([dcb[sk]], np.asarray(b[sk])))
                if not getattr(self, 'dc_reliable', {}).get(sk, True):
                    full[0] = 0.0          # flagged DC points are not corrected
                out[sk] = full
            return out

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

        self._unfold_residual = {}
        print("[unfold] self-consistent comb-bias correction "
              "(applied bias RMS -> residual RMS, harmonics):")
        for sk, rk in _SYS_TO_SPEC.items():
            corrected = raw0[sk] - b2[sk]
            corrected[nan_mask[sk]] = np.nan
            self.reconstructed_spectra[rk] = corrected
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
                # Unfolded reconstruction: the bias has been subtracted; quote the
                # fixed-point iteration increment as the residual systematic.
                sys = {sk: self._unfold_residual[sk][1:] for sk in _SYS_TO_SPEC}
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
                                            s1212_echo_wmax=self.config.wmax)

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
        """Saves the reconstructed spectra (including DC at w=0) to a .npz file."""
        # Save specs.npz at the data folder root (consumed by downstream scripts)
        path = os.path.join(project_root(), self.config.data_folder, "specs.npz")
        save_dict = dict(
            wk=self.wk,
            spam_protocol=getattr(self.config, 'spam_protocol', 'none'),
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
        """Runs the full reconstruction pipeline."""
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
    """Main function to run the spectra reconstruction.

    Parameters
    ----------
    data_folder : str, optional
        Run folder to reconstruct (relative to the project root). Defaults to the
        active regime's canonical NoSPAM folder; the SPAM pipeline scripts pass
        their protocol-suffixed folders here.
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
    main()
