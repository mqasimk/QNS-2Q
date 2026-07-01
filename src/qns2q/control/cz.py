"""
CZ-gate pulse-sequence optimization (v2) -- pipeline Stage 3a.

Physics role
------------
This is the "control" arm's CZ-gate optimizer for the two-qubit QNS-2Q
pipeline (its sibling `control/idle.py` does the same job for the identity/
idling gate -- Stage 3b). Given the two-qubit dephasing environment as a
3x3 spectral matrix -- self-spectra `S_11`, `S_22`, `S_1212` for qubit 1,
qubit 2, and the Ising ("ZZ") coupling channel, plus the three cross spectra
`S_1_2`, `S_1_12`, `S_2_12` -- this file searches for the pulsed control
sequence (a dynamical-decoupling-style train of instantaneous pi-pulses on
each qubit, applied while an always-on Ising coupling J is switched on)
that realizes the entangling exp(-i*pi/4*Z1*Z2) phase at the heart of a CZ
gate while suppressing as much of the dephasing-induced infidelity as
possible. Where Stage 1-2 (`characterize/`) answer "what does the noise
look like", this file (`control/`) answers "what is the best gate we can
build against it" -- the noise-tailored (NT) half of the paper.

Pipeline position
------------------
```
noise/spectra.py (analytic ground-truth model)
        |
characterize/{experiments,reconstruct}.py   (Stage 1-2: simulate QNS
        |                                     experiments -> results.npz;
        |                                     reconstruct -> specs.npz)
        v
control/cz.py  (THIS FILE, Stage 3a)   <-- sibling -->   control/idle.py (3b)
        |
scripts/report_showcase_figs.py, scripts/harvest_design_numbers.py,
scripts/run_margin_band.py, qns2q.viz.cz_pulse_plot
        (downstream consumers of the .npz files this module writes)
```

Inputs
------
A run folder (selected by `qns2q.paths.run_folder()`, which resolves the
active `QNS2Q_REGIME`) holding either:
  - `specs.npz` + `params.npz` -- Stage 2's *reconstructed* spectra, i.e.
    the "characterized" view of the noise that a real blind QNS experiment
    would actually have access to; or
  - `simulated_spectra.npz` -- the analytic ground-truth model directly
    (`--simulated`; used for the acceptance-gate / sanity comparisons).
`CZOptConfig.__post_init__` loads these eagerly (see the note on dataclass
`__post_init__` just above the class) and builds two 4x4 matrices of
spectra on a frequency grid: `SMat` (from the characterized/simulated data
-- what the blind optimizer searches against) and `SMat_ideal` (always
built from the CURRENT analytic noise model -- the "ground truth" used only
to report the true, published infidelity of whatever sequence the blind
search picked).

Outputs
-------
Written by `run_optimization()` into the run folder:
  - `infs_known_cz_v2*.npz`, `infs_opt_cz_v2*.npz` -- infidelity-vs-gate-time
    curves for the best known-library (CDD/mqCDD) and NT (free-timing
    optimized) sequences.
  - `plotting_data/plotting_data_cz_v2*.npz` -- the full per-gate-time
    winner sequences, labels, and provenance metadata consumed by the
    plotting/report scripts listed above.

What it does
------------
1. Loads spectral noise data (via `CZOptConfig`).
2. Constructs libraries of known pulse sequences (CDD, mqCDD) -- pulse-
   sequence generators defined LOCALLY in this file. These are separate,
   lightweight switch-time generators used only for the overlap-integral
   infidelity estimate below; they are NOT the same functions as
   `model/trajectories.py`'s pulse generators, which drive the full
   3-qubit propagator simulation of Stage 1.
3. Evaluates sequence performance using overlap integrals -- a second-order
   ("Gaussian"/cumulant, i.e. filter-function) approximation to the
   dephasing-induced infidelity -- computed either exactly in the time
   domain or approximately on the QNS measurement comb (see
   `use_comb_approximation`).
4. Optimizes randomly-initialized ("NT") pulse timings using JAX-based
   gradient descent (SLSQP, a standard constrained nonlinear optimizer,
   fed exact gradients computed by JAX autodiff).
5. Optimizes the coupling strength J for each candidate sequence so it hits
   the CZ target phase, subject to a maximum achievable coupling `Jmax`.

Based on the pre-restructure cz_optimize.py (and the now-removed cz_optimize_legacy.py).

Author: [Q]
Date: [01/18/2026]
"""

import functools
import itertools
import os
import traceback
from dataclasses import dataclass, field

import jax
# jax.jit traces a plain Python function once (the first time it's called
# with a given input shape/dtype) and compiles that trace to fast XLA code;
# every later call with matching shapes reuses the compiled version instead
# of re-running Python. jax.config.update("jax_enable_x64", True) below
# switches JAX from its default 32-bit floats to 64-bit -- required here
# because the spectral overlap integrals span many orders of magnitude and
# 32-bit precision would corrupt the small differences the optimizer relies
# on for its gradient.
jax.config.update("jax_enable_x64", True)
# OPT-SPEEDUPS (a): persistent XLA compilation cache. With the spectra passed
# as runtime ARGUMENTS (not closure constants -- see cost_vag_*), the compiled
# programs are value-independent, so repeats/reruns at the same shapes
# deserialize (~0.1 s) instead of recompiling (~1 s each). In plain terms:
# without this, every fresh Python process (e.g. every re-run of this
# script, or every restart in the optimizer's search) would silently pay the
# ~1 s XLA compile cost again for each distinct cost-function shape; caching
# the compiled binaries to disk turns that into a one-time cost.
from qns2q.paths import project_root as _project_root
jax.config.update("jax_compilation_cache_dir",
                  os.path.join(_project_root(), ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import jax.numpy as jnp
import jax.scipy.integrate
import jax.scipy.signal
import numpy as np
import scipy.optimize

from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 MODEL_VERSION, line_priors)
from qns2q.control.tails import tail_extend_interp_complex
from qns2q.control.padding import pad_targets, pad_count, pad_delays
from qns2q.paths import run_folder, project_root


# Fixed RNG seed for the unseeded np.random restarts (pulse-count / delay
# initialization). Pinning it makes the published infidelity curves and the
# winning-sequence labels reproducible across re-runs. Recorded in the saved
# plotting_data so every figure carries its provenance.
RANDOM_SEED = 20260608


# OPT-SPEEDUPS (d): SLSQP (Sequential Least SQuares Programming, scipy's
# gradient-based constrained optimizer -- see scipy.optimize.minimize below)
# convergence knobs. tol=1e-10 / maxiter=1000 over-converged an objective
# whose spectral inputs carry 5-20% reconstruction uncertainty -- i.e. the
# optimizer was grinding out extra digits of precision on a cost function
# that is itself only accurate to ~10%, wasting wall-clock time for no real
# gain in the reported infidelity. The host-driven iteration stream was a
# major wall-time term (each iteration is a ~2 ms device call dispatched
# from scipy, i.e. a round-trip from Python to the JAX-compiled cost/
# gradient and back). Winners validated unchanged against the 2026-06-11
# pre-change run.
SLSQP_TOL = 1e-7
SLSQP_MAXITER = 300


# Memoizes the (sequence-independent) folded-correlation setup built by
# prepare_time_domain_overlap. That setup depends only on the spectrum/grid
# arrays and (tau, T_seq, M) -- NOT on the pulse sequence -- yet was rebuilt once
# per pulse-count pair within each gate-time block. Cached tuples are returned
# verbatim (bit-identical), so this is a pure speed-up. Cleared at the start of
# run_optimization to keep it bounded.
_OVERLAP_SETUP_CACHE = {}


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CZOptConfig:
    """Configuration + input data for the CZ gate optimization (PROTECTED:
    imported as `czmod.CZOptConfig` by `scripts/harvest_design_numbers.py`
    and `scripts/run_margin_band.py` -- do not rename or change this
    class's constructor signature/defaults).

    This is a plain (mutable, non-frozen) `@dataclass`: most fields below
    are ordinary configuration knobs you can override at construction time
    (`CZOptConfig(max_pulses=0, ...)`), but a second group (marked
    `field(init=False)`) are NOT arguments -- they are outputs computed by
    `__post_init__` from the ones above (and from the .npz files on disk).
    Constructing a `CZOptConfig` therefore does real disk I/O and can raise
    `FileNotFoundError` if the upstream pipeline stage hasn't been run yet;
    it is not a free/side-effect-free object construction the way a config
    object usually is.
    """
    fname: str = field(default_factory=run_folder)
    parent_dir: str = os.pardir
    # Max Ising ("ZZ") coupling strength J, in tau units (J*tau): the CZ
    # phase is accumulated as J is switched on for a controlled duration, so
    # Jmax caps how fast that phase can be built up. 0.05 = 2e6 rad/s x 25 ns
    # (legacy SI anchor -- see CLAUDE.md's tau=1 units convention).
    Jmax: float = 0.05
    # Which gate times Tg to scan: Tg = Tqns / 2**(i - 1) for i in this list,
    # so i=1 is the base QNS block length; positive i give shorter gates,
    # negative i extend to longer gate times (added later in the project to
    # widen the plotted infidelity-vs-gate-time curve).
    gate_time_factors: list = field(default_factory=lambda: [3, 2, 1, 0, -1, -2, -3])
    output_path_known: str = "infs_known_cz_v2.npz"
    output_path_opt: str = "infs_opt_cz_v2.npz"
    plot_filename: str = "infs_GateTime_cz_v2.pdf"
    
    include_cross_spectra: bool = True
    tau_divisor: int = 160
    use_simulated: bool = False
    # How to turn the discrete QNS measurement comb (spectrum values known
    # only at the harmonics wk) into a continuous SMat(w) usable at
    # arbitrary optimizer frequencies: 'interp' = linear interpolation
    # straight through the comb teeth (plus a power-law tail extension
    # beyond the last measured tooth); 'selfconsistent' = reuse the same
    # line/tail/head-aware analytic-shape fit that the unfold bias
    # correction uses (characterize.systematics.selfconsistent_spectra) --
    # i.e. a smoother, physics-informed guess at what happens BETWEEN comb
    # teeth instead of a bare straight line (OPT-SPECTRAL-MODEL: this tag
    # marks every place in cz.py/idle.py that implements or selects between
    # the two choices, so grep for it to find them all).
    spectral_model: str = "interp"
    # Per-qubit pulse-count cap for the NT (free-timing) search and the
    # known-sequence library. Every extra pulse adds an optimization
    # parameter and grows the (n1, n2) search grid, so a cap keeps the
    # published run's compute budget bounded; 0 removes the cap entirely,
    # falling back to the only remaining hard limit -- pulses have to fit in
    # the gate time given the minimum separation -- i.e. up to
    # T_seq/min_sep - 1 pulses per qubit (UNCAP-0611: this tag flags every
    # place that cap is threaded through, e.g. the "0 disables the cap"
    # branch in __post_init__ below).
    max_pulses: int = 150
    # Minimum allowed time between consecutive pulses on a qubit, in units
    # of tau (SHOWCASE-0612 scenario parameter). Physically this models
    # finite control-electronics bandwidth: real pi-pulses have nonzero
    # duration/rise-time, so they cannot be packed arbitrarily close
    # together the way an idealized instantaneous-pulse model allows --
    # e.g. min_sep_factor=8.0 models a 40 ns pi-pulse at the 5 ns showcase
    # tau anchor. Applied SYMMETRICALLY to the known library, the NT search
    # and the random inits (so no method gets an unfair timing advantage);
    # 1.0 = legacy idealized limit (separation = tau, i.e. no extra
    # constraint beyond the time-grid unit itself).
    min_sep_factor: float = 1.0
    # Ablation rung (c): build the CHARACTERIZED gate model from the
    # single-qubit spectra alone (S11/S22 kept; S1212 + every cross dropped,
    # as a 1Q-only QNS campaign would leave them) while the ideal benchmark
    # keeps the full truth -- prices what the TWO-QUBIT reconstruction adds.
    # Contrast include_cross_spectra=False, which drops crosses from BOTH.
    char_self_only: bool = False
    # Spectrum-informed pulse-count candidates (SHOWCASE-0612): rank uniform-
    # train (equally-spaced pulse) counts by a cheap first-order proxy --
    # just read off S11+S22 at that count's fundamental frequency w=pi*n/T,
    # no optimization needed -- of the CHARACTERIZED spectra (blind-protocol-
    # safe -- each arm ranks on what it knows, not the ground truth) and add
    # the top ("quietest") windows to the legacy {max, max-1, max-2, half+-1}
    # heuristic set. The reason this matters: on featured (multi-peak)
    # noise landscapes the legacy heuristic's fixed offsets can land its
    # candidate pulse counts entirely ON a spectral line, while the actually
    # quiet gap between lines sits at some untried pulse count the legacy
    # set never tries.
    informed_counts: bool = False
    # NT-search pair filter: skip (n1, n2) with |n1 - n2| > pair_gap when > 0
    # (CZ optima are near-diagonal; the full grid is quadratic in candidates).
    pair_gap: int = 0
    plot_data_name: str = "plotting_data_cz_v2.npz"

    # These will be loaded from the run files
    Tqns: float = field(init=False)
    mc: int = field(init=False)

    # Calculated properties
    path: str = field(init=False)
    specs: dict = field(init=False)
    w: jnp.ndarray = field(init=False)
    w_ideal: jnp.ndarray = field(init=False)
    wkqns: jnp.ndarray = field(init=False)
    SMat: jnp.ndarray = field(init=False)
    SMat_ideal: jnp.ndarray = field(init=False)
    T2q1: float = field(init=False, default=jnp.inf)
    T2q2: float = field(init=False, default=jnp.inf)
    tau: float = field(init=False)

    def __post_init__(self):
        # __post_init__ is a dataclass hook that Python runs automatically
        # right after the auto-generated __init__ assigns the fields above;
        # it's the natural place to compute derived fields (the
        # field(init=False) ones) from the plain arguments. Here it does
        # more than that -- it reads .npz files off disk and builds the
        # spectral matrices used by the whole rest of this module, which is
        # why simply constructing a CZOptConfig can fail (FileNotFoundError)
        # or print WARNINGs.
        if self.max_pulses == 0:
            self.max_pulses = 10**9
            print("[cz] max_pulses=0: pulse count limited only by the "
                  "minimum separation tau (n <= T_seq/tau - 1 per qubit)")
        self.path = os.path.join(project_root(), self.fname)
        
        if not os.path.exists(self.path):
             raise FileNotFoundError(f"Data directory not found at {self.path}")

        # Load Data
        if self.use_simulated:
             sim_path = os.path.join(self.path, "simulated_spectra.npz")
             if not os.path.exists(sim_path):
                 raise FileNotFoundError(f"Simulated spectra not found at {sim_path}")
             print(f"Loading simulated spectra from {sim_path}")
             self.specs = np.load(sim_path)
             self.params = self.specs
        else:
             specs_path = os.path.join(self.path, "specs.npz")
             print(f"Loading reconstructed spectra from {specs_path}")
             self.specs = np.load(specs_path)
             self.params = np.load(os.path.join(self.path, "params.npz"))
             if 'spam_protocol' in self.specs.files:
                 print(f"  specs spam_protocol: {self.specs['spam_protocol']}")
             # A False *_dc_ok flag means the reconstruction could not determine
             # that w=0 point (the stored value is a first-harmonic floor or an
             # insignificant fit) -- surfaced because the SMat consumes it.
             bad_dc = sorted(k for k in self.specs.files
                             if k.endswith('_dc_ok') and not bool(self.specs[k]))
             if bad_dc:
                 print(f"[cz] NOTE: flagged (undetermined) DC points in specs: "
                       f"{', '.join(k[:-len('_dc_ok')] for k in bad_dc)}")

        # OPT-PROVENANCE (this tag marks every consumer that checks the
        # noise-model version stamp -- also used in characterize/experiments.py
        # and characterize/reconstruct.py, so grep it to find them all): if
        # the loaded specs/params were generated under an OLDER noise model
        # than the one `qns2q.noise.spectra` implements right now, then
        # `_build_ideal_spectra` below (which always evaluates the CURRENT
        # model) would be comparing the optimizer's search data against a
        # different noise environment than the one it actually optimized
        # against -- an apples-to-oranges "mixed-model" comparison that would
        # silently make the reported gate improvement meaningless. This is
        # the failure mode the paper-tracker item CA-REPRO-NUMBERS ("make
        # sure every regenerated figure/number traces to ONE consistent
        # noise-model version") exists to catch; the warning below is the
        # concrete check for it.
        mv = None
        for src_ in (self.specs, self.params):
            if 'model_version' in src_:
                mv = str(src_['model_version'])
                break
        self.model_version = mv if mv is not None else 'unknown'
        if self.model_version != MODEL_VERSION:
            print(f"[cz] WARNING: spectra model_version={self.model_version!r} "
                  f"!= current noise model {MODEL_VERSION!r}; the ideal "
                  f"benchmark uses the CURRENT model -- regenerate Stage 1/2.")

        self.Tqns = float(self.params['T'])

        # Units guard: the pipeline works in tau units (T = 160 tau = 160). A
        # run/spectra file from the SI-era code has T ~ 4e-6 s; mixing it with
        # the tau-unit Jmax silently breaks the optimization. Regenerate Stage
        # 1/2 (or spectra_input) for such folders.
        if self.Tqns < 1.0:
            print(f"[cz] WARNING: loaded T={self.Tqns:g} looks like SI-era data "
                  f"(expected tau-unit T ~ 160). Regenerate the spectra for "
                  f"this folder before trusting the optimization.")

        # Filter gate_time_factors to ensure physical feasibility with Jmax
        min_Tg = np.pi / (4 * self.Jmax)
        valid_factors = []
        for i in self.gate_time_factors:
            Tg = self.Tqns / 2 ** (i - 1)
            if Tg >= min_Tg:
                valid_factors.append(i)
            else:
                print(f"Config: Excluding factor {i} (Tg={Tg:.2e} tau) - too short for Jmax (min {min_Tg:.2e} tau)")
        self.gate_time_factors = valid_factors

        self.tau = self.Tqns / self.tau_divisor
        if self.min_sep_factor < 1.0:
            raise ValueError("min_sep_factor < 1: the minimum separation "
                             "cannot undercut the time unit tau")
        self.min_sep = self.min_sep_factor * self.tau
        if self.min_sep_factor != 1.0:
            print(f"[cz] control-bandwidth scenario: min pulse separation = "
                  f"{self.min_sep_factor:g} tau (n <= T_seq/min_sep - 1 per "
                  f"qubit; applied to library, NT search and inits)")
        self.mc = int(self.params['truncate'])

        # Frequency Grid. The predicted-side grid must reach the pulse-spacing
        # Nyquist pi/tau (= 4*w_max_sys for the T=160/truncate=20 comb): short
        # optimized sequences put their filter passband ABOVE the comb's last
        # tooth, and a grid stopping at 2*w_max_sys silently ignored that band
        # (85% of the true NT gate error at Tg=80tau). The time-domain overlap
        # cost is unchanged (its dt is pinned to tau/4 via pad_factor).
        w_max_sys = 2 * jnp.pi * self.mc / self.Tqns
        self.w_max = 8 * w_max_sys
        self.N_w = 20000
        self.w = jnp.linspace(0, self.w_max, self.N_w)
        self.w_ideal = jnp.linspace(0, 2 * self.w_max, 2 * self.N_w)
        
        if 'wk' in self.specs:
            # Both simulated_spectra.npz and reconstructed specs.npz carry their own
            # frequency grid (the latter includes a DC point, so it is one longer than
            # `mc` harmonics); use it so interpolation xp/fp lengths always match.
            self.wkqns = jnp.array(self.specs['wk'])
        else:
            self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])

        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()
        self._calculate_T2()

    def _build_interpolated_spectra(self):
        """Builds `self.SMat`: the 4x4 matrix (indices Null,1,2,12) of noise
        spectra the BLIND optimizer actually searches against, evaluated on
        the dense frequency grid `self.w`.

        The QNS experiment only measures the spectrum at a finite "comb" of
        harmonic frequencies (the `wk` teeth) -- it cannot know the true
        continuous S(w) everywhere the optimizer might want to evaluate it.
        This function is what turns that discrete comb into something the
        optimizer can query at any frequency: linear interpolation between
        teeth by default (or the `selfconsistent` alternative, see
        `spectral_model` above).

        Beyond the comb's last tooth each component is extended with a power
        law fitted to the top teeth (control.tails) instead of right=0 -- the
        zero extension let the optimizer park its filter weight in the
        unmeasured band as if it were noise-free, i.e. it would let the
        search "cheat" by hiding all its residual filter weight above the
        last measured frequency, where the blind objective can't see any
        noise at all.

        The w=0 (DC/zero-frequency) point comes from the data grid whenever
        it carries a measured DC sample (OPT-DC-ORACLE: this tag marks every
        place -- also in control/idle.py -- that has to decide where the
        SMat's w=0 entry comes from). Concretely: a reconstructed
        `specs.npz` gets its DC point from an actual measured experiment
        (the noise-aware Ramsey-slope-fit / double-echo DC protocols of
        Stage 1-2 -- see CLAUDE.md's "DC-via-Ramsey-slope" note), so using it
        keeps the optimization blind (built only from what a real experiment
        measured). Only a DC-less grid falls back to inserting the analytic
        S(0) instead -- this happens for `simulated_spectra.npz`, where the
        file already IS the analytic ground-truth model evaluated at the
        teeth (so there is no "blindness" to protect), or for a legacy
        `specs.npz` that predates the DC protocol (warned at load time:
        regenerate Stage 2 for those)."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to Null, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        grid_has_dc = bool(float(self.wkqns[0]) == 0.0)
        if not grid_has_dc and not self.use_simulated:
            print("[cz] WARNING: specs grid carries no w=0 point -- inserting "
                  "the analytic-model DC (legacy behavior). Regenerate Stage 2 "
                  "for a measured DC point.")

        if self.spectral_model not in ('interp', 'selfconsistent'):
            raise ValueError(f"unknown spectral_model {self.spectral_model!r}")
        use_sc = (self.spectral_model == 'selfconsistent')
        if use_sc and self.use_simulated:
            raise ValueError("spectral_model='selfconsistent' models a "
                             "reconstructed comb; simulated_spectra.npz "
                             "already IS the analytic model")
        if use_sc:
            # OPT-SPECTRAL-MODEL: each channel comes from the same line/
            # tail/head-aware analytic-SHAPE model the unfold bias
            # correction uses (characterize.systematics.
            # selfconsistent_spectra), instead of a bare straight-line
            # interpolation between comb teeth: Gaussian lines placed at the
            # experimentally-known nuclear-difference frequencies (only
            # their HEIGHTS are fit to the comb data via NNLS, for S11/S22
            # only), power-law tails beyond the last tooth, and a saturated
            # power-law "head" below the first tooth (where there is no
            # comb data at all). This is still "assumption-light" in the
            # sense that matters for the paper's blind-protocol claim --
            # every number going in is either the reconstructed data itself
            # or an independently-known experimental prior (the line
            # positions), never the ground-truth spectral SHAPE -- so this
            # remains a legitimate thing for a real blind experiment to do.
            # NaN (unreconstructed) channels are zero-filled here; the
            # cross() drop below still excludes them from the SMat.
            from qns2q.characterize.systematics import selfconsistent_spectra
            from qns2q.noise.spectra import line_priors
            sc_recon = {k: np.nan_to_num(np.asarray(self.specs[k]))
                        for k in ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212')}
            sc_fns = selfconsistent_spectra(np.asarray(self.wkqns), sc_recon,
                                            lines=line_priors())
            w_np = np.asarray(self.w)

        def combine(key, dc_func):
            if use_sc:
                return jnp.asarray(np.asarray(sc_fns[key](w_np), dtype=complex))
            interp = tail_extend_interp_complex(self.w, self.wkqns,
                                                self.specs[key])
            if grid_has_dc:
                return interp
            return interp.at[0].set(dc_func(w0)[0])

        # Diagonal elements. NaN here means corrupted data, not a protocol
        # limitation -- fail loudly.
        for key in ("S11", "S22", "S1212"):
            if np.any(np.isnan(np.asarray(self.specs[key]))):
                raise ValueError(f"self-spectrum {key} contains NaN -- "
                                 f"corrupted specs.npz?")
        SMat = SMat.at[1, 1].set(combine("S11", S_11))
        SMat = SMat.at[2, 2].set(combine("S22", S_22))
        if self.char_self_only:
            # Ablation rung (c): a single-qubit-only QNS campaign leaves the
            # ZZ self-spectrum and every cross unreconstructed -- the blind
            # objective assumes zero there; SMat_ideal keeps the full truth,
            # so the benchmark prices what the 2Q reconstruction adds.
            print("[cz] char_self_only: S1212 + crosses DROPPED from the "
                  "characterized model (1Q-only QNS counterfactual); the "
                  "ideal benchmark retains them.")
            return SMat
        SMat = SMat.at[3, 3].set(combine("S1212", S_1212))

        # Off-diagonal (cross-spectrum) elements. Some SPAM protocols cannot
        # reconstruct every cross channel at all (e.g. the "robust" protocol
        # leaves S112/S212 as all-NaN -- see CLAUDE.md's SPAM-protocol
        # section). OPT-ROBUST-NAN (this tag also appears in control/idle.py,
        # marking the matching logic there): such a channel is dropped from
        # THIS characterized model with a printed notice -- so the blind
        # search behaves exactly like a real experiment that never measured
        # that channel, i.e. it optimizes as if that noise source were zero
        # -- while `_build_ideal_spectra` below always keeps the full ground
        # truth. Comparing the two then tells you what losing that channel
        # actually costs in true infidelity. By contrast, the separate
        # `include_cross_spectra=False` config flag is a different,
        # DELIBERATE counterfactual: it removes the cross channels from BOTH
        # models (characterized AND ideal), simulating a world where those
        # correlations simply don't exist rather than one where they exist
        # but couldn't be measured.
        if self.include_cross_spectra:
            def cross(key, dc_func):
                data = np.asarray(self.specs[key])
                n_nan = int(np.isnan(data).sum())
                if n_nan:
                    proto = (str(self.specs['spam_protocol'])
                             if 'spam_protocol' in self.specs else 'none')
                    print(f"[cz] NOTE: {key} not reconstructed ({n_nan}/"
                          f"{data.size} NaN, spam_protocol={proto}) -- dropped "
                          f"from the characterized gate model; the ideal "
                          f"benchmark retains it.")
                    return None
                return combine(key, dc_func)

            for r, c, key, fn in ((1, 2, "S12", S_1_2),
                                  (1, 3, "S112", S_1_12),
                                  (2, 3, "S212", S_2_12)):
                val = cross(key, fn)
                if val is not None:
                    SMat = SMat.at[r, c].set(val)
                    SMat = SMat.at[c, r].set(jnp.conj(val))

        return SMat

    def _build_ideal_spectra(self):
        """Builds `self.SMat_ideal`: the 4x4 matrix of TRUE (ground-truth)
        noise spectra, evaluated directly from the analytic model in
        `qns2q.noise.spectra` on the (finer, wider) `self.w_ideal` grid.

        Unlike `_build_interpolated_spectra`, this never looks at the
        reconstructed/simulated data on disk -- it is the "answer key" used
        only to report the TRUE infidelity of whatever sequence the blind
        search (which only sees `self.SMat`) ends up picking. This is what
        makes the published numbers meaningful: the search is blind, but the
        reported performance is honest."""
        SMat_ideal = jnp.zeros((4, 4, self.w_ideal.size), dtype=jnp.complex128)
        
        # Diagonal elements
        SMat_ideal = SMat_ideal.at[1, 1].set(S_11(self.w_ideal))
        SMat_ideal = SMat_ideal.at[2, 2].set(S_22(self.w_ideal))
        SMat_ideal = SMat_ideal.at[3, 3].set(S_1212(self.w_ideal))
        
        # Off-diagonal elements
        if self.include_cross_spectra:
            # 1-2
            SMat_ideal = SMat_ideal.at[1, 2].set(S_1_2(self.w_ideal))
            SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_1_2(self.w_ideal)))
            # 1-12 (Index 1-3)
            SMat_ideal = SMat_ideal.at[1, 3].set(S_1_12(self.w_ideal))
            SMat_ideal = SMat_ideal.at[3, 1].set(jnp.conj(S_1_12(self.w_ideal)))
            # 2-12 (Index 2-3)
            SMat_ideal = SMat_ideal.at[2, 3].set(S_2_12(self.w_ideal))
            SMat_ideal = SMat_ideal.at[3, 2].set(jnp.conj(S_2_12(self.w_ideal)))
        
        return SMat_ideal

    def _calculate_T2(self):
        """Calculates each qubit's free-induction-decay time T2* from the
        ideal (ground-truth) spectra, via the standard pure-dephasing
        relation T2 = 2/S(0) (the low-frequency/quasi-static noise power
        sets how fast an uncontrolled qubit dephases). Used only to print
        Tg/T2 context ratios in `run_optimization`'s progress log, not in
        any optimization math."""
        S11_0 = jnp.real(self.SMat_ideal[1, 1, 0])
        S22_0 = jnp.real(self.SMat_ideal[2, 2, 0])
        
        self.T2q1 = 2.0 / S11_0 if S11_0 > 0 else jnp.inf
        self.T2q2 = 2.0 / S22_0 if S22_0 > 0 else jnp.inf
        
        print(f"Calculated T2 times (Ideal): Q1={self.T2q1:.2e} tau, Q2={self.T2q2:.2e} tau")

# ==============================================================================
# Sequence Generation Utilities
# ==============================================================================

def remove_consecutive_duplicates(input_list):
    """Drops adjacent equal switch-times from a pulse-time list.

    The recursive CDD construction (`cdd` below) can place a pulse from an
    outer recursion level at exactly the same instant as one from an inner
    level. Physically, two pi-pulses applied back-to-back with zero time
    between them cancel (pi * pi = identity), so both timestamps should be
    removed as a pair rather than kept as two separate switch events -- that
    is what the `i += 2` branch below does; a non-duplicate time is kept and
    only the index advances by 1."""
    output_list = []
    i = 0
    while i < len(input_list):
        if i + 1 < len(input_list) and input_list[i] == input_list[i+1]:
            i += 2
        else:
            output_list.append(input_list[i])
            i += 1
    return output_list

def cdd(t0, T, n):
    """Recursive Concatenated Dynamical Decoupling (CDD) pulse-time
    generator: the standard construction that builds an order-`n` CDD
    sequence (equivalently, CPMG-like nested pulse trains used for
    dynamical decoupling) on the window [t0, t0+T] by placing a pulse at the
    midpoint and recursing on each half at one order lower. Returns a plain
    Python list of pulse TIMES (not yet deduplicated -- see
    `remove_consecutive_duplicates`); n=1 is the base case (a single
    midpoint pulse, i.e. a Hahn echo)."""
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)

def cddn(t0, T, n):
    """`cdd` plus deduplication plus the block-boundary times 0 and T, i.e.
    a ready-to-use order-`n` CDD sequence's full switch-time list (including
    the endpoints `construct_pulse_library` expects)."""
    out = remove_consecutive_duplicates(cdd(t0, T, n))
    if out[0] == 0.:
        return out + [T]
    else:
        return [0.] + out + [T]

def mqCDD(T, n, m):
    """Generates a two-qubit ("multi-qubit") nested CDD sequence: qubit 1
    gets a plain order-`n` CDD train, and qubit 2 gets an order-`m` CDD
    train independently re-started (nested) within EACH of qubit 1's
    inter-pulse intervals. This produces two-qubit sequence pairs running
    at different effective rates on each qubit -- part of the known-
    sequence library `construct_pulse_library` builds and searches, as an
    alternative to plain matched-order CDD(n, n) pairs."""
    # Do not remove duplicates yet to preserve interval structure for nesting
    tk1 = cdd(0., T, n)
    tk2 = []
    for i in range(len(tk1)-1):
        tk2 += cdd(tk1[i], tk1[i+1]-tk1[i], m)
    tk2 += cdd(tk1[-1], T-tk1[-1], m)
    
    tk1 = remove_consecutive_duplicates(tk1)
    tk2 = remove_consecutive_duplicates(tk2)

    if tk1[0] != 0.: tk1 = [0.] + tk1
    if tk2[0] != 0.: tk2 = [0.] + tk2
    
    return [tk1 + [T], tk2 + [T]]

# The two functions below are the first `@jax.jit` uses in this module's
# body (as opposed to the config-time helpers above). `@jax.jit` traces the
# function ONCE per distinct input shape/dtype the first time it's called,
# compiles that trace to fast XLA code, and reuses the compiled version on
# every later call with matching shapes -- so these behave like ordinary
# Python functions but run much faster after the first call, at the cost of
# only accepting array-like inputs (not e.g. Python control flow that
# depends on an array's runtime VALUE).

@jax.jit
def pulse_times_to_delays(tk):
    """Absolute pulse TIMES -> inter-pulse DELAYS (the free optimization
    parameters). Drops the fixed block-boundary times 0 and T (and the
    final delay up to T, which is determined by the others) so a sequence
    of n pulses is represented by n free numbers; the inverse is
    `delays_to_pulse_times`."""
    tk_arr = jnp.array(tk)
    if len(tk_arr) <= 2:
        return jnp.array([])
    diffs = jnp.diff(tk_arr)
    return diffs[:-1]

@jax.jit
def delays_to_pulse_times(delays, T):
    """Inverse of `pulse_times_to_delays`: n free delays -> the full
    (n+2)-length list of absolute pulse times [0, t1, t2, ..., T], with the
    last interval filled in so the total spans exactly the block time T."""
    if delays.size == 0:
        return jnp.array([0., T])
    last_delay = T - jnp.sum(delays)
    all_delays = jnp.concatenate([delays, jnp.array([last_delay])])
    times = jnp.cumsum(all_delays)
    return jnp.concatenate([jnp.array([0.]), times])

def get_random_delays(n, T, tau):
    """Draws n random positive delays that sum to <= T while respecting the
    minimum pulse separation `tau` -- used to seed the free-timing (NT)
    optimization when no explicit `seed_seq` is supplied. Uses plain
    `np.random` (not JAX's own PRNG), so its output is controlled by
    whatever `np.random.seed(...)` call was last made (pinned once per
    `run_optimization` call via `RANDOM_SEED`, not by this function itself)."""
    if n <= 0:
        return jnp.array([])
    slack = T - (n + 1) * tau
    if slack > 0:
        r = np.random.rand(int(n) + 1)
        r = r / np.sum(r)
        delays = tau + slack * r
        return jnp.array(delays[:n])
    else:
        return jnp.ones(n) * T / (n + 1)

@jax.jit
def make_tk12(tk1, tk2):
    """Merges qubit 1's and qubit 2's pulse times into the single switch-
    time list of their PRODUCT toggling function -- i.e. the times at which
    EITHER qubit's frame flips, which is what controls the sign of the
    Ising (ZZ) coupling channel between them. This is why index 3 ("12") of
    the 4x4 overlap/spectrum matrices uses `pt12` = `make_tk12(pt1, pt2)`
    rather than either qubit's own pulse times."""
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """Builds the library of "known" (textbook, not gradient-optimized)
    two-qubit pulse sequences that the blind search compares its free-
    timing (NT) results against -- every (CDD order, CDD order) pair
    including mismatched/asynchronous orders, plus mqCDD variants -- for a
    single gate time `T_seq`.

    Parameters
    ----------
    T_seq : float
        Sequence block duration (the gate time being evaluated).
    tau_min : float
        Minimum allowed inter-pulse separation (`config.min_sep`); sequence
        orders that would violate it are excluded (both the single-qubit
        CDD generation loop and the mqCDD nesting loop stop increasing
        their order as soon as they would).
    max_pulses : int, optional
        Per-qubit pulse-count cap (`config.max_pulses`); sequences needing
        more pulses on either qubit than this are dropped in step 4.

    Returns
    -------
    pLib_delays : list of (jnp.ndarray, jnp.ndarray)
        Each entry is a (qubit-1 delays, qubit-2 delays) pair, ready for
        `evaluate_known_sequences_with_T`.
    pLib_descriptions : list of str
        Human-readable label per entry (e.g. "CDD(3, 3)", "mqCDD(n=2, m=4)"),
        used for the winning-sequence label saved to plotting_data.
    """
    # 1. Generate single-qubit CDD sequences
    cddLib = []
    cddOrd = 1
    while True:
        pul = jnp.array(cddn(0., T_seq, cddOrd))
        if jnp.any(jnp.diff(pul) < tau_min):
            break
        cddLib.append(pul)
        cddOrd += 1

    # 2. Create combinations (including synchronous sequences)
    pLib_times = list(itertools.product(cddLib, repeat=2))

    # 3. Generate mqCDD sequences
    mq_cdd_orders_log = []
    ncddOrd1 = 1
    while True:
        ncddOrd2 = 1
        pul_n = mqCDD(T_seq, ncddOrd1, ncddOrd2)[0]
        if jnp.any(jnp.diff(jnp.array(pul_n)) < tau_min):
            break 

        while True:
            pul = mqCDD(T_seq, ncddOrd1, ncddOrd2)
            if jnp.any(jnp.diff(jnp.array(pul[1])) < tau_min):
                break
            pLib_times.append((jnp.array(pul[0]), jnp.array(pul[1])))
            mq_cdd_orders_log.append((ncddOrd1, ncddOrd2))
            ncddOrd2 += 1
        ncddOrd1 += 1

    # 4. Prune and convert to delays
    pLib_delays = []
    pLib_descriptions = []
    num_cdd_perms = len(list(itertools.product(cddLib, repeat=2)))
    
    def find_cdd_index(tk, lib):
        for idx, seq in enumerate(lib):
            if len(seq) == len(tk) and jnp.allclose(seq, tk):
                return idx + 1
        return -1

    for i, (tk1, tk2) in enumerate(pLib_times):
        n1 = len(tk1) - 2
        n2 = len(tk2) - 2
        if n1 <= max_pulses and n2 <= max_pulses:
            d1 = pulse_times_to_delays(tk1)
            d2 = pulse_times_to_delays(tk2)
            pLib_delays.append((d1, d2))
            
            if i < num_cdd_perms:
                ord1 = find_cdd_index(tk1, cddLib)
                ord2 = find_cdd_index(tk2, cddLib)
                pLib_descriptions.append(f"CDD({ord1}, {ord2})")
            else:
                mq_idx = i - num_cdd_perms
                n, m = mq_cdd_orders_log[mq_idx]
                pLib_descriptions.append(f"mqCDD(n={n}, m={m})")

    return pLib_delays, pLib_descriptions

# ==============================================================================
# Core Calculation Functions (JAX-Compatible)
# ==============================================================================

# `functools.partial(jax.jit, static_argnames=[...])` (used repeatedly
# below): `static_argnames` tells JAX that those named arguments are plain
# Python values that affect the function's CONTROL FLOW or output SHAPE
# (e.g. an integer repeat count `M`, or a size `n_base_steps`), not numeric
# arrays to differentiate/trace -- JAX will compile a separate specialized
# program for each distinct value seen, rather than (incorrectly) trying to
# trace them as abstract array placeholders. `functools.partial` here is
# just "pre-fill jax.jit's `static_argnames` keyword", used as a decorator
# so the wrapped function is jitted with that setting from definition time.

@functools.partial(jax.jit, static_argnames=['M', 'n_base_steps'])
def precompute_R_folded(R_shifted, lags_R, M, T_base, dt, n_base_steps):
    """Folds (periodically sums) a single noise auto/cross-correlation
    function R(t) -- already given on a shifted lag grid `lags_R` -- across
    M repetitions of a base sequence block of length `T_base`, producing the
    effective correlation function seen by an M-times-repeated pulse block.
    Physically: repeating a sequence M times doesn't just repeat its filter
    function once M times independently -- correlations between pulses in
    DIFFERENT repeats also contribute, weighted by how many repeat-pairs are
    separated by each integer multiple of T_base (the triangular `weights`
    array, `M - |p|` for offset `p` repeats). This is the exact time-domain
    counterpart of the frequency-domain "M/T_seq" comb weighting in
    `evaluate_overlap_comb`. Returned on the (2*n_base_steps - 1)-point lag
    grid spanning one base-sequence period, split into real/imaginary parts
    and interpolated (via `jnp.interp`) because `lags_R` may not exactly
    line up with the shifted grid at each repeat offset."""
    lags_C = (jnp.arange(2 * n_base_steps - 1) - (n_base_steps - 1)) * dt
    p_vals = jnp.arange(-(M - 1), M)
    weights = float(M) - jnp.abs(p_vals)
    R_real = jnp.real(R_shifted)
    R_imag = jnp.imag(R_shifted)

    def get_folded_component(R_comp):
        def interp_slice(p, w):
            shifted_lags = lags_C + p * T_base
            return w * jnp.interp(shifted_lags, lags_R, R_comp, left=0., right=0.)
        slices = jax.vmap(interp_slice)(p_vals, weights)
        return jnp.sum(slices, axis=0)

    return get_folded_component(R_real) + 1j * get_folded_component(R_imag)

@functools.partial(jax.jit, static_argnames=['n_base_steps'])
def evaluate_overlap_folded(pulse_times_a, pulse_times_b, R_folded, dt, n_base_steps):
    """Exact (non-approximate) time-domain overlap integral I_{a,b} between
    two toggling-frame switching functions y_a(t), y_b(t) (defined by their
    pulse times) and a folded noise correlation function `R_folded` (from
    `precompute_R_folded`/`prepare_time_domain_overlap`). Concretely:
    samples each switching function on a uniform time grid as a +-1 square
    wave (`get_y_samples`, via `searchsorted` to count how many pulses
    precede each grid point), cross-correlates them, and integrates the
    product against `R_folded` -- the discretized version of
    I_{a,b} = integral over t, t' of y_a(t) y_b(t') R_{a,b}(t - t') dt dt'.
    This is the "folded"/time-domain evaluator; `evaluate_overlap_comb`
    below is the faster, comb-frequency-domain alternative for the same
    physical quantity."""
    t_grid = jnp.arange(n_base_steps) * dt
    def get_y_samples(pt):
        indices = jnp.searchsorted(pt, t_grid, side='right')
        return (-1.0) ** (indices - 1)
    y_a = get_y_samples(pulse_times_a)
    y_b = get_y_samples(pulse_times_b)
    C_vals = jax.scipy.signal.correlate(y_a, y_b, mode='full') * dt
    integral = jnp.sum(C_vals * R_folded) * dt
    return integral

@jax.jit
def get_spectral_amplitudes_jax(pulse_times, omega):
    """Fourier-domain "filter function" amplitude of a toggling-frame
    switching function (defined by `pulse_times`) at each frequency in
    `omega`: the integral of the +-1 square wave against exp(i*omega*t),
    computed in closed form per interval (each interval between
    consecutive pulses contributes `+-(exp(i*omega*t_end) -
    exp(i*omega*t_start))`, with alternating sign for the alternating +-1
    frame). Used by the comb-based overlap evaluator below, which needs
    this amplitude for both sequences at every comb harmonic."""
    exp_t = jnp.exp(1j * jnp.outer(pulse_times, omega))
    diffs = exp_t[1:] - exp_t[:-1]
    n_intervals = len(pulse_times) - 1
    signs = jnp.array((-1.0)**jnp.arange(n_intervals))
    return jnp.sum(signs[:, None] * diffs, axis=0)

@functools.partial(jax.jit, static_argnames=['M'])
def evaluate_overlap_comb(pulse_times_a, pulse_times_b, S_packed, omega_k, T_seq, M):
    """Fast, comb-frequency-domain approximation to the same overlap
    integral I_{a,b} that `evaluate_overlap_folded` computes exactly: instead
    of an explicit time-domain double integral, sums the filter-function
    amplitudes (`get_spectral_amplitudes_jax`) against the spectrum SAMPLED
    ONLY at the QNS measurement comb's harmonics `omega_k` (plus the w=0/DC
    term, handled separately via `get_dc` since the DC contribution isn't a
    finite-frequency term). Valid when the spectrum doesn't vary much across
    one comb tooth's width (see `use_comb_approximation` for the validity
    check); the `M / T_seq` prefactor accounts for M sequence repetitions
    the same way `precompute_R_folded`'s time-domain folding does."""
    def get_dc(pt):
        diffs = jnp.diff(pt)
        n = len(diffs)
        signs = jnp.array((-1.0)**jnp.arange(n))
        return jnp.sum(diffs * signs)
    dc_a = get_dc(pulse_times_a)
    dc_b = get_dc(pulse_times_b)
    S_0 = S_packed[0]
    term_dc = dc_a * dc_b * S_0

    if omega_k.size == 0:
        return term_dc * M / T_seq

    S_k = S_packed[1:]
    A_a = get_spectral_amplitudes_jax(pulse_times_a, omega_k)
    A_b = get_spectral_amplitudes_jax(pulse_times_b, omega_k)

    Ar, Ai = jnp.real(A_a), jnp.imag(A_a)
    Br, Bi = jnp.real(A_b), jnp.imag(A_b)
    Sr, Si = jnp.real(S_k), jnp.imag(S_k)

    P_real = Ar * Br + Ai * Bi
    P_imag = Ai * Br - Ar * Bi
    term_ac_real = (P_real * Sr - P_imag * Si) / (omega_k**2)
    sum_ac = jnp.sum(term_ac_real)
    total = term_dc + 2 * sum_ac
    return total * M / T_seq


def prepare_time_domain_overlap(SMat, w_grid, tau, T_seq, M):
    """Prepare folded correlation data for the EXACT time-domain overlap
    integral method (the setup consumed by `evaluate_overlap_folded` and
    `precompute_R_folded`). Converts the frequency-domain spectral matrix
    SMat(w) into a real-space noise correlation matrix R(t) via an inverse
    FFT, then folds it for M sequence repetitions -- once per (spectrum,
    grid, tau, T_seq, M) combination, since none of that setup depends on
    the actual pulse sequence being evaluated (only on the gate time and
    repeat count), which is what makes memoizing it in
    `_OVERLAP_SETUP_CACHE` a pure speed-up, not a behavior change.

    Applies adaptive zero-padding to achieve dt <= tau/4 for improved time
    resolution, mirrors the spectrum for Hermitian symmetry (a real-valued
    physical correlation function requires S(-w) = S*(w)), IFFTs to obtain
    R(t), and folds the correlation across M repetitions.

    Returns
    -------
    RMat_data : jnp.ndarray, shape (4, 4, 2*n_base_steps-1)
        Folded noise correlation matrix on the base-sequence lag grid.
    dt : float
        Time-domain grid spacing, in tau units (this pipeline works in
        tau=1 units throughout -- see CLAUDE.md).
    n_base_steps : int
        Number of time steps spanning one base sequence period T_seq.
    """
    # The cache key uses Python's id() (object identity, i.e. "is this the
    # very same array object", not "does it hold the same values") for SMat
    # and w_grid rather than comparing their contents. That's intentional
    # and safe here: within one CZOptConfig, config.SMat/config.w are built
    # once and never mutated, so re-using them always means re-using the
    # exact same object -- but it does mean two DIFFERENT arrays that
    # happen to hold identical values would (correctly, if wastefully) miss
    # the cache and get recomputed rather than being treated as equal.
    cache_key = (id(SMat), id(w_grid), float(tau), float(T_seq), int(M))
    cached = _OVERLAP_SETUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    N = w_grid.shape[0]
    dw = float(w_grid[1] - w_grid[0])
    w_max = float(w_grid[-1])

    # Adaptive padding: ensure dt <= tau / 4
    # After padding:  array has (1 + pad_factor) * N points
    # After mirror:   N_sym ≈ 2 * (1 + pad_factor) * N
    # dt = 2π / (N_sym * dw) ≈ π / ((1 + pad_factor) * w_max)
    desired_dt = tau / 4
    pad_factor = max(int(np.ceil(np.pi / (w_max * desired_dt))) - 1, 1)

    SMat_padded = jnp.pad(SMat, ((0, 0), (0, 0), (0, pad_factor * N)))

    # Mirror for Hermitian symmetry: S(-ω) = S*(ω)
    SMat_sym = jnp.concatenate(
        [SMat_padded, jnp.conj(jnp.flip(SMat_padded[..., 1:-1], axis=-1))],
        axis=-1,
    )
    N_sym = SMat_sym.shape[-1]

    dt = 2 * np.pi / (N_sym * dw)
    print(f"  Time-domain setup: pad_factor={pad_factor}, dt={dt:.4f} tau, "
          f"tau/4={tau/4:.4f} tau")

    lags_R = (jnp.arange(N_sym) - N_sym // 2) * dt
    RMat_vals = jnp.fft.ifft(SMat_sym, axis=-1)
    RMat_scaled = RMat_vals / dt
    RMat_shifted = jnp.fft.fftshift(RMat_scaled, axes=-1)

    n_base_steps = int(np.ceil(T_seq / dt))

    @jax.jit
    def get_folded_matrix(RMat_in):
        R_flat = RMat_in.reshape(-1, RMat_in.shape[-1])
        folded_flat = jax.vmap(
            lambda r: precompute_R_folded(r, lags_R, M, T_seq, dt, n_base_steps)
        )(R_flat)
        return folded_flat.reshape(4, 4, -1)

    RMat_data = get_folded_matrix(RMat_shifted)

    result = (RMat_data, dt, n_base_steps)
    _OVERLAP_SETUP_CACHE[cache_key] = result
    return result


# ==============================================================================
# CZ Fidelity Calculation
# ==============================================================================

@jax.jit
def zzPTM():
    """The ideal (noise-free) CZ gate's Pauli Transfer Matrix (PTM): the
    16x16 real matrix R_ideal such that R_ideal[i,j] = (1/4) Tr(P_i U P_j
    U^dagger), where {P_i} runs over the 16 two-qubit Pauli operators (`p2q`
    = all tensor products of {I, X, Y, Z} = `p1q`) and U = exp(-i*pi/4*Z1Z2)
    is the ideal entangling unitary this gate implements. A PTM is just the
    Pauli-basis representation of how a (possibly noisy) quantum channel
    transforms Pauli operators; comparing a noisy gate's PTM to this ideal
    one (as `calculate_cz_fidelity` does) is the standard way to compute an
    average gate fidelity. Cached implicitly by JAX's jit (this function
    takes no arguments, so it traces/compiles to the same constant matrix
    every call)."""
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    U = jax.scipy.linalg.expm(-1j * jnp.kron(p1q[3], p1q[3]) * jnp.pi / 4)
    gamma = jnp.array([[(1 / 4) * jnp.trace(p2q[i] @ U @ p2q[j] @ U.conj().transpose()) for j in range(16)] for i in
                       range(16)])
    return jnp.real(gamma)

@jax.jit
def sgn(O, a, b):
    """Commutation-sign helper: returns +1 if the two-qubit operator `O`
    COMMUTES with the product Z_a @ Z_b, and -1 if it ANTICOMMUTES with it
    (for O a Pauli operator, Tr(O^dagger Z_a Z_b O Z_a Z_b)/4 is exactly
    +-1). `a, b` index the four "Z-type"/checkerboard two-qubit operators
    II, ZI, IZ, ZZ (`z2q` below) -- e.g. `sgn(O, 1, 0)` asks whether O
    commutes with Z (x) I on qubit 1 alone. This is the building block
    `calculate_cz_fidelity` uses to work out which Pauli components of the
    noisy channel pick up a dephasing-induced decay vs. which are
    untouched, and which pick up the coherent CZ rotation."""
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    return jnp.trace(O.conj().T @ z2q[a] @ z2q[b] @ O @ z2q[a] @ z2q[b]) / 4

@jax.jit
def calculate_cz_fidelity(I_matrix, J, M, dc_12):
    """
    Calculate the two-qubit CZ gate fidelity from overlap integrals and coupling.

    This function computes the average gate fidelity of a CZ gate implemented
    via a pulsed sequence with a tunable coupling strength $J$. It accounts
    for the unitary rotation (controlled-Z) and the dephasing noise captured
    by the overlap integrals $I_{a,b}$. The fidelity is calculated by
    constructing the Pauli Transfer Matrix (PTM) of the noisy gate and
    comparing it to the ideal CZ gate.

    Parameters
    ----------
    I_matrix : jax.Array
        A 4x4 matrix of overlap integrals $I_{a,b}$.
    J : float
        Coupling strength for the Ising interaction.
    M : int
        Number of sequence repetitions.
    dc_12 : float
        DC component of the Ising interaction control function.

    Returns
    -------
    float
        The calculated average gate fidelity for the CZ operation.
    """

    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    
    # Every operator in the exponent is DIAGONAL (the z2q are Z (x) Z products),
    # so the 4x4 expm reduces to an elementwise exp of its diagonal; and the
    # exponent depends on Oi only, so G is computed once per Oi rather than once
    # per (Oi, Oj) pair. 16 length-4 vector exps replace 256 matrix exponentials
    # per fidelity evaluation -- an exact rewrite of the same map.
    z2q_diag = jnp.diagonal(z2q, axis1=1, axis2=2)

    def g_diag(Oi):
        val = jnp.zeros(4, dtype=jnp.complex128)
        for i in range(3):
            for j in range(3):
                idx_i = i + 1
                idx_j = j + 1

                # Second-cumulant decay coefficient (paper Eq. c2_spectra, corrected
                # both-index parity). Mirrors calculate_idling_fidelity: with the
                # anticommutation flag c_a = (1 - sgn(O, a, 0))/2 in {0, 1}, a pair
                # (a, b) contributes only when O anticommutes with BOTH Z_a and Z_b,
                # carrying weight 1/2 c_a c_b on the bare overlap I_{a,b} (the 1/2 is
                # the dephasing-convention factor). I_matrix is the (1/2pi)-normalized
                # overlap from evaluate_overlap_comb (bare, no 1/2 folded in).
                c_a = 0.5 * (1.0 - sgn(Oi, idx_i, 0))
                c_b = 0.5 * (1.0 - sgn(Oi, idx_j, 0))
                coeff = 0.5 * c_a * c_b

                val += coeff * I_matrix[idx_i, idx_j] * (z2q_diag[idx_i] * z2q_diag[idx_j])

        rot_val = (1.0 - sgn(Oi, 1, 2)) * M * J * dc_12
        return jnp.exp(-1j * rot_val * z2q_diag[3] - val)

    G_diag = jax.vmap(g_diag)(p2q)                                  # (16, 4)
    # tr(Oi @ diag(g) @ Oj) = sum_{a,b} Oi[a,b] g[b] Oj[b,a]
    R_noisy = jnp.real(jnp.einsum('iab,ib,jba->ij', p2q, G_diag, p2q) * 0.25)
    
    # Fidelity = Tr(R_ideal.T @ R_noisy) / 16
    R_ideal = zzPTM()
    fid = jnp.trace(R_ideal.T @ R_noisy) / 16.0
    
    return fid

def use_comb_approximation(M, T_seq):
    """Whether the frequency-comb overlap approximation (`evaluate_overlap_comb`)
    is accurate enough to use at this (M, T_seq), instead of the exact but
    slower time-domain evaluator (`evaluate_overlap_folded`).

    The comb evaluator samples the spectrum S only at discrete "teeth"
    (delta functions in frequency), implicitly assuming S is smooth across
    the gap between neighboring teeth. But the TRUE M-fold-repeated filter
    function has its OWN finite tooth width ~ 2pi/(M*T_seq) (repeating a
    sequence M times narrows its passbands, and a finite M gives that
    passband a finite width rather than a true delta function). When that
    width becomes comparable to the width sigma of a real spectral line
    (e.g. one of the featured model's nuclear-difference lines), the comb
    approximation starts mis-weighting the line and the two evaluators
    disagree (OPT-COMB-M16: this tag marks the empirical accuracy figures
    below and their counterpart in control/idle.py; the diagnostic script
    that produced them, `scripts/diag_comb_vs_folded.py`, was removed in the
    2026-06-16 cleanup of dev-only tooling, but the thresholds it
    established are hardcoded here and remain the basis for this cutoff):
    8-14% relative error at Tg = 320 tau / M = 16, 3-7% at 640, up to 3.2%
    at 1280; <= 1.7% once 2pi/(M*T_seq) < sigma/8, which is the condition
    coded below. Smooth (bland/monotonic, no lines) spectra keep the legacy
    speed cutoff of M > 10 regardless of line width, since there's no line
    to mis-weight. (CZ currently always runs at M = 1, well below any of
    these cutoffs, so in practice this function currently always returns
    False for CZ; it's kept here as future-proofing, in lockstep with the
    same logic in idle.py where M > 1 is actually used.)"""
    if M <= 10:
        return False
    pri = line_priors()
    if pri is None:
        return True
    _, sigma = pri
    return 2 * np.pi / (M * T_seq) < sigma / 8


# ==============================================================================
# Optimization Logic
# ==============================================================================

def _pack_comb(SMat, w_grid, omega_k):
    """Re-samples the dense SMat(w) grid down onto the sparse comb harmonics
    `omega_k` (interpolating each of the 16 (4x4) channels), and packs the
    result as `[S(0), S(w_k1), S(w_k2), ...]` per channel -- exactly the
    layout `evaluate_overlap_comb`'s `S_packed` argument expects (index 0 =
    the DC/zero-frequency value, the rest = the finite-frequency comb
    values)."""
    S_flat = SMat.reshape(-1, SMat.shape[-1])

    def interp_row(fp):
        return (jnp.interp(omega_k, w_grid, jnp.real(fp), right=0.) +
                1j * jnp.interp(omega_k, w_grid, jnp.imag(fp), right=0.))

    S_h = jax.vmap(interp_row)(S_flat)
    return jnp.concatenate([S_flat[:, :1], S_h], axis=1).reshape(4, 4, -1)


def _cz_inf_from_imat(I_mat, pt12, Jmax, M):
    """Shared tail of the CZ cost/infidelity computation, called by both
    evaluators (`_cost_folded`/`_cost_comb` and the non-jit callers below):
    given the overlap-integral matrix `I_mat` and the joint (qubit-1 AND
    qubit-2) toggling pattern `pt12`, works out the coupling strength J
    needed to realize the CZ's pi/4 phase over M repetitions given how much
    "on time" the Ising interaction accumulates (`dc_12`, the DC/net area
    of the pt12 square wave), clips it to the physically allowed `Jmax`,
    evaluates the resulting fidelity, and adds a SMOOTH penalty (rather
    than a hard cutoff) when even `Jmax` can't reach the required phase --
    the smoothness matters because this is a term inside a function being
    differentiated by JAX for gradient-based optimization: a hard
    if/else cutoff would give zero gradient in the infeasible region and
    the optimizer would have no signal telling it which direction to move
    back toward feasibility."""
    diffs = jnp.diff(pt12)
    signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
    dc_12 = jnp.sum(diffs * signs)

    # Safe division: clamp |dc_12| away from zero to prevent gradient explosion
    abs_dc_12 = jnp.abs(dc_12)
    safe_dc = jnp.where(abs_dc_12 > 1e-15, dc_12, jnp.sign(dc_12 + 1e-30) * 1e-15)

    J_target = jnp.pi * 0.25 / (M * safe_dc)
    J = jnp.clip(J_target, -Jmax, Jmax)

    fid = calculate_cz_fidelity(I_mat, J, M, dc_12)

    # Smooth penalty for infeasible sequences (Jmax cannot achieve required CZ phase)
    required_dc = jnp.pi * 0.25 / (M * Jmax)
    feasibility_ratio = abs_dc_12 / required_dc
    penalty = jnp.where(feasibility_ratio >= 1.0, 0.0, (1.0 - feasibility_ratio) ** 2)

    return 1.0 - fid + penalty


def _delays_to_pts(delays_params, n_pulses1, T_seq):
    """Splits the flat SLSQP optimization-parameter vector `delays_params`
    (qubit 1's delays followed by qubit 2's) back into the four pulse-time
    arrays every overlap evaluator needs: `pt0` (the trivial whole-block
    boundary, used for the "Null" row/column 0 of the 4x4 overlap matrix),
    `pt1`, `pt2` (each qubit's own switching times), and `pt12`
    (`make_tk12(pt1, pt2)`, the joint switching pattern controlling the ZZ
    channel)."""
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    return [pt0, pt1, pt2, pt12]


def _cost_folded(delays_params, RMat_data, dt, T_seq, Jmax, M,
                 n_pulses1, n_base_steps):
    """CZ infidelity cost function (time-domain/folded evaluator) for a
    given pulse-timing parameter vector -- the objective SLSQP minimizes in
    `optimize_sequence`. Every input is a runtime ARGUMENT rather than a
    value baked into a Python closure (OPT-SPEEDUPS (b): the same design
    choice appears in control/idle.py's cost functions): if the spectra
    were instead captured from an enclosing scope, JAX would treat each new
    spectrum as requiring a BRAND NEW compiled program (a fresh closure is a
    fresh function identity to JAX's cache), so every gate time / restart
    would pay a full recompile. Passing them as explicit arguments instead
    means the compiled program only depends on argument SHAPES, so the
    `cost_vag_folded`/`cost_vag_comb` wrappers below are stable, reusable
    module-level objects: restarts and gate times reuse the same compiled
    HLO (High-Level Optimizer IR -- XLA's compiled program representation)
    per (shape, static-argument) combination instead of recompiling per
    fresh closure, and because that HLO doesn't depend on the actual
    spectrum VALUES, the persistent compilation cache (see the top of this
    file) can reuse it across separate runs and repeats too."""
    pts = _delays_to_pts(delays_params, n_pulses1, T_seq)
    I_mat = jnp.array([[evaluate_overlap_folded(pts[i], pts[j], RMat_data[i, j],
                                                dt, n_base_steps)
                        for j in range(4)] for i in range(4)])
    return _cz_inf_from_imat(I_mat, pts[3], Jmax, M)


def _cost_comb(delays_params, S_packed, omega_k, T_seq, Jmax, n_pulses1, M):
    """CZ cost on the comb evaluator (see _cost_folded for the design)."""
    pts = _delays_to_pts(delays_params, n_pulses1, T_seq)
    I_mat = jnp.array([[evaluate_overlap_comb(pts[i], pts[j], S_packed[i, j],
                                              omega_k, T_seq, M)
                        for j in range(4)] for i in range(4)])
    return _cz_inf_from_imat(I_mat, pts[3], Jmax, M)


# jax.value_and_grad(f) builds a new function that returns BOTH f's value
# AND its exact gradient (via JAX automatic differentiation) in a single
# traced/compiled call -- used here because scipy's SLSQP optimizer
# (called with jac=True in optimize_sequence below) needs both the cost and
# its gradient at every iteration, and computing them together is cheaper
# than two separate calls. The gradient is EXACT (to floating-point
# precision), not a finite-difference approximation, which is what makes
# gradient-based optimization of hundreds of pulse-timing parameters
# practical here.
cost_vag_folded = jax.jit(jax.value_and_grad(_cost_folded),
                          static_argnames=('n_pulses1', 'n_base_steps'))
cost_vag_comb = jax.jit(jax.value_and_grad(_cost_comb),
                        static_argnames=('n_pulses1', 'M'))

def optimize_sequence(config, M, T_seq, n1, n2, seed_seq=None, pad_to=None):
    """
    Perform gradient-based pulse timing optimization for a CZ gate.

    This function optimizes the pulse switch times for both qubits and
    the Ising coupling strength $J$ simultaneously to minimize the gate
    infidelity. It uses JAX-based automatic differentiation for gradients
    and the SLSQP algorithm for constrained optimization.

    Parameters
    ----------
    config : CZOptConfig
        Configuration object containing spectral data and optimization settings.
    M : int
        Number of repetitions.
    T_seq : float
        Total sequence time for one block.
    n1 : int
        Number of pulses for qubit 1.
    n2 : int
        Number of pulses for qubit 2.
    seed_seq : tuple of jax.Array, optional
        Initial pulse times for seeding the optimization.
    pad_to : tuple of int or None, optional
        Per-qubit shape-unification targets (control.padding): the jitted
        cost runs at the PADDED pulse counts, with exact-identity zero-delay
        pads appended inside the wrapper, while SLSQP keeps optimizing the
        original (n1 + n2)-dimensional problem -- one compiled program per
        parity class per gate time instead of one per (n1, n2) pair. Targets
        must match the parity of n1/n2; None entries disable padding.

    Returns
    -------
    tuple
        A tuple containing (best_seq, best_J, best_inf), where `best_seq`
        is a tuple of (pt1, pt2) absolute pulse times.
    """
    # Check feasibility of pulse count (min_sep = tau unless the
    # control-bandwidth scenario raises it)
    if (n1 + 1) * config.min_sep > T_seq or (n2 + 1) * config.min_sep > T_seq:
        return None, 1.0

    n1p = n1 if (pad_to is None or pad_to[0] is None) else max(int(pad_to[0]), n1)
    n2p = n2 if (pad_to is None or pad_to[1] is None) else max(int(pad_to[1]), n2)
    pad1 = np.zeros(n1p - n1)
    pad2 = np.zeros(n2p - n2)

    # Setup Evaluation Method
    use_comb = use_comb_approximation(M, T_seq)

    if use_comb:
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        omega_k = jnp.arange(1, max_k + 1) * w0
        S_packed = _pack_comb(config.SMat, config.w, omega_k)

        def vag(xp):
            return cost_vag_comb(xp, S_packed, omega_k, T_seq, config.Jmax,
                                 n_pulses1=n1p, M=M)
    else:
        RMat_data, dt, n_base_steps = prepare_time_domain_overlap(
            config.SMat, config.w, config.tau, T_seq, M
        )

        def vag(xp):
            return cost_vag_folded(xp, RMat_data, dt, T_seq, config.Jmax, M,
                                   n_pulses1=n1p, n_base_steps=n_base_steps)

    # Optimization. The pads are appended/stripped here in the wrapper; the
    # discarded gradient components belong to the frozen identity pads.
    def fun_wrapper(x):
        xp = jnp.asarray(np.concatenate([x[:n1], pad1, x[n1:], pad2]))
        v, g = vag(xp)
        g = np.asarray(g)
        return float(v), np.concatenate([g[:n1], g[n1p:n1p + n2]])

    if seed_seq is not None:
        d1 = pulse_times_to_delays(seed_seq[0])
        d2 = pulse_times_to_delays(seed_seq[1])
        initial_params = jnp.concatenate([d1, d2])
    else:
        d1 = get_random_delays(n1, T_seq, config.min_sep)
        d2 = get_random_delays(n2, T_seq, config.min_sep)
        initial_params = jnp.concatenate([d1, d2])

    bounds = [(config.min_sep, T_seq) for _ in range(len(initial_params))]
    # scipy.optimize.LinearConstraint(A, lb, ub) constrains lb <= A @ x <= ub
    # for the optimization vector x. Here A just sums qubit 1's delays (row
    # 0) and qubit 2's delays (row 1) separately, so this enforces "each
    # qubit's own delays must sum to at most T_seq - min_sep" -- i.e. leave
    # room for that qubit's own final (implicit) delay back to T_seq without
    # violating the minimum pulse separation.
    A = np.zeros((2, n1 + n2))
    A[0, :n1] = 1
    A[1, n1:] = 1
    linear_cons = scipy.optimize.LinearConstraint(A, -np.inf,
                                                  T_seq - config.min_sep)
    
    try:
        res = scipy.optimize.minimize(fun_wrapper, np.array(initial_params), method='SLSQP',
                                      bounds=bounds, constraints=linear_cons, jac=True,
                                      tol=SLSQP_TOL,
                                      options={'maxiter': SLSQP_MAXITER, 'disp': False})
        
        d1_opt = jnp.array(res.x[:n1])
        d2_opt = jnp.array(res.x[n1:])
        pt1 = delays_to_pulse_times(d1_opt, T_seq)
        pt2 = delays_to_pulse_times(d2_opt, T_seq)
        
        # Check if the optimized sequence can actually perform the gate
        pt12 = make_tk12(pt1, pt2)
        diffs = jnp.diff(pt12)
        signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
        dc_12 = jnp.sum(diffs * signs)
        if config.Jmax * M * jnp.abs(dc_12) < jnp.pi * 0.25:
             return None, 1.0
             
        return (pt1, pt2), res.fun
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, 1.0

def evaluate_known_sequences_with_T(config, M, T_seq, pLib):
    """Evaluates every sequence in the known-sequence library `pLib`
    (built by `construct_pulse_library`) at this (M, T_seq) against the
    CHARACTERIZED spectra (`config.SMat` -- the blind-optimizer's view, NOT
    the ground truth), each with its own optimal coupling J, and returns the
    single best (lowest-infidelity) entry.

    Parameters
    ----------
    config : CZOptConfig
    M : int
        Number of sequence repetitions.
    T_seq : float
        Gate/block time being evaluated.
    pLib : list of (jnp.ndarray, jnp.ndarray)
        Candidate (qubit-1 delays, qubit-2 delays) pairs, as returned by
        `construct_pulse_library`.

    Returns
    -------
    best_seq : tuple of jnp.ndarray or None
        The winning (pt1, pt2) ORIGINAL (unpadded) pulse times, or None if
        no library entry could reach the required CZ phase at this Jmax.
    best_inf : float
        That winner's infidelity on the characterized spectra (1.0 if none
        found).
    best_idx : int
        Its index into `pLib` (and hence into the matching description
        list), or -1 if none found.
    """
    best_inf = 1.0
    best_seq = None
    best_idx = -1

    # Setup Evaluation Method
    use_comb = use_comb_approximation(M, T_seq)

    if use_comb:
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        omega_k = jnp.arange(1, max_k + 1) * w0
        S_packed = _pack_comb(config.SMat, config.w, omega_k)

        def overlap_fn(pt_a, pt_b, r, c):
            return evaluate_overlap_comb(pt_a, pt_b, S_packed[r, c],
                                         omega_k, T_seq, M)
    else:
        RMat_data, dt, n_base_steps = prepare_time_domain_overlap(
            config.SMat, config.w, config.tau, T_seq, M
        )

        def overlap_fn(pt_a, pt_b, r, c):
            return evaluate_overlap_folded(pt_a, pt_b, RMat_data[r, c],
                                           dt, n_base_steps)

    # Shape-unify the library with exact-identity padding (control.padding):
    # entries collapse to <= 2 shapes per qubit (parity classes), so the
    # evaluator compiles a handful of programs instead of one per distinct
    # CDD/mqCDD length. The padded arrays are used for EVALUATION only; the
    # recorded winner keeps its original (unpadded) pulse times.
    targets1 = pad_targets([np.asarray(d1).shape[0] for d1, _ in pLib])
    targets2 = pad_targets([np.asarray(d2).shape[0] for _, d2 in pLib])

    for i, (d1, d2) in enumerate(pLib):
        d1p = pad_delays(d1, pad_count(np.asarray(d1).shape[0], targets1))
        d2p = pad_delays(d2, pad_count(np.asarray(d2).shape[0], targets2))
        pt1 = delays_to_pulse_times(d1p, T_seq)
        pt2 = delays_to_pulse_times(d2p, T_seq)
        pt12 = make_tk12(pt1, pt2)
        pt0 = jnp.array([0., T_seq])

        pts = [pt0, pt1, pt2, pt12]

        vals = []
        for r in range(4):
            row_vals = []
            for c in range(4):
                val = overlap_fn(pts[r], pts[c], r, c)
                row_vals.append(val)
            vals.append(row_vals)
        I_mat = jnp.array(vals)

        # J optimization
        diffs = jnp.diff(pt12)
        signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
        dc_12 = jnp.sum(diffs * signs)
        
        # Filter out sequences that cannot achieve the required phase with Jmax
        # This excludes decoupling sequences (like mqCDD) that average interaction to zero
        if config.Jmax * M * jnp.abs(dc_12) < jnp.pi * 0.25:
            continue

        safe_dc = jnp.where(jnp.abs(dc_12) > 1e-15, dc_12, jnp.sign(dc_12 + 1e-30) * 1e-15)
        J_target = jnp.pi * 0.25 / (M * safe_dc)
        J = jnp.clip(J_target, -config.Jmax, config.Jmax)
        
        fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
        inf = 1.0 - fid

        if inf < best_inf:
            best_inf = inf
            # Record the ORIGINAL (unpadded) sequence -- padding is an
            # evaluation-shape detail, not part of the winner.
            best_seq = (delays_to_pulse_times(d1, T_seq),
                        delays_to_pulse_times(d2, T_seq))
            best_idx = i

    return best_seq, best_inf, best_idx

def calculate_infidelity(seq, config, M, T_seq, use_ideal=False):
    """Computes the CZ-gate infidelity (1 - average gate fidelity) of a
    single fixed pulse sequence `seq`, at a given gate time and repeat
    count, against either the characterized (blind) or ideal (ground-truth)
    spectra. PROTECTED: called externally as `czmod.calculate_infidelity`
    from `scripts/harvest_design_numbers.py` and `scripts/run_margin_band.py`
    to re-score already-chosen winner sequences (e.g. under
    reconstruction-uncertainty-perturbed spectra for the margin bands, or
    under a restricted spectral model for the knowledge-ladder rungs) --
    do not change this function's call signature.

    Parameters
    ----------
    seq : (jnp.ndarray, jnp.ndarray) or None
        The (pt1, pt2) absolute pulse times to score; None (e.g. "no
        feasible sequence found") short-circuits to the worst-case
        infidelity of 1.0.
    config : CZOptConfig
    M : int
        Number of sequence repetitions.
    T_seq : float
        Gate/block time this sequence is being evaluated at.
    use_ideal : bool, optional
        False (default): score against `config.SMat`, the characterized/
        blind-search spectra -- this is what the optimizer itself uses
        internally to compare candidates. True: score against
        `config.SMat_ideal`, the ground-truth analytic spectra -- this is
        the PUBLISHED-NUMBER path (the true infidelity of whichever
        sequence the blind search picked), and always uses the exact
        time-domain evaluator regardless of `use_comb_approximation`, so
        quoted numbers never carry the comb approximation's few-percent
        error; the comb evaluator remains a search-side-only speed
        optimization (OPT-COMB-M16 hardening: this tag also marks the
        matching guard in control/idle.py).

    Returns
    -------
    float
        The gate infidelity (0 = perfect, 1 = worst case / infeasible
        sequence for the given Jmax).
    """
    if seq is None: return 1.0

    SMat = config.SMat_ideal if use_ideal else config.SMat
    w_grid = config.w_ideal if use_ideal else config.w

    use_comb = (not use_ideal) and use_comb_approximation(M, T_seq)
    
    if use_comb:
        w0 = 2 * jnp.pi / T_seq
        max_k = int(w_grid[-1] / w0)
        omega_k = jnp.arange(1, max_k + 1) * w0
        RMat_data = _pack_comb(SMat, w_grid, omega_k)

        overlap_fn = lambda pt_a, pt_b, data: evaluate_overlap_comb(pt_a, pt_b, data, omega_k, T_seq, M)

    else:
        RMat_data, dt, n_base_steps = prepare_time_domain_overlap(
            SMat, w_grid, config.tau, T_seq, M
        )

        overlap_fn = lambda pt_a, pt_b, data: evaluate_overlap_folded(pt_a, pt_b, data, dt, n_base_steps)

    pt1, pt2 = seq
    pt12 = make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    pts = [pt0, pt1, pt2, pt12]
    
    vals = []
    for i in range(4):
        row_vals = []
        for j in range(4):
            val = overlap_fn(pts[i], pts[j], RMat_data[i, j])
            row_vals.append(val)
        vals.append(row_vals)
    I_mat = jnp.array(vals)
    
    diffs = jnp.diff(pt12)
    signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
    dc_12 = jnp.sum(diffs * signs)

    # Guard: return max infidelity for infeasible sequences
    if config.Jmax * M * jnp.abs(dc_12) < jnp.pi * 0.25:
        return 1.0

    safe_dc = jnp.where(jnp.abs(dc_12) > 1e-15, dc_12, jnp.sign(dc_12 + 1e-30) * 1e-15)
    J_target = jnp.pi * 0.25 / (M * safe_dc)
    J = jnp.clip(J_target, -config.Jmax, config.Jmax)

    fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
    return 1.0 - fid

# ==============================================================================
# Visualization
# ==============================================================================
# (Empty despite the header: no plotting code actually lives in this file.
# Plots are made downstream, by qns2q.viz and scripts/report_showcase_figs.py,
# from the .npz files run_optimization() below writes out.)

# ==============================================================================
# Main Execution
# ==============================================================================

def run_optimization(config):
    """Top-level driver for the CZ-gate search: for every gate time in
    `config.gate_time_factors`, finds the best-performing sequence from (1)
    the known CDD/mqCDD library and (2) a free-timing (NT) gradient search
    over several candidate pulse counts, both scored BLIND (against
    `config.SMat`) but then reported at their TRUE, ideal-spectra infidelity
    (`config.SMat_ideal`) -- and saves the resulting infidelity-vs-gate-time
    curves plus the full per-gate-time winner sequences to the run folder's
    `infs_known_cz_v2*.npz` / `infs_opt_cz_v2*.npz` / `plotting_data/
    plotting_data_cz_v2*.npz` (read downstream by scripts/report_showcase_figs.py,
    scripts/harvest_design_numbers.py and scripts/run_margin_band.py). This is
    the function `__main__` below calls once argument parsing is done."""
    # Pin the RNG so the random pulse-count selection and delay seeding are
    # reproducible (SEED-OPT: this tag also marks the matching seed pin in
    # control/idle.py). Without this the headline CZ infidelities and
    # winning sequences drift between runs (np.random is otherwise only
    # seeded by whatever the OS/interpreter did last, so re-running this
    # function twice in the same process, or in two different processes,
    # could silently pick different "random" pulse-count restarts and
    # report a different number for the same physical setup).
    np.random.seed(RANDOM_SEED)
    _OVERLAP_SETUP_CACHE.clear()

    yaxis_opt, xaxis_opt = [], []
    yaxis_known, xaxis_known = [], []
    yaxis_nopulse = []
    # Per-gate-time winner sequences/labels (the margin-band tool re-evaluates
    # these fixed winners under recon-uncertainty-perturbed spectra).
    labels_known, labels_opt = [], []
    sequences_known, sequences_opt = [], []

    print(f"Running CZ Optimization (v2) [seed={RANDOM_SEED}]...")
    
    best_opt_seq_overall = None
    best_opt_inf_overall = 1.0
    best_known_seq_overall = None
    best_known_inf_overall = 1.0
    T_seq_best_opt = None
    T_seq_best_known = None
    
    for i in config.gate_time_factors:
        Tg = config.Tqns / 2 ** (i - 1)
        if Tg < config.tau:
            continue
            
        print(f"\nGate Time: {Tg:.1f} tau (Tg/T2q1={Tg/config.T2q1:.4f}, Tg/T2q2={Tg/config.T2q2:.4f})")

        # For now, M=1
        M = 1
        T_seq = Tg / M

        # No Pulse Calculation
        pt_nopulse = jnp.array([0., T_seq])
        seq_nopulse = (pt_nopulse, pt_nopulse)
        inf_nopulse = calculate_infidelity(seq_nopulse, config, M, T_seq, use_ideal=True)
        yaxis_nopulse.append(inf_nopulse)
        print(f"  No Pulse (Ideal): {inf_nopulse:.6e}")
        
        # 1. Known Sequences
        max_pulses_per_rep = config.max_pulses
        pLib_delays, pLib_desc = construct_pulse_library(T_seq, config.min_sep, max_pulses_per_rep)
        
        best_known_seq = None
        best_known_inf = 1.0
        
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences_with_T(config, M, T_seq, pLib_delays)
            label_k = pLib_desc[idx]
            print(f"  Best Known (Char): {best_known_inf:.6e} ({label_k})")

            # Recalculate with ideal
            best_known_inf_ideal = calculate_infidelity(best_known_seq, config, M, T_seq, use_ideal=True)
            print(f"  Best Known (Ideal): {best_known_inf_ideal:.6e}")

            if best_known_inf_ideal < best_known_inf_overall:
                best_known_inf_overall = best_known_inf_ideal
                best_known_seq_overall = best_known_seq
                T_seq_best_known = T_seq
        else:
            best_known_inf_ideal = 1.0
            label_k = "N/A"

        yaxis_known.append(best_known_inf_ideal)
        xaxis_known.append(Tg)
        labels_known.append(label_k)
        sequences_known.append(best_known_seq)
        
        # 2. Random Optimization
        best_opt_inf = 1.0
        best_opt_seq = None
        
        # Calculate nps dynamically
        max_n_physical = int(T_seq / config.min_sep) - 1
        
        # Determine candidates based on physical limit and config limit
        # We back off by 1 from physical max to ensure optimization slack (unless max_n is small)
        upper_bound_physical = max(1, max_n_physical - 1)
        effective_max = min(config.max_pulses, upper_bound_physical)
        
        # Ensure we don't exceed hard physical limit
        if effective_max > max_n_physical:
            effective_max = max_n_physical
            
        n_candidates_set = set()
        if effective_max > 0:
            n_candidates_set.add(effective_max)
            if effective_max > 1:
                n_candidates_set.add(effective_max - 1)
            if effective_max > 2:
                n_candidates_set.add(effective_max - 2)

        # Also search around half the number of allowed pulses
        half_max = effective_max // 2
        if half_max > 0:
            n_candidates_set.add(half_max)
            if half_max > 1:
                n_candidates_set.add(half_max - 1)
            if half_max + 1 < effective_max:
                n_candidates_set.add(half_max + 1)

        if config.informed_counts and effective_max >= 2:
            # Rank every feasible uniform-train count by the characterized
            # spectra at its fundamental w = pi*n/T_seq (first-order window
            # proxy; S11 + S22 dominate the self error). Take the best
            # windows, with a min separation so the picks don't cluster, and
            # add +-1 neighbors for the free-timing search to refine.
            ns = np.arange(2, effective_max + 1)
            w_fund = jnp.asarray(np.pi * ns / T_seq)
            proxy = np.asarray(
                jnp.interp(w_fund, config.w, jnp.real(config.SMat[1, 1]))
                + jnp.interp(w_fund, config.w, jnp.real(config.SMat[2, 2])))
            picked = []
            for idx in np.argsort(proxy):
                n_pick = int(ns[idx])
                if all(abs(n_pick - p) > 2 for p in picked):
                    picked.append(n_pick)
                if len(picked) >= 3:
                    break
            informed_set = set()
            for p in list(picked):
                informed_set.update(
                    q for q in (p - 1, p, p + 1) if 1 <= q <= effective_max)
            n_candidates_set |= informed_set
            print(f"  Spectrum-informed windows (characterized): {sorted(picked)}")
        else:
            informed_set = set()

        n_candidates = sorted(list(n_candidates_set), reverse=True)

        if not n_candidates:
             print(f"    Skipping: T_seq too small for pulses")
             continue

        print(f"  Auto-selected pulse counts: {n_candidates}")

        # One compiled cost per parity class for the whole (n1, n2) sweep
        # (control.padding shape unification).
        opt_pad_targets = pad_targets(n_candidates)

        for n1 in n_candidates:
            for n2 in n_candidates:
                # Check feasibility (redundant with max_n logic but safe)
                if (n1 + 1) * config.min_sep > T_seq or (n2 + 1) * config.min_sep > T_seq:
                     continue
                if config.pair_gap > 0 and abs(n1 - n2) > config.pair_gap:
                     continue

                print(f"  Optimizing n=({n1}, {n2})...")
                # Spectrum-informed window pairs start from the EQUIDISTANT
                # (uniform-train) configuration -- the solution class the
                # window ranking is valid for; SLSQP then refines it. A random
                # init in a featured landscape routinely converges to a local
                # optimum ABOVE the plain uniform train (probe 4: 4.2e-4 vs
                # the CPMG-28 bound 2.6e-4). All other pairs keep the random
                # exploration init.
                seed = None
                if n1 in informed_set and n2 in informed_set:
                    seed = (delays_to_pulse_times(
                                jnp.full(n1, T_seq / (n1 + 1)), T_seq),
                            delays_to_pulse_times(
                                jnp.full(n2, T_seq / (n2 + 1)), T_seq))
                seq, inf = optimize_sequence(config, M, T_seq, n1, n2,
                                             seed_seq=seed,
                                             pad_to=(opt_pad_targets[n1 % 2],
                                                     opt_pad_targets[n2 % 2]))
                if seq is not None and inf < best_opt_inf:
                    best_opt_inf = inf
                    best_opt_seq = seq
                    print(f"    New Best Opt: {inf:.6e}")


        # Recalculate best opt with ideal
        if best_opt_seq is not None:
             best_opt_inf_ideal = calculate_infidelity(best_opt_seq, config, M, T_seq, use_ideal=True)
             print(f"  Best Opt (Ideal): {best_opt_inf_ideal:.6e}")
             
             if best_opt_inf_ideal < best_opt_inf_overall:
                 best_opt_inf_overall = best_opt_inf_ideal
                 best_opt_seq_overall = best_opt_seq
                 T_seq_best_opt = T_seq
        else:
             best_opt_inf_ideal = 1.0
             
        # Compare known vs opt (using characterized inf for selection). The
        # recorded per-Tg "opt" sequence is the blind winner the curve point
        # represents (falls back to the known sequence when that won blind).
        if best_known_inf < best_opt_inf:
             print(f"  Known sequence was better (on characterized spectrum).")
             final_inf = best_known_inf_ideal # Use ideal for plot
             final_seq, final_label = best_known_seq, label_k
        else:
             final_inf = best_opt_inf_ideal # Use ideal for plot
             final_seq = best_opt_seq
             if best_opt_seq is not None:
                 final_label = (f"NT({len(best_opt_seq[0]) - 2},"
                                f"{len(best_opt_seq[1]) - 2})")
             else:
                 final_label = "N/A"

        yaxis_opt.append(final_inf)
        xaxis_opt.append(Tg)
        labels_opt.append(final_label)
        sequences_opt.append(final_seq)

    # Save all plotting data
    min_gate_time = np.pi / (4 * config.Jmax)
    
    # Create a dedicated directory for plotting data
    plotting_dir = os.path.join(config.path, "plotting_data")
    os.makedirs(plotting_dir, exist_ok=True)
    
    save_dict = {
        'taxis': np.array(xaxis_known),
        'infs_known': np.array(yaxis_known),
        'infs_opt': np.array(yaxis_opt),
        'infs_nopulse': np.array(yaxis_nopulse),
        'tau': config.tau,
        'min_gate_time': min_gate_time,
        'seed': RANDOM_SEED,
        # Frequency grid kept for reference; the full-resolution SMat is omitted
        # (the plot scripts recompute spectra from the analytic noise model
        # (qns2q.noise.spectra), so saving the 4x4x20000 matrix here is ~5 MB
        # of dead weight per file).
        'w': np.array(config.w),
        'w_max': float(config.w_max),
        'M': M,
        'Tg': T_seq_best_known if T_seq_best_known is not None else T_seq_best_opt,
        'gate_type': 'cz',
        # OPT-PROVENANCE: noise-model version the input spectra were generated
        # under (the viz overlays warn when it differs from the current model).
        'model_version': config.model_version,
        'spectral_model': config.spectral_model,
        # UNCAP-0611 provenance: the pulse-count cap this run searched under
        # (10**9 = separation-limited).
        'max_pulses': int(config.max_pulses),
        # SHOWCASE-0612 provenance: control-bandwidth scenario + ablation rung
        'min_sep_factor': float(config.min_sep_factor),
        'char_self_only': bool(config.char_self_only),
        'informed_counts': bool(config.informed_counts),
        'pair_gap': int(config.pair_gap),
    }

    if best_known_seq_overall is not None:
        save_dict['best_known_seq_pt1'] = np.array(best_known_seq_overall[0])
        save_dict['best_known_seq_pt2'] = np.array(best_known_seq_overall[1])
        save_dict['T_seq_best_known'] = T_seq_best_known

    if best_opt_seq_overall is not None:
        save_dict['best_opt_seq_pt1'] = np.array(best_opt_seq_overall[0])
        save_dict['best_opt_seq_pt2'] = np.array(best_opt_seq_overall[1])
        save_dict['T_seq_best_opt'] = T_seq_best_opt

    # Per-Tg winners (object arrays, indexed like taxis; entries are
    # (pt1, pt2) tuples or None). Load with allow_pickle=True.
    def _seq_obj_array(seqs):
        out = np.empty(len(seqs), dtype=object)
        for k, s in enumerate(seqs):
            out[k] = None if s is None else (np.array(s[0]), np.array(s[1]))
        return out
    save_dict['labels_known'] = np.array(labels_known)
    save_dict['labels_opt'] = np.array(labels_opt)
    save_dict['sequences_known'] = _seq_obj_array(sequences_known)
    save_dict['sequences_opt'] = _seq_obj_array(sequences_opt)

    np.savez(os.path.join(plotting_dir, config.plot_data_name), **save_dict)
    print(f"Saved all plotting data to {os.path.join(plotting_dir, config.plot_data_name)}")

    np.savez(os.path.join(config.path, config.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))
    np.savez(os.path.join(config.path, config.output_path_known), infs_known=np.array(yaxis_known),
             taxis=np.array(xaxis_known))

    print(f"\nTo generate plots, run:\n  python plot_optimization.py --data-dir {config.path} --gate-type cz")

if __name__ == '__main__':
    import argparse

    # Defaults match the manuscript text (§V.C and the fig:infidelity_vs_time /
    # tab:fidelity_summary captions): the CZ optimization runs on the SPAM-free
    # reconstructed spectra (specs.npz) of the active regime's NoSPAM folder.
    # --protocol redirects `fname` to one of the SPAM-arm run folders instead
    # (DraftRun_SPAM_<regime>_<protocol>, built by qns2q.paths.run_folder --
    # OPT-ARM-PLUMBING: this tag marks every place, here and in
    # control/idle.py, that wires --folder/--protocol through to the actual
    # run-folder path); --simulated instead optimizes directly on the
    # ground-truth file (simulated_spectra.npz), used as the acceptance-gate
    # sanity probe (does the optimizer do the right thing when handed the
    # true spectra with no reconstruction noise at all).
    parser = argparse.ArgumentParser(description="CZ pulse-sequence optimization")
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--folder', help="run-folder name under the repo root "
                     "(default: the active regime's NoSPAM folder)")
    src.add_argument('--protocol',
                     choices=('reference', 'raw', 'mitigated', 'robust'),
                     help="read the SPAM arm DraftRun_SPAM_<regime>_<protocol>")
    parser.add_argument('--simulated', action='store_true',
                        help="optimize on simulated_spectra.npz (ground truth) "
                             "instead of the reconstructed specs.npz")
    parser.add_argument('--no-cross', action='store_true',
                        help="counterfactual: drop the cross-spectra from BOTH "
                             "the characterized and ideal gate models (channels "
                             "a protocol cannot reconstruct, e.g. robust "
                             "S112/S212, are auto-dropped from the "
                             "characterized model alone)")
    parser.add_argument('--spectral-model',
                        choices=('interp', 'selfconsistent'),
                        default='interp',
                        help="characterized-SMat construction: linear interp "
                             "through the teeth (+tails); or the unfold "
                             "model's line/tail/head-aware spectra "
                             "(OPT-SPECTRAL-MODEL)")
    parser.add_argument('--max-pulses', type=int, default=150,
                        help="per-qubit pulse-count cap (default 150, the "
                             "published-run value); 0 = separation-limited, "
                             "i.e. only the minimum separation bounds the "
                             "count (UNCAP-0611)")
    parser.add_argument('--min-sep', type=float, default=1.0,
                        help="minimum pulse separation in units of tau "
                             "(SHOWCASE-0612 control-bandwidth scenario; "
                             "applied symmetrically to the library, the NT "
                             "search and the inits; default 1.0 = legacy)")
    parser.add_argument('--self-only', action='store_true',
                        help="ablation rung (c): characterized model from "
                             "S11/S22 alone (S1212 + crosses dropped, as a "
                             "1Q-only QNS campaign leaves them); the ideal "
                             "benchmark keeps the full truth")
    parser.add_argument('--informed-counts', action='store_true',
                        help="add spectrum-informed pulse-count candidates "
                             "(top quiet windows of the CHARACTERIZED "
                             "spectra) to the legacy heuristic set "
                             "(SHOWCASE-0612)")
    parser.add_argument('--pair-gap', type=int, default=0,
                        help="NT search: skip (n1, n2) pairs with "
                             "|n1 - n2| > this (0 = full grid)")
    parser.add_argument('--factors', type=str, default=None,
                        help="comma-separated gate-time factors to run "
                             "(default: the full 3,2,1,0,-1,-2,-3 sweep); "
                             "Tg = T/2^(f-1)")
    parser.add_argument('--tag', type=str, default="",
                        help="suffix for all output files, so a rerun does "
                             "not overwrite the published outputs")
    cli = parser.parse_args()
    fname = cli.folder or (run_folder(spam=True, protocol=cli.protocol)
                           if cli.protocol else run_folder())
    sfx = f"_{cli.tag}" if cli.tag else ""
    kwargs = dict(fname=fname, use_simulated=cli.simulated,
                  include_cross_spectra=not cli.no_cross,
                  spectral_model=cli.spectral_model,
                  max_pulses=cli.max_pulses,
                  min_sep_factor=cli.min_sep,
                  char_self_only=cli.self_only,
                  informed_counts=cli.informed_counts,
                  pair_gap=cli.pair_gap,
                  output_path_known=f"infs_known_cz_v2{sfx}.npz",
                  output_path_opt=f"infs_opt_cz_v2{sfx}.npz",
                  plot_data_name=f"plotting_data_cz_v2{sfx}.npz")
    if cli.factors is not None:
        kwargs['gate_time_factors'] = [int(f) for f in cli.factors.split(',')]
    config = CZOptConfig(**kwargs)
    run_optimization(config)
