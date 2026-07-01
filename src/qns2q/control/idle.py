"""
Pulse Sequence Optimization for Two-Qubit Idling Gates (v4).

**Physics role / pipeline position.** This is Stage 3b of the QNS-2Q pipeline
(see repo-root CLAUDE.md), on the "control" arm: given the noise power spectral
densities already reconstructed by the "characterize" arm (Stage 1
``characterize/experiments.py`` -> Stage 2 ``characterize/reconstruct.py``),
it designs pulse sequences that hold the two-qubit + bath-qubit register idle
(the "identity gate" / quantum-memory operation, as opposed to an entangling
operation) for a target duration ``Tg`` while suppressing the dephasing that
the environment noise would otherwise accumulate. Physically this is
dynamical decoupling (DD): interleave pi-pulses so that the qubits'
sensitivity to (mostly slow, 1/f-like) Z-axis dephasing noise keeps flipping
sign and the accumulated phase partially cancels. Its sibling module,
``control/cz.py``, runs the same overlap-integral infidelity machinery for
the two-qubit CZ entangling gate instead -- read that file's docstring for
the CZ-specific parts; here the target operation is "do nothing" rather than
an entangling unitary, so the fidelity formula and pulse-library construction
differ from cz.py even though the surrounding optimization scaffolding is
close to identical.

Concretely this script:
1. Loads reconstructed (or, with ``--simulated``, ground-truth analytic)
   spectral noise data via ``Config``.
2. Constructs libraries of known pulse sequences (CDD, mqCDD) as a baseline.
3. Evaluates sequence performance using overlap integrals of the pulse
   sequence's "switching function" against the noise autocorrelation,
   computed either exactly in the time domain or via a frequency-comb
   shortcut (see ``use_comb_approximation``).
4. Optimizes free ("NT" = noise-tailored, i.e. not from a textbook family)
   pulse-timing sequences using JAX-based gradients feeding SciPy's SLSQP.
5. Efficiently handles M-fold repeated sequences via time-domain folding
   (``prepare_time_domain_overlap`` / ``evaluate_overlap_folded``) or the
   frequency-comb approximation (``evaluate_overlap_comb``), and sweeps M
   (number of base-sequence repetitions within Tg) when run as ``__main__``.

**Inputs.** ``Config`` (below) loads ``specs.npz`` + ``params.npz`` (Stage 2
output) or ``simulated_spectra.npz`` (the analytic ground truth written by
``python -m qns2q.noise.spectra``) from a run folder -- by default the active
regime's NoSPAM folder (``qns2q.paths.run_folder()``), or a SPAM arm via
``--protocol``. It also imports the analytic spectral functions directly from
``qns2q.noise.spectra`` to build the "ideal" ground-truth benchmark spectra
(``Config.SMat_ideal``) regardless of what characterization could actually
measure, plus helpers from ``qns2q.control.tails`` (extrapolating reconstructed
spectra beyond the QNS comb's last resolvable tooth) and
``qns2q.control.padding`` (see the note at ``pad_targets`` import, below).

**Outputs.** Per-M summary files ``infs_known_id_M<M>*.npz`` /
``infs_opt_id_M*.npz``, a combined ``plotting_data/plotting_data_id_v4*.npz``,
and (only when the whole M-sweep in ``__main__`` completes)
``optimization_data_all_M*.npz`` -- all written under the run folder. These
feed the manuscript's gate-comparison figure (``scripts/report_showcase_figs.py``,
see FIGURE_PROVENANCE.md) and are also read directly (as a Python module, not
just its saved files) by ``scripts/harvest_design_numbers.py`` and
``scripts/showcase_storage_panel.py``, which import this file as
``from qns2q.control import idle as idmod`` and call ``idmod.Config``,
``idmod.calculate_infidelity``, ``idmod.calculate_idling_fidelity``,
``idmod.prepare_time_domain_overlap``, ``idmod.evaluate_overlap_folded`` and
``idmod.make_tk12`` -- those six names are this module's protected external
surface; do not change their names or call signatures.

Run from the repo root as ``PYTHONPATH=src python -m qns2q.control.idle``
(see CLAUDE.md "Running the Pipeline", Stage 3b).

Author: [Q]
Date: [01/18/2026]
"""

import functools
import itertools
import os
import traceback

import jax
import jax.numpy as jnp
import jax.scipy.integrate
# JAX defaults to 32-bit floats; the noise spectra span many orders of
# magnitude and the optimizer differentiates through them, so 64-bit
# precision is required everywhere in this pipeline (silently losing it would
# show up as noisy/wrong gradients, not a crash).
jax.config.update("jax_enable_x64", True)
# OPT-SPEEDUPS (a): persistent XLA compilation cache (see cz.py for the fuller
# writeup). JAX normally has to re-compile (JIT: "just-in-time"-compile a
# Python function to fast device code) every function the first time it sees
# a given input shape, in every fresh process -- expensive (~1 s) and paid
# again on every rerun. Caching the compiled programs to disk under
# .jax_cache/ lets a rerun at the same array shapes just deserialize
# (~0.1 s) instead of recompiling; it is purely a wall-clock optimization and
# never changes what gets computed.
from qns2q.paths import project_root as _project_root
jax.config.update("jax_compilation_cache_dir",
                  os.path.join(_project_root(), ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import jax.scipy.signal
import numpy as np
import scipy.optimize

from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 MODEL_VERSION, line_priors)
from qns2q.control.tails import tail_extend_interp_complex
from qns2q.control.padding import pad_targets, pad_count, pad_delays
from qns2q.paths import run_folder, project_root

# Fixed RNG seed for the unseeded np.random restarts (random pulse counts and
# delay seeding). Pinning it makes the published idling infidelity curves and
# winning-sequence labels reproducible across re-runs. Recorded in the saved
# optimization data so every figure carries its provenance.
RANDOM_SEED = 20260608

# OPT-SPEEDUPS (d): SLSQP (Sequential Least Squares Programming, scipy's
# constrained gradient optimizer used below) convergence knobs. Tighter values
# (tol=1e-10, maxiter=1000) over-converge an objective built from spectra that
# already carry 5-20% reconstruction uncertainty -- squeezing extra digits out
# of noisy inputs, at real wall-clock cost, since each iteration is a ~2 ms
# device call dispatched from Python/SciPy. These looser knobs were checked to
# reproduce the pre-change (2026-06-11) winning sequences and infidelities
# (see cz.py, which shares this rationale).
SLSQP_TOL = 1e-7
SLSQP_MAXITER = 300

# Memoizes the (sequence-independent) folded-correlation setup built by
# prepare_time_domain_overlap. That setup depends only on the spectrum/grid
# arrays and (tau, T_seq, M) -- NOT on the pulse sequence. Cached tuples are
# returned verbatim (bit-identical), so this is a pure speed-up. Cleared at the
# start of each run_optimization_pipeline call (per M) to keep it bounded.
_OVERLAP_SETUP_CACHE = {}

# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """
    Configuration and data management for idling/DD pulse-sequence optimization.

    NOTE FOR READERS OF THIS CODEBASE: unlike every other pipeline-stage config
    in this repo (``CZOptConfig`` in ``control/cz.py``, ``QNSExperimentConfig``,
    ``SpectraReconConfig``, ...), this one is a plain Python class with a hand
    -written ``__init__``, NOT a ``@dataclass``. There is no functional
    difference that matters here -- both patterns just group named parameters
    with defaults -- but it means the parameter list/defaults live in the
    ``__init__`` signature below rather than in dataclass field declarations,
    and it does not get the free ``__repr__``/``field(default_factory=...)``
    conveniences a ``@dataclass`` would. This is a historical inconsistency,
    not a deliberate design choice -- if you are looking for where "M" or
    "max_pulses" defaults are set, look at the ``__init__`` signature, not a
    class body full of ``field(...)`` declarations.

    Constructing a ``Config`` does real disk I/O and can fail fast: like the
    dataclass configs elsewhere in this repo, ``__init__`` eagerly loads the
    upstream ``.npz`` files (``specs.npz``/``params.npz`` or
    ``simulated_spectra.npz``) for the given run folder and immediately builds
    the derived spectral matrices below, so a missing/incompatible prior stage
    is caught at construction time rather than deep inside an optimization
    loop.

    This class handles the loading of reconstructed spectral data, physical system
    parameters, and the configuration of the optimization engine. It constructs
    the interpolated spectral matrices (S-matrix) used for infidelity calculations:
    ``SMat`` is the model built from what the (possibly SPAM-limited, possibly
    finite-sample) characterization actually measured -- what the blind
    optimizer searches against -- while ``SMat_ideal`` is always built straight
    from the analytic noise model of ``qns2q.noise.spectra``, i.e. the "ground
    truth" used only to SCORE a chosen sequence's true infidelity, never to
    pick it (keeping the optimizer blind to information a real experiment
    would not have).

    Parameters
    ----------
    fname : str, optional
        Data folder name with results from Stage 1 & 2. Defaults to the active regime's run folder (qns2q.paths.run_folder()).
    include_cross_spectra : bool, optional
        Whether to include cross-correlated noise terms ($S_{12}, S_{1,12}$, etc.). Default is True.
    Tg : float, optional
        Target gate (idle/hold) time, in units of tau (the minimum pulse
        separation; see repo CLAUDE.md "Units: tau = 1") -- NOT seconds,
        despite historically being documented that way here.
    tau_divisor : int, optional
        Divisor of the QNS time $T$ to determine the minimum pulse interval $\tau$. Default is 160.
    M : int, optional
        Number of sequence repetitions (blocks) tiling the gate time Tg (each
        block has duration T_seq = Tg / M); the ``__main__`` sweep below tries
        M = 1, 2, 4, ..., 128 and keeps whichever gives the lowest true
        infidelity. Default is 1.
    max_pulses : int, optional
        Maximum allowed pulses across all repetitions (0 = uncapped, limited
        only by the minimum pulse separation; see the UNCAP-0611 comment
        below). Default is 100.
    num_random_trials : int, optional
        Number of random sequence initializations for optimization. Default is 10.
    use_known_as_seed : bool, optional
        If True, use the best known sequence from the library as an optimization seed. Default is False.
    output_path_known : str, optional
        Filename for saving known sequence results.
    output_path_opt : str, optional
        Filename for saving optimized sequence results.
    plot_filename : str, optional
        Filename for the infidelity vs gate time plot.
    reps_known, reps_opt : list of int, optional
        Legacy/unused parameters (kept only so old call sites do not break;
        see the "Legacy/Unused" comment where they are stored in __init__).
    use_simulated : bool, optional
        If True, load simulated target spectra instead of reconstructed ones. Default is False.
    gate_time_factors : list of int, optional
        Powers of 2 to scale the gate time relative to $T_{qns}$.
    spectral_model : {'interp', 'selfconsistent'}, optional
        How the characterized ``SMat`` is built from the reconstructed comb:
        'interp' linearly interpolates through the measured teeth (+ tail
        extrapolation past the last one); 'selfconsistent' instead uses the
        line/tail/head-aware model from
        ``characterize.systematics.selfconsistent_spectra`` (OPT-SPECTRAL-MODEL,
        see ``_build_interpolated_spectra`` below). Default 'interp'.
    max_dim : int, optional
        SLSQP tractability guard: when > 0, clip any single noise-tailored
        (NT) search trial to at most this many optimization variables
        (n1 + n2 pulse-timing parameters combined); 0 = no guard. See
        UNCAP-0611 below.
    min_sep_factor : float, optional
        Minimum pulse separation as a multiple of tau (a finite-control-
        bandwidth scenario: > 1.0 models pi-pulses that are not
        instantaneous); 1.0 = legacy/idealized instantaneous pulses. Default
        1.0.
    char_self_only : bool, optional
        Ablation switch: if True, build the characterized model from the two
        single-qubit self-spectra only (drop S1212 and every cross-spectrum,
        as a 1-qubit-only QNS campaign would have to), while ``SMat_ideal``
        still keeps the full truth -- this prices what the two-qubit part of
        the reconstruction is worth. Default False.
    informed_counts : bool, optional
        If True, add a few pulse-count candidates chosen by looking at where
        the characterized self-spectra are quietest (rather than relying
        purely on random sampling) to the NT search's trial list. Default
        False.
    plot_data_name : str, optional
        Filename for the consolidated per-M plotting-data npz.
    """
    def __init__(self,
                 fname=None,
                 include_cross_spectra=True,
                 Tg=2240.0,   # tau units (= 4*14us at the legacy tau=25ns anchor)
                 tau_divisor=160,
                 M=1,
                 max_pulses=100,
                 num_random_trials=10,
                 use_known_as_seed=False,
                 output_path_known="infs_known_id.npz",
                 output_path_opt="infs_opt_id.npz",
                 plot_filename="infs_GateTime_id.pdf",
                 reps_known=None,
                 reps_opt=None,
                 use_simulated=False,
                 gate_time_factors=None,
                 spectral_model="interp",
                 max_dim=0,
                 min_sep_factor=1.0,
                 char_self_only=False,
                 informed_counts=False,
                 plot_data_name="plotting_data_id_v4.npz"
                 ):

        """
        Initialize configuration. See the class docstring above for what each
        parameter means; this constructor does real file I/O (loads the
        upstream .npz files and builds the spectral matrices) rather than
        just storing values.
        """
        # Paths
        if fname is None:
            fname = run_folder()
        self.path = os.path.join(project_root(), fname)
        
        if not os.path.exists(self.path):
             raise FileNotFoundError(f"Data directory not found at {self.path}")

        # Load Data
        if use_simulated:
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
                 print(f"[idle] NOTE: flagged (undetermined) DC points in specs: "
                       f"{', '.join(k[:-len('_dc_ok')] for k in bad_dc)}")
        self.use_simulated = use_simulated
        # 'interp' = linear interpolation through the comb teeth (+ tails);
        # 'selfconsistent' = the unfold model's line/tail/head-aware spectra
        # (OPT-SPECTRAL-MODEL: see _build_interpolated_spectra below for what
        # that alternate model actually does).
        self.spectral_model = spectral_model

        # OPT-PROVENANCE: spectra generated under a different noise model than
        # the current one make the ideal benchmark (SMat_ideal, built from the
        # CURRENT model) a mixed-model comparison -- the trap behind
        # CA-REPRO-NUMBERS. Concretely: if specs.npz was produced by an older
        # version of qns2q.noise.spectra (e.g. before a calibration constant
        # changed), comparing its "characterized" infidelity against an
        # "ideal" benchmark built from TODAY's analytic model silently mixes
        # two different noise models -- the reported gap would reflect a
        # model change, not reconstruction quality. The version stamps are
        # compared below and a mismatch prints a loud warning rather than
        # failing silently.
        mv = None
        for src_ in (self.specs, self.params):
            if 'model_version' in src_:
                mv = str(src_['model_version'])
                break
        self.model_version = mv if mv is not None else 'unknown'
        if self.model_version != MODEL_VERSION:
            print(f"[idle] WARNING: spectra model_version={self.model_version!r} "
                  f"!= current noise model {MODEL_VERSION!r}; the ideal "
                  f"benchmark uses the CURRENT model -- regenerate Stage 1/2.")

        # System Parameters
        self.Tqns = jnp.float64(self.params['T'])
        # Units guard: tau-unit data has T = 160; SI-era data has T ~ 4e-6 s.
        if float(self.Tqns) < 1.0:
            print(f"[idle] WARNING: loaded T={float(self.Tqns):g} looks like "
                  f"SI-era data (expected tau-unit T ~ 160). Regenerate the "
                  f"spectra for this folder before trusting the optimization.")
        self.mc = int(self.params['truncate'])
        self.include_cross_spectra = include_cross_spectra
        
        # Optimization Parameters
        self.Tg = Tg
        self.tau_divisor = tau_divisor
        self.tau = self.Tqns / tau_divisor
        # Control-bandwidth scenario (SHOWCASE-0612): minimum pulse separation
        # in units of tau, applied symmetrically to the library, the NT search
        # and the inits. 1.0 = legacy (separation = tau). The time-domain
        # overlap resolution stays keyed to tau. Physically, min_sep_factor > 1
        # models the fact that a real pi-pulse takes finite time to apply (it
        # is not an instantaneous kick), so two pulses cannot be scheduled
        # arbitrarily close together; raising this value shrinks how many
        # pulses can fit in a given hold time, which is the whole point of
        # this "reduced control bandwidth" scenario.
        if min_sep_factor < 1.0:
            raise ValueError("min_sep_factor < 1: the minimum separation "
                             "cannot undercut the time unit tau")
        self.min_sep_factor = min_sep_factor
        self.min_sep = min_sep_factor * self.tau
        if min_sep_factor != 1.0:
            print(f"[idle] control-bandwidth scenario: min pulse separation = "
                  f"{min_sep_factor:g} tau")
        # Ablation rung (c) (SHOWCASE-0612): characterized model from the
        # single-qubit spectra alone; the ideal benchmark keeps full truth.
        # This asks "how much would the gate design suffer if the experiment
        # had only ever run single-qubit QNS, and never learned the two-qubit
        # ZZ spectrum or any cross-correlation?" -- one rung of the paper's
        # "knowledge ladder" of what characterization buys you.
        self.char_self_only = char_self_only
        # Spectrum-informed pulse-count candidates (SHOWCASE-0612; see cz.py
        # for the fuller rationale): besides trying random pulse counts, also
        # deliberately try a few pulse counts whose fundamental frequency
        # lands in a quiet part of the CHARACTERIZED spectrum -- a cheap,
        # physically-motivated way to make sure the random search does not
        # miss an obviously-good window purely by bad luck.
        self.informed_counts = informed_counts

        self.M = M
        if max_pulses == 0:
            max_pulses = 10**9
            print("[idle] max_pulses=0: pulse count limited only by the "
                  "minimum separation tau (n <= T_seq/tau - 1 per qubit "
                  "per repetition)")
        self.max_pulses = max_pulses
        # SLSQP tractability guard (UNCAP-0611): when > 0, clip the random
        # NT search so a single trial never exceeds max_dim optimization
        # variables (n1 + n2). Clipped blocks are announced in the log --
        # never a silent cap. Why this exists: max_pulses=0 above intentionally
        # removes the historical pulse-count cap so the search is limited only
        # by physics (the minimum pulse separation), but SLSQP's per-iteration
        # cost grows with the number of free variables (n1 + n2); at very
        # short T_seq / long M this "uncapped" count can get large enough that
        # a single optimization restart becomes impractically slow. max_dim
        # gives a way to bound that cost again without silently reintroducing
        # a hidden pulse-count limit -- any clip is printed.
        self.max_dim = max_dim
        self.plot_data_name = plot_data_name
        self.num_random_trials = num_random_trials
        self.use_known_as_seed = use_known_as_seed
        
        self.output_path_known = output_path_known
        self.output_path_opt = output_path_opt
        self.plot_filename = plot_filename
        
        # Extended gate time factors to include larger gate times
        # Factors are powers of 2 divisor of Tqns.
        # Tqns is typically 160 tau.
        # Factor -1 -> Tg = Tqns * 2
        # Factor -2 -> Tg = Tqns * 4
        # Factor -3 -> Tg = Tqns * 8
        # Factor -4 -> Tg = Tqns * 16
        self.gate_time_factors = gate_time_factors if gate_time_factors is not None else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        
        # Derived Parameters
        self.T_seq = self.Tg / self.M
        self.max_pulses_per_rep = int(self.max_pulses / self.M)
        
        # Legacy/Unused
        self.reps_known = reps_known if reps_known is not None else [i for i in range(100, 401, 10)]
        self.reps_opt = reps_opt if reps_opt is not None else [i for i in range(100, 401, 20)]
        
        # Frequency Grids
        # Start from 0 to include DC component
        # Extend frequency range to ensure fine time resolution for correlation
        w_max_sys = 2 * jnp.pi * self.mc / self.Tqns
        # Reach the pulse-spacing Nyquist pi/tau (4*w_max_sys for the
        # T=160/truncate=20 comb): optimized sequences put filter weight above
        # the comb's last tooth; a shorter grid silently ignored that band.
        self.w_max = 8 * w_max_sys
        self.N_w = 20000
        
        self.w = jnp.linspace(0, self.w_max, self.N_w)
        self.w_ideal = jnp.linspace(0, 2 * self.w_max, 2 * self.N_w)
        
        if 'wk' in self.specs:
            self.wkqns = jnp.array(self.specs['wk'])
        else:
            self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])
        
        # Spectral Matrices
        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()
        self._calculate_T2()

    def _build_interpolated_spectra(self):
        """Constructs the matrix of interpolated spectra from QNS data.

        The w=0 point comes from the data grid whenever it carries a DC sample
        (reconstructed specs.npz: the noise-aware slope-fit / double-echo DC
        experiments land there, OPT-DC-ORACLE -- i.e. this is a real measured
        point, not an "oracle" value smuggled in from the analytic ground
        truth; the tag records that this WAS a place a bug could sneak the
        true answer in unnoticed, and confirms it doesn't). Only a DC-less
        grid falls back to inserting the analytic S(0) -- simulated_spectra.npz,
        where the file IS the analytic model evaluated at the teeth, or a
        legacy specs.npz (warned at load: regenerate Stage 2). The
        distinction is first-order here: for M > 10 the comb evaluator's
        term_dc reads SMat[..., 0] directly."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to 0, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        grid_has_dc = bool(float(self.wkqns[0]) == 0.0)
        if not grid_has_dc and not self.use_simulated:
            print("[idle] WARNING: specs grid carries no w=0 point -- inserting "
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
            # OPT-SPECTRAL-MODEL: each channel from the same line/tail/head-
            # aware model the unfold bias correction uses (characterize.
            # systematics.selfconsistent_spectra). See cz.py for the full
            # rationale; the blind protocol is preserved (data + experimental
            # priors only). Plainly: instead of just drawing a straight line
            # between the reconstructed comb teeth ('interp'), this rebuilds
            # each spectrum using the same physically-motivated line-shape
            # model (peak positions/widths from experiment, heights fit to
            # the data) that the reconstruction's own bias-correction step
            # uses -- it only ever uses information a real experiment could
            # have, so the blind (no-look-at-truth) comparison stays valid.
            from qns2q.characterize.systematics import selfconsistent_spectra
            from qns2q.noise.spectra import line_priors
            sc_recon = {k: np.nan_to_num(np.asarray(self.specs[k]))
                        for k in ('S11', 'S22', 'S1212', 'S12', 'S112', 'S212')}
            sc_fns = selfconsistent_spectra(np.asarray(self.wkqns), sc_recon,
                                            lines=line_priors())
            w_np = np.asarray(self.w)

        def combine(key, dc_func):
            """Channel curve on the dense grid; w=0 from the grid's DC sample
            when present, else from the analytic model (power-law tail beyond
            the last tooth either way -- control.tails / the sc model)."""
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
            print("[idle] char_self_only: S1212 + crosses DROPPED from the "
                  "characterized model (1Q-only QNS counterfactual); the "
                  "ideal benchmark retains them.")
            return SMat
        SMat = SMat.at[3, 3].set(combine("S1212", S_1212))

        # Off-diagonal elements. A channel the protocol could not reconstruct
        # (robust: all-NaN S112/S212) is dropped from this CHARACTERIZED model
        # with a notice -- the blind objective then assumes zero there -- while
        # _build_ideal_spectra keeps the full truth, so the ideal benchmark
        # prices what losing the channel costs (OPT-ROBUST-NAN: e.g. the
        # SPAM-robust protocol cannot reconstruct the two 3-body cross-spectra
        # at all -- this is the code path that decides what to do about the
        # resulting gap in the input data, namely "pretend it's zero for the
        # optimizer, but still judge the result against the true nonzero
        # value"). By contrast, include_cross_spectra=False removes the
        # channels from BOTH models (the gate-v style counterfactual world).
        if self.include_cross_spectra:
            def cross(key, dc_func):
                data = np.asarray(self.specs[key])
                n_nan = int(np.isnan(data).sum())
                if n_nan:
                    proto = (str(self.specs['spam_protocol'])
                             if 'spam_protocol' in self.specs else 'none')
                    print(f"[idle] NOTE: {key} not reconstructed ({n_nan}/"
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
        """Constructs the matrix of ideal analytical spectra.

        Unlike ``_build_interpolated_spectra`` (which is built from what the
        characterization measured/reconstructed, and can be missing channels
        or subject to reconstruction noise), this always evaluates the
        analytic noise model of ``qns2q.noise.spectra`` directly on the dense
        ``w_ideal`` grid -- the ground truth used only to score a chosen
        sequence's TRUE infidelity (``calculate_infidelity(..., use_ideal=True)``),
        never to select it.
        """
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
        """Calculates T2 times for each qubit based on ideal spectra."""
        # T2 = 2 / S(0)
        # S11(0) is at index [1,1,0]
        # S22(0) is at index [2,2,0]
        
        S11_0 = jnp.real(self.SMat_ideal[1, 1, 0])
        S22_0 = jnp.real(self.SMat_ideal[2, 2, 0])
        
        self.T2q1 = 2.0 / S11_0 if S11_0 > 0 else jnp.inf
        self.T2q2 = 2.0 / S22_0 if S22_0 > 0 else jnp.inf
        
        print(f"Calculated T2 times (Ideal): Q1={self.T2q1:.2e} tau, Q2={self.T2q2:.2e} tau")


# ==============================================================================
# Sequence Generation Utilities
# ==============================================================================
# A "pulse sequence" throughout this file means the sorted list of times at
# which a pi-pulse is applied to one qubit during one base block of duration
# T, ALWAYS written including both boundary points: [0, t1, t2, ..., tn, T].
# The physical picture: each pi-pulse flips the sign of that qubit's
# instantaneous sensitivity to Z-axis dephasing noise ("the switching
# function" y(t) = (-1)^(number of pulses so far)); a well-chosen sequence
# makes y(t) oscillate fast enough that the noise's slow (small-w) power
# averages to a small net phase over the block. Two equivalent
# representations are used and converted between with
# ``pulse_times_to_delays``/``delays_to_pulse_times``: the absolute times
# above, and the "delays" between consecutive pulses (which is what the
# SLSQP optimizer actually varies, since the OPTIMIZATION variables need to
# be an unconstrained-looking vector of intervals rather than a sorted list
# of absolute times).

def remove_consecutive_duplicates(input_list):
    """Removes consecutive duplicate elements from a list.

    Used to clean up the CDD/mqCDD recursive construction below: nested CDD
    trees generate the same boundary time twice where two sub-sequences
    meet (e.g. the midpoint of one CDD block is also the start of the next),
    and two coincident pi-pulses are physically the identity (they cancel),
    so both copies are simply dropped rather than kept as a redundant
    "do-nothing" pair.
    """
    output_list = []
    i = 0
    while i < len(input_list):
        if i + 1 < len(input_list) and input_list[i] == input_list[i+1]:
            i += 2 # Skip both duplicates
        else:
            output_list.append(input_list[i])
            i += 1
    return output_list

def cdd(t0, T, n):
    """Recursive generation of a CDD_n (Concatenated Dynamical Decoupling,
    order n) sequence: bisect the interval [t0, t0+T] with a pulse at its
    midpoint, then recursively apply the same construction to each half at
    one lower order. This is the standard textbook DD family used as a
    baseline against the numerically optimized ("NT") sequences searched for
    elsewhere in this file; higher n cancels higher orders of the noise
    spectrum's low-frequency (quasi-static) content at the cost of more
    pulses. Returns interior pulse times only (no explicit 0/T boundary
    points) -- ``cddn`` below adds those.
    """
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)

def cddn(t0, T, n):
    """Generates the order-n CDD sequence WITH boundary points included,
    i.e. in the [0, t1, ..., tn, T]-style representation used throughout
    this file (see the module-section note above ``remove_consecutive_duplicates``)."""
    out = remove_consecutive_duplicates(cdd(t0, T, n))
    if out[0] == 0.:
        return out + [T]
    else:
        return [0.] + out + [T]

def mqCDD(T, n, m):
    """Generates a "multi-qubit CDD" pair: a CDD_n sequence for one qubit,
    with a CDD_m sequence nested independently inside each of its intervals
    for the other qubit. Physically this lets the two qubits decouple at
    different orders/rates while still sharing the same overall block
    period T -- a richer known-sequence-library entry than running the same
    CDD order on both qubits. Returns [seq_qubit1, seq_qubit2], each in the
    boundary-inclusive [0, ..., T] representation.
    """
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

# @jax.jit below (first use in this file): marks a function for XLA
# just-in-time compilation to fast device code the first time it is called
# with a given input shape/dtype, then reuses the compiled version on later
# calls with matching shapes -- a plain speed optimization, not a change in
# what the function computes. It does mean the function body must be
# "traceable" (pure array math, no data-dependent Python control flow on
# traced values), which is why e.g. ``pulse_times_to_delays`` below branches
# on ``len(tk_arr)`` (a static Python length) rather than on array VALUES.
@jax.jit
def pulse_times_to_delays(tk):
    """
    Converts absolute pulse times to delay intervals.
    Input: [0, t1, t2, ..., tn, T]
    Output: [t1, t2-t1, ..., tn-tn-1]
    """
    tk_arr = jnp.array(tk)
    if len(tk_arr) <= 2:
        return jnp.array([])
    diffs = jnp.diff(tk_arr)
    return diffs[:-1]

@jax.jit
def delays_to_pulse_times(delays, T):
    """
    Converts delay intervals back to absolute pulse times.
    Input: [d1, d2, ...]
    Output: [0, d1, d1+d2, ..., T]
    """
    if delays.size == 0:
        return jnp.array([0., T])
    last_delay = T - jnp.sum(delays)
    all_delays = jnp.concatenate([delays, jnp.array([last_delay])])
    times = jnp.cumsum(all_delays)
    return jnp.concatenate([jnp.array([0.]), times])

def get_random_delays(n, T, tau):
    """Generates random delay intervals summing to < T with minimum separation tau.

    NOTE: as of this pass, this function has no callers anywhere in the repo
    (``optimize_random_sequences`` below seeds its random trials with
    ``get_equidistant_delays`` instead, then lets SLSQP move the pulses).
    ``control/cz.py`` defines and uses its own identical-purpose
    ``get_random_delays``. Left in place rather than removed since deleting
    it is outside the scope of a documentation pass -- flagged here, and in
    this task's report, for a maintainer to decide whether to prune it.
    """
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

def get_equidistant_delays(n, T):
    """Generates equidistant (evenly spaced) delay intervals -- the starting
    point handed to SLSQP for each "random" (n1, n2) trial in
    ``optimize_random_sequences`` below (only the pulse COUNT is randomized;
    the initial guess for the timing itself is a uniform train, which SLSQP
    then moves via gradient descent)."""
    if n <= 0:
        return jnp.array([])
    return jnp.ones(int(n)) * T / (n + 1)

@jax.jit
def make_tk12(tk1, tk2):
    """
    Combines two pulse sequences into a single sequence for the 12 interaction.
    Assumes inputs are [0, t1..., T] and [0, t2..., T].

    Physically: the two-qubit ZZ coupling term is only refocused by a pulse
    on EITHER qubit (a single-qubit pi-pulse still flips the sign of Z1*Z2),
    so the "12" (Ising/ZZ) channel's effective switching function toggles at
    the union of both qubits' pulse times, merged and sorted here.
    """
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """
    Constructs a library of known pulse sequences (CDD permutations and mqCDD)
    to idle for one base block of duration T_seq: this is the "textbook DD"
    baseline that ``evaluate_known_sequences`` scores and that the free
    ("NT") search in ``optimize_random_sequences`` is trying to beat.

    Parameters
    ----------
    T_seq : float
        Duration of one base sequence block, in units of tau.
    tau_min : float
        Minimum allowed spacing between consecutive pulses (``config.min_sep``
        upstream) -- CDD/mqCDD orders are grown until the next order would
        violate this, i.e. sequence generation stops at whatever depth is
        still physically realizable, not at a fixed pulse count.
    max_pulses : int, optional
        Per-qubit pulse-count cap applied when pruning the generated library
        (default 50).

    Returns:
        pLib_delays: List of (d1, d2) delay tuples.
        pLib_descriptions: List of strings describing each sequence.
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

    # 2. Create permutations. Unlike cz.py's product(): identical-order (n, n)
    # pairs give y_12 = +1 (ZZ fully undecoupled) -- essential for the CZ drive
    # but never useful for idling, so they are deliberately excluded here.
    pLib_times = list(itertools.permutations(cddLib, 2))

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
    num_cdd_perms = len(list(itertools.permutations(cddLib, 2)))
    
    # Helper to identify CDD order
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

# @functools.partial(jax.jit, static_argnames=[...]) below (first use here):
# same JIT compilation as plain @jax.jit, but some arguments (named in
# static_argnames) are Python ints/values used to control array SHAPES or
# Python-level control flow (e.g. M, n_base_steps here) rather than being
# traced as array data. JAX needs those pinned at trace time (a change in
# their value triggers a fresh compile, e.g. once per distinct M in the
# outer M-sweep) precisely because shapes must be static for compilation --
# they cannot themselves be ordinary runtime array arguments.
@functools.partial(jax.jit, static_argnames=['M', 'n_base_steps'])
def precompute_R_folded(R_shifted, lags_R, M, T_base, dt, n_base_steps):
    """
    Precomputes the folded noise autocorrelation R_folded(u) on the domain of C_1(u).
    R_folded(u) = sum_{p=-(M-1)}^{M-1} (M - |p|) * R(u + p*T_base)

    Physical picture: repeating a base pulse block M times means the noise
    correlation function R (the inverse Fourier transform of the spectrum,
    computed once per (SMat, tau, T_seq, M) by ``prepare_time_domain_overlap``
    and cached there) gets sampled at every pairwise combination of the M
    repeats' time origins, which is exactly this triangular-weighted sum of
    shifted copies of R. Precomputing it here means the per-restart
    optimization cost (``evaluate_overlap_folded`` below) does not have to
    redo this fold on every SLSQP iteration.
    """
    # Lags for C1 (base sequence correlation)
    # y has length n_base_steps. C1 has length 2*n_base_steps - 1.
    lags_C = (jnp.arange(2 * n_base_steps - 1) - (n_base_steps - 1)) * dt

    p_vals = jnp.arange(-(M - 1), M)
    weights = jnp.float64(M) - jnp.abs(p_vals)

    # R is complex
    R_real = jnp.real(R_shifted)
    R_imag = jnp.imag(R_shifted)

    def get_folded_component(R_comp):
        def interp_slice(p, w):
            shifted_lags = lags_C + p * T_base
            # Interpolate R at shifted lags
            return w * jnp.interp(shifted_lags, lags_R, R_comp, left=0., right=0.)

        # jax.vmap ("vectorizing map") below: runs interp_slice once per
        # value of p_vals/weights, all at once as a single batched/vectorized
        # device operation, instead of writing a Python for-loop over p that
        # JAX would have to unroll. Same result as the loop, just faster and
        # more memory-efficient on GPU/TPU.
        slices = jax.vmap(interp_slice)(p_vals, weights)
        # Sum over p
        return jnp.sum(slices, axis=0)

    R_folded_real = get_folded_component(R_real)
    R_folded_imag = get_folded_component(R_imag)
    
    return R_folded_real + 1j * R_folded_imag

@functools.partial(jax.jit, static_argnames=['n_base_steps'])
def evaluate_overlap_folded(pulse_times_a, pulse_times_b, R_folded, dt, n_base_steps):
    """
    Calculates the overlap integral using the folded noise autocorrelation.
    I = integral C_1(u) * R_folded(u) du

    This is the exact (no frequency-comb shortcut) time-domain evaluator: it
    samples each pulse sequence's switching function y(t) on a fine time
    grid, cross-correlates the two (giving C_1(u), the filter-function-like
    overlap of the two switching functions at time lag u), and integrates
    that against the precomputed folded noise autocorrelation R_folded from
    ``precompute_R_folded``/``prepare_time_domain_overlap``. The resulting
    scalar I_{a,b} is one entry of the 4x4 overlap-integral matrix that
    ``calculate_idling_fidelity`` turns into a fidelity. Called externally
    (as ``idmod.evaluate_overlap_folded``) by ``scripts/showcase_storage_panel.py``,
    so its call signature is part of this module's protected surface.
    """
    # Construct y on [0, T_base]
    t_grid = jnp.arange(n_base_steps) * dt
    
    def get_y_samples(pt):
        # pt is [0, t1, ..., T_base]
        # y is periodic with T_base (but we only need one period here)
        # We assume pt are within [0, T_base]
        # searchsorted returns index i such that pt[i-1] <= t < pt[i]
        indices = jnp.searchsorted(pt, t_grid, side='right')
        return (-1.0) ** (indices - 1)
        
    y_a = get_y_samples(pulse_times_a)
    y_b = get_y_samples(pulse_times_b)
    
    # Compute C_1(u)
    # mode='full' returns output of size 2*n_base_steps - 1
    C_vals = jax.scipy.signal.correlate(y_a, y_b, mode='full') * dt
    
    # Integral
    # R_folded is already aligned with C_vals
    integral = jnp.sum(C_vals * R_folded) * dt
    
    return integral

@jax.jit
def get_spectral_amplitudes_jax(pulse_times, omega):
    """
    Computes A(omega) = sum (-1)^j (exp(i w t_{j+1}) - exp(i w t_j)).
    Returns vector of size len(omega).

    This is the (dimensionless) filter-function amplitude of one pulse
    sequence's switching function y(t), evaluated at the frequency-comb
    harmonics omega -- the building block ``evaluate_overlap_comb`` (below)
    combines pairwise to approximate the overlap integral without doing an
    explicit time-domain correlation.
    """
    # pulse_times: [0, t1, ..., T]
    # omega: [w1, ..., wK]
    
    # We want exp(i * outer(pt, w))
    # pt is (N+1,), w is (K,)
    # outer is (N+1, K)
    
    exp_t = jnp.exp(1j * jnp.outer(pulse_times, omega))
    
    # diffs: (N, K)
    diffs = exp_t[1:] - exp_t[:-1]
    
    # signs: (N,)
    n_intervals = len(pulse_times) - 1
    signs = jnp.array((-1.0)**jnp.arange(n_intervals))
    
    # Sum over intervals
    # signs[:, None] is (N, 1)
    # result is (K,)
    return jnp.sum(signs[:, None] * diffs, axis=0)

@functools.partial(jax.jit, static_argnames=['M'])
def evaluate_overlap_comb(pulse_times_a, pulse_times_b, S_packed, omega_k, T_seq, M):
    """
    Calculates overlap integral using frequency comb approximation.
    S_packed: [S(0), S(w1), ..., S(wK)]
    omega_k: [w1, ..., wK]

    Alternative to ``evaluate_overlap_folded``: instead of correlating the
    two switching functions in the time domain, this treats an M-fold
    repeated sequence as sampling the noise spectrum only at its own
    harmonic "comb" frequencies (multiples of 2*pi/T_seq), which is a valid
    approximation when the comb tooth spacing is fine compared to any
    spectral feature the noise has (see ``use_comb_approximation`` for when
    that holds and why it is only ever used on the search side, never for
    the published/"ideal" infidelity numbers).
    """
    # 1. DC Component
    # y(0) = sum (-1)^j (t_{j+1} - t_j)
    # This is equivalent to sum(delays * signs)
    # We can compute it from pulse_times
    
    def get_dc(pt):
        diffs = jnp.diff(pt)
        n = len(diffs)
        signs = jnp.array((-1.0)**jnp.arange(n))
        return jnp.sum(diffs * signs)
        
    dc_a = get_dc(pulse_times_a)
    dc_b = get_dc(pulse_times_b)
    S_0 = S_packed[0]
    
    term_dc = dc_a * dc_b * S_0
    
    # 2. AC Components (Harmonics)
    # If omega_k is empty, skip
    if omega_k.size == 0:
        return term_dc * M / T_seq
        
    S_k = S_packed[1:]
    
    A_a = get_spectral_amplitudes_jax(pulse_times_a, omega_k)
    A_b = get_spectral_amplitudes_jax(pulse_times_b, omega_k)
    
    # Decompose into real arithmetic to avoid complex gradient instabilities
    # We want Re( A_a * conj(A_b) * S_k ) / w^2
    
    Ar, Ai = jnp.real(A_a), jnp.imag(A_a)
    Br, Bi = jnp.real(A_b), jnp.imag(A_b)
    Sr, Si = jnp.real(S_k), jnp.imag(S_k)
    
    # Real part of A_a * conj(A_b) = (Ar + iAi)(Br - iBi)
    # = (ArBr + AiBi) + i(AiBr - ArBi)
    P_real = Ar * Br + Ai * Bi
    P_imag = Ai * Br - Ar * Bi
    
    # Real part of (P_real + i P_imag) * (Sr + i Si)
    # = P_real * Sr - P_imag * Si
    term_ac_real = (P_real * Sr - P_imag * Si) / (omega_k**2)
    
    sum_ac = jnp.sum(term_ac_real)
    
    # Total integral
    # I = M/T * (Term_DC + 2 * Sum_AC)
    total = term_dc + 2 * sum_ac
    
    return total * M / T_seq


def prepare_time_domain_overlap(SMat, w_grid, tau, T_seq, M):
    """Prepare folded correlation data for time-domain overlap integral method.

    Applies adaptive zero-padding to achieve dt <= tau/4 for improved time
    resolution, mirrors spectrum for Hermitian symmetry, IFFTs to obtain R(τ),
    and folds the correlation across M repetitions.

    Physically: the noise spectrum S(w) (input) and its time-domain
    autocorrelation R(u) (what the overlap integral in
    ``evaluate_overlap_folded`` actually needs) are a Fourier-transform pair;
    this function does that S(w) -> R(u) conversion numerically (via FFT) on
    a fine enough time grid (dt <= tau/4, so that features on the pulse-timing
    scale tau are resolved) and then pre-folds R across the M repeated base
    blocks (see ``precompute_R_folded``) so the expensive part is done once
    per (spectrum, tau, T_seq, M) setup rather than once per SLSQP iteration
    or per candidate sequence. Results are memoized in the module-level
    ``_OVERLAP_SETUP_CACHE`` dict (a plain manual cache, not
    ``functools.lru_cache``, because the key needs custom handling -- see
    where ``_OVERLAP_SETUP_CACHE`` is defined above) since this setup only
    depends on the spectrum/grid/(tau, T_seq, M), never on the pulse sequence
    itself. Called externally (as ``idmod.prepare_time_domain_overlap``) by
    ``scripts/showcase_storage_panel.py``, so its call signature is part of
    this module's protected surface.

    Returns
    -------
    RMat_data : jnp.ndarray, shape (4, 4, 2*n_base_steps-1)
        Folded noise correlation matrix on the base-sequence lag grid.
    dt : float
        Time-domain grid spacing, in units of tau (NOT seconds, despite the
        historical SI-era phrasing this file otherwise uses -- see repo
        CLAUDE.md "Units: tau = 1").
    n_base_steps : int
        Number of time steps spanning one base sequence period T_seq.
    """
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

    # Mirror for Hermitian symmetry: S(-ω) = S*(ω). A physical (real-valued
    # in the time domain) noise process has a spectrum that is Hermitian in
    # this sense; since only S(w>=0) is stored/reconstructed, the negative-
    # frequency half is reconstructed here from that symmetry before the
    # inverse FFT, rather than being separately measured.
    SMat_sym = jnp.concatenate(
        [SMat_padded, jnp.conj(jnp.flip(SMat_padded[..., 1:-1], axis=-1))],
        axis=-1,
    )
    N_sym = SMat_sym.shape[-1]

    dt = 2 * np.pi / (N_sym * dw)
    print(f"  Time-domain setup: pad_factor={pad_factor}, dt={dt:.4f} tau, "
          f"tau/4={tau/4:.4f} tau")

    lags_R = (jnp.arange(N_sym) - N_sym // 2) * dt
    # jnp.fft.ifft: inverse Fast Fourier Transform, i.e. numerically convert
    # the frequency-domain S(w) samples into the time-domain autocorrelation
    # R(u) samples (fftshift below just reorders the FFT's native
    # zero-frequency-first layout into a plain ascending-lag array).
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


@jax.jit
def calculate_idling_fidelity(I_matrix):
    r"""
    Calculate the two-qubit idling gate fidelity $F_I(T)$ from overlap integrals.

    This function uses a numerically stable formula to compute the average
    gate fidelity by summing the contributions from all 16 two-qubit Pauli
    operators. The fidelity is derived from the first-order cumulant expansion
    of the gate evolution under noise.

    Parameters
    ----------
    I_matrix : jax.Array
        A 4x4 matrix of overlap integrals $I_{a,b} = \int \int dt_1 dt_2 C_{a,b}(t_1, t_2) R_{a,b}(t_1-t_2)$,
        where $a, b \in \{0, 1, 2, 12\}$.

    Returns
    -------
    float
        The calculated average idling gate fidelity.
    """
    # Commutation rules with Z: 0(I):comm, 1(X):anti, 2(Y):anti, 3(Z):comm
    comm_with_z = jnp.array([0, 1, 1, 0])

    F_total = 0.0

    # Iterate over all 16 Pauli operators P_k = P1 \otimes P2. These are
    # plain Python for-loops over range(4) (a static Python int, not a traced
    # JAX array), so under @jax.jit they get UNROLLED into 16 copies of the
    # loop body at compile time -- fine here since 16 is small and fixed;
    # this is different from ``jax.vmap`` (used elsewhere in this file),
    # which is the idiom to reach for when the loop trip count would be
    # large or itself traced.
    for p1 in range(4):
        for p2 in range(4):
            # Determine commutation of P_k with Z_1, Z_2, Z_12
            c1 = comm_with_z[p1]
            c2 = comm_with_z[p2]
            c12 = (c1 + c2) % 2
            
            # Anticommutation flags c_a = (1 - sgn(P_k, a, 0))/2 in {0, 1} for the
            # register channels a in {0, 1, 2, 12}.
            c_vec = jnp.array([0.0, c1, c2, c12])
            
            # Theta_l = coeff of Z_l in C^{(2)}_{P_k}/2! = sum_j c_j c_{j XOR l} I_{j, j XOR l}.
            # A pair (j, j XOR l) contributes only when P_k anticommutes with BOTH
            # Z_j and Z_{j XOR l} (paper Eq. c2_spectra, both-index parity); the 1/2 of
            # the dephasing convention is carried by the E_i factors below.
            Thetas = jnp.zeros(4, dtype=jnp.complex128)
            for l in range(4):
                col_indices = jnp.arange(4) ^ l
                val = jnp.sum(c_vec * c_vec[col_indices]
                              * I_matrix[jnp.arange(4), col_indices])
                Thetas = Thetas.at[l].set(val)
            
            # Ensure Thetas are real (overlap integrals are real)
            T0, T1, T2, T12 = jnp.real(Thetas[0]), jnp.real(Thetas[1]), jnp.real(Thetas[2]), jnp.real(Thetas[3])
            
            # Numerically stable calculation of C_1^{(k)}
            # Use exponential expansion to avoid 0 * Inf instability when T0 is large
            # Also clamp the exponent to <= 0 to prevent unphysical growth (F > 1) 
            # which causes overflow (-inf cost) during optimization exploration.
            E1 = 0.5 * (-T0 - T1 - T2 - T12)
            E2 = 0.5 * (-T0 - T1 + T2 + T12)
            E3 = 0.5 * (-T0 + T1 - T2 + T12)
            E4 = 0.5 * (-T0 + T1 + T2 - T12)
            
            C_1 = 0.25 * (jnp.exp(jnp.minimum(E1, 0.0)) +
                          jnp.exp(jnp.minimum(E2, 0.0)) +
                          jnp.exp(jnp.minimum(E3, 0.0)) +
                          jnp.exp(jnp.minimum(E4, 0.0)))
            
            F_total += C_1
            
    return jnp.real(F_total)



def use_comb_approximation(M, T_seq):
    """Whether the frequency-comb overlap approximation is valid at (M, T_seq).

    In plain terms: the comb approximation (``evaluate_overlap_comb``) treats
    the noise spectrum as if it only mattered at a discrete set of harmonic
    frequencies; that is a good approximation when the spectrum is smooth on
    the scale of the spacing between those harmonics, but a bad one near a
    sharp spectral line (the featured/showcase noise model's nuclear-
    difference lines) if the M-fold-repeated sequence's frequency comb is
    coarser than the line width -- the comb can then land its samples on the
    wrong side of a peak and mis-weight it substantially.

    The comb samples S at delta teeth; the TRUE M-fold filter tooth has width
    ~ 2pi/(M*T_seq). When that width is comparable to the nuclear-line width
    sigma the comb mis-weights the lines: 8-14% infidelity error at
    Tg = 320 tau / M = 16, 3-7% at Tg = 640, up to 3.2% at Tg = 1280; every
    point passing 2pi/Tg < sigma/8 measures <= 1.7% (OPT-COMB-M16 diagnostic
    + boundary sweep; this was quantified with a since-removed one-off
    diagnostic script, scripts/diag_comb_vs_folded.py -- the numeric
    thresholds it produced are recorded here since the script itself no
    longer exists after CLEANUP-0616). Smooth (bland) spectra keep the legacy
    speed cutoff M > 10. Note the published-number path (calculate_infidelity
    with use_ideal=True) is always folded regardless."""
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
    """Interpolate the dense SMat onto the comb harmonics omega_k and prepend
    S(0): returns [S(0), S(w_k)] packed per channel, the exact input format
    ``evaluate_overlap_comb`` expects for its DC + AC-harmonics split."""
    S_flat = SMat.reshape(-1, SMat.shape[-1])

    def interp_row(fp):
        return (jnp.interp(omega_k, w_grid, jnp.real(fp), right=0.) +
                1j * jnp.interp(omega_k, w_grid, jnp.imag(fp), right=0.))

    S_h = jax.vmap(interp_row)(S_flat)
    return jnp.concatenate([S_flat[:, :1], S_h], axis=1).reshape(4, 4, -1)


def _delays_to_pts(delays_params, n_pulses1, T_seq):
    """Split the flat SLSQP optimization vector (qubit-1 delays followed by
    qubit-2 delays) back into the four pulse-time sequences the overlap
    evaluators need: [identity, qubit1, qubit2, combined ZZ channel]."""
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
    # Dummy pulse sequence for index 0 (Identity): SMat[0,:] is 0, so the
    # overlap vanishes regardless of sequence.
    pt0 = jnp.array([0., T_seq])
    return [pt0, pt1, pt2, pt12]


def _cost_folded(delays_params, RMat_data, dt, T_seq, n_pulses1, n_base_steps):
    """Idling cost (1 - F/16) on the folded evaluator: builds the 4x4 overlap
    matrix (over the {identity, qubit1, qubit2, ZZ} channels, see
    ``_delays_to_pts``) and converts it to an infidelity via
    ``calculate_idling_fidelity``. This is the scalar function SLSQP
    minimizes; ``cost_vag_folded`` below wraps it to also return the
    gradient. Every input is a runtime ARGUMENT rather than a value baked
    into a Python closure (OPT-SPEEDUPS (b)): concretely, if the spectrum
    arrays were captured as closure constants instead, JAX would treat every
    new spectrum (a new run, a new restart with re-seeded config, etc.) as a
    brand-new function needing its own fresh compile; passing them as
    ordinary arguments instead means the value_and_grad wrappers below are
    stable, reusable, module-level objects -- restarts and (Tg, M) blocks
    share compiled programs keyed only on (shape, static args), and the
    persistent compilation cache (OPT-SPEEDUPS (a), see the imports at the
    top of the file) can hit across separate process runs too."""
    pts = _delays_to_pts(delays_params, n_pulses1, T_seq)
    I_mat = jnp.array([[evaluate_overlap_folded(pts[i], pts[j], RMat_data[i, j],
                                                dt, n_base_steps)
                        for j in range(4)] for i in range(4)])
    return 1.0 - calculate_idling_fidelity(I_mat) / 16.0


def _cost_comb(delays_params, S_packed, omega_k, T_seq, n_pulses1, M):
    """Idling cost on the comb evaluator (see _cost_folded for the design)."""
    pts = _delays_to_pts(delays_params, n_pulses1, T_seq)
    I_mat = jnp.array([[evaluate_overlap_comb(pts[i], pts[j], S_packed[i, j],
                                              omega_k, T_seq, M)
                        for j in range(4)] for i in range(4)])
    return 1.0 - calculate_idling_fidelity(I_mat) / 16.0


# jax.value_and_grad(f): returns a new function that evaluates BOTH f(x) and
# its gradient d f/dx in one pass (via automatic differentiation), which is
# exactly what a gradient-based optimizer (SLSQP with jac=True, used below)
# needs; wrapping that again in jax.jit compiles the value+gradient
# computation together. Defined once at module load (not inside a function)
# so it is that single stable, reusable, compiled object referred to above.
cost_vag_folded = jax.jit(jax.value_and_grad(_cost_folded),
                          static_argnames=('n_pulses1', 'n_base_steps'))
cost_vag_comb = jax.jit(jax.value_and_grad(_cost_comb),
                        static_argnames=('n_pulses1', 'M'))

def optimize_random_sequences(config, M, n_pulses_list, seed_seq=None):
    """Optimizes random (and optionally seeded) noise-tailored (NT) pulse
    sequences for a given repetition count M: for each (n1, n2) pulse-count
    pair in ``n_pulses_list`` (plus one seeded restart from ``seed_seq`` when
    given), runs SLSQP from an equidistant-pulse starting guess and keeps
    whichever restart reaches the lowest infidelity. Returns
    ``(best_seq, best_inf)`` -- ``best_seq`` is a (pt1, pt2) pair of absolute
    pulse times, or None if every trial was infeasible/failed."""
    T_seq = config.Tg / M
    best_inf = 1.0
    best_seq = None

    # Setup Evaluation Method
    use_comb = use_comb_approximation(M, T_seq)

    if use_comb:
        print(f"Using Frequency Comb Approximation (M={M})...")
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        omega_k = jnp.arange(1, max_k + 1) * w0
        S_packed = _pack_comb(config.SMat, config.w, omega_k)

        def vag(xp, n1p):
            return cost_vag_comb(xp, S_packed, omega_k, T_seq,
                                 n_pulses1=n1p, M=M)
    else:
        print(f"Using Folded Noise Matrix (M={M})...")
        RMat_data, dt, n_base_steps = prepare_time_domain_overlap(
            config.SMat, config.w, config.tau, T_seq, M
        )

        def vag(xp, n1p):
            return cost_vag_folded(xp, RMat_data, dt, T_seq,
                                   n_pulses1=n1p, n_base_steps=n_base_steps)

    # One compiled cost per parity class for the whole trial list
    # (control.padding shape unification).
    all_ns = [int(n) for pair in n_pulses_list for n in pair]
    if seed_seq is not None:
        all_ns += [len(seed_seq[0]) - 2, len(seed_seq[1]) - 2]
    opt_targets = pad_targets(all_ns)

    def run_single_optimization(n1, n2, initial_params, label):
        # `nonlocal` lets this inner function update the ENCLOSING function's
        # best_inf/best_seq variables directly (rather than returning a value
        # that the caller has to merge in), which is convenient here since
        # run_single_optimization is called many times below (once per seed/
        # random restart) purely for its side effect of possibly improving
        # the running best.
        nonlocal best_inf, best_seq

        # Bounds (min_sep = tau unless the control-bandwidth scenario raises it)
        bounds = [(config.min_sep, T_seq) for _ in range(len(initial_params))]

        n1p = pad_count(n1, opt_targets)
        n2p = pad_count(n2, opt_targets)
        pad1 = np.zeros(n1p - n1)
        pad2 = np.zeros(n2p - n2)

        # The pads are appended/stripped here in the wrapper (exact identity,
        # control.padding); SLSQP sees the original (n1 + n2)-dim problem.
        def fun_wrapper(x):
            xp = jnp.asarray(np.concatenate([x[:n1], pad1, x[n1:], pad2]))
            v, g = vag(xp, n1p)
            g = np.asarray(g)
            return float(v), np.concatenate([g[:n1], g[n1p:n1p + n2]])

        # Linear constraints: scipy.optimize.LinearConstraint(A, lb, ub)
        # enforces lb <= A @ x <= ub. Here A picks out "sum of qubit-1
        # delays" (row 0) and "sum of qubit-2 delays" (row 1), and the upper
        # bound caps each sum at T_seq - min_sep -- physically, the delays
        # for one qubit must leave room for at least one more min_sep-sized
        # gap before hitting the end of the block (the -inf lower bound means
        # this is a one-sided/"at most" constraint, not an equality).
        A = np.zeros((2, n1 + n2))
        A[0, :n1] = 1
        A[1, n1:] = 1
        linear_cons = scipy.optimize.LinearConstraint(A, -np.inf,
                                                      T_seq - config.min_sep)

        try:
            # jac=True tells SciPy that fun_wrapper itself returns
            # (value, gradient) together (as produced by jax.value_and_grad
            # above) instead of SciPy having to estimate the gradient by
            # finite differences -- this is both much faster and more
            # accurate for a many-parameter, JAX-differentiable objective.
            res = scipy.optimize.minimize(fun_wrapper, np.asarray(initial_params), method='SLSQP',
                                          bounds=bounds, constraints=linear_cons, jac=True,
                                          tol=SLSQP_TOL,
                                          options={'maxiter': SLSQP_MAXITER, 'disp': False})

            inf = res.fun

            if inf < best_inf:
                best_inf = inf
                d1_opt = jnp.array(res.x[:n1])
                d2_opt = jnp.array(res.x[n1:])
                pt1 = delays_to_pulse_times(d1_opt, T_seq)
                pt2 = delays_to_pulse_times(d2_opt, T_seq)
                best_seq = (pt1, pt2)

            print(f"  {label} (n={n1},{n2}): Infidelity = {inf:.6e}")

        except Exception as e:
            print(f"  {label} failed for n={n1},{n2}: {e}")
            # traceback.print_exc()

    # 1. Seeded Optimization
    if seed_seq is not None:
        pt1_s, pt2_s = seed_seq
        n1_s = len(pt1_s) - 2
        n2_s = len(pt2_s) - 2
        d1_s = pulse_times_to_delays(pt1_s)
        d2_s = pulse_times_to_delays(pt2_s)
        init_p = jnp.concatenate([d1_s, d2_s])
        init_p = init_p + np.random.normal(0, 1e-10, size=init_p.shape)
        run_single_optimization(n1_s, n2_s, init_p, "Seeded Opt")

    # 2. Random Optimization
    for n1, n2 in n_pulses_list:
        if (n1 + 1) * config.min_sep > T_seq or (n2 + 1) * config.min_sep > T_seq:
            print(f"Skipping Random Opt (n={n1},{n2}): Over-constrained (Too many pulses for T_seq)")
            continue

        d1 = get_equidistant_delays(n1, T_seq)
        d2 = get_equidistant_delays(n2, T_seq)
        initial_params = jnp.concatenate([d1, d2])
        run_single_optimization(n1, n2, initial_params, "Random Opt")
            
    return best_seq, best_inf

def evaluate_known_sequences(config, M, pLib):
    """Evaluates every (d1, d2) delay-pair sequence in the known-sequence
    library ``pLib`` (built by ``construct_pulse_library``) against the
    CHARACTERIZED spectra and returns the best one: ``(best_seq, best_inf,
    best_idx)``, where ``best_seq`` is a (pt1, pt2) pair of absolute pulse
    times and ``best_idx`` indexes back into ``pLib`` (and its parallel
    description list from ``construct_pulse_library``)."""
    T_seq = config.Tg / M
    best_inf = 1.0
    best_seq = None
    best_idx = -1
    
    print(f"Evaluating {len(pLib)} known sequences...")

    # Setup Evaluation Method
    use_comb = use_comb_approximation(M, T_seq)

    if use_comb:
        print(f"Using Frequency Comb Approximation (M={M})...")
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        omega_k = jnp.arange(1, max_k + 1) * w0
        S_packed = _pack_comb(config.SMat, config.w, omega_k)

        def overlap_fn(pt_a, pt_b, r, c):
            return evaluate_overlap_comb(pt_a, pt_b, S_packed[r, c],
                                         omega_k, T_seq, M)
    else:
        print(f"Using Folded Noise Matrix (M={M})...")
        RMat_data, dt, n_base_steps = prepare_time_domain_overlap(
            config.SMat, config.w, config.tau, T_seq, M
        )

        def overlap_fn(pt_a, pt_b, r, c):
            return evaluate_overlap_folded(pt_a, pt_b, RMat_data[r, c],
                                           dt, n_base_steps)

    # Shape-unify the library with exact-identity padding (control.padding):
    # <= 2 shapes per qubit (parity classes) instead of one compile per
    # distinct CDD/mqCDD length. Padded arrays are used for EVALUATION only;
    # the recorded winner keeps its original (unpadded) pulse times.
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

        fid = calculate_idling_fidelity(I_mat)
        inf = 1.0 - fid / 16.0

        if inf < best_inf:
            best_inf = inf
            # Record the ORIGINAL (unpadded) sequence -- padding is an
            # evaluation-shape detail, not part of the winner.
            best_seq = (delays_to_pulse_times(d1, T_seq),
                        delays_to_pulse_times(d2, T_seq))
            best_idx = i

    return best_seq, best_inf, best_idx

def calculate_infidelity(seq, config, M, T_seq, use_ideal=False):
    """Scores a chosen (pt1, pt2) idling sequence's infidelity 1 - F/16, given
    an already-decided ``config`` and repetition count M.

    Parameters
    ----------
    seq : tuple of (jnp.ndarray, jnp.ndarray) or None
        (pt1, pt2) absolute pulse times for qubit 1 and qubit 2 (each in the
        boundary-inclusive [0, ..., T_seq] representation); None (no valid
        sequence was found upstream) short-circuits to infidelity 1.0.
    config : Config
        Supplies the characterized (``SMat``) and ideal/ground-truth
        (``SMat_ideal``) spectral matrices, plus tau.
    M : int
        Number of repetitions of this base block.
    T_seq : float
        Duration of one base block, in units of tau.
    use_ideal : bool, optional
        If False (default), scores against the CHARACTERIZED spectra
        ``config.SMat`` -- what the optimizer itself sees. If True, scores
        against the analytic ground truth ``config.SMat_ideal`` instead --
        this is the "true infidelity" reported in the paper's figures/tables
        for a blind winner, i.e. what the sequence would actually achieve in
        the real (fully characterized in the limit) noise environment.

    Called externally (as ``idmod.calculate_infidelity``) by
    ``scripts/harvest_design_numbers.py``, so its call signature is part of
    this module's protected surface.
    """
    if seq is None: return 1.0

    SMat = config.SMat_ideal if use_ideal else config.SMat
    w_grid = config.w_ideal if use_ideal else config.w

    # use_ideal=True is the published-number path (the true-infidelity
    # benchmark of the blind winner): always take the exact folded evaluator
    # there, so quoted numbers carry NO comb approximation. The comb remains a
    # search-side speed optimization only (OPT-COMB-M16 hardening).
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
    
    fid = calculate_idling_fidelity(I_mat)
    return 1.0 - fid / 16.0

# ==============================================================================
# Main Execution
# ==============================================================================

def run_optimization_pipeline(config):
    """
    Runs the full optimization pipeline for a SINGLE fixed repetition count
    (``config.M``), scanning over ``config.gate_time_factors`` (a range of
    total hold times Tg). At each gate time it: (1) scores the "no pulse at
    all" baseline, (2) evaluates the known-sequence library
    (``construct_pulse_library`` + ``evaluate_known_sequences``), and (3)
    runs the free/NT search (``optimize_random_sequences``) -- all against
    the CHARACTERIZED spectra, then rescored against the IDEAL/ground-truth
    spectra for the numbers actually recorded/plotted. Saves per-gate-time
    curves plus the single best-over-gate-time known and NT sequences to
    ``.npz`` files under the run folder (see the save_dict / np.savez calls
    below), and returns a dict of the same curves/sequences for the
    ``__main__`` M-sweep below to aggregate across M.

    This function is called once per M value by the ``__main__`` block; it
    does NOT itself loop over M (compare ``optimization_data_all_M*.npz``,
    which the caller builds by calling this once per M and combining the
    results).
    """
    # Drop cached folded-correlation setups from a previous M so the cache stays
    # bounded (it is keyed on this per-M config's spectrum arrays).
    _OVERLAP_SETUP_CACHE.clear()
    print(f"\n{'='*60}")
    print(f"RUNNING OPTIMIZATION PIPELINE")
    print(f"M (Repetitions): {config.M}")
    print(f"Total Pulse Limit: {config.max_pulses}")
    print(f"Pulse Limit Per Repetition: {config.max_pulses_per_rep}")
    print(f"{'='*60}")
    
    yaxis_opt, xaxis_opt = [], []
    yaxis_known, xaxis_known = [], []
    yaxis_nopulse = []
    
    # Store labels for plotting
    labels_known = []
    labels_opt = []

    # Store sequences
    sequences_known = []
    sequences_opt = []
    
    best_known_seq_overall = None
    best_opt_seq_overall = None
    T_seq_best_known = None
    T_seq_best_opt = None
    best_known_inf_overall = 1.0
    best_opt_inf_overall = 1.0
    
    # Store best label for overall best
    best_known_label_overall = ""
    best_opt_label_overall = ""

    for i in config.gate_time_factors:
        # Use Tqns as base for gate time scaling, similar to cz_optimize
        Tg = config.Tqns / 2**(i-1)
        if Tg < config.tau:
            continue
            
        # Update config for this iteration
        config.Tg = Tg
        config.T_seq = Tg / config.M
        
        print(f"\nGate Time: {Tg:.1f} tau (Tg/T2q1={Tg/config.T2q1:.4f}, Tg/T2q2={Tg/config.T2q2:.4f})")
        
        # No Pulse Calculation
        pt_nopulse = jnp.array([0., config.T_seq])
        seq_nopulse = (pt_nopulse, pt_nopulse)
        inf_nopulse = calculate_infidelity(seq_nopulse, config, config.M, config.T_seq, use_ideal=True)
        yaxis_nopulse.append(inf_nopulse)
        print(f"  No Pulse (Ideal): {inf_nopulse:.6e}")
        
        # 1. Known Sequences
        pLib_delays, pLib_desc = construct_pulse_library(config.T_seq, config.min_sep, config.max_pulses_per_rep)
        
        best_known_seq = None
        best_known_inf = 1.0
        idx = -1
        
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences(config, config.M, pLib_delays)
            print(f"  Best Known (Char): {best_known_inf:.6e} ({pLib_desc[idx]})")
            
            # Recalculate with ideal
            best_known_inf_ideal = calculate_infidelity(best_known_seq, config, config.M, config.T_seq, use_ideal=True)
            print(f"  Best Known (Ideal): {best_known_inf_ideal:.6e}")
            
            label_k = f"{pLib_desc[idx]}^{config.M}"
            labels_known.append(label_k)
            
            if best_known_inf_ideal < best_known_inf_overall:
                best_known_inf_overall = best_known_inf_ideal
                best_known_seq_overall = best_known_seq
                T_seq_best_known = config.T_seq
                best_known_label_overall = label_k
        else:
            best_known_inf_ideal = 1.0
            labels_known.append("N/A")
            print("  No valid known sequences found.")
            
        yaxis_known.append(best_known_inf_ideal)
        xaxis_known.append(Tg)
        sequences_known.append(best_known_seq)
            
        # 2. Random Optimization
        print("  Running Random Optimization...")
        best_opt_seq = None
        
        # Random candidate selection
        max_n_physical = int(config.T_seq / config.min_sep) - 1
        upper_bound_physical = max(1, max_n_physical - 1)
        effective_max = min(config.max_pulses_per_rep, upper_bound_physical)
        if config.max_dim > 0 and 2 * effective_max > config.max_dim:
            print(f"  NOTE: SLSQP dim guard clips the NT search at this "
                  f"(Tg, M): per-qubit max {effective_max} -> "
                  f"{config.max_dim // 2} (max_dim={config.max_dim}); the "
                  f"separation limit is reachable at M >= "
                  f"{int(np.ceil(config.Tg / (config.tau * (config.max_dim // 2 + 2))))}")
            effective_max = config.max_dim // 2

        n_pulses_list = []
        if config.informed_counts and effective_max >= 2:
            # Spectrum-informed windows (SHOWCASE-0612, mirrors cz.py): rank
            # uniform-train counts by the CHARACTERIZED self-spectra at the
            # fundamental and guarantee the best windows are tried, as sync
            # and near-sync pairs; the random draws below still explore.
            ns = np.arange(2, effective_max + 1)
            w_fund = jnp.asarray(np.pi * ns / config.T_seq)
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
            print(f"  Spectrum-informed windows (characterized): {sorted(picked)}")
            for p in picked:
                n_pulses_list.append((p, p))
                if p > 1:
                    n_pulses_list.append((p, p - 1))
        for _ in range(config.num_random_trials):
            n1 = np.random.randint(0, effective_max + 1)
            n2 = np.random.randint(0, effective_max + 1)
            n_pulses_list.append((n1, n2))
        
        if not n_pulses_list:
             print("    Skipping: T_seq too small for pulses")
             best_opt_inf_ideal = 1.0
             labels_opt.append("N/A")
        else:
             print(f"  Randomly selected pulse counts: {n_pulses_list}")
             seed_seq = best_known_seq if config.use_known_as_seed else None
             best_opt_seq, best_opt_inf = optimize_random_sequences(config, config.M, n_pulses_list, seed_seq=seed_seq)
             
             # Recalculate with ideal
             if best_opt_seq:
                 best_opt_inf_ideal = calculate_infidelity(best_opt_seq, config, config.M, config.T_seq, use_ideal=True)
                 print(f"  Best Optimized (Ideal): {best_opt_inf_ideal:.6e}")
                 
                 n1_opt = len(best_opt_seq[0]) - 2
                 n2_opt = len(best_opt_seq[1]) - 2
                 label_o = f"NT({n1_opt},{n2_opt})^{config.M}"
                 labels_opt.append(label_o)
                 
                 if best_opt_inf_ideal < best_opt_inf_overall:
                     best_opt_inf_overall = best_opt_inf_ideal
                     best_opt_seq_overall = best_opt_seq
                     T_seq_best_opt = config.T_seq
                     best_opt_label_overall = label_o
             else:
                 best_opt_inf_ideal = 1.0
                 labels_opt.append("N/A")
                 print("  No optimized sequence found.")
        
        yaxis_opt.append(best_opt_inf_ideal)
        xaxis_opt.append(Tg)
        sequences_opt.append(best_opt_seq)

    # Save Results
    # Create a dedicated directory for plotting data
    plotting_dir = os.path.join(config.path, "plotting_data")
    os.makedirs(plotting_dir, exist_ok=True)

    # Note: min_gate_time (= pi/(4*Jmax)) is a CZ entangling-gate bound and is
    # not meaningful for the idling gate, so it is intentionally not saved here
    # (MINGATE-METADATA: cz.py's saved plotting_data DOES carry a
    # min_gate_time field, since a CZ gate has a minimum physical duration
    # set by the coupling strength Jmax; there is no such lower bound for
    # simply idling, so a reader diffing the two saved-file schemas should
    # not expect this key to appear here).
    save_dict = {
        'taxis': np.array(xaxis_known),
        'infs_known': np.array(yaxis_known),
        'infs_opt': np.array(yaxis_opt),
        'infs_nopulse': np.array(yaxis_nopulse),
        'tau': config.tau,
        'seed': RANDOM_SEED,
        # Frequency grid kept for reference; the full-resolution SMat is omitted
        # (the plot scripts recompute spectra from the analytic noise model
        # (qns2q.noise.spectra), so saving the 4x4x20000 matrix here is ~5 MB
        # of dead weight per file).
        'w': np.array(config.w),
        'w_max': float(config.w_max),
        'M': int(config.M),
        'Tg': float(config.Tg),
        'gate_type': 'id',
        # OPT-PROVENANCE: noise-model version the input spectra were generated
        # under (the viz overlays warn when it differs from the current model).
        # Saved so a later reader/plot script can tell, just from this file,
        # whether it is safe to compare against today's analytic model.
        'model_version': config.model_version,
        'spectral_model': config.spectral_model,
        # UNCAP-0611 provenance: the caps this run searched under
        # (max_pulses=10**9 = separation-limited; max_dim=0 = no guard).
        # Recorded so a reader of the saved npz can tell whether a given
        # infidelity curve came from an uncapped search or a capped one,
        # without having to know which CLI flags were used to produce it.
        'max_pulses': int(config.max_pulses),
        'max_dim': int(config.max_dim),
    }

    if best_known_seq_overall is not None:
        save_dict['best_known_seq_pt1'] = np.array(best_known_seq_overall[0])
        save_dict['best_known_seq_pt2'] = np.array(best_known_seq_overall[1])
        save_dict['T_seq_best_known'] = T_seq_best_known

    if best_opt_seq_overall is not None:
        save_dict['best_opt_seq_pt1'] = np.array(best_opt_seq_overall[0])
        save_dict['best_opt_seq_pt2'] = np.array(best_opt_seq_overall[1])
        save_dict['T_seq_best_opt'] = T_seq_best_opt

    np.savez(os.path.join(plotting_dir, config.plot_data_name), **save_dict)
    print(f"Saved all plotting data to {os.path.join(plotting_dir, config.plot_data_name)}")

    np.savez(os.path.join(config.path, config.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))
    np.savez(os.path.join(config.path, config.output_path_known), infs_known=np.array(yaxis_known),
             taxis=np.array(xaxis_known))

    # Final Comparison summary
    print("\n" + "=" * 80)
    print(f"{'FINAL COMPARISON (Best Overall)':^80}")
    print("=" * 80)

    inf_k_str = f"{best_known_inf_overall:.6e}" if best_known_seq_overall else "N/A"
    inf_o_str = f"{best_opt_inf_overall:.6e}" if best_opt_seq_overall else "N/A"
    print(f"{'Infidelity':<25} | {inf_k_str:<25} | {inf_o_str:<25}")

    print(f"{'Sequence':<25} | {best_known_label_overall:<25} | {best_opt_label_overall:<25}")
    print("=" * 80)

    # NOTE: this message is stale -- plot_optimization.py no longer exists in
    # this repo (the actual figure generator is scripts/report_showcase_figs.py,
    # see FIGURE_PROVENANCE.md); left as printed since it is informational
    # only and does not affect any saved data.
    print(f"\nTo generate plots, run:\n  python plot_optimization.py --data-dir {config.path} --gate-type id")

    # Return data for aggregate plotting
    return {
        'gate_times': xaxis_known,
        'known': list(zip(yaxis_known, labels_known)),
        'opt': list(zip(yaxis_opt, labels_opt)),
        'nopulse': yaxis_nopulse,
        'sequences_known': sequences_known,
        'sequences_opt': sequences_opt
    }

# `if __name__ == "__main__":` below is the standard Python idiom for "only
# run this block when the file is EXECUTED directly (e.g. `python -m
# qns2q.control.idle`), not when it is merely IMPORTED as a module" -- this
# is why scripts/harvest_design_numbers.py and scripts/showcase_storage_panel.py
# can safely `from qns2q.control import idle as idmod` to reuse this file's
# functions/classes without triggering a full, expensive optimization run as
# a side effect of the import. Everything below (argparse CLI, the M-sweep
# loop) is this module's command-line entry point, not part of its
# importable API.
if __name__ == "__main__":
    import argparse

    # Defaults match the manuscript: the idling M-sweep runs on the SPAM-free
    # reconstructed spectra (specs.npz) of the active regime's NoSPAM folder.
    # --protocol points at a SPAM arm instead (OPT-ARM-PLUMBING: the CLI
    # plumbing that maps a --protocol name to the corresponding SPAM-arm run
    # folder via qns2q.paths.run_folder(spam=True, protocol=...), mirrored
    # from cz.py so both optimizers accept the same SPAM-arm selection
    # flags); --simulated optimizes on the ground-truth file.
    parser = argparse.ArgumentParser(description="Idling-gate (DD) optimization M-sweep")
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
    parser.add_argument('--min-sep', type=float, default=1.0,
                        help="minimum pulse separation in units of tau "
                             "(SHOWCASE-0612 control-bandwidth scenario; "
                             "default 1.0 = legacy)")
    parser.add_argument('--self-only', action='store_true',
                        help="ablation rung (c): characterized model from "
                             "S11/S22 alone (S1212 + crosses dropped); the "
                             "ideal benchmark keeps the full truth")
    parser.add_argument('--informed-counts', action='store_true',
                        help="guarantee spectrum-informed pulse-count windows "
                             "(top quiet windows of the CHARACTERIZED "
                             "spectra) in the NT trial list (SHOWCASE-0612)")
    parser.add_argument('--max-pulses', type=int, default=1000,
                        help="total pulse-count cap across all repetitions "
                             "(default 1000, the published-run value); 0 = "
                             "separation-limited, i.e. only the minimum "
                             "separation tau bounds the per-repetition count "
                             "(UNCAP-0611)")
    parser.add_argument('--max-dim', type=int, default=0,
                        help="SLSQP tractability guard: clip any single NT "
                             "trial to this many optimization variables "
                             "(n1+n2); 0 = no guard. Clips are announced in "
                             "the log.")
    parser.add_argument('--tag', type=str, default="",
                        help="suffix for all output files, so a rerun does "
                             "not overwrite the published outputs")
    cli = parser.parse_args()
    cli_fname = cli.folder or (run_folder(spam=True, protocol=cli.protocol)
                               if cli.protocol else None)
    sfx = f"_{cli.tag}" if cli.tag else ""

    try:
        # Pin the RNG so the random pulse-count selection and delay seeding are
        # reproducible across the whole M sweep (SEED-OPT: without this,
        # np.random draws from whatever state Python's global RNG happens to
        # be in, so the "random" pulse counts tried -- and hence which NT
        # sequence wins -- would differ from run to run, making the
        # published infidelity numbers and winning-sequence labels
        # irreproducible). Seeded once here so each M draws from a distinct
        # but reproducible stream.
        np.random.seed(RANDOM_SEED)
        print(f"[seed={RANDOM_SEED}]")

        # Iterate through M values: 1, 2, 4, ..., 128 (2**i for i in 0..7)
        M_values = [2**i for i in range(8)] # Adjust range as needed
        results_by_M = {}
        
        # For Infidelity vs M plot (at longest gate time)
        longest_gate_time_results = []

        print(f"{'M':<10} | {'Known Inf':<20} | {'Opt Inf':<20}")
        print("-" * 56)
        
        # We need a config object to access path and tau later
        last_config = None
        
        for m in M_values:
            print(f"\n{'='*40}")
            print(f"Running Optimization for M = {m}")
            print(f"{'='*40}")

            # Initialize configuration with testing parameters
            config = Config(
                fname=cli_fname,
                include_cross_spectra=not cli.no_cross,
                spectral_model=cli.spectral_model,
                use_known_as_seed=False,
                M=m,
                max_pulses=cli.max_pulses,
                max_dim=cli.max_dim,
                min_sep_factor=cli.min_sep,
                char_self_only=cli.self_only,
                informed_counts=cli.informed_counts,
                num_random_trials=20,
                tau_divisor=160,
                use_simulated=cli.simulated,
                gate_time_factors=[-5, -4, -3, -2, -1, 0], # Range of gate times
                output_path_known=f"infs_known_id_M{m}{sfx}.npz",
                output_path_opt=f"infs_opt_id_M{m}{sfx}.npz",
                plot_filename=f"infs_GateTime_id_M{m}{sfx}.pdf",
                plot_data_name=f"plotting_data_id_v4{sfx}.npz"
            )
            last_config = config
            print(f"Configuration loaded for M={m}.")

            # Run the pipeline
            res = run_optimization_pipeline(config)
            results_by_M[m] = res
            
            # Extract data for longest gate time (last element)
            inf_k_long, label_k_long = res['known'][-1]
            inf_o_long, label_o_long = res['opt'][-1]
            inf_np_long = res['nopulse'][-1]
            
            longest_gate_time_results.append((m, inf_k_long, label_k_long, inf_o_long, label_o_long, inf_np_long))

        print("\n" + "="*60)
        print("SUMMARY OF RESULTS (Longest Gate Time)")
        print(f"{'M':<10} | {'Known Inf':<20} | {'Opt Inf':<20} | {'No Pulse Inf':<20}")
        print("-" * 78)

        known_infs = []
        known_labels = []
        opt_infs = []
        opt_labels = []
        nopulse_infs = []

        for m, k, lk, o, lo, np_inf in longest_gate_time_results:
            print(f"{m:<10} | {k:<20.6e} | {o:<20.6e} | {np_inf:<20.6e}")
            known_infs.append(k)
            known_labels.append(lk)
            opt_infs.append(o)
            opt_labels.append(lo)
            nopulse_infs.append(np_inf)
        print("="*60)
        
        if last_config:
            # Save all data generated in the optimization: unlike the
            # per-M plotting_data npz saved inside run_optimization_pipeline
            # (which gets overwritten by each new M since its filename does
            # not depend on m), this file below aggregates EVERY M's curves
            # and winning sequences into one place, with each M's arrays
            # prefixed "M{m}_" below.
            save_all_path = os.path.join(last_config.path, f"optimization_data_all_M{sfx}.npz")
            data_to_save = {}
            data_to_save['M_values'] = np.array(M_values)
            data_to_save['seed'] = RANDOM_SEED
            data_to_save['max_pulses'] = int(last_config.max_pulses)
            data_to_save['max_dim'] = int(last_config.max_dim)
            data_to_save['min_sep_factor'] = float(last_config.min_sep_factor)
            data_to_save['char_self_only'] = bool(last_config.char_self_only)
            data_to_save['informed_counts'] = bool(last_config.informed_counts)
            # Frequency grid kept for reference; SMat omitted (plot scripts
            # recompute spectra from the analytic noise model -- avoids ~5 MB
            # of dead weight).
            data_to_save['tau'] = float(last_config.tau)
            data_to_save['model_version'] = last_config.model_version
            data_to_save['spectral_model'] = last_config.spectral_model
            data_to_save['w'] = np.array(last_config.w)
            data_to_save['w_max'] = float(last_config.w_max)

            def to_numpy_seq(seq):
                if seq is None: return None
                return (np.array(seq[0]), np.array(seq[1]))

            # dtype=object below: a plain np.array() requires every element
            # to have the same shape; these per-gate-time sequence lists mix
            # None (no sequence found at that gate time) with (pt1, pt2)
            # pairs of DIFFERENT lengths (different pulse counts), so numpy
            # is told to store them as a 1-D array of arbitrary Python
            # objects instead of trying to stack them into a single
            # rectangular numeric array.
            for m, res in results_by_M.items():
                prefix = f"M{m}_"
                data_to_save[prefix + 'gate_times'] = np.array(res['gate_times'])

                infs_known, labels_known = zip(*res['known']) if res['known'] else ([], [])
                data_to_save[prefix + 'infs_known'] = np.array(infs_known)
                data_to_save[prefix + 'labels_known'] = np.array(labels_known)

                infs_opt, labels_opt = zip(*res['opt']) if res['opt'] else ([], [])
                data_to_save[prefix + 'infs_opt'] = np.array(infs_opt)
                data_to_save[prefix + 'labels_opt'] = np.array(labels_opt)

                data_to_save[prefix + 'infs_nopulse'] = np.array(res['nopulse'])

                seqs_known_np = [to_numpy_seq(s) for s in res['sequences_known']]
                seqs_opt_np = [to_numpy_seq(s) for s in res['sequences_opt']]

                data_to_save[prefix + 'sequences_known'] = np.array(seqs_known_np, dtype=object)
                data_to_save[prefix + 'sequences_opt'] = np.array(seqs_opt_np, dtype=object)

            np.savez(save_all_path, **data_to_save)
            print(f"Saved all optimization data to {save_all_path}")

            # NOTE: as above, plot_optimization.py no longer exists in this
            # repo; this message is stale and does not affect saved data
            # (see scripts/report_showcase_figs.py / FIGURE_PROVENANCE.md for
            # the actual figure-generation entry point).
            print(f"\nTo generate plots, run:")
            print(f"  python plot_optimization.py --data-dir {last_config.path} --gate-type id --all-m")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
