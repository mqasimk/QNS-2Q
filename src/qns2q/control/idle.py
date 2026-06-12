"""
Pulse Sequence Optimization for Two-Qubit Idling Gates (v4).

This script implements an optimization pipeline for dynamical decoupling sequences
to minimize infidelity in a two-qubit system. It supports:
1. Loading spectral noise data.
2. Constructing libraries of known pulse sequences (CDD, mqCDD).
3. Evaluating sequence performance using overlap integrals calculated in the time domain.
4. Optimizing random pulse sequences using JAX-based gradient descent.
5. Efficiently handling repeated sequences via time-folding strategies or frequency comb approximations.

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
jax.config.update("jax_enable_x64", True)
# OPT-SPEEDUPS (a): persistent XLA compilation cache (see cz.py).
from qns2q.paths import project_root as _project_root
jax.config.update("jax_compilation_cache_dir",
                  os.path.join(_project_root(), ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
import jax.scipy.signal
import numpy as np
import scipy.optimize

from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 MODEL_VERSION, line_priors)
from qns2q.control.tails import tail_extend_interp_complex, smoothfit_curve
from qns2q.control.padding import pad_targets, pad_count, pad_delays
from qns2q.paths import run_folder, project_root

# Fixed RNG seed for the unseeded np.random restarts (random pulse counts and
# delay seeding). Pinning it makes the published idling infidelity curves and
# winning-sequence labels reproducible across re-runs. Recorded in the saved
# optimization data so every figure carries its provenance.
RANDOM_SEED = 20260608

# OPT-SPEEDUPS (d): SLSQP convergence knobs (see cz.py for rationale).
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
    Configuration and data management for pulse sequence optimization.

    This class handles the loading of reconstructed spectral data, physical system
    parameters, and the configuration of the optimization engine. It constructs
    the interpolated spectral matrices (S-matrix) used for infidelity calculations.

    Parameters
    ----------
    fname : str, optional
        Data folder name with results from Stage 1 & 2. Defaults to the active regime's run folder (run_paths.run_folder()).
    include_cross_spectra : bool, optional
        Whether to include cross-correlated noise terms ($S_{12}, S_{1,12}$, etc.). Default is True.
    Tg : float, optional
        Target gate time in seconds.
    tau_divisor : int, optional
        Divisor of the QNS time $T$ to determine the minimum pulse interval $\tau$. Default is 160.
    M : int, optional
        Number of sequence repetitions (blocks). Default is 1.
    max_pulses : int, optional
        Maximum allowed pulses across all repetitions. Default is 100.
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
    use_simulated : bool, optional
        If True, load simulated target spectra instead of reconstructed ones. Default is False.
    gate_time_factors : list of int, optional
        Powers of 2 to scale the gate time relative to $T_{qns}$.
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
        Initialize configuration.
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
        # (OPT-SPECTRAL-MODEL).
        self.spectral_model = spectral_model

        # OPT-PROVENANCE: spectra generated under a different noise model than
        # the current one make the ideal benchmark (SMat_ideal, built from the
        # CURRENT model) a mixed-model comparison -- the trap behind
        # CA-REPRO-NUMBERS.
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
        # overlap resolution stays keyed to tau.
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
        self.char_self_only = char_self_only
        # Spectrum-informed pulse-count candidates (SHOWCASE-0612; see cz.py).
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
        # never a silent cap.
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
        experiments land there, OPT-DC-ORACLE). Only a DC-less grid falls back
        to inserting the analytic S(0) -- simulated_spectra.npz, where the file
        IS the analytic model evaluated at the teeth, or a legacy specs.npz
        (warned at load: regenerate Stage 2). The distinction is first-order
        here: for M > 10 the comb evaluator's term_dc reads SMat[..., 0]
        directly."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to 0, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        grid_has_dc = bool(float(self.wkqns[0]) == 0.0)
        if not grid_has_dc and not self.use_simulated:
            print("[idle] WARNING: specs grid carries no w=0 point -- inserting "
                  "the analytic-model DC (legacy behavior). Regenerate Stage 2 "
                  "for a measured DC point.")

        if self.spectral_model not in ('interp', 'selfconsistent', 'smoothfit'):
            raise ValueError(f"unknown spectral_model {self.spectral_model!r}")
        use_sc = (self.spectral_model == 'selfconsistent')
        use_smooth = (self.spectral_model == 'smoothfit')
        if use_sc and self.use_simulated:
            raise ValueError("spectral_model='selfconsistent' models a "
                             "reconstructed comb; simulated_spectra.npz "
                             "already IS the analytic model")
        if use_smooth:
            print("[idle] spectral_model='smoothfit': LINE-BLIND characterized "
                  "model (single power law through all teeth per self-"
                  "spectrum; crosses interp) -- ablation rung (b)")
        if use_sc:
            # OPT-SPECTRAL-MODEL: each channel from the same line/tail/head-
            # aware model the unfold bias correction uses (characterize.
            # systematics.selfconsistent_spectra). See cz.py for the full
            # rationale; the blind protocol is preserved (data + experimental
            # priors only).
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
            if use_smooth and key in ("S11", "S22", "S1212"):
                dc_val = (float(np.real(np.asarray(self.specs[key])[0]))
                          if grid_has_dc else float(np.real(dc_func(w0)[0])))
                return smoothfit_curve(self.w, self.wkqns, self.specs[key],
                                       dc_val=dc_val).astype(jnp.complex128)
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
        # prices what losing the channel costs (OPT-ROBUST-NAN). By contrast,
        # include_cross_spectra=False removes the channels from BOTH models
        # (the gate-v style counterfactual world).
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
        """Constructs the matrix of ideal analytical spectra."""
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

def remove_consecutive_duplicates(input_list):
    """Removes consecutive duplicate elements from a list."""
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
    """Recursive generation of CDD sequence."""
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)

def cddn(t0, T, n):
    """Generates CDD_n sequence with boundary points."""
    out = remove_consecutive_duplicates(cdd(t0, T, n))
    if out[0] == 0.:
        return out + [T]
    else:
        return [0.] + out + [T]

def mqCDD(T, n, m):
    """Generates multi-qubit CDD sequence."""
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
    """Generates random delay intervals summing to < T with minimum separation tau."""
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
    """Generates equidistant delay intervals."""
    if n <= 0:
        return jnp.array([])
    return jnp.ones(int(n)) * T / (n + 1)

@jax.jit
def make_tk12(tk1, tk2):
    """
    Combines two pulse sequences into a single sequence for the 12 interaction.
    Assumes inputs are [0, t1..., T] and [0, t2..., T].
    """
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """
    Constructs a library of known pulse sequences (CDD permutations and mqCDD).
    
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

@functools.partial(jax.jit, static_argnames=['M', 'n_base_steps'])
def precompute_R_folded(R_shifted, lags_R, M, T_base, dt, n_base_steps):
    """
    Precomputes the folded noise autocorrelation R_folded(u) on the domain of C_1(u).
    R_folded(u) = sum_{p=-(M-1)}^{M-1} (M - |p|) * R(u + p*T_base)
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
        
        # vmap over p to compute all slices
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

    Returns
    -------
    RMat_data : jnp.ndarray, shape (4, 4, 2*n_base_steps-1)
        Folded noise correlation matrix on the base-sequence lag grid.
    dt : float
        Time-domain grid spacing (seconds).
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

    # Iterate over all 16 Pauli operators P_k = P1 \otimes P2
    for p1 in range(4):
        for p2 in range(4):
            # Determine commutation of P_k with Z_1, Z_2, Z_12
            c1 = comm_with_z[p1]
            c2 = comm_with_z[p2]
            c12 = (c1 + c2) % 2
            
            # lambda_a^{(k)} = sgn(P_k, a, 0) - 1
            lam = jnp.array([
                0.0,          # lambda_0
                -2.0 * c1,    # lambda_1
                -2.0 * c2,    # lambda_2
                -2.0 * c12    # lambda_12
            ])
            
            # Calculate Theta_l = sum_{j} lambda_j * I_{j, j XOR l}
            # Vectorized calculation for l=0,1,2,3
            Thetas = jnp.zeros(4, dtype=jnp.complex128)
            for l in range(4):
                # Indices for j XOR l
                col_indices = jnp.arange(4) ^ l
                # Sum over j: lam[j] * I[j, col_indices[j]]
                val = -jnp.sum(lam * I_matrix[jnp.arange(4), col_indices])
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

    The comb samples S at delta teeth; the TRUE M-fold filter tooth has width
    ~ 2pi/(M*T_seq). When that width is comparable to the nuclear-line width
    sigma the comb mis-weights the lines: 8-14% infidelity error at
    Tg = 320 tau / M = 16, 3-7% at Tg = 640, up to 3.2% at Tg = 1280; every
    point passing 2pi/Tg < sigma/8 measures <= 1.7% (OPT-COMB-M16 diagnostic
    + boundary sweep, scripts/diag_comb_vs_folded.py). Smooth (bland) spectra
    keep the legacy speed cutoff M > 10. Note the published-number path
    (calculate_infidelity with use_ideal=True) is always folded regardless."""
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
    """[S(0), S(w_k)] packed per channel for evaluate_overlap_comb."""
    S_flat = SMat.reshape(-1, SMat.shape[-1])

    def interp_row(fp):
        return (jnp.interp(omega_k, w_grid, jnp.real(fp), right=0.) +
                1j * jnp.interp(omega_k, w_grid, jnp.imag(fp), right=0.))

    S_h = jax.vmap(interp_row)(S_flat)
    return jnp.concatenate([S_flat[:, :1], S_h], axis=1).reshape(4, 4, -1)


def _delays_to_pts(delays_params, n_pulses1, T_seq):
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
    """Idling cost (1 - F/16) on the folded evaluator. Every input is a
    runtime ARGUMENT (OPT-SPEEDUPS (b)): the value_and_grad wrappers below
    are stable module-level objects, so restarts and (Tg, M) blocks reuse
    compiled programs per (shape, static) instead of recompiling per fresh
    closure, and the HLO is spectrum-value-independent (persistent cache
    works across runs/repeats)."""
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


cost_vag_folded = jax.jit(jax.value_and_grad(_cost_folded),
                          static_argnames=('n_pulses1', 'n_base_steps'))
cost_vag_comb = jax.jit(jax.value_and_grad(_cost_comb),
                        static_argnames=('n_pulses1', 'M'))

def optimize_random_sequences(config, M, n_pulses_list, seed_seq=None):
    """Optimizes random sequences for a given repetition count M."""
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

        # Linear constraints
        A = np.zeros((2, n1 + n2))
        A[0, :n1] = 1
        A[1, n1:] = 1
        linear_cons = scipy.optimize.LinearConstraint(A, -np.inf,
                                                      T_seq - config.min_sep)

        try:
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
    """Evaluates all sequences in the library."""
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
    Runs the full optimization pipeline based on the provided configuration.
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
    # (MINGATE-METADATA). id_plots.py reads it only if present.
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
        'model_version': config.model_version,
        'spectral_model': config.spectral_model,
        # UNCAP-0611 provenance: the caps this run searched under
        # (max_pulses=10**9 = separation-limited; max_dim=0 = no guard).
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

if __name__ == "__main__":
    import argparse

    # Defaults match the manuscript: the idling M-sweep runs on the SPAM-free
    # reconstructed spectra (specs.npz) of the active regime's NoSPAM folder.
    # --protocol points at a SPAM arm instead (OPT-ARM-PLUMBING); --simulated
    # optimizes on the ground-truth file.
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
                        choices=('interp', 'selfconsistent', 'smoothfit'),
                        default='interp',
                        help="characterized-SMat construction: linear interp "
                             "through the teeth (+tails); the unfold "
                             "model's line/tail/head-aware spectra "
                             "(OPT-SPECTRAL-MODEL); or the LINE-BLIND single "
                             "power law (ablation rung (b), SHOWCASE-0612)")
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
        # reproducible across the whole M sweep (SEED-OPT). Seeded once here so
        # each M draws from a distinct but reproducible stream.
        np.random.seed(RANDOM_SEED)
        print(f"[seed={RANDOM_SEED}]")

        # Iterate through M values: 1, 2, 4, ..., 512
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
            # Save all data generated in the optimization
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

            print(f"\nTo generate plots, run:")
            print(f"  python plot_optimization.py --data-dir {last_config.path} --gate-type id --all-m")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
