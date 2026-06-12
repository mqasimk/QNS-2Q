"""
Pulse Sequence Optimization for CZ Gates (v2).

This script implements an optimization pipeline for CZ gate sequences
to minimize infidelity in a two-qubit system. It supports:
1. Loading spectral noise data.
2. Constructing libraries of known pulse sequences (CDD, mqCDD).
3. Evaluating sequence performance using overlap integrals calculated in the time domain.
4. Optimizing random pulse sequences using JAX-based gradient descent.
5. Optimizing the coupling strength J dynamically.

Based on id_optimize.py (and the now-removed cz_optimize_legacy.py).

Author: [Q]
Date: [01/18/2026]
"""

import functools
import itertools
import os
import traceback
from dataclasses import dataclass, field

import jax
jax.config.update("jax_enable_x64", True)
# OPT-SPEEDUPS (a): persistent XLA compilation cache. With the spectra passed
# as runtime ARGUMENTS (not closure constants -- see cost_vag_*), the compiled
# programs are value-independent, so repeats/reruns at the same shapes
# deserialize (~0.1 s) instead of recompiling (~1 s each).
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
from qns2q.control.tails import tail_extend_interp_complex, smoothfit_curve
from qns2q.control.padding import pad_targets, pad_count, pad_delays
from qns2q.paths import run_folder, project_root


# Fixed RNG seed for the unseeded np.random restarts (pulse-count / delay
# initialization). Pinning it makes the published infidelity curves and the
# winning-sequence labels reproducible across re-runs. Recorded in the saved
# plotting_data so every figure carries its provenance.
RANDOM_SEED = 20260608


# OPT-SPEEDUPS (d): SLSQP convergence knobs. tol=1e-10 / maxiter=1000
# over-converged an objective whose spectral inputs carry 5-20%
# reconstruction uncertainty; the host-driven iteration stream was a major
# wall-time term (each iteration is a ~2 ms device call dispatched from
# scipy). Winners validated unchanged against the 2026-06-11 pre-change run.
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
    """Configuration for the CZ gate optimization."""
    fname: str = field(default_factory=run_folder)
    parent_dir: str = os.pardir
    # Max Ising coupling in tau units (J*tau): 0.05 = 2e6 rad/s x 25 ns (legacy SI anchor)
    Jmax: float = 0.05
    # Extended gate time factors to include larger gate times (-1, 0)
    gate_time_factors: list = field(default_factory=lambda: [3, 2, 1, 0, -1, -2, -3])
    output_path_known: str = "infs_known_cz_v2.npz"
    output_path_opt: str = "infs_opt_cz_v2.npz"
    plot_filename: str = "infs_GateTime_cz_v2.pdf"
    
    include_cross_spectra: bool = True
    tau_divisor: int = 160
    use_simulated: bool = False
    # 'interp' = linear interpolation through the comb teeth (+ tails);
    # 'selfconsistent' = the unfold model's line/tail/head-aware spectra
    # (OPT-SPECTRAL-MODEL); 'smoothfit' = the LINE-BLIND single power law
    # through all teeth (ablation rung (b), SHOWCASE-0612 -- prices what the
    # line-aware reconstruction adds; tails.smoothfit_curve).
    spectral_model: str = "interp"
    # Per-qubit pulse-count cap for the NT search and the known-sequence
    # library. 0 = no cap beyond the minimum-separation feasibility limit,
    # i.e. up to T_seq/min_sep - 1 pulses per qubit (UNCAP-0611).
    max_pulses: int = 150
    # Minimum pulse separation in units of tau (SHOWCASE-0612 scenario
    # parameter): finite control bandwidth -- e.g. 8.0 models 40 ns pi-pulses
    # at the 5 ns showcase anchor. Applied SYMMETRICALLY to the known library,
    # the NT search and the random inits; 1.0 = legacy (separation = tau).
    min_sep_factor: float = 1.0
    # Ablation rung (c): build the CHARACTERIZED gate model from the
    # single-qubit spectra alone (S11/S22 kept; S1212 + every cross dropped,
    # as a 1Q-only QNS campaign would leave them) while the ideal benchmark
    # keeps the full truth -- prices what the TWO-QUBIT reconstruction adds.
    # Contrast include_cross_spectra=False, which drops crosses from BOTH.
    char_self_only: bool = False
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
        """Constructs the matrix of interpolated spectra from QNS data.

        Beyond the comb's last tooth each component is extended with a power
        law fitted to the top teeth (control.tails) instead of right=0 -- the
        zero extension let the optimizer park its filter weight in the
        unmeasured band as if it were noise-free.

        The w=0 point comes from the data grid whenever it carries a DC sample
        (reconstructed specs.npz: the noise-aware slope-fit / double-echo DC
        experiments land there, OPT-DC-ORACLE). Only a DC-less grid falls back
        to inserting the analytic S(0) -- simulated_spectra.npz, where the file
        IS the analytic model evaluated at the teeth, or a legacy specs.npz
        (warned at load: regenerate Stage 2)."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to Null, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        grid_has_dc = bool(float(self.wkqns[0]) == 0.0)
        if not grid_has_dc and not self.use_simulated:
            print("[cz] WARNING: specs grid carries no w=0 point -- inserting "
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
            print("[cz] spectral_model='smoothfit': LINE-BLIND characterized "
                  "model (single power law through all teeth per self-"
                  "spectrum; crosses interp) -- ablation rung (b)")
        if use_sc:
            # OPT-SPECTRAL-MODEL: each channel from the same line/tail/head-
            # aware model the unfold bias correction uses (characterize.
            # systematics.selfconsistent_spectra): Gaussian lines at the
            # experimentally-known nuclear-difference centers (heights
            # NNLS-fitted from the comb, S11/S22 only), power-law tails,
            # saturated power-law head below tooth 1. Assumption-light --
            # everything is the reconstructed data or an experimental prior,
            # so the blind protocol is preserved. NaN (unreconstructed)
            # channels are zero-filled here; the cross() drop below still
            # excludes them from the SMat.
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
            print("[cz] char_self_only: S1212 + crosses DROPPED from the "
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
        S11_0 = jnp.real(self.SMat_ideal[1, 1, 0])
        S22_0 = jnp.real(self.SMat_ideal[2, 2, 0])
        
        self.T2q1 = 2.0 / S11_0 if S11_0 > 0 else jnp.inf
        self.T2q2 = 2.0 / S22_0 if S22_0 > 0 else jnp.inf
        
        print(f"Calculated T2 times (Ideal): Q1={self.T2q1:.2e} tau, Q2={self.T2q2:.2e} tau")

# ==============================================================================
# Sequence Generation Utilities
# ==============================================================================

def remove_consecutive_duplicates(input_list):
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
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)

def cddn(t0, T, n):
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
    tk_arr = jnp.array(tk)
    if len(tk_arr) <= 2: 
        return jnp.array([])
    diffs = jnp.diff(tk_arr)
    return diffs[:-1]

@jax.jit
def delays_to_pulse_times(delays, T):
    if delays.size == 0:
        return jnp.array([0., T])
    last_delay = T - jnp.sum(delays)
    all_delays = jnp.concatenate([delays, jnp.array([last_delay])])
    times = jnp.cumsum(all_delays)
    return jnp.concatenate([jnp.array([0.]), times])

def get_random_delays(n, T, tau):
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
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """
    Constructs a library of known pulse sequences (CDD permutations and mqCDD).
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

@functools.partial(jax.jit, static_argnames=['M', 'n_base_steps'])
def precompute_R_folded(R_shifted, lags_R, M, T_base, dt, n_base_steps):
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
    exp_t = jnp.exp(1j * jnp.outer(pulse_times, omega))
    diffs = exp_t[1:] - exp_t[:-1]
    n_intervals = len(pulse_times) - 1
    signs = jnp.array((-1.0)**jnp.arange(n_intervals))
    return jnp.sum(signs[:, None] * diffs, axis=0)

@functools.partial(jax.jit, static_argnames=['M'])
def evaluate_overlap_comb(pulse_times_a, pulse_times_b, S_packed, omega_k, T_seq, M):
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


# ==============================================================================
# CZ Fidelity Calculation
# ==============================================================================

@jax.jit
def zzPTM():
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    U = jax.scipy.linalg.expm(-1j * jnp.kron(p1q[3], p1q[3]) * jnp.pi / 4)
    gamma = jnp.array([[(1 / 4) * jnp.trace(p2q[i] @ U @ p2q[j] @ U.conj().transpose()) for j in range(16)] for i in
                       range(16)])
    return jnp.real(gamma)

@jax.jit
def sgn(O, a, b):
    # a, b are indices 0, 1, 2, 3 corresponding to II, ZI, IZ, ZZ
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

                # Second-cumulant decay coefficient. Mirrors calculate_idling_fidelity
                # (id_optimize.py): the per-channel weight is
                #   -1/2 * (sgn(O, a, 0) - 1) * I_{a,b},  summed over a, b in {1, 2, 12}.
                # I_matrix is the (1/2pi)-normalized overlap returned by
                # evaluate_overlap_comb, so it enters directly (no extra factor of 2),
                # and every (a, b) pair contributes (no gating by sgn(O, a, b)).
                coeff = -0.5 * (sgn(Oi, idx_i, 0) - 1.0)

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
    """Whether the frequency-comb overlap approximation is valid at (M, T_seq).

    The comb samples S at delta teeth; the TRUE M-fold filter tooth has width
    ~ 2pi/(M*T_seq). When that width is comparable to the nuclear-line width
    sigma the comb mis-weights the lines (OPT-COMB-M16 diagnostic + boundary
    sweep, scripts/diag_comb_vs_folded.py: 8-14% at Tg = 320 tau / M = 16,
    3-7% at 640, up to 3.2% at 1280; <= 1.7% past 2pi/Tg < sigma/8). Smooth
    (bland) spectra keep the legacy speed cutoff M > 10. (CZ currently runs
    M = 1, so this is future-proofing kept in lockstep with idle.py.)"""
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


def _cz_inf_from_imat(I_mat, pt12, Jmax, M):
    """Shared tail of the CZ cost: J from the achievable phase + penalty."""
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
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    return [pt0, pt1, pt2, pt12]


def _cost_folded(delays_params, RMat_data, dt, T_seq, Jmax, M,
                 n_pulses1, n_base_steps):
    """CZ cost on the folded evaluator. Every input is a runtime ARGUMENT
    (OPT-SPEEDUPS (b)): the value_and_grad wrappers below are stable
    module-level objects, so restarts and gate times reuse compiled programs
    per (shape, static) instead of recompiling per fresh closure -- and the
    HLO is spectrum-value-independent, so the persistent compilation cache
    works across runs and repeats."""
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

# ==============================================================================
# Main Execution
# ==============================================================================

def run_optimization(config):
    # Pin the RNG so the random pulse-count selection and delay seeding are
    # reproducible (SEED-OPT). Without this the headline CZ infidelities and
    # winning sequences drift between runs.
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
                if (n1 + 1) * config.tau > T_seq or (n2 + 1) * config.tau > T_seq:
                     continue

                print(f"  Optimizing n=({n1}, {n2})...")
                # Use best known as seed if available and pulse counts match?
                # For now, just random init
                seq, inf = optimize_sequence(config, M, T_seq, n1, n2,
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
    # --protocol points at a SPAM arm instead (OPT-ARM-PLUMBING); --simulated
    # optimizes on the ground-truth file (the acceptance-gate probe).
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
                        choices=('interp', 'selfconsistent', 'smoothfit'),
                        default='interp',
                        help="characterized-SMat construction: linear interp "
                             "through the teeth (+tails); the unfold "
                             "model's line/tail/head-aware spectra "
                             "(OPT-SPECTRAL-MODEL); or the LINE-BLIND single "
                             "power law (ablation rung (b), SHOWCASE-0612)")
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
                  output_path_known=f"infs_known_cz_v2{sfx}.npz",
                  output_path_opt=f"infs_opt_cz_v2{sfx}.npz",
                  plot_data_name=f"plotting_data_cz_v2{sfx}.npz")
    if cli.factors is not None:
        kwargs['gate_time_factors'] = [int(f) for f in cli.factors.split(',')]
    config = CZOptConfig(**kwargs)
    run_optimization(config)
