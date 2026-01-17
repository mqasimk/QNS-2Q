"""
Pulse Sequence Optimization for Two-Qubit Idling Gates (v4).

This script implements an optimization pipeline for dynamical decoupling sequences
to minimize infidelity in a two-qubit system. It supports:
1. Loading spectral noise data.
2. Constructing libraries of known pulse sequences (CDD, mqCDD).
3. Evaluating sequence performance using overlap integrals calculated in the time domain.
4. Optimizing random pulse sequences using JAX-based gradient descent.
5. Efficiently handling repeated sequences via time-folding strategies or frequency comb approximations.

Author: [Your Name/Organization]
Date: [Current Date]
"""

import functools
import itertools
import os
import traceback

import jax
import jax.numpy as jnp
import jax.scipy.integrate
jax.config.update("jax_enable_x64", True)
import jax.scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.optimize

from spectraIn import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
import plot_utils

# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """
    Configuration class for pulse optimization.
    
    Handles loading of spectral data, system parameters, and optimization settings.
    Constructs the interpolated and ideal spectral matrices used for calculations.
    """
    def __init__(self, 
                 fname="DraftRun_NoSPAM_Feature",
                 include_cross_spectra=True,
                 Tg=4 * 14 * 1e-6, # Default, will be overridden by loop
                 tau_divisor=160, 
                 
                 # Optimization/Testing Parameters
                 M=1,                      # Repetition count
                 max_pulses=100,           # Total max pulses allowed
                 num_random_trials=10,     # Number of random sequences to optimize
                 use_known_as_seed=False,  # Use best known sequence as seed
                 
                 # Output settings
                 output_path_known="infs_known_id_v4.npz",
                 output_path_opt="infs_opt_id_v4.npz",
                 plot_filename="infs_GateTime_id_v4.pdf",
                 
                 # Advanced/Unused Parameters (kept for compatibility)
                 reps_known=None, 
                 reps_opt=None,
                 use_simulated=False,
                 gate_time_factors=None
                 ):
        """
        Initialize configuration.
        """
        # Paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.path = os.path.join(project_root, fname)
        
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
             self.specs = np.load(os.path.join(self.path, "specs.npz"))
             self.params = np.load(os.path.join(self.path, "params.npz"))
        
        # System Parameters
        self.Tqns = float(self.params['T'])
        self.mc = int(self.params['truncate'])
        self.gamma = float(self.params['gamma'])
        self.gamma12 = float(self.params['gamma_12'])
        self.include_cross_spectra = include_cross_spectra
        
        # Optimization Parameters
        self.Tg = Tg
        self.tau_divisor = tau_divisor
        self.tau = self.Tqns / tau_divisor
        
        self.M = M
        self.max_pulses = max_pulses
        self.num_random_trials = num_random_trials
        self.use_known_as_seed = use_known_as_seed
        
        self.output_path_known = output_path_known
        self.output_path_opt = output_path_opt
        self.plot_filename = plot_filename
        
        self.gate_time_factors = gate_time_factors if gate_time_factors is not None else [-1, 0, 1, 2, 3, 4, 5, 6]
        
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
        # Multiplier to ensure dt is small enough (e.g. ~10-100ns)
        self.w_max = 4 * w_max_sys
        self.N_w = 20000
        
        self.w = jnp.linspace(0, self.w_max, self.N_w)
        self.w_ideal = jnp.linspace(0, 2 * self.w_max, 2 * self.N_w)
        
        if use_simulated and 'wk' in self.specs:
            self.wkqns = jnp.array(self.specs['wk'])
        else:
            self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])
        
        # Spectral Matrices
        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()

    def _build_interpolated_spectra(self):
        """Constructs the matrix of interpolated spectra from QNS data."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to 0, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        
        def interp_c(fp):
            """Interpolates complex data."""
            # Use right=0. to assume noise decays to zero at high frequencies beyond data
            return (jnp.interp(self.w, self.wkqns, jnp.real(fp), right=0.) +
                   1j * jnp.interp(self.w, self.wkqns, jnp.imag(fp), right=0.))

        def combine(spec_data, dc_func, *args):
            """Interpolates and inserts exact DC value."""
            interp = interp_c(spec_data)
            dc_val = dc_func(w0, *args)[0]
            return interp.at[0].set(dc_val)
        
        # Diagonal elements
        SMat = SMat.at[1, 1].set(combine(self.specs["S11"], S_11))
        SMat = SMat.at[2, 2].set(combine(self.specs["S22"], S_22))
        SMat = SMat.at[3, 3].set(combine(self.specs["S1212"], S_1212))
        
        # Off-diagonal elements
        if self.include_cross_spectra:
            # 1-2
            s12 = combine(self.specs["S12"], S_1_2, self.gamma)
            SMat = SMat.at[1, 2].set(s12)
            SMat = SMat.at[2, 1].set(jnp.conj(s12))
            # 1-12 (Index 1-3)
            s112 = combine(self.specs["S112"], S_1_12, self.gamma12)
            SMat = SMat.at[1, 3].set(s112)
            SMat = SMat.at[3, 1].set(jnp.conj(s112))
            # 2-12 (Index 2-3)
            s212 = combine(self.specs["S212"], S_2_12, self.gamma12 - self.gamma)
            SMat = SMat.at[2, 3].set(s212)
            SMat = SMat.at[3, 2].set(jnp.conj(s212))
        
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
            SMat_ideal = SMat_ideal.at[1, 2].set(S_1_2(self.w_ideal, self.gamma))
            SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_1_2(self.w_ideal, self.gamma)))
            # 1-12 (Index 1-3)
            SMat_ideal = SMat_ideal.at[1, 3].set(S_1_12(self.w_ideal, self.gamma12))
            SMat_ideal = SMat_ideal.at[3, 1].set(jnp.conj(S_1_12(self.w_ideal, self.gamma12)))
            # 2-12 (Index 2-3)
            SMat_ideal = SMat_ideal.at[2, 3].set(S_2_12(self.w_ideal, self.gamma12 - self.gamma))
            SMat_ideal = SMat_ideal.at[3, 2].set(jnp.conj(S_2_12(self.w_ideal, self.gamma12 - self.gamma)))
        
        return SMat_ideal


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
    tk1 = remove_consecutive_duplicates(cdd(0., T, n))
    tk2 = []
    for i in range(len(tk1)-1):
        tk2 += remove_consecutive_duplicates(cdd(tk1[i], tk1[i+1]-tk1[i], m))
    tk2 += remove_consecutive_duplicates(cdd(tk1[-1], T-tk1[-1], m))
    
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

    # 2. Create permutations
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
    weights = float(M) - jnp.abs(p_vals)
    
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
    
    return jnp.real(integral)

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
    # Take real part of DC component explicitly
    S_0 = jnp.real(S_packed[0])
    
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

@jax.jit
def calculate_idling_fidelity(I_matrix):
    """
    Calculates the Idling Gate Fidelity F1(T) based on the overlap integrals.
    Uses a numerically stable formula involving exponentials of sums.
    
    I_matrix indices: 0->0, 1->1, 2->2, 3->12.
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


# ==============================================================================
# Optimization Logic
# ==============================================================================

def cost_function(delays_params, n_pulses1, RMat_data, T_seq, tau_min, overlap_fn):
    """
    Cost function: 1 - Normalized Fidelity.
    overlap_fn: A callable that takes (pt_a, pt_b, data) and returns the overlap integral.
    """
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
    
    # Dummy pulse sequence for index 0 (Identity). 
    # Since SMat[0,:] is 0, the overlap will be 0 regardless of sequence.
    pt0 = jnp.array([0., T_seq]) 
    
    pts = [pt0, pt1, pt2, pt12]
    
    # Calculate I_matrix (4x4)
    vals = []
    for i in range(4):
        row_vals = []
        for j in range(4):
            val = overlap_fn(pts[i], pts[j], RMat_data[i, j])
            row_vals.append(val)
        vals.append(row_vals)
    
    I_mat = jnp.array(vals)
    
    fid = calculate_idling_fidelity(I_mat)
    norm_fid = fid / 16.0
    infidelity = 1.0 - norm_fid
    
    return infidelity

def optimize_random_sequences(config, M, n_pulses_list, seed_seq=None):
    """Optimizes random sequences for a given repetition count M."""
    T_seq = config.Tg / M
    best_inf = 1.0
    best_seq = None
    
    # Setup Evaluation Method
    use_comb = (M > 10)
    
    if use_comb:
        print(f"Using Frequency Comb Approximation (M={M})...")
        # 1. Determine Harmonics
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        k_vals = jnp.arange(1, max_k + 1)
        omega_k = k_vals * w0
        
        # 2. Interpolate Spectra
        S_flat = config.SMat.reshape(-1, config.SMat.shape[-1])
        
        def interp_row(fp):
            return (jnp.interp(omega_k, config.w, jnp.real(fp), right=0.) +
                   1j * jnp.interp(omega_k, config.w, jnp.imag(fp), right=0.))
                   
        S_harmonics_flat = jax.vmap(interp_row)(S_flat) # (16, K)
        S_DC_flat = S_flat[:, 0]
        
        # Pack: (16, K+1) -> (4, 4, K+1)
        S_packed_flat = jnp.concatenate([S_DC_flat[:, None], S_harmonics_flat], axis=1)
        RMat_data = S_packed_flat.reshape(4, 4, -1)
        
        @jax.jit
        def overlap_fn(pt_a, pt_b, data):
            return evaluate_overlap_comb(pt_a, pt_b, data, omega_k, T_seq, M)
            
    else:
        print(f"Using Folded Noise Matrix (M={M})...")
        N = config.w.shape[0]
        dw = config.w[1] - config.w[0]
        dt = (2 * np.pi / (N * dw))
        lags_R = (jnp.arange(N) - N//2) * dt
        
        RMat_vals = jnp.fft.ifft(config.SMat, axis=-1)
        RMat_scaled = RMat_vals / dt
        RMat_shifted = jnp.fft.fftshift(RMat_scaled, axes=-1)
        
        n_base_steps = int(np.ceil(T_seq / dt))
        
        @jax.jit
        def get_folded_matrix(RMat_in):
            R_flat = RMat_in.reshape(-1, RMat_in.shape[-1])
            folded_flat = jax.vmap(lambda r: precompute_R_folded(r, lags_R, M, T_seq, dt, n_base_steps))(R_flat)
            return folded_flat.reshape(4, 4, -1)

        RMat_data = get_folded_matrix(RMat_shifted)
        
        @jax.jit
        def overlap_fn(pt_a, pt_b, data):
            return evaluate_overlap_folded(pt_a, pt_b, data, dt, n_base_steps)
    
    def run_single_optimization(n1, n2, initial_params, label):
        nonlocal best_inf, best_seq

        # Bounds
        bounds = [(config.tau, T_seq) for _ in range(len(initial_params))]
        
        # Partial cost function for JIT
        cost_for_n = functools.partial(cost_function, n_pulses1=n1, 
                                       RMat_data=RMat_data, 
                                       T_seq=T_seq, tau_min=config.tau, 
                                       overlap_fn=overlap_fn)
        
        val_and_grad = jax.jit(jax.value_and_grad(cost_for_n))

        def fun_wrapper(x):
            v, g = val_and_grad(x)
            return float(v), np.array(g)

        # Linear constraints
        A = np.zeros((2, n1 + n2))
        A[0, :n1] = 1
        A[1, n1:] = 1
        linear_cons = scipy.optimize.LinearConstraint(A, -np.inf, T_seq - config.tau)
        
        try:
            res = scipy.optimize.minimize(fun_wrapper, np.array(initial_params), method='SLSQP',
                                          bounds=bounds, constraints=linear_cons, jac=True,
                                          tol=1e-14, options={'maxiter': 1000, 'disp': False})
            
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
        if (n1 + 1) * config.tau > T_seq or (n2 + 1) * config.tau > T_seq:
            print(f"Skipping Random Opt (n={n1},{n2}): Over-constrained (Too many pulses for T_seq)")
            continue

        d1 = get_random_delays(n1, T_seq, config.tau)
        d2 = get_random_delays(n2, T_seq, config.tau)
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
    use_comb = (M > 10)
    
    if use_comb:
        print(f"Using Frequency Comb Approximation (M={M})...")
        w0 = 2 * jnp.pi / T_seq
        max_k = int(config.w_max / w0)
        k_vals = jnp.arange(1, max_k + 1)
        omega_k = k_vals * w0
        
        S_flat = config.SMat.reshape(-1, config.SMat.shape[-1])
        
        def interp_row(fp):
            return (jnp.interp(omega_k, config.w, jnp.real(fp), right=0.) +
                   1j * jnp.interp(omega_k, config.w, jnp.imag(fp), right=0.))
                   
        S_harmonics_flat = jax.vmap(interp_row)(S_flat)
        S_DC_flat = S_flat[:, 0]
        S_packed_flat = jnp.concatenate([S_DC_flat[:, None], S_harmonics_flat], axis=1)
        RMat_data = S_packed_flat.reshape(4, 4, -1)
        
        @jax.jit
        def overlap_fn(pt_a, pt_b, data):
            return evaluate_overlap_comb(pt_a, pt_b, data, omega_k, T_seq, M)
            
    else:
        print(f"Using Folded Noise Matrix (M={M})...")
        N = config.w.shape[0]
        dw = config.w[1] - config.w[0]
        dt = (2 * np.pi / (N * dw))
        lags_R = (jnp.arange(N) - N//2) * dt
        
        RMat_vals = jnp.fft.ifft(config.SMat, axis=-1)
        RMat_scaled = RMat_vals / dt
        RMat_shifted = jnp.fft.fftshift(RMat_scaled, axes=-1)
        
        n_base_steps = int(np.ceil(T_seq / dt))
        
        @jax.jit
        def get_folded_matrix(RMat_in):
            R_flat = RMat_in.reshape(-1, RMat_in.shape[-1])
            folded_flat = jax.vmap(lambda r: precompute_R_folded(r, lags_R, M, T_seq, dt, n_base_steps))(R_flat)
            return folded_flat.reshape(4, 4, -1)

        RMat_data = get_folded_matrix(RMat_shifted)
        
        @jax.jit
        def overlap_fn(pt_a, pt_b, data):
            return evaluate_overlap_folded(pt_a, pt_b, data, dt, n_base_steps)
        
    for i, (d1, d2) in enumerate(pLib):
        pt1 = delays_to_pulse_times(d1, T_seq)
        pt2 = delays_to_pulse_times(d2, T_seq)
        pt12 = make_tk12(pt1, pt2)
        pt0 = jnp.array([0., T_seq])
        
        pts = [pt0, pt1, pt2, pt12]
        
        vals = []
        for r in range(4):
            row_vals = []
            for c in range(4):
                val = overlap_fn(pts[r], pts[c], RMat_data[r, c])
                row_vals.append(val)
            vals.append(row_vals)
        I_mat = jnp.array(vals)
        
        fid = calculate_idling_fidelity(I_mat)
        inf = 1.0 - fid / 16.0
        
        if inf < best_inf:
            best_inf = inf
            best_seq = (pt1, pt2)
            best_idx = i
            
    return best_seq, best_inf, best_idx

def calculate_infidelity(seq, config, M, T_seq, use_ideal=False):
    if seq is None: return 1.0
    
    SMat = config.SMat_ideal if use_ideal else config.SMat
    w_grid = config.w_ideal if use_ideal else config.w
    
    use_comb = (M > 10)
    
    if use_comb:
        w0 = 2 * jnp.pi / T_seq
        limit_w = w_grid[-1]
        max_k = int(limit_w / w0)
        
        k_vals = jnp.arange(1, max_k + 1)
        omega_k = k_vals * w0
        
        S_flat = SMat.reshape(-1, SMat.shape[-1])
        def interp_row(fp):
            return (jnp.interp(omega_k, w_grid, jnp.real(fp), right=0.) +
                   1j * jnp.interp(omega_k, w_grid, jnp.imag(fp), right=0.))
        S_harmonics_flat = jax.vmap(interp_row)(S_flat)
        S_DC_flat = S_flat[:, 0]
        S_packed_flat = jnp.concatenate([S_DC_flat[:, None], S_harmonics_flat], axis=1)
        RMat_data = S_packed_flat.reshape(4, 4, -1)
        
        overlap_fn = lambda pt_a, pt_b, data: evaluate_overlap_comb(pt_a, pt_b, data, omega_k, T_seq, M)
        
    else:
        N = w_grid.shape[0]
        dw = w_grid[1] - w_grid[0]
        dt = (2 * np.pi / (N * dw))
        lags_R = (jnp.arange(N) - N//2) * dt
        
        RMat_vals = jnp.fft.ifft(SMat, axis=-1)
        RMat_scaled = RMat_vals / dt
        RMat_shifted = jnp.fft.fftshift(RMat_scaled, axes=-1)
        
        n_base_steps = int(np.ceil(T_seq / dt))
        
        def get_folded_matrix(RMat_in):
            R_flat = RMat_in.reshape(-1, RMat_in.shape[-1])
            folded_flat = jax.vmap(lambda r: precompute_R_folded(r, lags_R, M, T_seq, dt, n_base_steps))(R_flat)
            return folded_flat.reshape(4, 4, -1)
        RMat_data = get_folded_matrix(RMat_shifted)
        
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
    print(f"\n{'='*60}")
    print(f"RUNNING OPTIMIZATION PIPELINE")
    print(f"M (Repetitions): {config.M}")
    print(f"Total Pulse Limit: {config.max_pulses}")
    print(f"Pulse Limit Per Repetition: {config.max_pulses_per_rep}")
    print(f"{'='*60}")
    
    yaxis_opt, xaxis_opt = [], []
    yaxis_known, xaxis_known = [], []
    yaxis_nopulse = []
    
    best_known_seq_overall = None
    best_opt_seq_overall = None
    T_seq_best_known = None
    T_seq_best_opt = None
    best_known_inf_overall = 1.0
    best_opt_inf_overall = 1.0
    
    for i in config.gate_time_factors:
        Tg = config.Tqns / 2**(i-1)
        if Tg < config.tau:
            continue
            
        # Update config for this iteration
        config.Tg = Tg
        config.T_seq = Tg / config.M
        
        print(f"\nGate Time: {Tg*1e6:.2f} us")
        
        # No Pulse Calculation
        pt_nopulse = jnp.array([0., config.T_seq])
        seq_nopulse = (pt_nopulse, pt_nopulse)
        inf_nopulse = calculate_infidelity(seq_nopulse, config, config.M, config.T_seq, use_ideal=True)
        yaxis_nopulse.append(inf_nopulse)
        print(f"  No Pulse (Ideal): {inf_nopulse:.6e}")
        
        # 1. Known Sequences
        pLib_delays, pLib_desc = construct_pulse_library(config.T_seq, config.tau, config.max_pulses_per_rep)
        
        best_known_seq = None
        best_known_inf = 1.0
        idx = -1
        
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences(config, config.M, pLib_delays)
            print(f"  Best Known (Char): {best_known_inf:.6e} ({pLib_desc[idx]})")
            
            # Recalculate with ideal
            best_known_inf_ideal = calculate_infidelity(best_known_seq, config, config.M, config.T_seq, use_ideal=True)
            print(f"  Best Known (Ideal): {best_known_inf_ideal:.6e}")
            
            if best_known_inf_ideal < best_known_inf_overall:
                best_known_inf_overall = best_known_inf_ideal
                best_known_seq_overall = best_known_seq
                T_seq_best_known = config.T_seq
        else:
            best_known_inf_ideal = 1.0
            print("  No valid known sequences found.")
            
        yaxis_known.append(best_known_inf_ideal)
        xaxis_known.append(Tg)
            
        # 2. Random Optimization
        print("  Running Random Optimization...")
        
        n_pulses_list = []
        for _ in range(config.num_random_trials):
            n1 = np.random.randint(1, config.max_pulses_per_rep + 1)
            n2 = np.random.randint(1, config.max_pulses_per_rep + 1)
            n_pulses_list.append((n1, n2))
                
        seed_seq = best_known_seq if config.use_known_as_seed else None
        best_opt_seq, best_opt_inf = optimize_random_sequences(config, config.M, n_pulses_list, seed_seq=seed_seq)
        
        # Recalculate with ideal
        if best_opt_seq:
            best_opt_inf_ideal = calculate_infidelity(best_opt_seq, config, config.M, config.T_seq, use_ideal=True)
            print(f"  Best Optimized (Ideal): {best_opt_inf_ideal:.6e}")
            
            if best_opt_inf_ideal < best_opt_inf_overall:
                best_opt_inf_overall = best_opt_inf_ideal
                best_opt_seq_overall = best_opt_seq
                T_seq_best_opt = config.T_seq
        else:
            best_opt_inf_ideal = 1.0
            print("  No optimized sequence found.")
        
        yaxis_opt.append(best_opt_inf_ideal)
        xaxis_opt.append(Tg)

    # Save Results
    np.savez(os.path.join(config.path, config.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))
    np.savez(os.path.join(config.path, config.output_path_known), infs_known=np.array(yaxis_known),
             taxis=np.array(xaxis_known))

    # Plot Infidelity vs Gate Time
    plot_utils.plot_infidelity_vs_gatetime(xaxis_known, yaxis_known, xaxis_opt, yaxis_opt, yaxis_nopulse, config.Tqns, os.path.join(config.path, config.plot_filename))

    # Final Comparison and Detailed Plots (for best overall sequences)
    print("\n" + "=" * 80)
    print(f"{'FINAL COMPARISON (Best Overall)':^80}")
    print("=" * 80)
    
    inf_k_str = f"{best_known_inf_overall:.6e}" if best_known_seq_overall else "N/A"
    inf_o_str = f"{best_opt_inf_overall:.6e}" if best_opt_seq_overall else "N/A"
    print(f"{'Infidelity':<25} | {inf_k_str:<25} | {inf_o_str:<25}")
    
    if best_known_seq_overall:
        nk1, nk2 = len(best_known_seq_overall[0]) - 2, len(best_known_seq_overall[1]) - 2
        pc_k_str = f"({nk1}, {nk2})"
    else:
        pc_k_str = "N/A"
        
    if best_opt_seq_overall:
        no1, no2 = len(best_opt_seq_overall[0]) - 2, len(best_opt_seq_overall[1]) - 2
        pc_o_str = f"({no1}, {no2})"
    else:
        pc_o_str = "N/A"
        
    print(f"{'Pulse Count (Q1, Q2)':<25} | {pc_k_str:<25} | {pc_o_str:<25}")
    print("=" * 80)

    # 3. Plotting Detailed Characteristics
    # We plot for the best sequences found across all gate times.
    # If T_seq differs, we might need separate plots or just plot them separately.
    
    if best_known_seq_overall or best_opt_seq_overall:
        # If both exist and have same T_seq, plot together
        if best_known_seq_overall and best_opt_seq_overall and T_seq_best_known == T_seq_best_opt:
             T_seq = T_seq_best_known
             plot_utils.plot_comparison(config, best_known_seq_overall, best_opt_seq_overall, T_seq)
             plot_utils.plot_filter_functions(config, best_known_seq_overall, best_opt_seq_overall, T_seq)
             plot_utils.plot_filter_functions_with_spectra(config, best_known_seq_overall, best_opt_seq_overall, T_seq)
        else:
             # Plot separately if T_seq differs or one is missing
             if best_known_seq_overall:
                 plot_utils.plot_comparison(config, best_known_seq_overall, None, T_seq_best_known)
                 plot_utils.plot_filter_functions(config, best_known_seq_overall, None, T_seq_best_known)
                 plot_utils.plot_filter_functions_with_spectra(config, best_known_seq_overall, None, T_seq_best_known)
             if best_opt_seq_overall:
                 plot_utils.plot_comparison(config, None, best_opt_seq_overall, T_seq_best_opt)
                 plot_utils.plot_filter_functions(config, None, best_opt_seq_overall, T_seq_best_opt)
                 plot_utils.plot_filter_functions_with_spectra(config, None, best_opt_seq_overall, T_seq_best_opt)
        
        plot_utils.plot_noise_correlations(config)
        
        if best_known_seq_overall:
            plot_utils.plot_control_correlations(config, best_known_seq_overall, T_seq_best_known, config.M, "Best Known Sequence")
            plot_utils.plot_generalized_filter_functions(config, best_known_seq_overall, T_seq_best_known, "Best Known Sequence")
        if best_opt_seq_overall:
            plot_utils.plot_control_correlations(config, best_opt_seq_overall, T_seq_best_opt, config.M, "Best Optimized Sequence")
            plot_utils.plot_generalized_filter_functions(config, best_opt_seq_overall, T_seq_best_opt, "Best Optimized Sequence")

if __name__ == "__main__":
    try:
        # Initialize configuration with testing parameters
        config = Config(
            use_known_as_seed=False,
            M=1,
            max_pulses=400,
            num_random_trials=10,
            use_simulated=True # Enable simulated spectra by default for testing
        )
        print("Configuration loaded successfully.")
        
        # Run the pipeline
        run_optimization_pipeline(config)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
