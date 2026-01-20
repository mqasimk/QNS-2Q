"""
Pulse Sequence Optimization for CZ Gates (v2).

This script implements an optimization pipeline for CZ gate sequences
to minimize infidelity in a two-qubit system. It supports:
1. Loading spectral noise data.
2. Constructing libraries of known pulse sequences (CDD, mqCDD).
3. Evaluating sequence performance using overlap integrals calculated in the time domain.
4. Optimizing random pulse sequences using JAX-based gradient descent.
5. Optimizing the coupling strength J dynamically.

Based on ID_opt_v4.py and CZopt.py.

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
import jax.numpy as jnp
import jax.scipy.integrate
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

@dataclass
class CZOptConfig:
    """Configuration for the CZ gate optimization."""
    fname: str = "DraftRun_NoSPAM_Feature"
    parent_dir: str = os.pardir
    Jmax: float = 2e6
    # Extended gate time factors to include larger gate times (-1, 0)
    gate_time_factors: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    output_path_known: str = "infs_known_cz_v2.npz"
    output_path_opt: str = "infs_opt_cz_v2.npz"
    plot_filename: str = "infs_GateTime_cz_v2.pdf"
    
    include_cross_spectra: bool = True
    tau_divisor: int = 160 / 2
    use_simulated: bool = False
    max_pulses: int = 400
    
    # These will be loaded from the run files
    Tqns: float = field(init=False)
    mc: int = field(init=False)
    gamma: float = field(init=False)
    gamma12: float = field(init=False)

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
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.path = os.path.join(project_root, self.fname)
        
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
             self.specs = np.load(os.path.join(self.path, "specs.npz"))
             self.params = np.load(os.path.join(self.path, "params.npz"))

        self.Tqns = float(self.params['T'])

        # Filter gate_time_factors to ensure physical feasibility with Jmax
        min_Tg = np.pi / (4 * self.Jmax)
        valid_factors = []
        for i in self.gate_time_factors:
            Tg = self.Tqns / 2 ** (i - 1)
            if Tg >= min_Tg:
                valid_factors.append(i)
            else:
                print(f"Config: Excluding factor {i} (Tg={Tg:.2e} s) - too short for Jmax (min {min_Tg:.2e} s)")
        self.gate_time_factors = valid_factors

        self.tau = self.Tqns / self.tau_divisor
        self.mc = int(self.params['truncate'])
        self.gamma = float(self.params['gamma'])
        self.gamma12 = float(self.params['gamma_12'])

        # Frequency Grid
        w_max_sys = 2 * jnp.pi * self.mc / self.Tqns
        self.w_max = 2 * w_max_sys
        self.N_w = 20000
        self.w = jnp.linspace(0, self.w_max, self.N_w)
        self.w_ideal = jnp.linspace(0, 2 * self.w_max, 2 * self.N_w)
        
        if self.use_simulated and 'wk' in self.specs:
            self.wkqns = jnp.array(self.specs['wk'])
        else:
            self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])

        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()
        self._calculate_T2()

    def _build_interpolated_spectra(self):
        """Constructs the matrix of interpolated spectra from QNS data."""
        # 4x4 Matrix: Indices 0, 1, 2, 3 correspond to Null, 1, 2, 12
        SMat = jnp.zeros((4, 4, self.w.size), dtype=jnp.complex128)
        w0 = jnp.array([0.0])
        
        def interp_c(fp):
            return (jnp.interp(self.w, self.wkqns, jnp.real(fp), right=0.) +
                   1j * jnp.interp(self.w, self.wkqns, jnp.imag(fp), right=0.))

        def combine(spec_data, dc_func, *args):
            interp = interp_c(spec_data)
            dc_val = dc_func(w0, *args)[0]
            return interp.at[0].set(dc_val)
        
        # Diagonal elements
        SMat = SMat.at[1, 1].set(combine(self.specs["S11"], S_11))
        SMat = SMat.at[2, 2].set(combine(self.specs["S22"], S_22))
        SMat = SMat.at[3, 3].set(combine(self.specs["S1212"], S_1212))
        
        # Off-diagonal elements
        if self.include_cross_spectra:
            s12 = combine(self.specs["S12"], S_1_2, self.gamma)
            SMat = SMat.at[1, 2].set(s12)
            SMat = SMat.at[2, 1].set(jnp.conj(s12))
            
            s112 = combine(self.specs["S112"], S_1_12, self.gamma12)
            SMat = SMat.at[1, 3].set(s112)
            SMat = SMat.at[3, 1].set(jnp.conj(s112))
            
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

    def _calculate_T2(self):
        """Calculates T2 times for each qubit based on ideal spectra."""
        S11_0 = jnp.real(self.SMat_ideal[1, 1, 0])
        S22_0 = jnp.real(self.SMat_ideal[2, 2, 0])
        
        self.T2q1 = 2.0 / S11_0 if S11_0 > 0 else jnp.inf
        self.T2q2 = 2.0 / S22_0 if S22_0 > 0 else jnp.inf
        
        print(f"Calculated T2 times (Ideal): Q1={self.T2q1:.2e} s, Q2={self.T2q2:.2e} s")

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
    return jnp.trace(jnp.linalg.inv(O) @ z2q[a] @ z2q[b] @ O @ z2q[a] @ z2q[b]) / 4

@jax.jit
def calculate_cz_fidelity(I_matrix, J, M, dc_12):
    """
    Calculates CZ gate fidelity.
    I_matrix: 4x4 overlap integrals (indices 0:Null, 1:Z1, 2:Z2, 3:Z12).
    """
    z1q = jnp.array([jnp.array([[1, 0], [0, 1]]), jnp.array([[1, 0], [0, -1]])])
    z2q = jnp.array(
        [jnp.kron(z1q[0], z1q[0]), jnp.kron(z1q[1], z1q[0]), jnp.kron(z1q[0], z1q[1]), jnp.kron(z1q[1], z1q[1])])
    
    p1q = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    p2q = jnp.array([jnp.kron(p1q[i], p1q[j]) for i in range(4) for j in range(4)])
    
    def lambda_element(Oi, Oj):
        val_CO = jnp.zeros((4, 4), dtype=jnp.complex128)
        
        for i in range(3):
            for j in range(3):
                idx_i = i + 1
                idx_j = j + 1
                
                # pre_factor = -0.5 * (sgn(Oi, idx_i, idx_j) + 1)
                pre_factor = -0.5 * (sgn(Oi, idx_i, idx_j) + 1.0)
                
                # sgn_term = sgn(Oi, idx_i, 0) - 1
                sgn_term = sgn(Oi, idx_i, 0) - 1.0
                
                # overlap_term = 2.0 * I_matrix[idx_i, idx_j]
                # Factor of 2 accounts for 1/pi vs 1/2pi difference
                overlap_term = 2.0 * I_matrix[idx_i, idx_j]
                
                term = pre_factor * sgn_term * overlap_term
                
                val_CO += term * (z2q[idx_i] @ z2q[idx_j])
        
        rot_val = (1.0 - sgn(Oi, 1, 2)) * M * J * dc_12
        rot_op = rot_val * z2q[3]
        
        G = jax.scipy.linalg.expm(-1j * rot_op - val_CO)
        
        return jnp.real(jnp.trace(Oi @ G @ Oj) * 0.25)

    # Vectorize over all pairs of Pauli matrices
    lambda_map = jax.vmap(jax.vmap(lambda_element, in_axes=(None, 0)), in_axes=(0, None))
    
    R_noisy = lambda_map(p2q, p2q)
    
    # Fidelity = Tr(R_ideal.T @ R_noisy) / 16
    R_ideal = zzPTM()
    fid = jnp.trace(R_ideal.T @ R_noisy) / 16.0
    
    return fid

# ==============================================================================
# Optimization Logic
# ==============================================================================

def cost_function(delays_params, n_pulses1, RMat_data, T_seq, tau_min, overlap_fn, Jmax, M):
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
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
    
    # Calculate DC component of y12 for J optimization
    # y12 corresponds to pts[3]
    diffs = jnp.diff(pt12)
    signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
    dc_12 = jnp.sum(diffs * signs)
    
    # Optimize J
    # J = pi / (4 * M * dc_12)
    # Clip J
    J_target = jnp.pi * 0.25 / (M * dc_12)
    J = jnp.clip(J_target, -Jmax, Jmax)
    
    fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
    
    return 1.0 - fid

def optimize_sequence(config, M, T_seq, n1, n2, seed_seq=None):
    # Check feasibility of pulse count
    if (n1 + 1) * config.tau > T_seq or (n2 + 1) * config.tau > T_seq:
        return None, 1.0

    # Setup Evaluation Method
    use_comb = (M > 10)
    
    if use_comb:
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
        N = config.w.shape[0]
        dw = config.w[1] - config.w[0]
        
        # Pad SMat with zeros to double the range (improving time resolution dt)
        # This reduces discretization error in the time-domain convolution
        SMat_padded = jnp.pad(config.SMat, ((0,0), (0,0), (0, N)))
        
        # Mirror SMat to ensure Hermitian symmetry
        SMat_sym = jnp.concatenate([SMat_padded, jnp.conj(jnp.flip(SMat_padded[..., 1:-1], axis=-1))], axis=-1)
        N_sym = SMat_sym.shape[-1]
        
        dt = (2 * np.pi / (N_sym * dw))
        lags_R = (jnp.arange(N_sym) - N_sym//2) * dt
        RMat_vals = jnp.fft.ifft(SMat_sym, axis=-1)
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

    # Optimization
    cost_fn = functools.partial(cost_function, n_pulses1=n1, RMat_data=RMat_data, 
                                T_seq=T_seq, tau_min=config.tau, overlap_fn=overlap_fn, 
                                Jmax=config.Jmax, M=M)
    val_and_grad = jax.jit(jax.value_and_grad(cost_fn))
    
    def fun_wrapper(x):
        v, g = val_and_grad(x)
        return float(v), np.array(g)

    if seed_seq is not None:
        d1 = pulse_times_to_delays(seed_seq[0])
        d2 = pulse_times_to_delays(seed_seq[1])
        initial_params = jnp.concatenate([d1, d2])
    else:
        d1 = get_random_delays(n1, T_seq, config.tau)
        d2 = get_random_delays(n2, T_seq, config.tau)
        initial_params = jnp.concatenate([d1, d2])

    bounds = [(config.tau, T_seq) for _ in range(len(initial_params))]
    A = np.zeros((2, n1 + n2))
    A[0, :n1] = 1
    A[1, n1:] = 1
    linear_cons = scipy.optimize.LinearConstraint(A, -np.inf, T_seq - config.tau)
    
    try:
        res = scipy.optimize.minimize(fun_wrapper, np.array(initial_params), method='SLSQP',
                                      bounds=bounds, constraints=linear_cons, jac=True,
                                      tol=1e-14, options={'maxiter': 2000, 'disp': False})
        
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
    use_comb = (M > 10)
    
    if use_comb:
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
        N = config.w.shape[0]
        dw = config.w[1] - config.w[0]
        
        # Pad SMat with zeros to double the range (improving time resolution dt)
        SMat_padded = jnp.pad(config.SMat, ((0,0), (0,0), (0, N)))
        
        # Mirror SMat to ensure Hermitian symmetry
        SMat_sym = jnp.concatenate([SMat_padded, jnp.conj(jnp.flip(SMat_padded[..., 1:-1], axis=-1))], axis=-1)
        N_sym = SMat_sym.shape[-1]
        
        dt = (2 * np.pi / (N_sym * dw))
        lags_R = (jnp.arange(N_sym) - N_sym//2) * dt
        RMat_vals = jnp.fft.ifft(SMat_sym, axis=-1)
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
        
        # J optimization
        diffs = jnp.diff(pt12)
        signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
        dc_12 = jnp.sum(diffs * signs)
        
        # Filter out sequences that cannot achieve the required phase with Jmax
        # This excludes decoupling sequences (like mqCDD) that average interaction to zero
        if config.Jmax * M * jnp.abs(dc_12) < jnp.pi * 0.25:
            continue
            
        J_target = jnp.pi * 0.25 / (M * dc_12)
        J = jnp.clip(J_target, -config.Jmax, config.Jmax)
        
        fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
        inf = 1.0 - fid
        
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
        
        SMat_curr = SMat
        if not use_ideal:
             # Pad to reduce discretization error (match Ideal resolution approx)
             SMat_curr = jnp.pad(SMat, ((0,0), (0,0), (0, N)))
        
        # Mirror SMat to ensure Hermitian symmetry
        SMat_sym = jnp.concatenate([SMat_curr, jnp.conj(jnp.flip(SMat_curr[..., 1:-1], axis=-1))], axis=-1)
        N_sym = SMat_sym.shape[-1]
        
        dt = (2 * np.pi / (N_sym * dw))
        lags_R = (jnp.arange(N_sym) - N_sym//2) * dt
        
        RMat_vals = jnp.fft.ifft(SMat_sym, axis=-1)
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
    
    diffs = jnp.diff(pt12)
    signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
    dc_12 = jnp.sum(diffs * signs)
    J_target = jnp.pi * 0.25 / (M * dc_12)
    J = jnp.clip(J_target, -config.Jmax, config.Jmax)
    
    fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
    return 1.0 - fid

# ==============================================================================
# Visualization
# ==============================================================================

def plot_comparison(config, known_seq, opt_seq, T_seq):
    """Plots the switching functions y(t) for comparison."""
    has_known = known_seq is not None
    has_opt = opt_seq is not None
    
    cols = 0
    if has_known: cols += 1
    if has_opt: cols += 1
    
    if cols == 0:
        return

    fig, axs = plt.subplots(3, cols, figsize=(6 * cols, 10), sharex=True, squeeze=False)
    
    def get_switching_function(pulse_times, T, num_points=1000):
        pulse_times = np.array(pulse_times) # Ensure numpy for plotting
        t_grid = np.linspace(0, T, num_points)
        y = np.ones_like(t_grid)
        if len(pulse_times) > 2:
            internal_pulses = pulse_times[1:-1]
            for t_pulse in internal_pulses:
                y[t_grid >= t_pulse] *= -1
        return t_grid, y

    def plot_col(col_idx, seq, title_prefix):
        pt1, pt2 = seq
        pt12 = make_tk12(pt1, pt2)
        
        t, y1 = get_switching_function(pt1, T_seq)
        _, y2 = get_switching_function(pt2, T_seq)
        _, y12 = get_switching_function(pt12, T_seq)
        
        # Row 0: y1
        axs[0, col_idx].step(t*1e6, y1, 'k-', where='post')
        axs[0, col_idx].set_title(f"{title_prefix}\nQubit 1 Switching Function ($y_1$)")
        axs[0, col_idx].set_ylabel("$y_1(t)$")
        axs[0, col_idx].set_ylim(-1.2, 1.2)
        axs[0, col_idx].grid(True, alpha=0.3)
        
        # Row 1: y2
        axs[1, col_idx].step(t*1e6, y2, 'k-', where='post')
        axs[1, col_idx].set_title("Qubit 2 Switching Function ($y_2$)")
        axs[1, col_idx].set_ylabel("$y_2(t)$")
        axs[1, col_idx].set_ylim(-1.2, 1.2)
        axs[1, col_idx].grid(True, alpha=0.3)
        
        # Row 2: y12
        axs[2, col_idx].step(t*1e6, y12, 'k-', where='post')
        axs[2, col_idx].set_title("Interaction Switching Function ($y_{12}$)")
        axs[2, col_idx].set_ylabel("$y_{12}(t)$")
        axs[2, col_idx].set_xlabel(r"Time ($\mu$s)")
        axs[2, col_idx].set_ylim(-1.2, 1.2)
        axs[2, col_idx].grid(True, alpha=0.3)

    current_col = 0
    if has_known:
        plot_col(current_col, known_seq, "Best Known Sequence")
        current_col += 1
    
    if has_opt:
        plot_col(current_col, opt_seq, "Best Optimized Sequence")
        current_col += 1

    plt.tight_layout()
    
    save_path = os.path.join(config.path, "sequence_comparison_cz.pdf")
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.close(fig)

# ==============================================================================
# Main Execution
# ==============================================================================

def run_optimization(config):
    yaxis_opt, xaxis_opt = [], []
    yaxis_known, xaxis_known = [], []
    yaxis_nopulse = []
    
    print(f"Running CZ Optimization (v2)...")
    
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
            
        print(f"\nGate Time: {Tg*1e6:.2f} us (Tg/T2q1={Tg/config.T2q1:.4f}, Tg/T2q2={Tg/config.T2q2:.4f})")

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
        pLib_delays, pLib_desc = construct_pulse_library(T_seq, config.tau, max_pulses_per_rep)
        
        best_known_seq = None
        best_known_inf = 1.0
        
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences_with_T(config, M, T_seq, pLib_delays)
            print(f"  Best Known (Char): {best_known_inf:.6e} ({pLib_desc[idx]})")
            
            # Recalculate with ideal
            best_known_inf_ideal = calculate_infidelity(best_known_seq, config, M, T_seq, use_ideal=True)
            print(f"  Best Known (Ideal): {best_known_inf_ideal:.6e}")
            
            if best_known_inf_ideal < best_known_inf_overall:
                best_known_inf_overall = best_known_inf_ideal
                best_known_seq_overall = best_known_seq
                T_seq_best_known = T_seq
        else:
            best_known_inf_ideal = 1.0
        
        yaxis_known.append(best_known_inf_ideal)
        xaxis_known.append(Tg)
        
        # 2. Random Optimization
        best_opt_inf = 1.0
        best_opt_seq = None
        
        # Calculate nps dynamically
        max_n_physical = int(T_seq / config.tau) - 1
        
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
        
        for n1 in n_candidates:
            for n2 in n_candidates:
                # Check feasibility (redundant with max_n logic but safe)
                if (n1 + 1) * config.tau > T_seq or (n2 + 1) * config.tau > T_seq:
                     continue
                
                print(f"  Optimizing n=({n1}, {n2})...")
                # Use best known as seed if available and pulse counts match?
                # For now, just random init
                seq, inf = optimize_sequence(config, M, T_seq, n1, n2)
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
             
        # Compare known vs opt (using characterized inf for selection)
        if best_known_inf < best_opt_inf:
             print(f"  Known sequence was better (on characterized spectrum).")
             final_inf = best_known_inf_ideal # Use ideal for plot
        else:
             final_inf = best_opt_inf_ideal # Use ideal for plot
        
        yaxis_opt.append(final_inf)
        xaxis_opt.append(Tg)

    np.savez(os.path.join(config.path, config.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))
    np.savez(os.path.join(config.path, config.output_path_known), infs_known=np.array(yaxis_known),
             taxis=np.array(xaxis_known))
    
    # Plot
    min_gate_time = np.pi / (4 * config.Jmax)
    plot_utils.plot_infidelity_vs_gatetime(xaxis_known, yaxis_known, xaxis_opt, yaxis_opt, yaxis_nopulse, config.tau, os.path.join(config.path, config.plot_filename), min_gate_time=min_gate_time)
    
    # Plot best sequences
    if best_known_seq_overall or best_opt_seq_overall:
        # Use the T_seq corresponding to the best sequence
        # If we want to compare them on the same plot, they might have different T_seq.
        # plot_comparison handles one T_seq.
        # We will plot them separately if T_seq differs, or just plot the best overall.
        
        if best_known_seq_overall and best_opt_seq_overall and T_seq_best_known == T_seq_best_opt:
             plot_utils.plot_comparison(config, best_known_seq_overall, best_opt_seq_overall, T_seq_best_known, filename_suffix="_cz")
        else:
             if best_known_seq_overall:
                 plot_utils.plot_comparison(config, best_known_seq_overall, None, T_seq_best_known, filename_suffix="_cz_known")
             if best_opt_seq_overall:
                 plot_utils.plot_comparison(config, None, best_opt_seq_overall, T_seq_best_opt, filename_suffix="_cz_opt")

if __name__ == '__main__':
    config = CZOptConfig(use_simulated=True)
    run_optimization(config)
