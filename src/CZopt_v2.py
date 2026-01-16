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
"""

import functools
import itertools
import os
import traceback
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.integrate
import jax.scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from spectraIn import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CZOptConfig:
    """Configuration for the CZ gate optimization."""
    fname: str = "DraftRun_NoSPAM_Feature"
    parent_dir: str = os.pardir
    Jmax: float = 3e6
    # Extended gate time factors to include larger gate times (-1, 0)
    gate_time_factors: list = field(default_factory=lambda: [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    output_path_known: str = "infs_known_cz_v2.npz"
    output_path_opt: str = "infs_opt_cz_v2.npz"
    plot_filename: str = "infs_GateTime_cz_v2.pdf"
    
    include_cross_spectra: bool = True
    tau_divisor: int = 80
    use_simulated: bool = False
    
    # These will be loaded from the run files
    Tqns: float = field(init=False)
    mc: int = field(init=False)
    gamma: float = field(init=False)
    gamma12: float = field(init=False)

    # Calculated properties
    path: str = field(init=False)
    specs: dict = field(init=False)
    w: jnp.ndarray = field(init=False)
    wkqns: jnp.ndarray = field(init=False)
    SMat: jnp.ndarray = field(init=False)
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
        self.tau = self.Tqns / self.tau_divisor
        self.mc = int(self.params['truncate'])
        self.gamma = float(self.params['gamma'])
        self.gamma12 = float(self.params['gamma_12'])

        # Frequency Grid
        w_max_sys = 2 * jnp.pi * self.mc / self.Tqns
        self.w_max = 4 * w_max_sys
        self.N_w = 20000
        self.w = jnp.linspace(0, self.w_max, self.N_w)
        
        if self.use_simulated and 'wk' in self.specs:
            self.wkqns = jnp.array(self.specs['wk'])
        else:
            self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])

        self.SMat = self._build_interpolated_spectra()
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

    def _calculate_T2(self):
        # Approximate T2 calculation using ideal spectra for reference
        # This is just for logging/info
        pass # Skip for now to avoid re-implementing T2 logic with new SMat structure

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
    return jnp.real(integral)

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
    S_0 = jnp.real(S_packed[0])
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
                                      tol=1e-14, options={'maxiter': 1000, 'disp': False})
        
        d1_opt = jnp.array(res.x[:n1])
        d2_opt = jnp.array(res.x[n1:])
        pt1 = delays_to_pulse_times(d1_opt, T_seq)
        pt2 = delays_to_pulse_times(d2_opt, T_seq)
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
        
        # J optimization
        diffs = jnp.diff(pt12)
        signs = jnp.array((-1.0)**jnp.arange(len(diffs)))
        dc_12 = jnp.sum(diffs * signs)
        J_target = jnp.pi * 0.25 / (M * dc_12)
        J = jnp.clip(J_target, -config.Jmax, config.Jmax)
        
        fid = calculate_cz_fidelity(I_mat, J, M, dc_12)
        inf = 1.0 - fid
        
        if inf < best_inf:
            best_inf = inf
            best_seq = (pt1, pt2)
            best_idx = i
            
    return best_seq, best_inf, best_idx

# ==============================================================================
# Main Execution
# ==============================================================================

def run_optimization(config):
    yaxis_opt, xaxis_opt = [], []
    yaxis_known, xaxis_known = [], []
    
    print(f"Running CZ Optimization (v2)...")
    
    for i in config.gate_time_factors:
        Tg = config.Tqns / 2 ** (i - 1)
        if Tg < config.tau:
            continue
            
        print(f"\nGate Time: {Tg*1e6:.2f} us")
        
        # For now, M=1
        M = 1
        T_seq = Tg / M
        
        # 1. Known Sequences
        max_pulses_per_rep = int(100 / M) # Arbitrary limit for library
        pLib_delays, pLib_desc = construct_pulse_library(T_seq, config.tau, max_pulses_per_rep)
        
        best_known_seq = None
        best_known_inf = 1.0
        
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences_with_T(config, M, T_seq, pLib_delays)
            print(f"  Best Known: {best_known_inf:.6e} ({pLib_desc[idx]})")
        
        yaxis_known.append(best_known_inf)
        xaxis_known.append(Tg)
        
        # 2. Random Optimization
        best_opt_inf = 1.0
        best_opt_seq = None
        
        # Calculate nps dynamically
        max_n = int(T_seq / config.tau) - 1
        n_candidates = []
        for k in [1, 2]:
             val = max_n - k
             if val > 0: n_candidates.append(val)
        
        if not n_candidates:
             if max_n >= 1: n_candidates = [1]
             else:
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
        
        # Compare known vs opt
        if best_known_inf < best_opt_inf:
             print(f"  Known sequence was better.")
             final_inf = best_known_inf
        else:
             final_inf = best_opt_inf
        
        yaxis_opt.append(final_inf)
        xaxis_opt.append(Tg)

    np.savez(os.path.join(config.path, config.output_path_opt), infs_opt=np.array(yaxis_opt),
             taxis=np.array(xaxis_opt))
    np.savez(os.path.join(config.path, config.output_path_known), infs_known=np.array(yaxis_known),
             taxis=np.array(xaxis_known))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(xaxis_known, yaxis_known, 'bs-', label='Known (v2)')
    plt.plot(xaxis_opt, yaxis_opt, 'ko-', label='Optimized (v2)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Gate Time (s)')
    plt.ylabel('Infidelity')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(config.path, config.plot_filename))
    print(f"Saved plot to {config.plot_filename}")

if __name__ == '__main__':
    config = CZOptConfig(use_simulated=True)
    run_optimization(config)
