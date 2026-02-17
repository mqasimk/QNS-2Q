"""
Pulse Sequence Optimization for Two-Qubit Idling Gates.

This script implements an optimization pipeline for dynamical decoupling sequences
to minimize infidelity in a two-qubit system. It supports:
1. Loading spectral noise data.
2. Constructing libraries of known pulse sequences (CDD, mqCDD).
3. Evaluating sequence performance using overlap integrals.
4. Optimizing random pulse sequences using JAX-based gradient descent.
5. Efficiently handling repeated sequences via time-folding strategies.

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
import jaxopt
import matplotlib.pyplot as plt
import numpy as np

from spectra_input import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """
    Configuration class for pulse optimization.
    
    Handles loading of spectral data, system parameters, and optimization settings.
    Constructs the interpolated and ideal spectral matrices used for calculations.
    """
    def __init__(self, fname="DraftRun_NoSPAM_Boring", include_cross_spectra=False,
                 Tg=4 * 4 * 14 * 1e-6, reps_known=None, reps_opt=None, 
                 tau_divisor=160, max_pulses=80, use_known_as_seed=False):
        """
        Initialize configuration.

        Args:
            fname (str): Name of the data directory.
            include_cross_spectra (bool): Whether to include cross-correlation spectra.
            Tg (float): Total gate time.
            reps_known (list): List of repetition counts for known sequences.
            reps_opt (list): List of repetition counts for optimization.
            tau_divisor (int): Divisor to determine minimum pulse separation (tau).
            max_pulses (int): Maximum allowed pulses in a sequence.
            use_known_as_seed (bool): Whether to use the best known sequence as a seed for optimization.
        """
        # Paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.path = os.path.join(project_root, fname)
        
        if not os.path.exists(self.path):
             raise FileNotFoundError(f"Data directory not found at {self.path}")

        # Load Data
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
        self.reps_known = reps_known if reps_known is not None else [i for i in range(100, 401, 10)]
        self.reps_opt = reps_opt if reps_opt is not None else [i for i in range(100, 401, 20)]
        self.max_pulses = max_pulses
        self.tau_divisor = tau_divisor
        self.tau = self.Tqns / tau_divisor
        self.use_known_as_seed = use_known_as_seed
        
        # Frequency Grids
        # Start from 0 to include DC component
        self.w = jnp.linspace(0, 2 * jnp.pi * self.mc / self.Tqns, 10000)
        self.w_ideal = jnp.linspace(0, 4 * jnp.pi * self.mc / self.Tqns, 20000)
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
            return jnp.interp(self.w, self.wkqns, jnp.real(fp)) + 1j * jnp.interp(self.w, self.wkqns, jnp.imag(fp))

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

@jax.jit
def _get_chi_on_grid(spectrum, omega_grid, t_grid):
    """Helper to compute chi(t) on a given time grid."""
    # Construct two-sided spectrum, assuming S(-w) = S(w)*
    w_p = omega_grid
    w_m = -jnp.flip(omega_grid)
    S_p = spectrum
    S_m = jnp.conj(jnp.flip(spectrum))
    
    # w_full does not contain 0 because omega_grid passed here is AC part (w > 0)
    w_p_sq = w_p**2
    w_m_sq = w_m**2
    
    integrand_p = (S_p / w_p_sq)[:, None] * jnp.exp(1j * w_p[:, None] * t_grid[None, :])
    integrand_m = (S_m / w_m_sq)[:, None] * jnp.exp(1j * w_m[:, None] * t_grid[None, :])
    
    # Integrate over two-sided frequency range
    return (jax.scipy.integrate.trapezoid(integrand_p, x=w_p, axis=0) / (2 * jnp.pi)
            + jax.scipy.integrate.trapezoid(integrand_m, x=w_m, axis=0) / (2 * jnp.pi))

@functools.partial(jax.jit, static_argnames=['M'])
def evaluate_overlap_small_M(pulse_times_a, pulse_times_b, spectrum, omega_grid, M, T_base):
    """
    Strategy 1: Exact Time-Folding (Small M).
    """
    # 1. Compute Jump Coefficients
    def get_sigmas(pulse_times):
        N = len(pulse_times) - 2
        if N < 0: return jnp.array([])
        s0 = jnp.array([-1.0])
        if N > 0:
            indices = jnp.arange(1, N + 1)
            sj = 2.0 * ((-1.0) ** (indices - 1))
        else:
            sj = jnp.array([])
        s_last = jnp.array([(-1.0) ** N])
        return jnp.concatenate([s0, sj, s_last])

    sigma_a = get_sigmas(pulse_times_a)
    sigma_b = get_sigmas(pulse_times_b)

    t_diffs = pulse_times_a[:, None] - pulse_times_b[None, :]
    
    # Extract DC and AC parts
    S_dc = spectrum[0]
    spectrum_ac = spectrum[1:]
    omega_grid_ac = omega_grid[1:]
    
    t_grid_large = jnp.linspace(-M * T_base, M * T_base, 10000)
    chi_base_vals = _get_chi_on_grid(spectrum_ac, omega_grid_ac, t_grid_large)

    chi_M_vals = jnp.zeros_like(t_diffs, dtype=jnp.complex128)
    for p in range(-(M - 1), M):
        weight = float(M - abs(p))
        shifted_times = t_diffs + p * T_base

        val_real = jnp.interp(shifted_times, t_grid_large, jnp.real(chi_base_vals))
        val_imag = jnp.interp(shifted_times, t_grid_large, jnp.imag(chi_base_vals))
        chi_M_vals += weight * (val_real + 1j * val_imag)
        
    # DC Contribution
    F_a_0 = jnp.dot(sigma_a, pulse_times_a)
    F_b_0 = jnp.dot(sigma_b, pulse_times_b)
    dc_contribution = (M**2) * F_a_0 * F_b_0 * S_dc / (2*jnp.pi)

    return jnp.einsum('i,j,ij->', sigma_a, sigma_b, chi_M_vals) + dc_contribution

@functools.partial(jax.jit, static_argnames=['M', 'num_harmonics'])
def evaluate_overlap_large_M(pulse_times_a, pulse_times_b, spectrum, omega_grid, M, T_base, num_harmonics):
    """
    Strategy 2: Frequency Comb (Large M).
    """
    # 1. Compute Jump Coefficients
    def get_sigmas(pulse_times):
        N = len(pulse_times) - 2
        if N < 0: return jnp.array([])
        s0 = jnp.array([-1.0])
        if N > 0:
            indices = jnp.arange(1, N + 1)
            sj = 2.0 * ((-1.0) ** (indices - 1))
        else:
            sj = jnp.array([])
        s_last = jnp.array([(-1.0) ** N])
        return jnp.concatenate([s0, sj, s_last])

    sigma_a = get_sigmas(pulse_times_a)
    sigma_b = get_sigmas(pulse_times_b)
    
    t_diffs = pulse_times_a[:, None] - pulse_times_b[None, :]
    
    dw = 2 * jnp.pi / T_base
    k_vals = jnp.arange(1, num_harmonics + 1)
    w_k = k_vals * dw
    
    S_k = jnp.interp(w_k, omega_grid, spectrum)
    C_k = (M / T_base) * (S_k / w_k**2)
    
    t_flat = t_diffs.flatten()
    exp_pos = jnp.exp(1j * w_k[:, None] * t_flat[None, :])
    exp_neg = jnp.exp(-1j * w_k[:, None] * t_flat[None, :])
    
    term1 = jnp.dot(C_k, exp_pos)
    term2 = jnp.dot(jnp.conj(C_k), exp_neg)
    
    chi_flat = term1 + term2
    chi_M_vals = chi_flat.reshape(t_diffs.shape)
    
    # DC Contribution
    S_dc = spectrum[0]
    F_a_0 = jnp.dot(sigma_a, pulse_times_a)
    F_b_0 = jnp.dot(sigma_b, pulse_times_b)
    dc_contribution = (M**2) * F_a_0 * F_b_0 * S_dc / (2*jnp.pi)
    
    return jnp.einsum('i,j,ij->', sigma_a, sigma_b, chi_M_vals) + dc_contribution

@jax.jit
def calculate_idling_fidelity(I_matrix):
    """
    Calculates the Idling Gate Fidelity F1(T) based on the overlap integrals.
    Matches the derivation in the notes.
    
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
            Thetas = jnp.zeros(4)
            for l in range(4):
                # Indices for j XOR l
                col_indices = jnp.arange(4) ^ l
                # Sum over j: lam[j] * I[j, col_indices[j]]
                val = jnp.sum(lam * I_matrix[jnp.arange(4), col_indices])
                Thetas = Thetas.at[l].set(val)
            
            T0, T1, T2, T12 = Thetas[0], Thetas[1], Thetas[2], Thetas[3]
            
            # C_1^{(k)}
            term = jnp.exp(T0*0.5) * (
                jnp.cosh(T1*0.5) * jnp.cosh(T2*0.5) * jnp.cosh(T12*0.5) -
                jnp.sinh(T1*0.5) * jnp.sinh(T2*0.5) * jnp.sinh(T12*0.5)
            )
            
            F_total += term
            
    return jnp.real(F_total)


# ==============================================================================
# Optimization Logic
# ==============================================================================

def cost_function(delays_params, n_pulses1, SMat, w_grid, T_seq, tau_min, overlap_fn):
    """
    Cost function: 1 - Normalized Fidelity + Penalty.
    overlap_fn: A callable that takes (pt_a, pt_b, spectrum, w_grid) and returns the overlap integral.
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
            val = overlap_fn(pts[i], pts[j], SMat[i, j], w_grid)
            row_vals.append(val)
        vals.append(row_vals)
    
    I_mat = jnp.array(vals)
    
    fid = calculate_idling_fidelity(I_mat)
    norm_fid = fid / 16.0
    infidelity = 1.0 - norm_fid
    
    # Constraints Penalty
    last_delay1 = T_seq - jnp.sum(delays1)
    last_delay2 = T_seq - jnp.sum(delays2)
    penalty = jax.nn.relu(tau_min - last_delay1) + jax.nn.relu(tau_min - last_delay2)
    
    return infidelity + penalty * 1e6

def optimize_random_sequences(config, M, n_pulses_list, seed_seq=None):
    """Optimizes random sequences for a given repetition count M."""
    T_seq = config.Tg / M
    best_inf = 1.0
    best_seq = None
    
    # Select Overlap Strategy based on M
    if M <= 10:
        overlap_fn = functools.partial(evaluate_overlap_small_M, M=M, T_base=T_seq)
    else:
        w_max = config.w[-1]
        dw = 2 * np.pi / T_seq
        num_harmonics = int(w_max / dw) + 1
        overlap_fn = functools.partial(evaluate_overlap_large_M, M=M, T_base=T_seq, num_harmonics=num_harmonics)
    
    def run_single_optimization(n1, n2, initial_params, label):
        nonlocal best_inf, best_seq

        # Bounds
        lower_bounds = jnp.ones_like(initial_params) * config.tau
        upper_bounds = jnp.ones_like(initial_params) * T_seq
        bounds = (lower_bounds, upper_bounds)
        
        # Partial cost function for JIT
        cost_for_n = functools.partial(cost_function, n_pulses1=n1, 
                                       SMat=config.SMat, w_grid=config.w, 
                                       T_seq=T_seq, tau_min=config.tau, 
                                       overlap_fn=overlap_fn)
        
        optimizer = jaxopt.ScipyBoundedMinimize(fun=cost_for_n, method='L-BFGS-B', 
                                                maxiter=1000, options={'disp': False})
        
        try:
            res = optimizer.run(initial_params, bounds=bounds)
            inf = res.state.fun_val
            
            if inf < best_inf:
                best_inf = inf
                d1_opt = res.params[:n1]
                d2_opt = res.params[n1:]
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
        run_single_optimization(n1_s, n2_s, init_p, "Seeded Opt")

    # 2. Random Optimization
    for n1, n2 in n_pulses_list:
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
    
    # Select Overlap Strategy based on M
    if M <= 10:
        overlap_fn = functools.partial(evaluate_overlap_small_M, M=M, T_base=T_seq)
    else:
        w_max = config.w[-1]
        dw = 2 * np.pi / T_seq
        num_harmonics = int(w_max / dw) + 1
        overlap_fn = functools.partial(evaluate_overlap_large_M, M=M, T_base=T_seq, num_harmonics=num_harmonics)
        
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
                val = overlap_fn(pts[r], pts[c], config.SMat[r, c], config.w)
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
        axs[2, col_idx].set_xlabel("Time ($\mu$s)")
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
    
    save_path = os.path.join(config.path, "sequence_comparison.pdf")
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.close(fig)


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    try:
        config = Config(use_known_as_seed=True)
        print("Configuration loaded successfully.")
        
        # Test for a specific M
        M_test = 1
        T_seq_test = config.Tg / M_test
        max_p_test = 40
        max_p_per_rep = int(max_p_test / M_test)
        
        print(f"\n{'='*60}")
        print(f"TESTING CONFIGURATION: M={M_test}")
        print(f"Total Pulse Limit: {max_p_test}")
        print(f"Pulse Limit Per Repetition: {max_p_per_rep}")
        print(f"{'='*60}")
        
        # 1. Known Sequences
        pLib_delays, pLib_desc = construct_pulse_library(T_seq_test, config.tau, max_p_per_rep)
        
        best_known_seq = None
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences(config, M_test, pLib_delays)
            
            print("\n" + "-" * 60)
            print("BEST KNOWN SEQUENCE RESULTS")
            print("-" * 60)
            
            if best_known_seq:
                n_k1 = len(best_known_seq[0]) - 2
                n_k2 = len(best_known_seq[1]) - 2
                print(f"{'Infidelity':<25}: {best_known_inf:.6e}")
                print(f"{'Sequence Type':<25}: {pLib_desc[idx]}")
                print(f"{'Pulse Count (Q1, Q2)':<25}: ({n_k1}, {n_k2})")
            else:
                print("No known sequence found with infidelity < 1.0.")
            print("-" * 60)
        else:
            print("No valid known sequences found.")
            
        # 2. Random Optimization
        print("\nRunning Random Optimization...")
        
        n_pulses_list = []
        # for _ in range(8):
        #     n1 = np.random.randint(1, max_p_per_rep + 1)
        #     n2 = np.random.randint(1, max_p_per_rep + 1)
        #     n_pulses_list.append((n1, n2))
                
        seed_seq = best_known_seq if config.use_known_as_seed else None
        best_opt_seq, best_opt_inf = optimize_random_sequences(config, M_test, n_pulses_list, seed_seq=seed_seq)
        
        print("\n" + "-" * 60)
        print("BEST OPTIMIZED SEQUENCE RESULTS")
        print("-" * 60)
        
        if best_opt_seq:
            n_o1 = len(best_opt_seq[0]) - 2
            n_o2 = len(best_opt_seq[1]) - 2
            print(f"{'Infidelity':<25}: {best_opt_inf:.6e}")
            print(f"{'Pulse Count (Q1, Q2)':<25}: ({n_o1}, {n_o2})")
        else:
            print(f"{'Infidelity':<25}: {best_opt_inf:.6e}")
            print("No optimized sequence found.")
        print("-" * 60)

        # Final Comparison
        print("\n" + "=" * 80)
        print(f"{'FINAL COMPARISON':^80}")
        print("=" * 80)
        print(f"{'Metric':<25} | {'Best Known Sequence':<25} | {'Best Optimized Sequence':<25}")
        print("-" * 80)
        
        inf_k_str = f"{best_known_inf:.6e}" if best_known_seq else "N/A"
        inf_o_str = f"{best_opt_inf:.6e}" if best_opt_seq else "N/A"
        print(f"{'Infidelity':<25} | {inf_k_str:<25} | {inf_o_str:<25}")
        
        if best_known_seq:
            nk1, nk2 = len(best_known_seq[0]) - 2, len(best_known_seq[1]) - 2
            pc_k_str = f"({nk1}, {nk2})"
            desc_k = pLib_desc[idx]
        else:
            pc_k_str = "N/A"
            desc_k = "N/A"
            
        if best_opt_seq:
            no1, no2 = len(best_opt_seq[0]) - 2, len(best_opt_seq[1]) - 2
            pc_o_str = f"({no1}, {no2})"
        else:
            pc_o_str = "N/A"
            
        print(f"{'Pulse Count (Q1, Q2)':<25} | {pc_k_str:<25} | {pc_o_str:<25}")
        print(f"{'Description':<25} | {desc_k:<25} | {'Random Optimization':<25}")
        
        def get_min_sep(seq):
            if seq is None: return None
            return min(float(jnp.min(jnp.diff(seq[0]))), float(jnp.min(jnp.diff(seq[1]))))
            
        sep_k = get_min_sep(best_known_seq)
        sep_o = get_min_sep(best_opt_seq)
        sep_k_str = f"{sep_k*1e6:.4f} us" if sep_k is not None else "N/A"
        sep_o_str = f"{sep_o*1e6:.4f} us" if sep_o is not None else "N/A"
        print(f"{'Min Pulse Separation':<25} | {sep_k_str:<25} | {sep_o_str:<25}")
        
        sep_k_tau = f"{sep_k/config.tau:.2f} tau" if sep_k is not None else "N/A"
        sep_o_tau = f"{sep_o/config.tau:.2f} tau" if sep_o is not None else "N/A"
        print(f"{'Min Separation (tau)':<25} | {sep_k_tau:<25} | {sep_o_tau:<25}")
        print("=" * 80)

        # 3. Plotting
        if best_known_seq or best_opt_seq:
            plot_comparison(config, best_known_seq, best_opt_seq, T_seq_test)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
