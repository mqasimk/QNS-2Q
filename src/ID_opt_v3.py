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

from spectraIn import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """
    Configuration class for pulse optimization.
    
    Handles loading of spectral data, system parameters, and optimization settings.
    Constructs the interpolated and ideal spectral matrices used for calculations.
    """
    def __init__(self, fname="DraftRun_NoSPAM_Boring", include_cross_spectra=True,
                 Tg=4 * 4 * 14 * 1e-6, reps_known=None, reps_opt=None, 
                 tau_divisor=160, max_pulses=120):
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
        
        # Frequency Grids
        self.w = jnp.linspace(1e-5, 2 * jnp.pi * self.mc / self.Tqns, 10000)
        self.w_ideal = jnp.linspace(1e-5, 4 * jnp.pi * self.mc / self.Tqns, 20000)
        self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])
        
        # Spectral Matrices
        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()

    def _build_interpolated_spectra(self):
        """Constructs the matrix of interpolated spectra from QNS data."""
        SMat = jnp.zeros((3, 3, self.w.size), dtype=jnp.complex64)
        
        # Diagonal elements
        SMat = SMat.at[0, 0].set(jnp.interp(self.w, self.wkqns, self.specs["S11"]))
        SMat = SMat.at[1, 1].set(jnp.interp(self.w, self.wkqns, self.specs["S22"]))
        SMat = SMat.at[2, 2].set(jnp.interp(self.w, self.wkqns, self.specs["S1212"]))
        
        # Off-diagonal elements
        if self.include_cross_spectra:
            SMat = SMat.at[0, 1].set(jnp.interp(self.w, self.wkqns, self.specs["S12"]))
            SMat = SMat.at[1, 0].set(jnp.interp(self.w, self.wkqns, np.conj(self.specs["S12"])))
            SMat = SMat.at[0, 2].set(jnp.interp(self.w, self.wkqns, self.specs["S112"]))
            SMat = SMat.at[2, 0].set(jnp.interp(self.w, self.wkqns, np.conj(self.specs["S112"])))
            SMat = SMat.at[1, 2].set(jnp.interp(self.w, self.wkqns, self.specs["S212"]))
            SMat = SMat.at[2, 1].set(jnp.interp(self.w, self.wkqns, np.conj(self.specs["S212"])))
        
        return SMat

    def _build_ideal_spectra(self):
        """Constructs the matrix of ideal analytical spectra."""
        SMat_ideal = jnp.zeros((3, 3, self.w_ideal.size), dtype=jnp.complex64)
        
        # Diagonal elements
        SMat_ideal = SMat_ideal.at[0, 0].set(S_11(self.w_ideal))
        SMat_ideal = SMat_ideal.at[1, 1].set(S_22(self.w_ideal))
        SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(self.w_ideal))
        
        # Off-diagonal elements
        if self.include_cross_spectra:
            SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(self.w_ideal, self.gamma))
            SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(self.w_ideal, self.gamma)))
            SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(self.w_ideal, self.gamma12))
            SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(self.w_ideal, self.gamma12)))
            SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(self.w_ideal, self.gamma12 - self.gamma))
            SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(self.w_ideal, self.gamma12 - self.gamma)))
        
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

def _get_chi_on_grid(spectrum, omega_grid, t_grid):
    """Helper to compute chi(t) on a given time grid."""
    # Construct two-sided spectrum, assuming S(-w) = S(w)*
    w_full = jnp.concatenate([-jnp.flip(omega_grid), omega_grid])
    S_full = jnp.concatenate([jnp.conj(jnp.flip(spectrum)), spectrum])
    
    # Add epsilon to avoid singularity at w=0
    w_sq = w_full**2 + 1e-12
    
    integrand = (S_full / w_sq)[:, None] * jnp.exp(1j * w_full[:, None] * t_grid[None, :])
    
    # Integrate over two-sided frequency range
    return jax.scipy.integrate.trapezoid(integrand, x=w_full, axis=0) / (2 * jnp.pi)

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
    
    t_grid_large = jnp.linspace(-M * T_base, M * T_base, 4000)
    chi_base_vals = _get_chi_on_grid(spectrum, omega_grid, t_grid_large)
    
    chi_M_vals = jnp.zeros_like(t_diffs, dtype=jnp.complex64)
    for p in range(-(M - 1), M):
        weight = float(M - abs(p))
        shifted_times = t_diffs + p * T_base
        
        val_real = jnp.interp(shifted_times, t_grid_large, jnp.real(chi_base_vals))
        val_imag = jnp.interp(shifted_times, t_grid_large, jnp.imag(chi_base_vals))
        chi_M_vals += weight * (val_real + 1j * val_imag)
        
    return jnp.einsum('i,j,ij->', sigma_a, sigma_b, chi_M_vals)

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
    
    return jnp.einsum('i,j,ij->', sigma_a, sigma_b, chi_M_vals)

def calculate_idling_fidelity(I_matrix):
    """
    Calculates the Idling Gate Fidelity F1(T) based on the overlap integrals.
    Sum of 16 coefficients corresponding to Pauli basis operators.
    """
    # Commutation rules with Z: 0(I):comm, 1(X):anti, 2(Y):anti, 3(Z):comm
    comm_rules = [0, 1, 1, 0] 
    F_total = 0.0
    
    for p1 in range(4):
        for p2 in range(4):
            c1 = comm_rules[p1]
            c2 = comm_rules[p2]
            c12 = (c1 + c2) % 2
            
            # Lambda = -2 if anti-commute, 0 if commute
            lam = jnp.array([-2.0 * c1, -2.0 * c2, -2.0 * c12])
            
            # Theta = lam @ I_matrix
            Thetas = lam @ I_matrix
            T1, T2, T12 = Thetas[0], Thetas[1], Thetas[2]
            
            term = (jnp.cosh(T1/2) * jnp.cosh(T2/2) * jnp.cosh(T12/2) - 
                    jnp.sinh(T1/2) * jnp.sinh(T2/2) * jnp.sinh(T12/2))
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
    pts = [pt1, pt2, pt12]
    
    # Calculate I_matrix
    vals = []
    for i in range(3):
        row_vals = []
        for j in range(3):
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

def optimize_random_sequences(config, M, n_pulses_list):
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
    
    for n1, n2 in n_pulses_list:
        # Initial Guess
        d1 = get_random_delays(n1, T_seq, config.tau)
        d2 = get_random_delays(n2, T_seq, config.tau)
        initial_params = jnp.concatenate([d1, d2])
        
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
                                                maxiter=200, options={'disp': False})
        
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
                
            print(f"  Random Opt (n={n1},{n2}): Infidelity = {inf:.6e}")
            
        except Exception as e:
            print(f"  Optimization failed for n={n1},{n2}: {e}")
            # traceback.print_exc()
            
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
        pts = [pt1, pt2, pt12]
        
        vals = []
        for r in range(3):
            row_vals = []
            for c in range(3):
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

def plot_comparison(known_seq, opt_seq, T_seq):
    """Plots the switching functions y(t) for comparison."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    def get_switching_function(pulse_times, T, num_points=1000):
        t_grid = np.linspace(0, T, num_points)
        y = np.ones_like(t_grid)
        internal_pulses = pulse_times[1:-1]
        for t_pulse in internal_pulses:
            y[t_grid >= t_pulse] *= -1
        return t_grid, y

    # Known Sequence
    if known_seq:
        k_pt1, k_pt2 = known_seq
        t1, y1 = get_switching_function(k_pt1, T_seq)
        t2, y2 = get_switching_function(k_pt2, T_seq)
        
        axs[0].step(t1*1e6, y1, 'r-', where='post', label='Qubit 1')
        axs[0].step(t2*1e6, y2, 'b--', where='post', label='Qubit 2')
        axs[0].set_title(f"Best Known Sequence")
        axs[0].set_ylabel("y(t)")
        axs[0].set_ylim(-1.2, 1.2)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
    else:
        axs[0].set_title("No Valid Known Sequence Found")

    # Optimized Sequence
    if opt_seq:
        o_pt1, o_pt2 = opt_seq
        t1_opt, y1_opt = get_switching_function(o_pt1, T_seq)
        t2_opt, y2_opt = get_switching_function(o_pt2, T_seq)
        
        axs[1].step(t1_opt*1e6, y1_opt, 'r-', where='post', label='Qubit 1')
        axs[1].step(t2_opt*1e6, y2_opt, 'b--', where='post', label='Qubit 2')
        axs[1].set_title(f"Best Optimized Sequence")
        axs[1].set_ylabel("y(t)")
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylim(-1.2, 1.2)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].set_title("No Optimized Sequence Found")

    plt.tight_layout()
    plt.savefig("sequence_comparison.pdf")
    print("Saved comparison plot to sequence_comparison.pdf")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    try:
        config = Config()
        print("Configuration loaded successfully.")
        
        # Test for a specific M
        M_test = 1
        T_seq_test = config.Tg / M_test
        max_p_test = 800
        
        print(f"\n--- Testing M={M_test} ---")
        
        # 1. Known Sequences
        pLib_delays, pLib_desc = construct_pulse_library(T_seq_test, config.tau, max_p_test)
        
        best_known_seq = None
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences(config, M_test, pLib_delays)
            print(f"Best Known Sequence Infidelity: {best_known_inf:.6e}")
            if idx != -1:
                print(f"Sequence Type: {pLib_desc[idx]}")
        else:
            print("No valid known sequences found.")
            
        # 2. Random Optimization
        print("\nRunning Random Optimization...")
        n_pulses_list = [(50, 50), (100, 100), (200, 200)]
        best_opt_seq, best_opt_inf = optimize_random_sequences(config, M_test, n_pulses_list)
        print(f"Best Optimized Sequence Infidelity: {best_opt_inf:.6e}")

        # 3. Plotting
        if best_known_seq or best_opt_seq:
            plot_comparison(best_known_seq, best_opt_seq, T_seq_test)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
