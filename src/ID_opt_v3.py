import functools
import itertools
import jax
import jax.numpy as jnp
import jax.scipy.integrate
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import os
from spectraIn import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12

class Config:
    """
    Configuration class for pulse optimization.
    Loads spectral data and parameters, and constructs spectral matrices.
    """
    def __init__(self, fname="DraftRun_NoSPAM_Boring", include_cross_spectra=False,
                 Tg=4 * 4 * 14 * 1e-6, reps_known=None, reps_opt=None, 
                 tau_divisor=160, max_pulses=400):
        # Determine the path to the data directory relative to this script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.path = os.path.join(project_root, fname)
        
        # Load data
        if not os.path.exists(self.path):
             raise FileNotFoundError(f"Data directory not found at {self.path}")

        self.specs = np.load(os.path.join(self.path, "specs.npz"))
        self.params = np.load(os.path.join(self.path, "params.npz"))
        
        # Extract parameters
        self.Tqns = float(self.params['T'])
        self.mc = int(self.params['truncate'])
        self.gamma = float(self.params['gamma'])
        self.gamma12 = float(self.params['gamma_12'])
        self.include_cross_spectra = include_cross_spectra
        
        # Optimization parameters
        if reps_opt is None:
            reps_opt = [i for i in range(100, 401, 20)]
        if reps_known is None:
            reps_known = [i for i in range(100, 401, 10)]
            
        self.Tg = Tg
        self.reps_known = reps_known
        self.reps_opt = reps_opt
        self.max_pulses = max_pulses
        self.tau_divisor = tau_divisor
        self.tau = self.Tqns / tau_divisor
        
        # Define frequency grids
        self.w = jnp.linspace(1e-5, 2 * jnp.pi * self.mc / self.Tqns, 10000)
        self.w_ideal = jnp.linspace(1e-5, 4 * jnp.pi * self.mc / self.Tqns, 20000)
        self.wkqns = jnp.array([2 * jnp.pi * (n + 1) / self.Tqns for n in range(self.mc)])
        
        # Construct spectral matrices
        self.SMat = self._build_interpolated_spectra()
        self.SMat_ideal = self._build_ideal_spectra()

    def _build_interpolated_spectra(self):
        """Constructs the matrix of interpolated spectra from QNS data."""
        SMat = jnp.zeros((3, 3, self.w.size), dtype=jnp.complex64)
        SMat = SMat.at[0, 0].set(jnp.interp(self.w, self.wkqns, self.specs["S11"]))
        SMat = SMat.at[1, 1].set(jnp.interp(self.w, self.wkqns, self.specs["S22"]))
        SMat = SMat.at[2, 2].set(jnp.interp(self.w, self.wkqns, self.specs["S1212"]))
        
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
        SMat_ideal = SMat_ideal.at[0, 0].set(S_11(self.w_ideal))
        SMat_ideal = SMat_ideal.at[1, 1].set(S_22(self.w_ideal))
        SMat_ideal = SMat_ideal.at[2, 2].set(S_1212(self.w_ideal))
        
        if self.include_cross_spectra:
            SMat_ideal = SMat_ideal.at[0, 1].set(S_1_2(self.w_ideal, self.gamma))
            SMat_ideal = SMat_ideal.at[1, 0].set(jnp.conj(S_1_2(self.w_ideal, self.gamma)))
            SMat_ideal = SMat_ideal.at[0, 2].set(S_1_12(self.w_ideal, self.gamma12))
            SMat_ideal = SMat_ideal.at[2, 0].set(jnp.conj(S_1_12(self.w_ideal, self.gamma12)))
            SMat_ideal = SMat_ideal.at[1, 2].set(S_2_12(self.w_ideal, self.gamma12 - self.gamma))
            SMat_ideal = SMat_ideal.at[2, 1].set(jnp.conj(S_2_12(self.w_ideal, self.gamma12 - self.gamma)))
        
        return SMat_ideal

# --- Sequence Generation Utilities ---

def remove_consecutive_duplicates(input_list):
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
    if n == 1:
        return [t0, t0 + T*0.5]
    else:
        return [t0] + cdd(t0, T*0.5, n-1) + [t0 + T*0.5] + cdd(t0 + T*0.5, T*0.5, n-1)

def cddn_util(t0, T, n):
    return remove_consecutive_duplicates(cdd(t0, T, n))

def cddn(t0, T, n):
    out = cddn_util(t0, T, n)
    if out[0] == 0.:
        return out + [T]
    else:
        return [0.] + out + [T]

def mqCDD(T, n, m):
    tk1 = cddn_util(0., T, n)
    tk2 = []
    for i in range(len(tk1)-1):
        tk2 += cddn_util(tk1[i], tk1[i+1]-tk1[i], m)
    tk2 += cddn_util(tk1[-1], T-tk1[-1], m)
    if tk1[0] != 0.:
        tk1 = [0.] + tk1
    if tk2[0] != 0.:
        tk2 = [0.] + tk2
    return [tk1 + [T], tk2 + [T]]

def pulse_times_to_delays(tk):
    """
    Converts a sequence of pulse times [0, t1, t2, ..., tn, T] to delays [t1, t2-t1, ..., tn-tn-1].
    The last delay (T-tn) is implicit and not returned, matching the optimization parameter format.
    """
    tk_arr = jnp.array(tk)
    if len(tk_arr) <= 2: # Only 0 and T, no pulses
        return jnp.array([])
    # tk is [0, t1, ..., tn, T]
    # diffs are [t1, t2-t1, ..., tn-tn-1, T-tn]
    # We want the first n delays.
    diffs = jnp.diff(tk_arr)
    return diffs[:-1]

def delays_to_pulse_times(delays, T):
    """
    Converts delays [t1, t2-t1, ...] to pulse times [0, t1, t2, ..., T].
    """
    if delays.size == 0:
        return jnp.array([0., T])
    
    # Calculate the last delay
    last_delay = T - jnp.sum(delays)
    
    # Concatenate all delays
    all_delays = jnp.concatenate([delays, jnp.array([last_delay])])
    
    # Cumulative sum to get times
    times = jnp.cumsum(all_delays)
    
    # Prepend 0
    return jnp.concatenate([jnp.array([0.]), times])

def get_random_delays(n, T, tau):
    """Generates n random delays summing to less than T, with min separation tau."""
    if n <= 0:
        return jnp.array([])
    
    slack = T - (n + 1) * tau
    if slack > 0:
        r = np.random.rand(int(n) + 1)
        r = r / np.sum(r)
        delays = tau + slack * r
        return jnp.array(delays[:n])
    else:
        # Fallback if slack is non-positive (should be checked before calling)
        return jnp.ones(n) * T / (n + 1)

def construct_pulse_library(T_seq, tau_min, max_pulses=50):
    """
    Constructs a library of known pulse sequences (cddn permutations and mqCDD) in terms of delays.
    Returns:
        pLib_delays: List of (d1, d2) delay tuples.
        pLib_descriptions: List of strings describing each sequence.
    """
    # 1. Generate single-qubit cddn sequences
    cddLib = []
    cddOrd = 1
    while True:
        pul = jnp.array(cddn(0., T_seq, cddOrd))
        # Check pulse separation (pul contains 0 and T, so check diffs)
        diffs = jnp.diff(pul)
        if jnp.any(diffs < tau_min):
            break
        cddLib.append(pul)
        cddOrd += 1

    # 2. Create permutations of the single-qubit sequences
    pLib_times = list(itertools.permutations(cddLib, 2))

    # 3. Generate and add multi-qubit (mqCDD) sequences
    mq_cdd_orders_log = []
    ncddOrd1 = 1
    while True:
        ncddOrd2 = 1
        # Check outer sequence
        pul_n = mqCDD(T_seq, ncddOrd1, ncddOrd2)[0]
        if jnp.any(jnp.diff(jnp.array(pul_n)) < tau_min):
            break 

        while True:
            pul = mqCDD(T_seq, ncddOrd1, ncddOrd2)
            # Check inner sequence
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
    
    for i, (tk1, tk2) in enumerate(pLib_times):
        n1 = len(tk1) - 2
        n2 = len(tk2) - 2
        if n1 <= max_pulses and n2 <= max_pulses:
            d1 = pulse_times_to_delays(tk1)
            d2 = pulse_times_to_delays(tk2)
            pLib_delays.append((d1, d2))
            
            if i < num_cdd_perms:
                # CDD Permutation
                L = len(cddLib)
                # i = a * (L-1) + b_prime
                # itertools.permutations order: (0,1), (0,2), ..., (1,0), (1,2), ...
                # This mapping is tricky to get exactly right without re-generating.
                # Let's just find which cddLib entry matches tk1 and tk2.
                
                # Helper to find index
                def find_cdd_index(tk, lib):
                    for idx, seq in enumerate(lib):
                        if len(seq) == len(tk) and jnp.allclose(seq, tk):
                            return idx + 1 # Return order
                    return -1
                
                ord1 = find_cdd_index(tk1, cddLib)
                ord2 = find_cdd_index(tk2, cddLib)
                pLib_descriptions.append(f"CDD({ord1}, {ord2})")
                
            else:
                mq_idx = i - num_cdd_perms
                n, m = mq_cdd_orders_log[mq_idx]
                pLib_descriptions.append(f"mqCDD(n={n}, m={m})")

    return pLib_delays, pLib_descriptions

def make_tk12(tk1, tk2):
    """
    Combines two pulse sequences into a single sequence for the 12 interaction.
    JIT-compatible version: assumes internal pulses are distinct or handles duplicates by keeping them.
    """
    # tk1: [0, t1_1, ..., t1_n1, T]
    # tk2: [0, t2_1, ..., t2_n2, T]
    
    # Extract internal pulses
    # Slicing is fine if shapes are static (which they are in optimization loop)
    int1 = tk1[1:-1]
    int2 = tk2[1:-1]
    
    # Combine and sort
    combined_int = jnp.concatenate([int1, int2])
    combined_int = jnp.sort(combined_int)
    
    # Reconstruct full sequence: [0, sorted_internal, T]
    # Note: tk1[-1] is T
    return jnp.concatenate([jnp.array([0.]), combined_int, jnp.array([tk1[-1]])])

# --- Overlap Integral Calculation ---

def calculate_overlap_integral(pulse_times_a, pulse_times_b, spectrum, omega_grid):
    """
    Calculates the spectral overlap integral I_{a,b}(T) using the time-domain method.
    
    Args:
        pulse_times_a (array): Pulse times for qubit a, including 0 and T.
        pulse_times_b (array): Pulse times for qubit b, including 0 and T.
        spectrum (array): The spectral density S_{a,b}(omega).
        omega_grid (array): The frequency grid corresponding to the spectrum.
        
    Returns:
        float/complex: The value of the overlap integral.
    """
    # 1. Compute Jump Coefficients (sigma)
    # Assumes pulse_times includes 0 and T.
    # y(t) starts at +1.
    # sigma_0 = -1
    # sigma_j = 2 * (-1)^(j-1) for j=1..N
    # sigma_{N+1} = (-1)^N
    
    def get_sigmas(pulse_times):
        N = len(pulse_times) - 2 # Number of internal pulses
        if N < 0: # Should not happen if 0 and T are present
            return jnp.array([])
        
        # Indices 0 to N+1
        # j=0
        s0 = jnp.array([-1.0])
        # j=1 to N
        if N > 0:
            indices = jnp.arange(1, N + 1)
            sj = 2.0 * ((-1.0) ** (indices - 1))
        else:
            sj = jnp.array([])
        # j=N+1
        s_last = jnp.array([(-1.0) ** N])
        
        return jnp.concatenate([s0, sj, s_last])

    sigma_a = get_sigmas(pulse_times_a)
    sigma_b = get_sigmas(pulse_times_b)
    
    # 2. Compute Chi Interpolator
    # chi(t) = (1/pi) * int (S(w)/w^2) * exp(i*w*t) dw
    # We compute this on a fine time grid and interpolate.
    
    # Use jnp.maximum for JAX compatibility instead of python max
    T_max = jnp.maximum(pulse_times_a[-1], pulse_times_b[-1])
    # Create a time grid from -T to T. 
    # Resolution should be sufficient for the highest frequency in omega_grid.
    # Max freq ~ 2*pi*mc/Tqns. 
    # Let's use a safe number of points.
    num_t_points = 2000
    t_grid = jnp.linspace(-T_max, T_max, num_t_points)
    
    # Vectorized integration
    # integrand shape: (num_freqs, num_t_points)
    integrand = (spectrum / omega_grid**2)[:, None] * jnp.exp(1j * omega_grid[:, None] * t_grid[None, :])
    
    # Integrate over frequency (axis 0)
    # Using trapezoidal rule
    chi_vals = jax.scipy.integrate.trapezoid(integrand, x=omega_grid, axis=0) / jnp.pi
    
    # 3. Vectorized Double Summation
    # Calculate time differences t_j - t_k
    # shape: (len(a), len(b))
    t_diffs = pulse_times_a[:, None] - pulse_times_b[None, :]
    
    # Interpolate chi values at these differences
    # jnp.interp expects real x. t_grid is real.
    # chi_vals is complex. jnp.interp works on complex y in recent JAX, 
    # but to be safe/compatible, we can interpolate real and imag parts separately.
    chi_matrix_real = jnp.interp(t_diffs, t_grid, jnp.real(chi_vals))
    chi_matrix_imag = jnp.interp(t_diffs, t_grid, jnp.imag(chi_vals))
    chi_matrix = chi_matrix_real + 1j * chi_matrix_imag
    
    # Sum over j, k: sigma_a[j] * sigma_b[k] * chi(t_j - t_k)
    I_ab = jnp.einsum('i,j,ij->', sigma_a, sigma_b, chi_matrix)
    
    return I_ab

def calculate_idling_fidelity(I_matrix):
    """
    Calculates the Idling Gate Fidelity F1(T) based on the overlap integrals.
    
    Args:
        I_matrix (array): 3x3 matrix of overlap integrals corresponding to indices 1, 2, 12.
                          [[I11, I12, I1_12], [I21, I22, I2_12], [I12_1, I12_2, I12_12]]
                          Indices: 0 -> Z1, 1 -> Z2, 2 -> Z12
    
    Returns:
        float: The fidelity F1(T) (sum of 16 coefficients).
    """
    # Pauli matrices commutation rules with Z
    # 0(I): comm, 1(X): anti, 2(Y): anti, 3(Z): comm
    # comm = 0, anti = 1
    comm_rules = [0, 1, 1, 0] 
    
    F_total = 0.0
    
    # Iterate over all 16 Paulis (p1, p2)
    for p1 in range(4):
        for p2 in range(4):
            # Determine commutation with Z1, Z2, Z12
            # Z1 (Z tensor I): depends on p1
            c1 = comm_rules[p1] # 1 if anti, 0 if comm
            
            # Z2 (I tensor Z): depends on p2
            c2 = comm_rules[p2]
            
            # Z12 (Z tensor Z): anti if c1+c2 is odd
            c12 = (c1 + c2) % 2
            
            # Calculate lambdas
            # lambda = sgn - 1. 
            # sgn = 1 if comm (c=0), -1 if anti (c=1).
            # lambda = 1 - 1 = 0 if comm.
            # lambda = -1 - 1 = -2 if anti.
            # So lambda = -2 * c
            
            lam = jnp.array([-2.0 * c1, -2.0 * c2, -2.0 * c12])
            
            # Calculate Thetas
            # Theta_l = sum_j lambda_j * I_{j,l}
            # I_matrix indices: 0->1, 1->2, 2->12
            # We need Theta_1 (l=0), Theta_2 (l=1), Theta_12 (l=2)
            
            # Theta vector [Theta_1, Theta_2, Theta_12]
            # I_matrix is [ [I11, I12, I1_12], [I21, I22, I2_12], ... ]
            # Theta = lam @ I_matrix
            
            Thetas = lam @ I_matrix
            
            T1 = Thetas[0]
            T2 = Thetas[1]
            T12 = Thetas[2]
            
            # Theta_0 assumed 0
            
            # C1(k)
            # term = cosh(T1/2)cosh(T2/2)cosh(T12/2) - sinh(T1/2)sinh(T2/2)sinh(T12/2)
            
            term = (jnp.cosh(T1/2) * jnp.cosh(T2/2) * jnp.cosh(T12/2) - 
                    jnp.sinh(T1/2) * jnp.sinh(T2/2) * jnp.sinh(T12/2))
            
            F_total += term
            
    return jnp.real(F_total)

# --- Optimization Functions ---

def cost_function(delays_params, n_pulses1, SMat, w_grid, T_seq, tau_min):
    """
    Cost function for optimization: 1 - Normalized Fidelity + Penalty.
    """
    delays1 = delays_params[:n_pulses1]
    delays2 = delays_params[n_pulses1:]
    
    # Reconstruct pulse times
    pt1 = delays_to_pulse_times(delays1, T_seq)
    pt2 = delays_to_pulse_times(delays2, T_seq)
    pt12 = make_tk12(pt1, pt2)
    
    pts = [pt1, pt2, pt12]
    
    # Calculate I_matrix
    I_mat = jnp.zeros((3, 3), dtype=complex)
    
    # We need to use jax.lax.scan or unroll loops for JIT compatibility if we were fully JITing,
    # but for now let's stick to Python loops as the matrix is small (3x3).
    # However, calculate_overlap_integral uses JAX ops, so it's fine.
    
    # Note: calculate_overlap_integral is somewhat expensive.
    # We can optimize by vectorizing the outer loops if needed, but 3x3 is small.
    
    # To make it JIT-able, we should avoid side effects and use JAX arrays.
    # But calculate_overlap_integral returns a scalar.
    
    # Let's manually unroll for clarity and JAX safety
    vals = []
    for i in range(3):
        row_vals = []
        for j in range(3):
            val = calculate_overlap_integral(pts[i], pts[j], SMat[i, j], w_grid)
            row_vals.append(val)
        vals.append(row_vals)
    
    I_mat = jnp.array(vals)
    
    fid = calculate_idling_fidelity(I_mat)
    norm_fid = fid / 16.0
    infidelity = 1.0 - norm_fid
    
    # Penalty for constraints
    # 1. Last delay >= tau_min
    last_delay1 = T_seq - jnp.sum(delays1)
    last_delay2 = T_seq - jnp.sum(delays2)
    
    penalty = 0.0
    penalty += jax.nn.relu(tau_min - last_delay1)
    penalty += jax.nn.relu(tau_min - last_delay2)
    
    return infidelity + penalty * 1e6

def optimize_random_sequences(config, M, n_pulses_list):
    """
    Optimizes random sequences for a given repetition count M.
    """
    T_seq = config.Tg / M
    
    best_inf = 1.0
    best_seq = None
    
    for n1, n2 in n_pulses_list:
        # Generate random initial guess
        d1 = get_random_delays(n1, T_seq, config.tau)
        d2 = get_random_delays(n2, T_seq, config.tau)
        initial_params = jnp.concatenate([d1, d2])
        
        # Bounds: delays >= tau
        lower_bounds = jnp.ones_like(initial_params) * config.tau
        upper_bounds = jnp.ones_like(initial_params) * T_seq # Loose upper bound
        bounds = (lower_bounds, upper_bounds)
        
        # Define a partial cost function where n_pulses1 is fixed (static)
        # This is crucial for JIT compilation of slicing operations
        cost_for_n = functools.partial(cost_function, n_pulses1=n1, 
                                       SMat=config.SMat, w_grid=config.w, 
                                       T_seq=T_seq, tau_min=config.tau)
        
        # Instantiate optimizer for this specific function signature
        optimizer = jaxopt.ScipyBoundedMinimize(fun=cost_for_n, method='L-BFGS-B', 
                                                maxiter=200, options={'disp': False})
        
        try:
            res = optimizer.run(initial_params, bounds=bounds)
            
            inf = res.state.fun_val
            params = res.params
            
            if inf < best_inf:
                best_inf = inf
                # Convert back to pulse times for storage
                d1_opt = params[:n1]
                d2_opt = params[n1:]
                pt1 = delays_to_pulse_times(d1_opt, T_seq)
                pt2 = delays_to_pulse_times(d2_opt, T_seq)
                best_seq = (pt1, pt2)
                
            print(f"  Random Opt (n={n1},{n2}): Infidelity = {inf:.6e}")
            
        except Exception as e:
            print(f"  Optimization failed for n={n1},{n2}: {e}")
            import traceback
            traceback.print_exc()
            
    return best_seq, best_inf

def evaluate_known_sequences(config, M, pLib):
    """
    Evaluates all sequences in the library and finds the best one.
    """
    T_seq = config.Tg / M
    best_inf = 1.0
    best_seq = None
    best_idx = -1
    
    print(f"Evaluating {len(pLib)} known sequences...")
    
    for i, (d1, d2) in enumerate(pLib):
        # Reconstruct pulse times
        pt1 = delays_to_pulse_times(d1, T_seq)
        pt2 = delays_to_pulse_times(d2, T_seq)
        pt12 = make_tk12(pt1, pt2)
        
        pts = [pt1, pt2, pt12]
        
        # Calculate I_matrix
        vals = []
        for r in range(3):
            row_vals = []
            for c in range(3):
                val = calculate_overlap_integral(pts[r], pts[c], config.SMat[r, c], config.w)
                row_vals.append(val)
            vals.append(row_vals)
        I_mat = jnp.array(vals)
        
        fid = calculate_idling_fidelity(I_mat)
        norm_fid = fid / 16.0
        inf = 1.0 - norm_fid
        
        if inf < best_inf:
            best_inf = inf
            best_seq = (pt1, pt2)
            best_idx = i
            
    return best_seq, best_inf, best_idx

def plot_comparison(known_seq, opt_seq, T_seq, title="Sequence Comparison"):
    """
    Plots the switching functions y(t) for comparison.
    y(t) starts at +1 and flips at each pulse time.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    def get_switching_function(pulse_times, T, num_points=1000):
        t_grid = np.linspace(0, T, num_points)
        y = np.ones_like(t_grid)
        # pulse_times includes 0 and T. Internal pulses are at indices 1:-1
        internal_pulses = pulse_times[1:-1]
        for t_pulse in internal_pulses:
            y[t_grid >= t_pulse] *= -1
        return t_grid, y

    # Known Sequence
    k_pt1, k_pt2 = known_seq
    t1, y1 = get_switching_function(k_pt1, T_seq)
    t2, y2 = get_switching_function(k_pt2, T_seq)
    
    axs[0].step(t1*1e6, y1, 'r-', where='post', label='Qubit 1')
    axs[0].step(t2*1e6, y2, 'b--', where='post', label='Qubit 2')
    axs[0].set_title(f"Best Known Sequence (Switching Function)")
    axs[0].set_ylabel("y(t)")
    axs[0].set_ylim(-1.2, 1.2)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Optimized Sequence
    if opt_seq:
        o_pt1, o_pt2 = opt_seq
        t1_opt, y1_opt = get_switching_function(o_pt1, T_seq)
        t2_opt, y2_opt = get_switching_function(o_pt2, T_seq)
        
        axs[1].step(t1_opt*1e6, y1_opt, 'r-', where='post', label='Qubit 1')
        axs[1].step(t2_opt*1e6, y2_opt, 'b--', where='post', label='Qubit 2')
        axs[1].set_title(f"Best Optimized Sequence (Switching Function)")
        axs[1].set_ylabel("y(t)")
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylim(-1.2, 1.2)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sequence_comparison.pdf")
    print("Saved comparison plot to sequence_comparison.pdf")

if __name__ == "__main__":
    # Test the configuration and library construction
    try:
        config = Config()
        print("Configuration loaded successfully.")
        
        # Test for a specific M
        M_test = 20
        T_seq_test = config.Tg / M_test
        max_p_test = int(config.max_pulses / M_test)
        
        print(f"\n--- Testing M={M_test} ---")
        
        # 1. Known Sequences
        pLib_delays, pLib_desc = construct_pulse_library(T_seq_test, config.tau, max_p_test)
        
        best_known_seq = None
        if pLib_delays:
            best_known_seq, best_known_inf, idx = evaluate_known_sequences(config, M_test, pLib_delays)
            print(f"Best Known Sequence Infidelity: {best_known_inf:.6e}")
            print(f"Sequence Type: {pLib_desc[idx]}")
        else:
            print("No valid known sequences found.")
            
        # 2. Random Optimization
        print("\nRunning Random Optimization...")
        # Try a few pulse number combinations
        n_pulses_list = [(4, 5), (6, 7), (8, 9)]
        best_opt_seq, best_opt_inf = optimize_random_sequences(config, M_test, n_pulses_list)
        print(f"Best Optimized Sequence Infidelity: {best_opt_inf:.6e}")
        
        # 3. Plotting
        if best_known_seq:
            plot_comparison(best_known_seq, best_opt_seq, T_seq_test)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
