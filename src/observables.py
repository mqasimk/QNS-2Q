"""
This module provides functions for calculating quantum observables from simulated experimental data,
including single-qubit and two-qubit expectation values, and concurrence-related metrics.
It is designed to work with QuTiP for quantum object representations and JAX for numerical calculations.
"""

from typing import List, Tuple, Callable
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from qutip_qip.operations import snot

from trajectories import make_init_state, make_y


# --- Helper functions for creating operators ---


def _get_hadamard_operators() -> List[qt.Qobj]:
    """
    Creates a list of 3-qubit Hadamard gate operators for measurement basis rotation.
    The operators are [H \otimes I \otimes I, I \otimes H \otimes I, H \otimes H \otimes I].
    """
    h_single = snot()
    id_q = qt.identity(2)

    h0 = qt.tensor(h_single, id_q, id_q)
    h1 = qt.tensor(id_q, h_single, id_q)
    h2 = qt.tensor(h_single, h_single, id_q)

    return [h0, h1, h2]


def _get_rx_operators() -> List[qt.Qobj]:
    """
    Creates a list of 3-qubit Rx gate operators for Y-basis measurement.
    The operators are [Rx \otimes I \otimes I, I \otimes Rx \otimes I, Rx \otimes Rx \otimes I].
    """
    rx_matrix = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)
    rx_single = qt.Qobj(rx_matrix, dims=[[2], [2]])
    id_q = qt.identity(2)

    rx0 = qt.tensor(rx_single, id_q, id_q)
    rx1 = qt.tensor(id_q, rx_single, id_q)
    rx2 = qt.tensor(rx_single, rx_single, id_q)

    return [rx0, rx1, rx2]


# --- Core Observable and Probability Functions ---


def povms(a_m: np.ndarray, delta: np.ndarray) -> List[qt.Qobj]:
    """
    Constructs the POVM operators for two-qubit measurements, accounting for measurement fidelity.

    Args:
        a_m: Array of measurement scaling factors.
        delta: Array of measurement offsets.

    Returns:
        A list of four 3-qubit POVM operators.
    """
    a1 = (a_m[0] + delta[0] + 1) * 0.5
    b1 = (a_m[0] - delta[0] + 1) * 0.5
    a2 = (a_m[1] + delta[1] + 1) * 0.5
    b2 = (a_m[1] - delta[1] + 1) * 0.5
    zp = qt.basis(2, 0)
    zm = qt.basis(2, 1)
    p1_0 = a1 * zp * zp.dag() + (1 - b1) * zm * zm.dag()
    p1_1 = (1 - a1) * zp * zp.dag() + b1 * zm * zm.dag()
    p2_0 = a2 * zp * zp.dag() + (1 - b2) * zm * zm.dag()
    p2_1 = (1 - a2) * zp * zp.dag() + b2 * zm * zm.dag()
    p1_0 = qt.tensor(p1_0, qt.identity(2), qt.identity(2))
    p1_1 = qt.tensor(p1_1, qt.identity(2), qt.identity(2))
    p2_0 = qt.tensor(qt.identity(2), p2_0, qt.identity(2))
    p2_1 = qt.tensor(qt.identity(2), p2_1, qt.identity(2))
    return [p1_0, p1_1, p2_0, p2_1]


@jax.jit
def compute_probs_jax(rho_batch, gate, M_ops):
    # rho_batch: (N, 8, 8)
    # gate: (8, 8)
    # M_ops: (4, 8, 8) -> [M00, M01, M10, M11]

    # Rotate: rho' = G rho G^dag
    # G (8,8), rho (N,8,8)
    # Using matmul broadcasting: (N, 8, 8)
    rho_prime = jnp.matmul(jnp.matmul(gate, rho_batch), gate.conj().T)

    # Measure: Tr(M * rho')
    # M_ops (4, 8, 8), rho' (N, 8, 8)
    # We want output (N, 4)
    # result[n, m] = Tr(M_ops[m] @ rho_prime[n])
    # einsum: m=measurement, n=batch, i,j=matrix indices
    probs = jnp.einsum('mij,nji->nm', M_ops, rho_prime)
    return jnp.real(probs)


def get_expect_val_from_probs(probs: np.ndarray, cm: np.ndarray, qubit_idx: int = -1) -> float:
    """
    Calculates the expectation value from a probability distribution using a confusion matrix.

    Args:
        probs: An array of outcome probabilities.
        cm: The confusion matrix for error mitigation.
        qubit_idx: The qubit index (1 or 2) for single-qubit expectation values,
                   or -1 for two-qubit correlations.

    Returns:
        The calculated expectation value.
    """
    pi = jnp.mean(probs, axis=0)
    p_corr = np.linalg.inv(cm) @ pi
    if qubit_idx == 1:
        p = p_corr[0] + p_corr[1]
    elif qubit_idx == 2:
        p = p_corr[0] + p_corr[2]
    else:  # Two-qubit correlation
        p = p_corr[0] + p_corr[3]
    return 2.0 * p - 1.0


def get_expect_val_per_shot(probs: np.ndarray, cm: np.ndarray, qubit_idx: int = -1) -> np.ndarray:
    """
    Calculates the expectation value per shot from probability distributions using a confusion matrix.

    Args:
        probs: (n_shots, 4) array of outcome probabilities.
        cm: Confusion matrix.
        qubit_idx: The qubit index (1 or 2) for single-qubit expectation values,
                   or -1 for two-qubit correlations.

    Returns:
        (n_shots,) array of expectation values.
    """
    # probs is (N, 4). cm_inv is (4, 4).
    # We want p_corr[i, :] = cm_inv @ probs[i, :].T
    # So p_corr = (np.linalg.inv(cm) @ probs.T).T
    p_corr = (np.linalg.inv(cm) @ probs.T).T

    if qubit_idx == 1:
        p = p_corr[:, 0] + p_corr[:, 1]
    elif qubit_idx == 2:
        p = p_corr[:, 0] + p_corr[:, 2]
    else:  # Two-qubit correlation
        p = p_corr[:, 0] + p_corr[:, 3]

    return 2.0 * p - 1.0


# --- Expectation Value Functions with Error Mitigation ---

def _compute_expectation(gate: qt.Qobj, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray,
                         cm: np.ndarray, qubit_idx: int = -1) -> float:
    """Helper to compute expectation values using vectorized JAX operations."""
    povms_list = povms(a_m, delta)

    # Convert QuTiP objects to JAX arrays
    gate_jax = jnp.array(gate.full())
    p_jax = [jnp.array(p.full()) for p in povms_list]

    # Construct the 4 composite measurement operators
    # M00 = p1_0 * p2_0
    M00 = p_jax[0] @ p_jax[2]
    # M01 = p1_0 * p2_1
    M01 = p_jax[0] @ p_jax[3]
    # M10 = p1_1 * p2_0
    M10 = p_jax[1] @ p_jax[2]
    # M11 = p1_1 * p2_1
    M11 = p_jax[1] @ p_jax[3]

    M_ops = jnp.stack([M00, M01, M10, M11])

    # Compute probabilities for the entire batch
    probs = compute_probs_jax(state, gate_jax, M_ops)

    return get_expect_val_from_probs(probs, cm, qubit_idx)


def e_x_hat(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x for a given qubit with measurement error mitigation."""
    h = _get_hadamard_operators()
    gate = h[0] if qubit == 1 else h[1]
    return _compute_expectation(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_y_hat(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y for a given qubit with measurement error mitigation."""
    rx = _get_rx_operators()
    gate = rx[0] if qubit == 1 else rx[1]
    return _compute_expectation(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_xx_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    return _compute_expectation(h[2], state, a_m, delta, cm)


def e_xy_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_y with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation(h[0] * rx[1], state, a_m, delta, cm)


def e_yx_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation(rx[0] * h[1], state, a_m, delta, cm)


def e_yy_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_y with measurement error mitigation."""
    rx = _get_rx_operators()
    return _compute_expectation(rx[2], state, a_m, delta, cm)


def _compute_expectation_with_stderr(gate: qt.Qobj, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray,
                                     cm: np.ndarray, qubit_idx: int = -1) -> Tuple[float, float]:
    """Helper to compute expectation values and standard error using vectorized JAX operations."""
    povms_list = povms(a_m, delta)

    # Convert QuTiP objects to JAX arrays
    gate_jax = jnp.array(gate.full())
    p_jax = [jnp.array(p.full()) for p in povms_list]

    # Construct the 4 composite measurement operators
    M00 = p_jax[0] @ p_jax[2]
    M01 = p_jax[0] @ p_jax[3]
    M10 = p_jax[1] @ p_jax[2]
    M11 = p_jax[1] @ p_jax[3]

    M_ops = jnp.stack([M00, M01, M10, M11])

    # Compute probabilities for the entire batch
    probs = compute_probs_jax(state, gate_jax, M_ops)
    probs_np = np.array(probs)

    vals = get_expect_val_per_shot(probs_np, cm, qubit_idx)

    mean_val = float(np.mean(vals))
    stderr_val = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    return mean_val, stderr_val


def e_x_hat_with_stderr(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_x for a given qubit."""
    h = _get_hadamard_operators()
    gate = h[0] if qubit == 1 else h[1]
    return _compute_expectation_with_stderr(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_y_hat_with_stderr(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_y for a given qubit."""
    rx = _get_rx_operators()
    gate = rx[0] if qubit == 1 else rx[1]
    return _compute_expectation_with_stderr(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_xx_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_x \otimes sigma_x."""
    h = _get_hadamard_operators()
    return _compute_expectation_with_stderr(h[2], state, a_m, delta, cm)


def e_xy_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_x \otimes sigma_y."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(h[0] * rx[1], state, a_m, delta, cm)


def e_yx_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_y \otimes sigma_x."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(rx[0] * h[1], state, a_m, delta, cm)


def e_yy_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    """Calculates the expectation value and stderr of sigma_y \otimes sigma_y."""
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(rx[2], state, a_m, delta, cm)


# --- Concurrence and Entanglement Metrics ---


def a(expec: Tuple[float, float]) -> float:
    """Calculates the quantity A related to concurrence."""
    return -0.25 * np.log(expec[0] ** 2 + expec[1] ** 2)


def d(pm: str, expec: Tuple[float, float, float, float]) -> float:
    """Calculates the quantity D related to concurrence."""
    ex1x2, ey1y2, ex1y2, ey1x2 = expec
    if pm == '+':
        return -0.25 * np.log(np.square(ex1y2 + ey1x2) + np.square(ex1x2 - ey1y2))
    if pm == '-':
        return -0.25 * np.log(np.square(ex1y2 - ey1x2) + np.square(ex1x2 + ey1y2))

    raise ValueError(f"Invalid input for pm: {pm}. Expected '+' or '-'.")


def frame_correct(sol: List[qt.Qobj]) -> List[qt.Qobj]:
    """
    Applies a frame correction to the final states.
    Note: The original implementation was specific to certain pulse sequences and is currently disabled.
    """
    return sol


# --- Main Calculation Functions for Concurrence ---


def parametric_bootstrap_error(means: List[float], stderrs: List[float], func: Callable, n_boot: int = 1000) -> float:
    """
    Estimates the standard error of a function of multiple variables using parametric bootstrapping.

    Args:
        means: List of mean values for input variables.
        stderrs: List of standard errors for input variables.
        func: Function that takes the variables and returns a scalar.
        n_boot: Number of bootstrap samples.

    Returns:
        Standard error of the function output.
    """
    rng = np.random.default_rng()
    samples = []
    for m, s in zip(means, stderrs):
        samples.append(rng.normal(m, s, n_boot))

    # Transpose to get list of arguments per bootstrap sample
    samples = np.array(samples).T

    # Evaluate function on all samples
    outputs = [func(*row) for row in samples]

    return float(np.std(outputs, ddof=1))


def c_12_0_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                 ct: float, cm: np.ndarray, n_shots: int, m: int, a_m: np.ndarray, delta: np.ndarray,
                 noise_mats: np.ndarray, a_sp: np.ndarray) -> Tuple[float, float]:
    """Calculates one point of the C_12_0_MT correlation with standard error."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_val(exp_val_hat_ftn_with_stderr):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        val, err = exp_val_hat_ftn_with_stderr(sol, a_m, delta, cm)
        return val / (a_sp[0] * a_sp[1]), err / (a_sp[0] * a_sp[1])

    ex1x2, ex1x2_err = get_exp_val(e_xx_hat_with_stderr)
    ey1y2, ey1y2_err = get_exp_val(e_yy_hat_with_stderr)
    ex1y2, ex1y2_err = get_exp_val(e_xy_hat_with_stderr)
    ey1x2, ey1x2_err = get_exp_val(e_yx_hat_with_stderr)

    def calc_d_sum(v1, v2, v3, v4):
        return np.real(d('+', (v1, v2, v3, v4)) + d('-', (v1, v2, v3, v4)))

    mean_val = calc_d_sum(ex1x2, ey1y2, ex1y2, ey1x2)
    
    means = [ex1x2, ey1y2, ex1y2, ey1x2]
    stderrs = [ex1x2_err, ey1y2_err, ex1y2_err, ey1x2_err]
    
    stderr_val = parametric_bootstrap_error(means, stderrs, calc_d_sum)

    return mean_val, stderr_val


def make_c_12_0_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                   cm: np.ndarray, sp_mit: bool, **kwargs) -> Tuple[List[float], List[float]]:
    """Generates the C_12_0_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    results = Parallel(n_jobs=1)(
        delayed(c_12_0_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
    means, stderrs = zip(*results)
    return list(means), list(stderrs)


def c_12_12_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                  ct: float, cm: np.ndarray, n_shots: int, m: int, a_m: np.ndarray, delta: np.ndarray,
                  noise_mats: np.ndarray, a_sp: np.ndarray) -> Tuple[float, float]:
    """Calculates one point of the C_12_12_MT correlation with standard error."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_val(exp_val_hat_ftn_with_stderr):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        val, err = exp_val_hat_ftn_with_stderr(sol, a_m, delta, cm)
        return val / (a_sp[0] * a_sp[1]), err / (a_sp[0] * a_sp[1])

    ex1x2, ex1x2_err = get_exp_val(e_xx_hat_with_stderr)
    ey1y2, ey1y2_err = get_exp_val(e_yy_hat_with_stderr)
    ex1y2, ex1y2_err = get_exp_val(e_xy_hat_with_stderr)
    ey1x2, ey1x2_err = get_exp_val(e_yx_hat_with_stderr)

    def calc_d_diff(v1, v2, v3, v4):
        return np.real(d('+', (v1, v2, v3, v4)) - d('-', (v1, v2, v3, v4)))

    mean_val = calc_d_diff(ex1x2, ey1y2, ex1y2, ey1x2)
    
    means = [ex1x2, ey1y2, ex1y2, ey1x2]
    stderrs = [ex1x2_err, ey1y2_err, ex1y2_err, ey1x2_err]
    
    stderr_val = parametric_bootstrap_error(means, stderrs, calc_d_diff)

    return mean_val, stderr_val


def make_c_12_12_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                    cm: np.ndarray, sp_mit: bool, **kwargs) -> Tuple[List[float], List[float]]:
    """Generates the C_12_12_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    results = Parallel(n_jobs=1)(
        delayed(c_12_12_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
    means, stderrs = zip(*results)
    return list(means), list(stderrs)


def c_a_b_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, cm: np.ndarray,
               n_shots: int, m: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> Tuple[float, float]:
    """Calculates one point of the C_a_b_MT correlation with standard error."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex, ex_err = e_x_hat_with_stderr(l, sol, a_m, delta, cm)
        ey, ey_err = e_y_hat_with_stderr(l, sol, a_m, delta, cm)
        return ex, ex_err, ey, ey_err

    exlp, exlp_err, eylp, eylp_err = get_exp_vals(rhop)
    exlm, exlm_err, eylm, eylm_err = get_exp_vals(rhom)

    def calc_ab_diff(v1, v2, v3, v4):
        # v1=exlp, v2=eylp, v3=exlm, v4=eylm
        ax = 0.5 * (v1 + v3) / a_sp[l - 1]
        bx = 0.5 * (v1 - v3) / (a_sp[0] * a_sp[1])
        ay = 0.5 * (v2 + v4) / a_sp[l - 1]
        by = 0.5 * (v2 - v4) / (a_sp[0] * a_sp[1])

        ap = a((float(ax + bx), float(ay + by)))
        am = a((float(ax - bx), float(ay - by)))
        return ap - am

    mean_val = calc_ab_diff(exlp, eylp, exlm, eylm)
    
    means = [exlp, eylp, exlm, eylm]
    stderrs = [exlp_err, eylp_err, exlm_err, eylm_err]
    
    stderr_val = parametric_bootstrap_error(means, stderrs, calc_ab_diff)

    return mean_val, stderr_val


def make_c_a_b_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  cm: np.ndarray, sp_mit: bool, **kwargs) -> Tuple[List[float], List[float]]:
    """Generates the C_a_b_MT correlation over a range of control times."""
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_b = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_b)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    results = Parallel(n_jobs=1)(
        delayed(c_a_b_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
    means, stderrs = zip(*results)
    return list(means), list(stderrs)


def c_a_0_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, cm: np.ndarray,
               n_shots: int, m: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> Tuple[float, float]:
    """Calculates one point of the C_a_0_MT correlation with standard error."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex, ex_err = e_x_hat_with_stderr(l, sol, a_m, delta, cm)
        ey, ey_err = e_y_hat_with_stderr(l, sol, a_m, delta, cm)
        return ex, ex_err, ey, ey_err

    exlp, exlp_err, eylp, eylp_err = get_exp_vals(rhop)
    exlm, exlm_err, eylm, eylm_err = get_exp_vals(rhom)

    def calc_ab_sum(v1, v2, v3, v4):
        # v1=exlp, v2=eylp, v3=exlm, v4=eylm
        ax = 0.5 * (v1 + v3) / a_sp[l - 1]
        bx = 0.5 * (v1 - v3) / (a_sp[0] * a_sp[1])
        ay = 0.5 * (v2 + v4) / a_sp[l - 1]
        by = 0.5 * (v2 - v4) / (a_sp[0] * a_sp[1])

        ap = a((float(ax + bx), float(ay + by)))
        am = a((float(ax - bx), float(ay - by)))
        return ap + am

    mean_val = calc_ab_sum(exlp, eylp, exlm, eylm)
    
    means = [exlp, eylp, exlm, eylm]
    stderrs = [exlp_err, eylp_err, exlm_err, eylm_err]
    
    stderr_val = parametric_bootstrap_error(means, stderrs, calc_ab_sum)

    return mean_val, stderr_val


def make_c_a_0_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  cm: np.ndarray, sp_mit: bool, **kwargs) -> Tuple[List[float], List[float]]:
    """Generates the C_a_0_MT correlation over a range of control times."""
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_b = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_b)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    results = Parallel(n_jobs=1)(
        delayed(c_a_0_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
    means, stderrs = zip(*results)
    return list(means), list(stderrs)
