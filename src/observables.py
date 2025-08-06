"""
This module provides functions for calculating quantum observables from simulated experimental data,
including single-qubit and two-qubit expectation values, and concurrence-related metrics.
It is designed to work with QuTiP for quantum object representations and JAX for numerical calculations.
"""

from typing import List, Tuple, Callable

import jax.numpy as jnp
import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from qutip.qip.gates import snot

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


def gen_probs(gate: qt.Qobj, rho: qt.Qobj, povms_list: List[qt.Qobj]) -> np.ndarray:
    """
    Generates the probabilities of the four outcomes for a two-qubit measurement.

    Args:
        gate: The quantum gate applied before measurement.
        rho: The density matrix of the state.
        povms_list: The list of POVM operators.

    Returns:
        An array of four probabilities [p00, p01, p10, p11].
    """
    final_state = gate * rho * gate.dag()
    p00 = np.real((povms_list[0] * povms_list[2] * final_state).tr())
    p01 = np.real((povms_list[0] * povms_list[3] * final_state).tr())
    p10 = np.real((povms_list[1] * povms_list[2] * final_state).tr())
    p11 = np.real((povms_list[1] * povms_list[3] * final_state).tr())
    return np.array([p00, p01, p10, p11])


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
    pi = probs.mean(axis=0)
    p_corr = np.linalg.inv(cm) @ pi
    if qubit_idx == 1:
        p = p_corr[0] + p_corr[1]
    elif qubit_idx == 2:
        p = p_corr[0] + p_corr[2]
    else:  # Two-qubit correlation
        p = p_corr[0] + p_corr[3]
    return 2.0 * p - 1.0


# --- Expectation Value Functions with Error Mitigation ---


def e_x_hat(qubit: int, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x for a given qubit with measurement error mitigation."""
    povms_list = povms(a_m, delta)
    h = _get_hadamard_operators()
    gate = h[0] if qubit == 1 else h[1]
    probs = np.array([gen_probs(gate, s, povms_list) for s in state])
    return get_expect_val_from_probs(probs, cm, qubit_idx=qubit)


def e_y_hat(qubit: int, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y for a given qubit with measurement error mitigation."""
    povms_list = povms(a_m, delta)
    rx = _get_rx_operators()
    gate = rx[0] if qubit == 1 else rx[1]
    probs = np.array([gen_probs(gate, s, povms_list) for s in state])
    return get_expect_val_from_probs(probs, cm, qubit_idx=qubit)


def _get_two_qubit_exp_val_hat(gate: qt.Qobj, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray,
                               cm: np.ndarray) -> float:
    """Helper function to calculate a two-qubit expectation value."""
    povms_list = povms(a_m, delta)
    probs = np.array([gen_probs(gate, s, povms_list) for s in state])
    return get_expect_val_from_probs(probs, cm)


def e_xx_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    return _get_two_qubit_exp_val_hat(h[2], state, a_m, delta, cm)


def e_xy_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_y with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(h[0] * rx[1], state, a_m, delta, cm)


def e_yx_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(rx[0] * h[1], state, a_m, delta, cm)


def e_yy_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_y with measurement error mitigation."""
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(rx[2], state, a_m, delta, cm)


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


def c_12_0_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                 ct: float, cm: np.ndarray, n_shots: int, m: int, a_m: np.ndarray, delta: np.ndarray,
                 noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_12_0_MT correlation."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_val(exp_val_hat_ftn):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        return exp_val_hat_ftn(sol, a_m, delta, cm) / (a_sp[0] * a_sp[1])

    ex1x2 = get_exp_val(e_xx_hat)
    ey1y2 = get_exp_val(e_yy_hat)
    ex1y2 = get_exp_val(e_xy_hat)
    ey1x2 = get_exp_val(e_yx_hat)

    d_plus = d('+', (ex1x2, ey1y2, ex1y2, ey1x2))
    d_minus = d('-', (ex1x2, ey1y2, ex1y2, ey1x2))

    return np.real(d_plus + d_minus)


def make_c_12_0_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                   cm: np.ndarray, sp_mit: bool, **kwargs) -> List[float]:
    """Generates the C_12_0_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(c_12_0_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def c_12_12_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                  ct: float, cm: np.ndarray, n_shots: int, m: int, a_m: np.ndarray, delta: np.ndarray,
                  noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_12_12_MT correlation."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_val(exp_val_hat_ftn):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        return exp_val_hat_ftn(sol, a_m, delta, cm) / (a_sp[0] * a_sp[1])

    ex1x2 = get_exp_val(e_xx_hat)
    ey1y2 = get_exp_val(e_yy_hat)
    ex1y2 = get_exp_val(e_xy_hat)
    ey1x2 = get_exp_val(e_yx_hat)

    d_plus = d('+', (ex1x2, ey1y2, ex1y2, ey1x2))
    d_minus = d('-', (ex1x2, ey1y2, ex1y2, ey1x2))

    return np.real(d_plus - d_minus)


def make_c_12_12_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                    cm: np.ndarray, sp_mit: bool, **kwargs) -> List[float]:
    """Generates the C_12_12_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = np.array([1., 1.]) if not sp_mit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(c_12_12_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def c_a_b_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, cm: np.ndarray,
               n_shots: int, m: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_a_b_MT correlation."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex = e_x_hat(l, sol, a_m, delta, cm)
        ey = e_y_hat(l, sol, a_m, delta, cm)
        return ex, ey

    exlp, eylp = get_exp_vals(rhop)
    exlm, eylm = get_exp_vals(rhom)

    ax = 0.5 * (exlp + exlm) / a_sp[l - 1]
    bx = 0.5 * (exlp - exlm) / (a_sp[0] * a_sp[1])
    ay = 0.5 * (eylp + eylm) / a_sp[l - 1]
    by = 0.5 * (eylp - eylm) / (a_sp[0] * a_sp[1])

    ap = a((float(ax + bx), float(ay + by)))
    am = a((float(ax - bx), float(ay - by)))

    return ap - am


def make_c_a_b_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  cm: np.ndarray, sp_mit: bool, **kwargs) -> List[float]:
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

    return Parallel(n_jobs=1)(
        delayed(c_a_b_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def c_a_0_mt_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, cm: np.ndarray,
               n_shots: int, m: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_a_0_MT correlation."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex = e_x_hat(l, sol, a_m, delta, cm)
        ey = e_y_hat(l, sol, a_m, delta, cm)
        return ex, ey

    exlp, eylp = get_exp_vals(rhop)
    exlm, eylm = get_exp_vals(rhom)

    ax = 0.5 * (exlp + exlm) / a_sp[l - 1]
    bx = 0.5 * (exlp - exlm) / (a_sp[0] * a_sp[1])
    ay = 0.5 * (eylp + eylm) / a_sp[l - 1]
    by = 0.5 * (eylp - eylm) / (a_sp[0] * a_sp[1])

    ap = a((float(ax + bx), float(ay + by)))
    am = a((float(ax - bx), float(ay - by)))

    return ap + am


def make_c_a_0_mt(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  cm: np.ndarray, sp_mit: bool, **kwargs) -> List[float]:
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

    return Parallel(n_jobs=1)(
        delayed(c_a_0_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
