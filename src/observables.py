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
from qutip.qip.operations import snot

from .trajectories import make_init_state, make_y


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

def POVMs(a_m: np.ndarray, delta: np.ndarray) -> List[qt.Qobj]:
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


def gen_probs(gate: qt.Qobj, rho: qt.Qobj, povms: List[qt.Qobj]) -> np.ndarray:
    """
    Generates the probabilities of the four outcomes for a two-qubit measurement.

    Args:
        gate: The quantum gate applied before measurement.
        rho: The density matrix of the state.
        povms: The list of POVM operators.

    Returns:
        An array of four probabilities [p00, p01, p10, p11].
    """
    final_state = gate * rho * gate.dag()
    p00 = np.real((povms[0] * povms[2] * final_state).tr())
    p01 = np.real((povms[0] * povms[3] * final_state).tr())
    p10 = np.real((povms[1] * povms[2] * final_state).tr())
    p11 = np.real((povms[1] * povms[3] * final_state).tr())
    return np.array([p00, p01, p10, p11])


def get_expect_val_from_probs(probs: np.ndarray, CM: np.ndarray, qubit_idx: int = -1) -> float:
    """
    Calculates the expectation value from a probability distribution using a confusion matrix.

    Args:
        probs: An array of outcome probabilities.
        CM: The confusion matrix for error mitigation.
        qubit_idx: The qubit index (1 or 2) for single-qubit expectation values,
                   or -1 for two-qubit correlations.

    Returns:
        The calculated expectation value.
    """
    Pi = probs.mean(axis=0)
    P = np.linalg.inv(CM) @ Pi
    if qubit_idx == 1:
        p = P[0] + P[1]
    elif qubit_idx == 2:
        p = P[0] + P[2]
    else:  # Two-qubit correlation
        p = P[0] + P[3]
    return 2.0 * p - 1.0


# --- Expectation Value Functions with Error Mitigation ---

def E_X_hat(qubit: int, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x for a given qubit with measurement error mitigation."""
    povms = POVMs(a_m, delta)
    h = _get_hadamard_operators()
    gate = h[0] if qubit == 1 else h[1]
    probs = np.array([gen_probs(gate, s, povms) for s in state])
    return get_expect_val_from_probs(probs, CM, qubit_idx=qubit)


def E_Y_hat(qubit: int, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y for a given qubit with measurement error mitigation."""
    povms = POVMs(a_m, delta)
    rx = _get_rx_operators()
    gate = rx[0] if qubit == 1 else rx[1]
    probs = np.array([gen_probs(gate, s, povms) for s in state])
    return get_expect_val_from_probs(probs, CM, qubit_idx=qubit)


def _get_two_qubit_exp_val_hat(gate: qt.Qobj, state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray,
                               CM: np.ndarray) -> float:
    """Helper function to calculate a two-qubit expectation value."""
    povms = POVMs(a_m, delta)
    probs = np.array([gen_probs(gate, s, povms) for s in state])
    return get_expect_val_from_probs(probs, CM)


def E_XX_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    return _get_two_qubit_exp_val_hat(h[2], state, a_m, delta, CM)


def E_XY_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x \otimes sigma_y with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(h[0] * rx[1], state, a_m, delta, CM)


def E_YX_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(rx[0] * h[1], state, a_m, delta, CM)


def E_YY_hat(state: List[qt.Qobj], a_m: np.ndarray, delta: np.ndarray, CM: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y \otimes sigma_y with measurement error mitigation."""
    rx = _get_rx_operators()
    return _get_two_qubit_exp_val_hat(rx[2], state, a_m, delta, CM)


# --- Concurrence and Entanglement Metrics ---

def A(expec: Tuple[float, float]) -> float:
    """Calculates the quantity A related to concurrence."""
    return -0.25 * np.log(expec[0] ** 2 + expec[1] ** 2)


def D(pm: str, expec: Tuple[float, float, float, float]) -> float:
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

def C_12_0_MT_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                 ct: float, CM: np.ndarray, n_shots: int, M: int, a_m: np.ndarray, delta: np.ndarray,
                 noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_12_0_MT correlation."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))

    def get_exp_val(exp_val_hat_ftn):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        return exp_val_hat_ftn(sol, a_m, delta, CM) / (a_sp[0] * a_sp[1])

    EX1X2 = get_exp_val(E_XX_hat)
    EY1Y2 = get_exp_val(E_YY_hat)
    EX1Y2 = get_exp_val(E_XY_hat)
    EY1X2 = get_exp_val(E_YX_hat)

    d_plus = D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2))
    d_minus = D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2))

    return np.real(d_plus + d_minus)


def make_C_12_0_MT(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                   CM: np.ndarray, spMit: bool, **kwargs) -> List[float]:
    """Generates the C_12_0_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_B = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    a_sp = np.array([1., 1.]) if not spMit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(C_12_0_MT_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, CM,
            n_shots=kwargs['n_shots'], M=kwargs['M'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def C_12_12_MT_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray, rho: jnp.ndarray,
                  ct: float, CM: np.ndarray, n_shots: int, M: int, a_m: np.ndarray, delta: np.ndarray,
                  noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_12_12_MT correlation."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))

    def get_exp_val(exp_val_hat_ftn):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
        return exp_val_hat_ftn(sol, a_m, delta, CM) / (a_sp[0] * a_sp[1])

    EX1X2 = get_exp_val(E_XX_hat)
    EY1Y2 = get_exp_val(E_YY_hat)
    EX1Y2 = get_exp_val(E_XY_hat)
    EY1X2 = get_exp_val(E_YX_hat)

    d_plus = D('+', (EX1X2, EY1Y2, EX1Y2, EY1X2))
    d_minus = D('-', (EX1X2, EY1Y2, EX1Y2, EY1X2))

    return np.real(d_plus - d_minus)


def make_C_12_12_MT(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                    CM: np.ndarray, spMit: bool, **kwargs) -> List[float]:
    """Generates the C_12_12_MT correlation over a range of control times."""
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_B = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_B)).full())
    a_sp = np.array([1., 1.]) if not spMit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(C_12_12_MT_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, rho, ct, CM,
            n_shots=kwargs['n_shots'], M=kwargs['M'], a_m=kwargs['a_m'],
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def C_a_b_MT_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, CM: np.ndarray,
               n_shots: int, M: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_a_b_MT correlation."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex = E_X_hat(l, sol, a_m, delta, CM)
        ey = E_Y_hat(l, sol, a_m, delta, CM)
        return ex, ey

    EXlp, EYlp = get_exp_vals(rhop)
    EXlm, EYlm = get_exp_vals(rhom)

    aX = 0.5 * (EXlp + EXlm) / a_sp[l - 1]
    bX = 0.5 * (EXlp - EXlm) / (a_sp[0] * a_sp[1])
    aY = 0.5 * (EYlp + EYlm) / a_sp[l - 1]
    bY = 0.5 * (EYlp - EYlm) / (a_sp[0] * a_sp[1])

    Ap = A((float(aX + bX), float(aY + bY)))
    Am = A((float(aX - bX), float(aY - bY)))

    return Ap - Am


def make_C_a_b_MT(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  CM: np.ndarray, spMit: bool, **kwargs) -> List[float]:
    """Generates the C_a_b_MT correlation over a range of control times."""
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_B = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    a_sp = np.array([1., 1.]) if not spMit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(C_a_b_MT_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, CM,
            n_shots=kwargs['n_shots'], M=kwargs['M'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )


def C_a_0_MT_i(solver_ftn: Callable, t_b: float, pulse: List[str], t_vec: np.ndarray,
               rho_pair: Tuple[jnp.ndarray, jnp.ndarray], ct: float, CM: np.ndarray,
               n_shots: int, M: int, a_m: np.ndarray, l: int, delta: np.ndarray,
               noise_mats: np.ndarray, a_sp: np.ndarray) -> float:
    """Calculates one point of the C_a_0_MT correlation."""
    rhop, rhom = rho_pair
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, M=M))

    def get_exp_vals(rho_in):
        sol = solver_ftn(y_uv, noise_mats, t_vec, rho_in, n_shots)
        ex = E_X_hat(l, sol, a_m, delta, CM)
        ey = E_Y_hat(l, sol, a_m, delta, CM)
        return ex, ey

    EXlp, EYlp = get_exp_vals(rhop)
    EXlm, EYlm = get_exp_vals(rhom)

    aX = 0.5 * (EXlp + EXlm) / a_sp[l - 1]
    bX = 0.5 * (EXlp - EXlm) / (a_sp[0] * a_sp[1])
    aY = 0.5 * (EYlp + EYlm) / a_sp[l - 1]
    bY = 0.5 * (EYlp - EYlm) / (a_sp[0] * a_sp[1])

    Ap = A((float(aX + bX), float(aY + bY)))
    Am = A((float(aX - bX), float(aY - bY)))

    return Ap + Am


def make_C_a_0_MT(solver_ftn: Callable, pulse: List[str], t_vec: np.ndarray, c_times: np.ndarray,
                  CM: np.ndarray, spMit: bool, **kwargs) -> List[float]:
    """Generates the C_a_0_MT correlation over a range of control times."""
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_B = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_B)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_B)).full())
    a_sp = np.array([1., 1.]) if not spMit else kwargs['a_sp']

    return Parallel(n_jobs=1)(
        delayed(C_a_0_MT_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, CM,
            n_shots=kwargs['n_shots'], M=kwargs['M'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
