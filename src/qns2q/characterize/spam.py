"""
SPAM (state-preparation and measurement) calibration, parameter estimation, and
SPAM-robust coefficient estimators for two-qubit frequency-comb QNS.

This module implements the two SPAM-handling strategies of the companion paper
(Secs. "SPAM-Robust frequency-comb QNS" and "SPAM-Mitigated QNS"):

SPAM-mitigated protocol
-----------------------
``simulate_calibration`` measures, at t=0 and through the *same* faulty POVMs used
by the experiments, the twisted readout estimates that are identifiable from
prepare-and-measure data:

* the readout asymmetries  delta_l            (exact, per-shot, via twisting), and
* the gauge-invariant products  P_l = alpha_M^(l) * alpha_SP^{z,(l)}.

The split of P_l into alpha_M vs alpha_SP^z is NOT identifiable from such data
(SPAM gauge freedom); the paper assumes it is supplied by a separate protocol
[khan2025]. ``estimate_spam`` therefore takes a ``split_error`` knob that emulates
the accuracy of that protocol while keeping the measured product exact. The
resulting ``SpamEstimate`` provides the estimated confusion matrix A_hat (inverted
by the standard estimators in ``model.observables``) and the estimated
alpha_hat_SP^z (used as the SP divisor ``a_sp_div``).

SPAM-robust protocol
--------------------
Raw (no confusion-matrix inversion) estimators built from

* twisting   : pre-measurement X1X2 bit-flips; tilde-E^pm = (E_hat -+ E_hat_flip)/2,
* wringing   : averaging the 'pp' and 'pp_wrung' (Z1Z2-conjugated) preparations,
* in-situ asymmetry removal : E^+[O12] = tilde-E^+[O12] - delta1_hat*delta2_hat,
  with delta_l_hat measured per-shot from the twisted single-qubit marginals.

The resulting coefficient estimates carry a time-independent SPAM offset that is
removed downstream by linear regression over the repetition number M
(``inversion.regress_observables_over_M``); the C_12_12 cross-coefficient is exactly
SPAM-free with no regression needed. Per the paper, the single-qubit coefficients
C_l_0 are only SPAM-robust when the qubit-Ising cross-spectra S_l,12 vanish, and
the C_a_b coefficients (-> S_1_12, S_2_12) are NOT robustly accessible at all.

Shot accounting mirrors the legacy estimators: every (observable, preparation)
pair gets its own ``n_shots`` noise realizations, as in a real experiment where
measurement settings cannot share shots. The twisted (flipped) run is measured on
the same final states as its unflipped partner -- a paired measurement that makes
the per-shot twisted asymmetry estimate exact in this simulator.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import jax.numpy as jnp
import numpy as np
import qutip as qt

from qns2q.model.observables import (povms, compute_probs_jax, _get_flip_operator,
                                     _get_hadamard_operators, _get_rx_operators,
                                     a, d, parametric_bootstrap_error)
from qns2q.model.trajectories import make_init_state, make_y


# ==============================================================================
# Confusion matrix, calibration, and SPAM parameter estimation (mitigated path)
# ==============================================================================


def confusion_matrix(a_m, delta):
    """Two-qubit confusion matrix A = A^(1) (x) A^(2) from (alpha_M, delta).

    [A^(l)]_ij = P(outcome i | state j) with the paper's parameterization
    (visibility alpha_M^(l), asymmetry delta_l). Identical to the CM built in
    QNSExperimentConfig.__post_init__ -- kept here as the single reusable form.
    """
    blocks = []
    for l in range(2):
        al, dl = float(a_m[l]), float(delta[l])
        blocks.append(np.array([[0.5 * (1 + al + dl), 0.5 * (1 - al + dl)],
                                [0.5 * (1 - al - dl), 0.5 * (1 + al - dl)]]))
    return np.kron(blocks[0], blocks[1])


@dataclass
class SpamEstimate:
    """SPAM parameters as estimated by the experimenter (NOT the injected truth)."""
    a_m: np.ndarray        # estimated alpha_M per qubit
    delta: np.ndarray      # estimated delta per qubit
    a_sp: np.ndarray       # estimated alpha_SP^z per qubit
    products: np.ndarray   # measured gauge-invariant products alpha_M*alpha_SP^z
    cm: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cm = confusion_matrix(self.a_m, self.delta)


def _ground_state_with_bath(a_sp, c):
    """Faulty two-qubit ground-state prep rho_in (x) maximally-mixed bath, as a
    (1, 8, 8) batch for the POVM machinery."""
    zp, zm = qt.basis(2, 0), qt.basis(2, 1)
    rhos = []
    for l in range(2):
        asp, cl = a_sp[l], c[l]
        rhos.append(0.5 * (1. + asp) * zp * zp.dag() + 0.5 * (1. - asp) * zm * zm.dag()
                    + 0.5 * cl * zp * zm.dag() + 0.5 * np.conj(cl) * zm * zp.dag())
    rho = qt.tensor(rhos[0], rhos[1], 0.5 * qt.identity(2))
    return jnp.array(rho.full())[None, :, :]


def _measurement_ops(a_m, delta):
    """Stacked composite measurement operators [M00, M01, M10, M11] (jax, (4,8,8))."""
    p_jax = [jnp.array(p.full()) for p in povms(a_m, delta)]
    return jnp.stack([p_jax[0] @ p_jax[2], p_jax[0] @ p_jax[3],
                      p_jax[1] @ p_jax[2], p_jax[1] @ p_jax[3]])


def _probs_to_e(probs: np.ndarray, qubit_idx: int) -> np.ndarray:
    """Per-shot raw expectation values from outcome probabilities (no CM inversion)."""
    probs = np.asarray(probs)
    if qubit_idx == 1:
        p = probs[:, 0] + probs[:, 1]
    elif qubit_idx == 2:
        p = probs[:, 0] + probs[:, 2]
    else:
        p = probs[:, 0] + probs[:, 3]
    return 2.0 * p - 1.0


def simulate_calibration(a_m_true, delta_true, a_sp_true, c_true):
    """Twisted t=0 readout calibration through the faulty prep and POVMs.

    Prepares the faulty ground state rho_in (no evolution) and measures Z_1, Z_2
    raw and with the pre-measurement X1X2 flip. The twisted combinations give

        delta_l_hat = (E_hat[Z_l] + E_hat_flip[Z_l]) / 2        (exact)
        P_l_hat     = (E_hat[Z_l] - E_hat_flip[Z_l]) / 2
                    = alpha_M^(l) * alpha_SP^{z,(l)}             (exact)

    In this simulator outcomes are exact POVM traces (no projective sampling), so
    these estimates are exact up to floating point.

    Returns
    -------
    (delta_hat, products_hat) : two length-2 float arrays.
    """
    rho = _ground_state_with_bath(a_sp_true, c_true)
    M_ops = _measurement_ops(a_m_true, delta_true)
    eye = jnp.eye(8, dtype=jnp.complex128)
    flip = jnp.array(_get_flip_operator().full())
    pr = compute_probs_jax(rho, eye, M_ops)
    prf = compute_probs_jax(rho, flip, M_ops)
    delta_hat, products_hat = [], []
    for l in (1, 2):
        e_raw = _probs_to_e(pr, l)[0]
        e_flip = _probs_to_e(prf, l)[0]
        delta_hat.append(0.5 * (e_raw + e_flip))
        products_hat.append(0.5 * (e_raw - e_flip))
    return np.array(delta_hat), np.array(products_hat)


def estimate_spam(a_m_true, delta_true, a_sp_true, c_true, split_error=0.0):
    """Full SPAM-parameter estimation as available to the experimenter.

    delta_hat and the products P_l are measured exactly by ``simulate_calibration``.
    The alpha_M-vs-alpha_SP^z split is supplied externally (khan2025); ``split_error``
    emulates its accuracy:

        alpha_hat_SP^z = alpha_SP^z * (1 + split_error)
        alpha_hat_M    = P_l_hat / alpha_hat_SP^z

    so the gauge-invariant product stays exact while the split is biased. The
    default ``split_error=0`` models a faithful external estimation.
    """
    delta_hat, products_hat = simulate_calibration(a_m_true, delta_true, a_sp_true, c_true)
    a_sp_hat = np.asarray(a_sp_true, dtype=float) * (1.0 + split_error)
    a_m_hat = products_hat / a_sp_hat
    return SpamEstimate(a_m=a_m_hat, delta=delta_hat, a_sp=a_sp_hat,
                        products=products_hat)


# ==============================================================================
# SPAM-robust estimators (raw + twisting + wringing + in-situ asymmetry removal)
# ==============================================================================


def _gates():
    """Measurement-basis rotation gates keyed by observable."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return {
        'x1': (h[0], 1), 'y1': (rx[0], 1), 'x2': (h[1], 2), 'y2': (rx[1], 2),
        'xx': (h[2], -1), 'yy': (rx[2], -1),
        'xy': (h[0] * rx[1], -1), 'yx': (rx[0] * h[1], -1),
    }


def _per_shot_twist_pair(sol, gate: qt.Qobj, qubit_idx: int, M_ops, flip_jax):
    """Per-shot (raw, flipped) expectation values of one observable on a batch of
    final states. Raw and flipped runs share the same noise shots (paired)."""
    g = jnp.array(gate.full())
    e_raw = _probs_to_e(compute_probs_jax(sol, g, M_ops), qubit_idx)
    e_flip = _probs_to_e(compute_probs_jax(sol, flip_jax @ g, M_ops), qubit_idx)
    return e_raw, e_flip


def _mean_err(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _robust_two_qubit_w(solver_ftn, y_uv, t_vec, n_shots, a_m, delta, a_sp, c,
                        noise_mats):
    """Twisted + wrung two-qubit estimator inputs for one (sequence, time) point.

    For each preparation in the wringing pair {'pp', 'pp_wrung'} and each two-qubit
    observable O in {XX, YY, XY, YX}, evolve a fresh batch of n_shots noise
    realizations and form, per shot,

        E^+[O] = (E_hat[O] + E_hat_flip[O])/2 - delta1_hat*delta2_hat
               = alpha_M^(1) alpha_M^(2) * E_bar[O],

    with delta_l_hat the per-shot twisted single-qubit marginals measured on the
    same shots (exact). The wringing average over the two preparations then kills
    the transverse-SP cross terms.

    Returns
    -------
    (means, errs) : dicts keyed by 'xx','yy','xy','yx' with the wrung means
        W[O] = W_+{E^+[O]} and their standard errors.
    """
    gates = _gates()
    M_ops = _measurement_ops(a_m, delta)
    flip_jax = jnp.array(_get_flip_operator().full())
    rho_b = 0.5 * qt.identity(2)

    per_prep = []
    for state in ('pp', 'pp_wrung'):
        rho0 = make_init_state(a_sp, c, state=state)
        rho = jnp.array((qt.tensor(rho0, rho_b)).full())
        prep_vals = {}
        for key in ('xx', 'yy', 'xy', 'yx'):
            gate, qidx = gates[key]
            sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
            e_raw, e_flip = _per_shot_twist_pair(sol, gate, qidx, M_ops, flip_jax)
            tplus = 0.5 * (e_raw + e_flip)
            # In-situ per-shot asymmetry product from the twisted single-qubit
            # marginals of the SAME shots (exact: tilde-E^+[O_l] = delta_l per shot).
            g1, q1 = gates['x1']
            g2, q2 = gates['x2']
            d1_raw, d1_flip = _per_shot_twist_pair(sol, g1, q1, M_ops, flip_jax)
            d2_raw, d2_flip = _per_shot_twist_pair(sol, g2, q2, M_ops, flip_jax)
            d1 = 0.5 * (d1_raw + d1_flip)
            d2 = 0.5 * (d2_raw + d2_flip)
            prep_vals[key] = _mean_err(tplus - d1 * d2)
        per_prep.append(prep_vals)

    means, errs = {}, {}
    for key in ('xx', 'yy', 'xy', 'yx'):
        (m0, e0), (m1, e1) = per_prep[0][key], per_prep[1][key]
        means[key] = 0.5 * (m0 + m1)
        errs[key] = 0.5 * np.sqrt(e0 ** 2 + e1 ** 2)
    return means, errs


def _c_two_qubit_robust_i(solver_ftn, t_b, pulse, t_vec, ct, n_shots, m, a_m, delta,
                          a_sp, c, noise_mats, combine: str) -> Tuple[float, float]:
    """One (control-time) point of the robust C_12_0 ('sum') / C_12_12 ('diff')."""
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))
    means, errs = _robust_two_qubit_w(solver_ftn, y_uv, t_vec, n_shots, a_m, delta,
                                      a_sp, c, noise_mats)

    def calc(v_xx, v_yy, v_xy, v_yx):
        dp = d('+', (v_xx, v_yy, v_xy, v_yx))
        dm = d('-', (v_xx, v_yy, v_xy, v_yx))
        return np.real(dp + dm) if combine == 'sum' else np.real(dp - dm)

    keys = ('xx', 'yy', 'xy', 'yx')
    mean_val = calc(*[means[k] for k in keys])
    stderr_val = parametric_bootstrap_error([means[k] for k in keys],
                                            [errs[k] for k in keys], calc)
    return mean_val, stderr_val


def make_c_12_0_mt_robust(solver_ftn, pulse, t_vec, c_times, cm, sp_mit,
                          **kwargs) -> Tuple[List[float], List[float]]:
    """SPAM-robust C_12_0(MT) = D^+ + D^- over a range of control times.

    Drop-in replacement for ``observables.make_c_12_0_mt`` (same signature; the
    ``cm``/``sp_mit`` arguments are ignored -- raw estimates by construction).
    Carries the time-independent SPAM intercept
    -1/2 ln[(aM1*aM2)^2((az1^2+ay1^2))((az2^2+ay2^2))], removed downstream by
    linear regression over the repetition number M.
    """
    results = [
        _c_two_qubit_robust_i(solver_ftn, kwargs['t_b'], pulse, t_vec, ct,
                              kwargs['n_shots'], kwargs['m'], kwargs['a_m'],
                              kwargs['delta'], kwargs['a_sp'], kwargs['c'],
                              kwargs['noise_mats'], 'sum')
        for ct in c_times
    ]
    means, stderrs = zip(*results)
    return list(means), list(stderrs)


def make_c_12_12_mt_robust(solver_ftn, pulse, t_vec, c_times, cm, sp_mit,
                           **kwargs) -> Tuple[List[float], List[float]]:
    """SPAM-robust C_12_12(MT) = D^+ - D^-: the SPAM intercept of D^+ and D^- is
    identical and cancels EXACTLY -- no M-regression needed for this coefficient."""
    results = [
        _c_two_qubit_robust_i(solver_ftn, kwargs['t_b'], pulse, t_vec, ct,
                              kwargs['n_shots'], kwargs['m'], kwargs['a_m'],
                              kwargs['delta'], kwargs['a_sp'], kwargs['c'],
                              kwargs['noise_mats'], 'diff')
        for ct in c_times
    ]
    means, stderrs = zip(*results)
    return list(means), list(stderrs)


def _c_a_0_robust_i(solver_ftn, t_b, pulse, t_vec, ct, n_shots, m, a_m, delta,
                    a_sp, c, noise_mats, l: int) -> Tuple[float, float]:
    """One point of the robust C_l_0: twisted-only estimator
    A_hat_l^pm = -1/4 ln{tilde-E^-[X_l]^2 + tilde-E^-[Y_l]^2}, summed over the
    psi_l^pm preparations."""
    gates = _gates()
    M_ops = _measurement_ops(a_m, delta)
    flip_jax = jnp.array(_get_flip_operator().full())
    rho_b = 0.5 * qt.identity(2)
    y_uv = jnp.array(make_y(t_b, pulse, ctime=ct, m=m))

    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'
    obs_x, obs_y = (f'x{l}', f'y{l}')

    vals, errs = [], []
    for state in (state_p, state_m):
        rho0 = make_init_state(a_sp, c, state=state)
        rho = jnp.array((qt.tensor(rho0, rho_b)).full())
        for key in (obs_x, obs_y):
            gate, qidx = gates[key]
            sol = solver_ftn(y_uv, noise_mats, t_vec, rho, n_shots)
            e_raw, e_flip = _per_shot_twist_pair(sol, gate, qidx, M_ops, flip_jax)
            mv, ev = _mean_err(0.5 * (e_raw - e_flip))   # tilde-E^- = alpha_M*E_bar
            vals.append(mv)
            errs.append(ev)

    def calc(exp_, eyp, exm, eym):
        return a((exp_, eyp)) + a((exm, eym))

    mean_val = calc(*vals)
    stderr_val = parametric_bootstrap_error(vals, errs, calc)
    return mean_val, stderr_val


def make_c_a_0_mt_robust(solver_ftn, pulse, t_vec, c_times, cm, sp_mit,
                         **kwargs) -> Tuple[List[float], List[float]]:
    """SPAM-robust C_l_0(MT) over a range of control times (twisting only).

    Per the paper this estimator is unbiased ONLY when the qubit-Ising
    cross-spectra vanish (S_l,12 = 0); with nonzero cross-correlations the
    transverse-SP mixing of the psi_l^pm channels leaks C_l,2(t) into the
    estimate. The SPAM intercept -1/2 ln[(aM_l)^2(az_l^2+ay_l^2)] is removed by
    M-regression downstream.
    """
    results = [
        _c_a_0_robust_i(solver_ftn, kwargs['t_b'], pulse, t_vec, ct,
                        kwargs['n_shots'], kwargs['m'], kwargs['a_m'],
                        kwargs['delta'], kwargs['a_sp'], kwargs['c'],
                        kwargs['noise_mats'], kwargs['l'])
        for ct in c_times
    ]
    means, stderrs = zip(*results)
    return list(means), list(stderrs)
