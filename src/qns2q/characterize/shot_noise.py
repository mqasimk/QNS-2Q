"""Deterministic measurement-error propagation from saved dephasing phases.

No trajectory solver, random generator, or plotting dependency is used here.
The acquisition model assigns independent physical measurements to each
observable/preparation setting, including settings reused by different
coefficient estimators. Calibration parameters are held fixed. The multinomial
covariance replaces, rather than adds to, the trajectory-only Monte Carlo error.
"""
from dataclasses import dataclass

import numpy as np


HARMONIC_EXPERIMENTS = (
    ('C_12_0_MT_1', 'sum', 0), ('C_12_0_MT_2', 'sum', 0),
    ('C_12_0_MT_3', 'sum', 0), ('C_12_12_MT_1', 'diff', 0),
    ('C_12_12_MT_2', 'diff', 0), ('C_1_0_MT_1', 'sum', 1),
    ('C_2_0_MT_1', 'sum', 2), ('C_12_0_MT_4', 'sum', 0),
    ('C_1_2_MT_1', 'diff', 1), ('C_1_2_MT_2', 'diff', 1),
    ('C_2_1_MT_1', 'diff', 2), ('C_2_1_MT_2', 'diff', 2),
)
DC_EXPERIMENTS = (
    ('C_1_0_FIDCDD3', 'sum', 1), ('C_2_0_CDD3FID', 'sum', 2),
    ('C_1_0_FIDFID', 'sum', 1), ('C_2_0_FIDFID', 'sum', 2),
    ('C_12_0_FID_FID', 'sum', 0), ('C_12_12_FID', 'diff', 0),
    ('C_1_12_FID', 'diff', 1), ('C_2_12_FID', 'diff', 2),
    ('C_1_0_CDD1CDD1', 'sum', 1), ('C_1_0_CDD1CPMG', 'sum', 1),
)
DIAGONAL_BASIS = np.array([[1, 1, -1, -1], [1, -1, 1, -1],
                           [1, -1, -1, 1]], dtype=float)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_RX = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
_RY = np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2)
READOUT_GATES = {
    'xx': np.kron(_H, _H), 'yy': np.kron(_RX, _RX),
    'xy': np.kron(_H, _RX), 'yx': np.kron(_RX, _H),
    'x1': np.kron(_H, _I), 'y1': np.kron(_RX, _I),
    'x2': np.kron(_I, _H), 'y2': np.kron(_I, _RX),
}
OUTCOME_SIGNS = {0: np.array([1, -1, -1, 1]),
                 1: np.array([1, 1, -1, -1]),
                 2: np.array([1, -1, 1, -1])}


@dataclass(frozen=True)
class Acquisition:
    name: str
    kind: str
    qubit: int
    index: int
    first_call: int
    repetitions: int
    dc: bool

    @property
    def width(self):
        return 2 if self.qubit else 4


def acquisition_plan(params):
    """Mirror the pinned non-robust experiment order, including DC sweeps."""
    if str(params.get('spam_protocol', 'none')) == 'robust':
        raise ValueError('The SPAM-robust acquisition is outside this workflow.')
    index = 0
    for dc, experiments, count in (
        (False, HARMONIC_EXPERIMENTS, int(params['truncate'])),
        (True, DC_EXPERIMENTS, int(params['M'])),
    ):
        for name, kind, qubit in experiments:
            for k in range(count):
                yield Acquisition(name, kind, qubit, k, index,
                                  k + 1 if dc else int(params['M']), dc)
                index += 2 if qubit else 4


def phase_transfer(coefficients):
    """Ensemble average of u_i conjugate(u_j), sufficient for every readout."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError('Expected a (noise_realizations, 3) phase array.')
    if not len(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError('Phase arrays must be nonempty and finite.')
    u = np.exp(-1j * (coefficients @ DIAGONAL_BASIS))
    return u.T @ u.conj() / len(u)


def prepared_state(params, state):
    """The two physical qubits; the inert bath traces out exactly."""
    asp = params['a_sp']
    c = params.get('c_prep', params['c'])
    rho = np.kron(*[np.array([[1 + asp[j], c[j]],
                             [np.conj(c[j]), 1 - asp[j]]]) / 2
                   for j in range(2)])
    ry1, ry2 = np.kron(_RY, _I), np.kron(_I, _RY)
    gates = {'pp': ry2 @ ry1, 'p0': ry1, 'p1': np.kron(_I, _X) @ ry1,
             '0p': ry2, '1p': np.kron(_X, _I) @ ry2}
    gate = gates[state]
    return gate @ rho @ gate.conj().T


def readout_probabilities(transfer, params, state, readout):
    rho = prepared_state(params, state) * transfer
    gate = READOUT_GATES[readout]
    probabilities = np.real(params['CM'] @ np.diag(gate @ rho @ gate.conj().T))
    if (not np.isfinite(probabilities).all() or probabilities.min() < -1e-12
            or abs(probabilities.sum() - 1) > 1e-11):
        raise ValueError('Invalid reconstructed readout probabilities.')
    # Only remove roundoff at probability-simplex boundaries.
    probabilities = np.maximum(probabilities, 0)
    return probabilities / probabilities.sum()


def multinomial_covariance(probabilities, shots):
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (4,) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError('Expected four normalized outcome probabilities.')
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError('shots must be a positive integer.')
    return (np.diag(p) - np.outer(p, p)) / shots


def coefficient_maps(qubit, prep_divisor):
    """Two two-dimensional contrasts entering the two logarithms."""
    div = np.asarray(prep_divisor, dtype=float)
    if div.shape != (2,) or np.any(div == 0):
        raise ValueError('Preparation divisors must be two nonzero numbers.')
    if not qubit:
        return (np.array([[1, -1, 0, 0], [0, 0, 1, 1]]) / div.prod(),
                np.array([[1, 1, 0, 0], [0, 0, 1, -1]]) / div.prod())
    a, b = 1 / div[qubit - 1], 1 / div.prod()
    return (np.array([[a+b, 0, a-b, 0], [0, a+b, 0, a-b]]) / 2,
            np.array([[a-b, 0, a+b, 0], [0, a-b, 0, a+b]]) / 2)


def coefficient_derivatives(means, kind, qubit, prep_divisor):
    """Value, gradient and Hessian of the pinned logarithmic estimator."""
    maps = coefficient_maps(qubit, prep_divisor)
    value, gradient, hessian = 0.0, np.zeros(4), np.zeros((4, 4))
    for sign, matrix in zip((1, 1 if kind == 'sum' else -1), maps):
        z = matrix @ np.asarray(means)
        radius2 = z @ z
        if radius2 <= 0 or not np.isfinite(radius2):
            raise ValueError('The logarithmic estimator has zero contrast.')
        value -= sign * .25 * np.log(radius2)
        gradient -= sign * .5 * matrix.T @ z / radius2
        hessian -= sign * .5 * matrix.T @ (
            np.eye(2) / radius2 - 2 * np.outer(z, z) / radius2**2) @ matrix
    return value, gradient, hessian


def quadratic_estimator_moments(probabilities, weights, shots, gradient, hessian):
    """Exact moments of the second-order Taylor polynomial, independent settings.

    The third moments and fourth cumulants of each sample mean follow exactly
    from its discrete readout distribution. No Gaussian approximation is needed
    for these polynomial moments; truncating the logarithmic estimator at second
    order is still an approximation, checked separately.
    """
    p = np.asarray(probabilities)
    weights = np.broadcast_to(weights, p.shape)
    means = np.sum(p * weights, axis=1)
    centered = weights - means[:, None]
    variance = np.sum(p * centered**2, axis=1) / shots
    third = np.sum(p * centered**3, axis=1) / shots**2
    fourth_cumulant = (np.sum(p * centered**4, axis=1)
                       - 3 * (variance * shots)**2) / shots**3
    covariance = np.diag(variance)
    first = float(gradient**2 @ variance)
    correction = float(gradient @ (np.diag(hessian) * third)
                       + .5 * np.trace(hessian @ covariance @ hessian @ covariance)
                       + .25 * np.diag(hessian)**2 @ fourth_cumulant)
    bias = float(.5 * np.diag(hessian) @ variance)
    return first, max(first + correction, 0), bias


def gaussian_quadrature_check(means, variances, kind, qubit, divisor, order=5):
    """Deterministic Gaussian quadrature checks log-Taylor truncation only.

    This does not claim calibrated finite-shot coverage or replace the discrete
    readout moments. It checks higher log-curvature terms at the same covariance.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    indices = np.indices((order,)*4).reshape(4, -1).T
    points = means + np.sqrt(2*variances) * nodes[indices]
    joint_weights = np.prod(weights[indices], axis=1) / np.pi**2
    values = np.zeros(len(indices))
    for sign, matrix in zip((1, 1 if kind == 'sum' else -1), coefficient_maps(qubit, divisor)):
        contrast = points @ matrix.T
        values -= sign * .25 * np.log(np.sum(contrast**2, axis=1))
    average = joint_weights @ values
    variance = joint_weights @ (values-average)**2
    _, gradient, hessian = coefficient_derivatives(means, kind, qubit, divisor)
    covariance = np.diag(variances)
    polynomial_variance = (gradient**2 @ variances
                           + .5*np.trace(hessian@covariance@hessian@covariance))
    return float(abs(variance-polynomial_variance) / max(variance, 1e-300))


def piecewise_trapezoid_filter(toggle, times, frequencies, sign=1):
    """The discrete trapezoid Fourier sum, evaluated by constant-toggle segments.

    Uniform samples and piecewise constant toggles allow geometric sums instead
    of a frequency-by-time exponential matrix. Endpoints retain the trapezoid
    half weights. The zero-frequency limit is handled by sinc, without division
    by frequency; aliased geometric singularities use direct sums.
    """
    y, t, w = np.asarray(toggle), np.asarray(times), np.asarray(frequencies)
    if y.ndim != 1 or t.shape != y.shape or len(t) < 2:
        raise ValueError('Expected matching one-dimensional toggle and time grids.')
    dt = (t[-1]-t[0])/(len(t)-1)
    tolerance = 64*np.finfo(float).eps*max(1, np.abs(t).max())
    if np.max(np.abs(t-(t[0]+dt*np.arange(len(t))))) > tolerance:
        raise ValueError('Segmented Fourier sums require a uniform time grid.')
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1])+1, len(y)]
    start, length = edges[:-1], np.diff(edges)
    midpoint = start + (length-1)/2
    out = np.empty(len(w), complex)
    for offset in range(0, len(w), 128):
        omega = sign*w[offset:offset+128]
        theta = omega*dt
        denominator = np.sinc(theta/(2*np.pi))
        safe = np.abs(denominator) > 1e-8
        subtotal = np.empty(len(omega), complex)
        a = theta[safe, None]
        segment = (np.exp(1j*a*midpoint) * length
                   * np.sinc(a*length/(2*np.pi)) / denominator[safe, None])
        subtotal[safe] = np.exp(1j*omega[safe]*t[0]) * (segment @ y[start])
        for j in np.flatnonzero(~safe):
            subtotal[j] = np.sum(y*np.exp(1j*omega[j]*t))
        endpoints = .5*(y[0]*np.exp(1j*omega*t[0])+y[-1]*np.exp(1j*omega*t[-1]))
        out[offset:offset+len(omega)] = dt*(subtotal-endpoints)
    return out


def analyze_acquisition(acquisition, transfers, params, shots):
    """Four independent observable estimates feeding one cumulant estimate."""
    q = acquisition.qubit
    if q:
        plus, minus = ('p0', 'p1') if q == 1 else ('0p', '1p')
        settings = [(plus, f'x{q}', 0), (plus, f'y{q}', 0),
                    (minus, f'x{q}', 1), (minus, f'y{q}', 1)]
    else:
        settings = [('pp', key, k) for k, key in enumerate(('xx', 'yy', 'xy', 'yx'))]
    protocol = str(params.get('spam_protocol', 'none'))
    if protocol == 'robust':
        raise ValueError('SPAM-robust processing is intentionally excluded.')
    correction = (params['est_cm'] if protocol == 'mitigated'
                  else np.eye(4) if protocol == 'raw' else params['CM'])
    weights = OUTCOME_SIGNS[q] @ np.linalg.inv(correction)
    means, variances, probabilities = [], [], []
    for state, readout, k in settings:
        p = readout_probabilities(transfers[k], params, state, readout)
        probabilities.append(p)
        means.append(weights @ p)
        variances.append(weights @ multinomial_covariance(p, shots) @ weights)
    means, variances = np.asarray(means), np.asarray(variances)
    divisor = (params.get('a_sp_div', params['a_sp'])
               if bool(params['spMit']) else np.ones(2))
    value, gradient, hessian = coefficient_derivatives(
        means, acquisition.kind, q, divisor)
    variance = float(gradient**2 @ variances)
    # Second-order Gaussian/delta diagnostics, not an assertion of coverage.
    covariance = np.diag(variances)
    bias2 = float(.5 * np.trace(hessian @ covariance))
    variance2 = float(.5 * np.trace(hessian @ covariance @ hessian @ covariance))
    contrast_snr = min(np.linalg.norm(matrix @ means) /
                       np.sqrt(max(np.linalg.eigvalsh(matrix @ covariance @ matrix.T).max(), 1e-300))
                       for matrix in coefficient_maps(q, divisor))
    first_variance, second_variance, finite_bias = quadratic_estimator_moments(
        probabilities, weights, shots, gradient, hessian)
    stderr = np.sqrt(max(variance, 0))
    quadrature_error = gaussian_quadrature_check(
        means, variances, acquisition.kind, q, divisor)
    return dict(value=float(value), stderr=float(np.sqrt(second_variance)),
                first_order_stderr=float(stderr), second_order_bias=finite_bias,
                gaussian_quadrature_relative_variance_discrepancy=quadrature_error, means=means,
                variances=variances, probabilities=np.asarray(probabilities),
                weights=weights, gradient=gradient, hessian=hessian,
                gaussian_second_order_bias=bias2,
                gaussian_second_order_variance=variance2,
                second_order_bias_in_stderr=abs(bias2) / max(stderr, 1e-300),
                second_order_stderr_fraction=np.sqrt(max(variance2, 0) / max(variance, 1e-300)),
                minimum_contrast_snr=float(contrast_snr))
