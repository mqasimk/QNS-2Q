"""
Turns simulated quantum states into the actual measured QNS observables.

Physics role / pipeline position
---------------------------------
This is the "measurement layer" of the **characterize** arm (Stage 1, QNS
experiment simulation) of the two-arm pipeline described in ``CLAUDE.md``. It
sits between the state solver and the spectral reconstruction:

    trajectories.solver_prop() -> (batch of simulated final states)
           |
           v
    model/observables.py  <-- YOU ARE HERE
        - povms(): build the (possibly SPAM-faulty) measurement operators
        - compute_probs_jax(): rotate + measure each simulated shot -> outcome
          probabilities
        - e_x_hat/e_y_hat/e_xx_hat/... : turn probabilities into single- and
          two-qubit Pauli expectation values, with confusion-matrix (M-error)
          and state-prep (SP-error) mitigation baked in
        - a(), d(): combine those expectation values into the concurrence-like
          coefficients A^pm_a(t), D^pm(t) of the companion paper
          ("Noise-tailored two-qubit gates from spectral reconstruction",
          Khan/Norris/Viola, Sec. "Two-qubit frequency-comb quantum noise
          spectroscopy")
        - make_c_12_0_mt / make_c_12_12_mt / make_c_a_0_mt / make_c_a_b_mt:
          sweep A^pm_a(t)/D^pm(t) over a set of control times to produce the
          four measured correlation coefficients C_{12,0}, C_{12,12}, C_{a,0},
          C_{a,b} (a in {1,2}) that the paper's Eqs. for C_{a,b}(t) define.
           |
           v
    {results.npz, params.npz}  (written by characterize/experiments.py)
           |
           v
    characterize/inversion.py / characterize/reconstruct.py (Stage 2: inverts
    the comb of C_{...}(MT) values across pulse sequences into the noise power
    spectral densities S_{a,b}(omega))

In short: trajectories.py answers "what quantum state do we end up in", and
this module answers "what would a real experimentalist actually read off a
detector, given realistic state-prep (SP) and measurement (M) errors" -- i.e.
it is where the SPAM-error model (see CLAUDE.md's "SPAM pipeline" section)
gets applied to turn a noiseless density matrix into a noisy classical number.

Callers / callees
------------------
- Imports ``make_init_state``, ``make_y``, ``PhasedState`` from
  ``qns2q.model.trajectories`` (state prep + pulse-sequence control matrices).
- Called by ``characterize/experiments.py`` (Stage 1 driver, for every
  ``spam_protocol`` except ``robust``: ``none`` legacy-oracle, ``raw``, and
  ``mitigated``) and ``characterize/single_qubit.py`` (the single-qubit QNS
  variant), which pick the ``make_c_*_mt`` functions by name out of a lookup
  table.
- ``characterize/spam.py`` (the "robust" SPAM protocol: twisting + wringing)
  reuses several of this module's low-level building blocks directly --
  ``povms``, ``compute_probs_jax``, ``_get_flip_operator``,
  ``_get_hadamard_operators``, ``_get_rx_operators``, ``a``, ``d``,
  ``parametric_bootstrap_error`` -- rather than duplicating them, so those
  names are a de-facto public API even though some start with an underscore;
  do not rename them without checking that module too.
- Exercised directly by ``tests/test_observables.py``, ``tests/test_spam.py``
  and ``tests/test_tau_invariance.py``.

This module leans on QuTiP (``qt.Qobj``) for building the small fixed
operators (Hadamard/Rx/POVMs, all on the 8x8 three-qubit Hilbert space, see
CLAUDE.md's "3-Qubit Hilbert Space Convention") and on JAX (``jax.numpy``) for
the actual batched numerical work over many Monte Carlo noise realizations
("shots"), since JAX's ``jit``/vectorized array ops are much faster than
looping in Python or using QuTiP objects per-shot.
"""

from typing import List, Tuple, Callable
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from qutip_qip.operations import snot

from qns2q.model.trajectories import make_init_state, make_y, PhasedState


# --- Helper functions for creating operators ---


def _get_hadamard_operators() -> List[qt.Qobj]:
    r"""
    Creates a list of 3-qubit Hadamard gate operators for measurement basis rotation.
    The operators are [H \otimes I \otimes I, I \otimes H \otimes I, H \otimes H \otimes I].

    Physical role: real hardware measures in the computational (Z) basis, so to
    read out sigma_x on a qubit you rotate it with a Hadamard *before* measuring
    Z (H Z H = X). These pre-built gates are applied as the ``gate`` argument of
    ``_compute_expectation``/``compute_probs_jax``. All three operators act on
    the same 8x8 (2-qubit + 1 bath-qubit) Hilbert space; the identity on the
    third tensor factor is the inert bath qubit (see module docstring / CLAUDE.md
    "3-Qubit Hilbert Space Convention" -- it exists so the same propagator
    machinery in ``trajectories.py`` can carry a bath degree of freedom, but no
    physical qubit is measured on it).
    """
    h_single = snot()
    id_q = qt.identity(2)

    h0 = qt.tensor(h_single, id_q, id_q)
    h1 = qt.tensor(id_q, h_single, id_q)
    h2 = qt.tensor(h_single, h_single, id_q)

    return [h0, h1, h2]


def _get_flip_operator() -> qt.Qobj:
    r"""
    Creates the 3-qubit pre-measurement bit-flip ("twisting") operator X \otimes X \otimes I.

    Applied AFTER the measurement-basis rotation and before the POVMs, this flips both
    measured qubits in the computational basis. Combining flipped and unflipped runs
    symmetrizes the asymmetric readout error delta_l (companion paper, "twisting"):
        E_tilde^pm[O] = (E_hat[O] pm E_hat_flipped[O]) / 2.

    In plain terms: a real detector's "0" and "1" readout thresholds are rarely
    perfectly balanced (that imbalance is delta_l in ``povms``/CLAUDE.md's SPAM
    section). Measuring once normally and once after flipping both qubits with
    X-gates, then averaging (the "+" combination) or differencing (the "-"
    combination), cancels that imbalance to first order without needing to know
    its exact value -- this is what "SPAM-robust" means for the ``robust``
    protocol in ``characterize/spam.py``. Despite the leading underscore this
    function is imported directly by ``characterize/spam.py``, so it is treated
    as part of the module's public surface -- do not rename it.
    """
    return qt.tensor(qt.sigmax(), qt.sigmax(), qt.identity(2))


def _get_rx_operators() -> List[qt.Qobj]:
    r"""
    Creates a list of 3-qubit Rx gate operators for Y-basis measurement.
    The operators are [Rx \otimes I \otimes I, I \otimes Rx \otimes I, Rx \otimes Rx \otimes I].

    Same idea as ``_get_hadamard_operators`` but for sigma_y instead of
    sigma_x: this Rx(pi/2) rotation maps the Y eigenbasis onto the Z
    (computational, measured) basis before the POVMs are applied.
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
        a_m: Per-qubit measurement *visibility* alpha_M (CLAUDE.md's "SPAM
            pipeline" section): a_m[0], a_m[1] for qubit 1, 2. alpha_M=1 is a
            perfect (noiseless) readout;
            alpha_M<1 shrinks the distinguishability between the "0" and "1"
            outcomes (readout visibility/fidelity loss).
        delta: Per-qubit measurement *asymmetry* delta_l: a nonzero delta means
            the detector is biased toward reporting one outcome more often than
            the other, independent of the true state (readout offset/bias).

    Returns:
        A list of four 3-qubit POVM operators [Pi_{1,0}, Pi_{1,1}, Pi_{2,0},
        Pi_{2,1}], one pair per qubit, each already tensored with the identity
        on the other qubit and on the inert bath qubit (see module docstring).
        With a_m=[1,1] and delta=[0,0] these reduce to the ideal projectors
        onto the computational basis; with a_m<1 and/or delta!=0 they encode
        the faulty single-qubit confusion matrix A^(l) of the companion paper
        (Sec. "Modeling SPAM errors").
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


# @jax.jit traces this function once (on its first call, for each distinct
# input shape/dtype) and compiles it to fast XLA code, so repeated calls over
# many noise shots/pulse sequences avoid Python-level interpreter overhead --
# this is why the file uses jax.numpy (jnp) instead of plain numpy throughout:
# jnp operations are what jax.jit knows how to trace and compile.
@jax.jit
def _probs_from_phases(u, rho, gate, M_ops):
    """Exact fast path for PhasedState batches (diagonal propagators).

    probs[n, m] = Tr[M_m G (rho o u_n u_n^dag) G^dag]
                = sum_ab [G^dag M_m G]_ab rho_ba u_nb conj(u_na)
    -- a quadratic form in the (N, 8) phase vectors; no (N, 8, 8) stacks.

    Why this exists: for pulse-free/diagonal-propagator experiments,
    ``trajectories.py`` can represent the whole batch of N noise shots as just
    N complex phase vectors (``PhasedState``, shape (N, 8)) instead of N dense
    8x8 density matrices -- ~24x less memory (see ``PhasedState``'s
    docstring). This function evaluates the same physical probabilities
    directly from that compact representation instead of first reconstructing
    the dense (N, 8, 8) stack, which would defeat the point of the compact
    representation. ``compute_probs_jax`` dispatches here automatically
    whenever it is handed a ``PhasedState``."""
    K = jnp.einsum('ia,mij,jb->mab', jnp.conj(gate), M_ops, gate)   # G^dag M G
    C = jnp.transpose(K, (0, 2, 1)) * rho[None, :, :]               # C[m,b,a] = K[m,a,b] rho[b,a]
    return jnp.real(jnp.einsum('nb,mba,na->nm', u, C, jnp.conj(u)))


# --- Projection (readout-sampling) noise, configured per run -----------------------
# n_meas = 0 reproduces the historical idealized behavior (exact expectation per
# noise realization). Finite n_meas draws multinomial outcome counts per shot, so
# the quoted bars include finite-measurement statistics. The dedicated Generator
# keeps the legacy np.random stream (noise keys, bootstrap) untouched.
_PROJECTION = {'n_meas': 0, 'rng': None}


def set_projection_sampling(n_meas: int, seed: int = None):
    """Turn on/off finite-shot ("projection") measurement noise for the rest of
    the process.

    n_meas=0 (the default, via the ``_PROJECTION`` dict above) means every
    outcome probability computed by ``compute_probs_jax`` is used exactly, as
    if you had infinitely many physical measurement repetitions per Monte
    Carlo noise shot -- the only randomness left is the noise trajectory
    itself. Passing n_meas>0 additionally simulates drawing n_meas real
    projective measurements from that probability distribution (see
    ``_sample_projection``), so the reported statistical error bars include
    genuine shot noise on top of the Monte Carlo noise-sampling error.

    This mutates the single module-level ``_PROJECTION`` dict rather than
    taking a parameter everywhere it's needed -- a deliberate "set it once per
    pipeline run" global switch (called once in ``characterize/experiments.py``
    from ``QNSExperimentConfig``) rather than threading an extra argument
    through every expectation-value function in this file. Because it is
    global mutable state, do not call it mid-run from multiple configs/threads.
    """
    _PROJECTION['n_meas'] = int(n_meas) if n_meas else 0
    _PROJECTION['rng'] = np.random.default_rng(seed) if n_meas else None


def _sample_projection(probs: np.ndarray) -> np.ndarray:
    """Replace exact outcome probabilities by multinomial frequencies.

    Draws ``n_meas`` samples from the categorical distribution ``probs`` (per
    row/shot) and returns the empirical frequencies -- i.e. simulates a finite
    number of real projective measurements on each Monte Carlo noise shot,
    instead of using the exact Born-rule probability. A no-op (returns
    ``probs`` unchanged) unless ``set_projection_sampling`` was called with a
    positive ``n_meas``.
    """
    n = _PROJECTION['n_meas']
    if not n:
        return probs
    p = np.clip(probs, 0.0, None)
    p = p / np.sum(p, axis=1, keepdims=True)
    counts = _PROJECTION['rng'].multinomial(n, p)
    return counts / float(n)


def compute_probs_jax(rho_batch, gate, M_ops):
    """Batched Born-rule readout: rotate each simulated shot into the
    measurement basis with ``gate`` and return its outcome probabilities under
    the composite POVMs ``M_ops``.

    Args:
        rho_batch: Either a ``PhasedState`` (compact per-shot phase
            representation, dispatched to the ``_probs_from_phases`` fast
            path) or a dense (N, 8, 8) array of N simulated post-evolution
            density matrices, one per Monte Carlo noise shot.
        gate: (8, 8) unitary that rotates the measurement basis onto the
            computational (Z) basis -- e.g. a Hadamard for an X measurement
            (see ``_get_hadamard_operators``), optionally pre-multiplied by
            the twisting bit-flip (``_get_flip_operator``).
        M_ops: (4, 8, 8) stack of the four composite two-qubit POVM elements
            [M00, M01, M10, M11] built in ``_compute_expectation`` from
            ``povms()``.

    Returns:
        (N, 4) array: ``probs[n, m]`` is the Born-rule probability of outcome
        m for shot n, i.e. Tr[M_ops[m] @ (gate @ rho_batch[n] @ gate^dagger)].
        This is the raw material ``get_expect_val_from_probs``/
        ``get_expect_val_per_shot`` turn into SPAM-corrected expectation
        values. Imported directly by ``characterize/spam.py`` for the
        SPAM-robust protocol -- do not change this signature.
    """
    if isinstance(rho_batch, PhasedState):
        return _probs_from_phases(rho_batch.u, rho_batch.rho, gate, M_ops)
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

    ``probs`` columns are always ordered [00, 01, 10, 11] over the two
    measured qubits (matching the ``M_ops`` order [M00, M01, M10, M11] built in
    ``_compute_expectation``): column 0+1 is "qubit 1 read 0" regardless of
    qubit 2's outcome, column 0+2 is "qubit 2 read 0" regardless of qubit 1's
    outcome, and column 0+3 is "both qubits agreed" (the ZZ-type two-qubit
    correlation). ``2*p - 1`` converts that computational-basis population
    (in [0, 1]) into a Pauli expectation value (in [-1, 1]).

    ``cm`` is the (4, 4) confusion matrix used to *mitigate* readout (M)
    errors: multiplying by its inverse undoes the mixing that faulty POVMs
    (see ``povms``) introduce between the four outcome probabilities, so the
    corrected populations ``p_corr`` approximate what an ideal detector would
    have reported. Passing ``cm=None`` (equivalent to the identity) means "do
    not mitigate M errors here" -- used by the ``raw``/``robust`` SPAM
    protocols (CLAUDE.md's "SPAM pipeline"), which handle M errors differently
    (or not at all) rather than through this confusion-matrix inversion.

    Args:
        probs: An array of outcome probabilities.
        cm: The confusion matrix for error mitigation.
        qubit_idx: The qubit index (1 or 2) for single-qubit expectation values,
                   or -1 for two-qubit correlations.

    Returns:
        The calculated expectation value.
    """
    pi = jnp.mean(probs, axis=0)
    cm = np.eye(4) if cm is None else cm
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

    Same confusion-matrix mitigation and outcome-population bookkeeping as
    ``get_expect_val_from_probs`` (see its docstring for the physical meaning
    of ``cm``/``qubit_idx``), but returns one value per Monte Carlo noise shot
    instead of averaging over shots first -- this is what lets
    ``_compute_expectation_with_stderr`` report a standard error across shots
    (``get_expect_val_from_probs`` only ever returns the shot-averaged mean).

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
    cm = np.eye(4) if cm is None else cm
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
                         cm: np.ndarray, qubit_idx: int = -1, twist: bool = False) -> float:
    """Helper to compute expectation values using vectorized JAX operations.

    With ``twist=True`` the pre-measurement bit-flip X1X2 is applied after the
    basis-rotation gate (the "twisting" run of the SPAM-robust protocol).

    Ties together every step of turning a batch of simulated states into one
    reported number: build the faulty POVMs from ``a_m``/``delta``
    (``povms``), compose them into the 4 two-qubit outcome operators
    ``M_ops``, get outcome probabilities per shot (``compute_probs_jax``),
    optionally add finite-shot noise (``_sample_projection``), average over
    shots and mitigate M errors via ``cm`` (``get_expect_val_from_probs``).
    This is the "M-mitigated" (confusion-matrix-inversion) path used by the
    ``none``/``mitigated`` SPAM protocols; the ``robust`` protocol instead
    calls ``compute_probs_jax``/``povms`` directly from ``characterize/spam.py``
    to build its own twisted+wrung estimators.
    """
    povms_list = povms(a_m, delta)

    if twist:
        gate = _get_flip_operator() * gate

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
    probs = _sample_projection(np.array(probs))

    return get_expect_val_from_probs(probs, cm, qubit_idx)


def e_x_hat(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_x for a given qubit with measurement error mitigation.

    ``qubit`` in {1, 2} selects which physical qubit's X_l is measured (companion
    paper Table "State preparations and ... observables measured for two-qubit
    QNS"). The single-qubit ``e_*_hat`` functions below all follow this same
    "pick the right basis-rotation gate(s), then delegate to
    ``_compute_expectation``" pattern.
    """
    h = _get_hadamard_operators()
    gate = h[0] if qubit == 1 else h[1]
    return _compute_expectation(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_y_hat(qubit: int, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    """Calculates the expectation value of sigma_y for a given qubit with measurement error mitigation."""
    rx = _get_rx_operators()
    gate = rx[0] if qubit == 1 else rx[1]
    return _compute_expectation(gate, state, a_m, delta, cm, qubit_idx=qubit)


def e_xx_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    r"""Calculates the expectation value of sigma_x \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    return _compute_expectation(h[2], state, a_m, delta, cm)


def e_xy_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    r"""Calculates the expectation value of sigma_x \otimes sigma_y with measurement error mitigation.

    Composing gates with QuTiP's ``*`` (``h[0] * rx[1]``) applies ``rx[1]``
    first and ``h[0]`` second (right-to-left, standard operator-product
    order): qubit 2 is rotated for a Y measurement, qubit 1 for an X
    measurement, before both are read out in the computational basis.
    """
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation(h[0] * rx[1], state, a_m, delta, cm)


def e_yx_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    r"""Calculates the expectation value of sigma_y \otimes sigma_x with measurement error mitigation."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation(rx[0] * h[1], state, a_m, delta, cm)


def e_yy_hat(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> float:
    r"""Calculates the expectation value of sigma_y \otimes sigma_y with measurement error mitigation."""
    rx = _get_rx_operators()
    return _compute_expectation(rx[2], state, a_m, delta, cm)


def _compute_expectation_with_stderr(gate: qt.Qobj, state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray,
                                     cm: np.ndarray, qubit_idx: int = -1, twist: bool = False) -> Tuple[float, float]:
    """Helper to compute expectation values and standard error using vectorized JAX operations.

    With ``twist=True`` the pre-measurement bit-flip X1X2 is applied after the
    basis-rotation gate (the "twisting" run of the SPAM-robust protocol).

    Identical construction to ``_compute_expectation``, except it keeps a
    per-shot expectation value (``get_expect_val_per_shot``, one number per
    Monte Carlo noise trajectory) instead of only the shot-averaged mean, so
    it can also report the standard error of the mean across shots
    (``ddof=1`` sample standard deviation over sqrt(n_shots)). This per-shot
    error is what ``parametric_bootstrap_error`` later propagates through the
    nonlinear ``a()``/``d()`` combiners.
    """
    povms_list = povms(a_m, delta)

    if twist:
        gate = _get_flip_operator() * gate

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
    probs_np = _sample_projection(np.array(probs))

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
    r"""Calculates the expectation value and stderr of sigma_x \otimes sigma_x."""
    h = _get_hadamard_operators()
    return _compute_expectation_with_stderr(h[2], state, a_m, delta, cm)


def e_xy_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    r"""Calculates the expectation value and stderr of sigma_x \otimes sigma_y."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(h[0] * rx[1], state, a_m, delta, cm)


def e_yx_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    r"""Calculates the expectation value and stderr of sigma_y \otimes sigma_x."""
    h = _get_hadamard_operators()
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(rx[0] * h[1], state, a_m, delta, cm)


def e_yy_hat_with_stderr(state: jnp.ndarray, a_m: np.ndarray, delta: np.ndarray, cm: np.ndarray) -> Tuple[float, float]:
    r"""Calculates the expectation value and stderr of sigma_y \otimes sigma_y."""
    rx = _get_rx_operators()
    return _compute_expectation_with_stderr(rx[2], state, a_m, delta, cm)


# --- Concurrence and Entanglement Metrics ---
#
# a() and d() implement the "A^pm_a(t)" and "D^pm(t)" combiners of the
# companion paper (Sec. "Two-qubit frequency-comb quantum noise spectroscopy",
# frequency-comb reconstruction subsection). Physically, the joint decay of a
# pair of transverse (X/Y) Pauli expectation values under dephasing is
# exp(-2*cumulant), so "-1/4 * log(sum of squares)" recovers the underlying
# cumulant (an accumulated second-order QNS coefficient, e.g. C_{1,0}(t) or
# C_{12,0}(t)) from the measured decay of coherence -- the same log-of-decay
# idea used to define concurrence-like entanglement measures for a Bell pair
# under dephasing, hence the "concurrence" name inherited from earlier code.
# make_c_12_0_mt/make_c_12_12_mt/make_c_a_0_mt/make_c_a_b_mt below call these
# to assemble the four measured correlation coefficients C_{12,0}, C_{12,12},
# C_{a,0}, C_{a,b} that Stage 2 (characterize/inversion.py) inverts into spectra.


def a(expec: Tuple[float, float]) -> float:
    """Calculates the quantity A related to concurrence.

    Implements A^pm_a(t) = -1/4 * ln{ E[X_a(t)]^2 + E[Y_a(t)]^2 } for the
    single-/two-body self-spectrum reconstruction, with ``expec = (E[X], E[Y])``
    the pair of transverse expectation values measured on the relevant fiducial
    state. Used directly by ``c_a_0_mt_i``/``c_a_b_mt_i`` below and also
    imported by ``characterize/spam.py`` for the SPAM-robust protocol's own
    (twisted/wrung) version of the same combination.
    """
    return -0.25 * np.log(expec[0] ** 2 + expec[1] ** 2)


def d(pm: str, expec: Tuple[float, float, float, float]) -> float:
    """Calculates the quantity D related to concurrence.

    Implements D^pm(t) = -1/4 * ln{ (E[X1Y2] pm E[Y1X2])^2 + (E[X1X2] mp E[Y1Y2])^2 }
    for the two-body (Ising) channel reconstruction, with ``expec =
    (E[X1X2], E[Y1Y2], E[X1Y2], E[Y1X2])`` all measured on the same
    |x+>x|x+> fiducial state. ``pm='+'`` uses "+" in the first term and "-" in
    the second (and vice versa for ``pm='-'``), matching the mp/pm sign
    convention in the paper. ``make_c_12_0_mt``/``make_c_12_12_mt`` combine
    ``d('+', ...)`` and ``d('-', ...)`` (sum -> C_{12,0}, difference ->
    C_{12,12}) to separate the two-body self-spectrum from the Q1-Q2 cross-
    spectrum.
    """
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

    Currently a no-op passthrough (returns ``sol`` unchanged); a repo-wide
    search found no remaining callers of this function, so it is likely a
    historical leftover kept in case the frame-correction logic it used to
    contain is needed again. Left in place rather than deleted since the
    grad-student translation pass is not the place to make that call -- flag
    it to a maintainer if you are cleaning up dead code.
    """
    return sol


# --- Main Calculation Functions for Concurrence ---


def _resolve_a_sp_div(sp_mit: bool, kwargs: dict) -> np.ndarray:
    """Resolve the SP-mitigation divisor for the coefficient estimators.

    The state PREP always uses the true kwargs['a_sp']; the estimators divide by
    this returned vector (the experimenter's estimate alpha_hat_SP^z). Legacy
    behavior is preserved: sp_mit=False -> no division; sp_mit=True without an
    explicit 'a_sp_div' -> oracle division by the true a_sp. The SPAM-mitigated
    protocol passes a_sp_div = the CALIBRATION-ESTIMATED alpha_SP^z instead.

    In plain terms, this picks which of the ``spam_protocol`` values from
    CLAUDE.md's "SPAM pipeline" section a caller is running:
    ``sp_mit=False`` is the ``raw`` protocol (no SP-error correction at all --
    the returned divisor is just 1, so it's a no-op below); ``sp_mit=True``
    with no ``a_sp_div`` override is the legacy ``none`` oracle protocol,
    which "cheats" by dividing by the *true, known* SP-visibility (only valid
    in simulation, where the ground truth is available); and ``sp_mit=True``
    with an explicit ``a_sp_div`` is the ``mitigated`` protocol, which divides
    by a *calibration-estimated* SP-visibility instead -- the realistic,
    experimentally-achievable case. ``kwargs`` here is the same
    "config-as-keyword-arguments" pattern used throughout ``make_c_*_mt``
    below: a dict of named experiment parameters passed straight through
    rather than listed as individual positional arguments, so those callers
    don't have to repeat every parameter name at each call site.
    """
    if not sp_mit:
        return np.array([1., 1.])
    div = kwargs.get('a_sp_div')
    return np.asarray(kwargs['a_sp'] if div is None else div, dtype=float)


def parametric_bootstrap_error(means: List[float], stderrs: List[float], func: Callable, n_boot: int = 1000) -> float:
    """
    Estimates the standard error of a function of multiple variables using parametric bootstrapping.

    Why this is needed: ``a()``/``d()`` are nonlinear (they involve squares and
    a logarithm) functions of the measured Pauli expectation values, so their
    error bars are not simply a combination of the input standard errors --
    ordinary linear error propagation would be wrong. Instead, this draws
    ``n_boot`` synthetic samples of each input variable from a Normal
    distribution centered on its measured mean with its measured standard
    error (i.e. assumes each input's own sampling distribution is
    approximately Gaussian, which is reasonable for a shot-averaged mean by
    the central limit theorem), evaluates ``func`` (e.g. ``calc_d_sum``/
    ``calc_ab_diff`` below) on each synthetic sample, and reports the spread
    (sample standard deviation) of the resulting outputs as the propagated
    standard error.

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
    """Calculates one point of the C_12_0_MT correlation with standard error.

    C_{12,0}(MT) is the companion paper's two-qubit coefficient that combines
    the qubit-1 and qubit-2 self-spectra (Eq. for ``C_{a,0}``/``C_{12,0}`` in
    "Two-qubit frequency-comb quantum noise spectroscopy"): it is measured by
    preparing both qubits in |x+> (the ``psi_12`` fiducial state of the
    paper's Table of state preparations), applying the given ``pulse``
    sequence for one block of ``m`` repetitions at control time ``ct``, and
    reading out all four two-qubit correlators X1X2, Y1Y2, X1Y2, Y1X2. Those
    four expectation values feed ``d('+', ...)`` and ``d('-', ...)`` (the
    paper's D^+(t)/D^-(t)); their *sum* is C_{12,0}(t) (this function), while
    their *difference* is C_{12,12}(t) (``c_12_12_mt_i`` below).

    Dividing each raw expectation value by ``a_sp[0] * a_sp[1]`` undoes the
    two-qubit state-prep bias exactly as derived in the paper's SPAM-mitigated
    two-qubit estimator (the product of both qubits' alpha_SP^z enters
    because both qubits were faultily prepared); which value ``a_sp`` actually
    holds (1, the true SP visibility, or a calibration estimate) is decided
    once by ``_resolve_a_sp_div`` in the caller (``make_c_12_0_mt``).
    """
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
    """Generates the C_12_0_MT correlation over a range of control times.

    Top-level entry point Stage 1 (``characterize/experiments.py``,
    ``characterize/single_qubit.py``) calls to sweep ``c_12_0_mt_i`` over every
    requested control time ``c_times`` (the "MT" of the name: M repetitions at
    total block time T, i.e. total evolution time ct), preparing the |x+>x|x+>
    state once and reusing it for every point in the sweep. Returns the
    parallel lists of (mean, stderr) pairs that Stage 1 writes into
    ``results.npz`` for Stage 2 (spectral reconstruction) to invert.

    ``**kwargs`` bundles the many experiment-config values (``a_sp``, ``c``,
    ``state``, ``t_b``, ``n_shots``, ``m``, ``a_m``, ``delta``,
    ``noise_mats``) as named keys instead of positional parameters -- this is
    a common Python idiom for functions with many configuration inputs that
    differ in number/name across call sites; the required keys are simply
    looked up by name (``kwargs['a_sp']`` etc.) rather than declared in the
    signature. ``sp_mit`` selects which SP-error divisor ``_resolve_a_sp_div``
    hands to every sweep point (see that function's docstring for the
    ``none``/``mitigated``/``raw`` protocol distinction).

    ``Parallel(n_jobs=1)``/``delayed(...)`` (joblib) is a job-scheduling
    wrapper: with ``n_jobs=1`` it simply runs ``c_12_0_mt_i`` once per control
    time in a plain Python loop (no actual parallelism here -- the batching
    over noise shots inside ``solver_ftn``/``compute_probs_jax`` is where the
    real speedup comes from), but written this way so bumping ``n_jobs`` would
    parallelize the sweep across processes without changing any call site.
    """
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = _resolve_a_sp_div(sp_mit, kwargs)

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
    """Calculates one point of the C_12_12_MT correlation with standard error.

    Structurally identical to ``c_12_0_mt_i`` (same fiducial state, same four
    measured correlators X1X2/Y1Y2/X1Y2/Y1X2), but combines D^+(t) and D^-(t)
    with a *difference* instead of a sum: C_{12,12}(t) is the coefficient that
    isolates the cross-spectrum between qubit 1 and qubit 2 (S_{1,2}), rather
    than the two self-spectra that C_{12,0} carries.
    """
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
    """Generates the C_12_12_MT correlation over a range of control times.

    Same sweep-over-``c_times`` structure as ``make_c_12_0_mt`` (see its
    docstring for the ``**kwargs``/joblib-``Parallel`` conventions shared by
    all four ``make_c_*_mt`` functions), but drives ``c_12_12_mt_i`` instead,
    so the returned coefficients feed the S_{1,2} cross-spectrum
    reconstruction rather than the self-spectra.
    """
    rho0 = make_init_state(kwargs['a_sp'], kwargs['c'], state=kwargs['state'])
    rho_b = 0.5 * qt.identity(2)
    rho = jnp.array((qt.tensor(rho0, rho_b)).full())
    a_sp = _resolve_a_sp_div(sp_mit, kwargs)

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
    """Calculates one point of the C_a_b_MT correlation with standard error.

    Unlike ``c_12_0_mt_i``/``c_12_12_mt_i`` (which measure two-qubit
    correlators on the |x+>x|x+> state), this measures single-qubit X_l, Y_l
    on qubit ``l`` (the physical qubit index, 1 or 2, mapping to the paper's
    "a" self-spectrum label) while the *other* ("partner") qubit is held in
    one of its two Z-basis branches: ``rho_pair = (rhop, rhom)`` are qubit
    ``l`` prepared in |x+> with the partner in |z+> and |z-> respectively (the
    fiducial states psi_a^+/psi_a^- of the companion paper's reconstruction
    Table). Summing/differencing over that partner-branch sign is what lets a
    *single*-qubit readout separate the self-spectrum S_{l,l} from the
    two-body Ising spectrum S_{12,12} that also couples into qubit l's
    coherence decay (Eq. for C_{a,0}(t) in "Two-qubit frequency-comb quantum
    noise spectroscopy": it sums S_{l,l} and S_{12,12} contributions).

    The ``ax``/``bx``/``ay``/``by`` combinations divide the summed
    (+/- branch average, ``ax``/``ay``) and differenced (+/- branch
    difference, ``bx``/``by``) raw expectation values by different SP-bias
    factors before recombining them into the eigen-branches fed to ``a()``:
    the average over the partner's sign scales with this qubit's own
    prep-bias alone (``a_sp[l-1]``), while the sign-difference is a residual
    that scales with the *product* of both qubits' prep-bias
    (``a_sp[0] * a_sp[1]``) because it is carried through the two-body Z1Z2
    channel that couples the two qubits' state preparations -- the same
    "single-qubit-bias vs. product-of-both-biases" split that appears
    elsewhere in the paper's SPAM-error analysis (Sec. "Modeling SPAM
    errors"), here applied to state-prep rather than measurement error. The
    resulting ``ap``/``am`` are then the paper's A^+_a(t)/A^-_a(t); their
    *difference* is C_{a,b}(t) (this function), their *sum* is C_{a,0}(t)
    (``c_a_0_mt_i`` below).
    """
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
    """Generates the C_a_b_MT correlation over a range of control times.

    Same sweep-over-``c_times`` structure as ``make_c_12_0_mt`` (see its
    docstring for the shared ``**kwargs``/joblib-``Parallel`` conventions),
    but for a chosen single qubit ``kwargs['l']`` in {1, 2} and driving
    ``c_a_b_mt_i``: builds the two partner-branch fiducial states ('p0'/'p1'
    for qubit 1, '0p'/'1p' for qubit 2 -- see ``trajectories.make_init_state``)
    once and reuses them for every control time in the sweep.
    """
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_b = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_b)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_b)).full())
    a_sp = _resolve_a_sp_div(sp_mit, kwargs)

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
    """Calculates one point of the C_a_0_MT correlation with standard error.

    Same measurement/state setup as ``c_a_b_mt_i`` (see its docstring for the
    partner-branch fiducial states and the ``ax``/``bx``/``ay``/``by``
    SP-bias split); the only difference is that ``calc_ab_sum`` below *sums*
    A^+_a(t) and A^-_a(t) rather than differencing them, giving C_{a,0}(t)
    (the self-spectrum S_{a,a} + two-body S_{12,12} combination) instead of
    C_{a,b}(t).
    """
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
    """Generates the C_a_0_MT correlation over a range of control times.

    Same sweep-over-``c_times`` structure as ``make_c_a_b_mt`` (see its and
    ``make_c_12_0_mt``'s docstrings for the shared conventions), driving
    ``c_a_0_mt_i`` instead so the returned coefficients carry the self-
    spectrum (S_{a,a} + S_{12,12}) combination rather than C_{a,b}.
    """
    l = kwargs['l']
    state_p = 'p0' if l == 1 else '0p'
    state_m = 'p1' if l == 1 else '1p'

    rho0p = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_p)
    rho0m = make_init_state(kwargs['a_sp'], kwargs['c'], state=state_m)
    rho_b = 0.5 * qt.identity(2)
    rhop = jnp.array((qt.tensor(rho0p, rho_b)).full())
    rhom = jnp.array((qt.tensor(rho0m, rho_b)).full())
    a_sp = _resolve_a_sp_div(sp_mit, kwargs)

    results = Parallel(n_jobs=1)(
        delayed(c_a_0_mt_i)(
            solver_ftn, kwargs['t_b'], pulse, t_vec, (rhop, rhom), ct, cm,
            n_shots=kwargs['n_shots'], m=kwargs['m'], a_m=kwargs['a_m'], l=l,
            delta=kwargs['delta'], noise_mats=kwargs['noise_mats'], a_sp=a_sp
        ) for ct in c_times
    )
    means, stderrs = zip(*results)
    return list(means), list(stderrs)
