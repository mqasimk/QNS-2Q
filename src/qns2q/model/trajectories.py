"""
Noise trajectory and quantum evolution simulation.

This module provides tools for generating temporally-correlated noise trajectories
and simulating the evolution of a two-qubit system under such noise. It uses
JAX for efficient vectorization and JIT compilation of the simulation pipeline.
"""

import numpy as np
import qutip as qt
import jax
import jax.numpy as jnp

from qns2q.noise import spectra as _model_spectra


def make_noise_mat_arr(act, **kwargs):
    """
    Generate or load the component noise-synthesis matrices.

    The noise model is the two-correlated-local-fields construction of
    ``qns2q.noise.spectra`` (see NOISE_MODEL_SPEC.md): channels are assembled
    per shot by ``make_channel_trajs`` from five independent component
    trajectories. This function precomputes the (sine, cosine) synthesis
    matrices for the five component streams:

        index 0: S_el_A   (electrical field at qubit 1, unshifted)
        index 1: S_el_B   shifted by dt_shift (the shared part of e_B)
        index 2: S_el_B   unshifted (the local part of e_B)
        index 3: S_nuc_1  (qubit-1 local nuclear)
        index 4: S_nuc_2  (qubit-2 local nuclear)

    Parameters
    ----------
    act : str
        Action to perform: 'load', 'make', or 'save'.
    **kwargs
        t_vec : jax.Array
            Time vector for evolution.
        w_grain : int
            Frequency discretization grain.
        wmax : float
            Maximum frequency cutoff.
        truncate : int
            Number of harmonics to include.
        components : tuple of callable, optional
            (S_el_A, S_el_B, S_nuc_1, S_nuc_2); defaults to the model's.
        dt_shift : float, optional
            Lag of e_B's shared part; defaults to spectra.DT_SHIFT.

    Returns
    -------
    jax.Array
        Array of shape [5][2][n_t][n_w] of sine/cosine synthesis matrices.
    """
    if kwargs.get('spec_vec') is not None or kwargs.get('gamma') is not None \
            or kwargs.get('gamma_12') is not None:
        raise TypeError(
            "spec_vec/gamma/gamma_12 were removed: the noise model components "
            "and cross-spectrum lag now live in qns2q.noise.spectra (see "
            "NOISE_MODEL_SPEC.md). Pass components=/dt_shift= to override.")
    t_vec = kwargs.get('t_vec')
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    truncate = kwargs.get('truncate')
    components = kwargs.get('components')
    if components is None:
        components = (_model_spectra.S_el_A, _model_spectra.S_el_B,
                      _model_spectra.S_nuc_1, _model_spectra.S_nuc_2)
    dt_shift = kwargs.get('dt_shift')
    if dt_shift is None:
        dt_shift = _model_spectra.DT_SHIFT
    # `midpoint=True` samples the noise-synthesis frequency grid at bin midpoints
    # (k+1/2)dw instead of the endpoints k*dw, which excludes the exact w=0 tone.
    # The w=0 tone otherwise injects a spurious *static* offset of variance
    # dw*S(0)/pi into every trajectory, biasing DC-sensitive observables (e.g. the
    # T2*/Ramsey decay) by O(dw). Default False preserves legacy seeded runs.
    midpoint = kwargs.get('midpoint', False)
    if act == 'load':
        return np.load('noise_mats.npy', allow_pickle=True)
    elif act == 'make':
        s_el_a, s_el_b, s_nuc_1, s_nuc_2 = components
        mk = lambda spec, shift: make_noise_mat(
            spec, t_vec, w_grain=w_grain, wmax=wmax, trunc_n=truncate,
            gamma=shift, midpoint=midpoint)
        return jnp.array([mk(s_el_a, 0.), mk(s_el_b, dt_shift), mk(s_el_b, 0.),
                          mk(s_nuc_1, 0.), mk(s_nuc_2, 0.)])
    elif act == 'save':
        mats = make_noise_mat_arr('make', **kwargs)
        np.save('noise_mats.npy', mats)
        return mats
    else:
        raise Exception("Invalid action input")


# @jax.jit
def sinM(spec, w, t, dw, gamma):
    """
    Utility function for noise matrix generation (sine component).

    Parameters
    ----------
    spec : callable
        Noise spectrum function.
    w : float
        Frequency.
    t : float
        Time.
    dw : float
        Frequency increment.
    gamma : float
        Time translation.

    Returns
    -------
    float
        Spectral amplitude component.
    """
    return jnp.sqrt(dw * spec(w) / jnp.pi) * jnp.sin(w * (t + gamma))


# @jax.jit
def cosM(spec, w, t, dw, gamma):
    """
    Utility function for noise matrix generation (cosine component).

    Parameters
    ----------
    spec : callable
        Noise spectrum function.
    w : float
        Frequency.
    t : float
        Time.
    dw : float
        Frequency increment.
    gamma : float
        Time translation.

    Returns
    -------
    float
        Spectral amplitude component.
    """
    return jnp.sqrt(dw * spec(w) / jnp.pi) * jnp.cos(w * (t + gamma))


def make_noise_mat(spec, t_vec, **kwargs):
    """
    Generate noise matrices for a given spectrum and time vector.

    Parameters
    ----------
    spec : callable
        Spectral density function.
    t_vec : jax.Array
        Time vector.
    **kwargs
        w_grain : int
            Frequency grain size.
        wmax : float
            Cutoff frequency.
        gamma : float
            Time shift.

    Returns
    -------
    tuple of jax.Array
        (sine_matrix, cosine_matrix)
    """
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    gamma = kwargs.get('gamma')
    midpoint = kwargs.get('midpoint', False)
    size_w = int(2 * w_grain)
    dw = wmax / w_grain
    if midpoint:
        # Bin-midpoint grid: excludes the exact w=0 tone (no spurious static term).
        w = (jnp.arange(size_w) + 0.5) * (2 * wmax) / size_w
    else:
        w = jnp.linspace(0, 2 * wmax, size_w)
    Sf = jax.vmap(jax.vmap(sinM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    Cf = jax.vmap(jax.vmap(cosM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    return Sf(spec, w, t_vec, dw, gamma), Cf(spec, w, t_vec, dw, gamma)


@jax.jit
def make_noise_traj(S, C, key):
    """
    Generate a noise trajectory using precomputed matrices and a random key.

    Parameters
    ----------
    S : jax.Array
        Sine matrix from `make_noise_mat`.
    C : jax.Array
        Cosine matrix from `make_noise_mat`.
    key : jax.Array
        JAX PRNG key (or pair of keys).

    Returns
    -------
    jax.Array
        Vector representing the noise trajectory over time.
    """
    key1 = jax.random.PRNGKey(key[0])
    A = jax.random.normal(key1, (jnp.size(S, 1), 1))
    key2 = jax.random.PRNGKey(key[1])
    B = jax.random.normal(key2, (jnp.size(S, 1), 1))
    traj = jnp.ravel(jnp.matmul(S, A) + jnp.matmul(C, B))
    return traj


# Mixing constants of the noise model (single source of truth: noise/spectra.py).
_C_SH = jnp.sqrt(_model_spectra.C2_SHARE)
_C_LOC = jnp.sqrt(1. - _model_spectra.C2_SHARE)
_A_J = _model_spectra.A_J
_B_J = _model_spectra.B_J


@jax.jit
def make_channel_trajs(noise_mats, key):
    """
    Assemble the three channel trajectories (zeta_1, zeta_2, zeta_12) for one shot.

    Five independent Gaussian streams drive the component matrices produced by
    ``make_noise_mat_arr`` ('shared' electrical, local-A, local-B, nuclear-1,
    nuclear-2); the channels are the linear mixture of NOISE_MODEL_SPEC.md
    section 5:

        e_A     = sqrt(C2)*g0 + sqrt(1-C2)*g_A
        e_B     = sqrt(C2)*g0(t + DT_SHIFT) + sqrt(1-C2)*g_B
        zeta_1  = e_A + n_1
        zeta_2  = e_B + n_2
        zeta_12 = A_J*e_A - B_J*e_B

    The same shared draws (stream 0) enter both e_A and e_B, which is what
    produces the partial inter-channel coherence with the measured (+, +, -)
    sign pattern.

    Parameters
    ----------
    noise_mats : jax.Array
        [5][2][n_t][n_w] synthesis matrices from `make_noise_mat_arr`.
    key : jax.Array
        Pair of integers seeding this shot's ten Gaussian draws.

    Returns
    -------
    jax.Array
        Array [3][n_t]: trajectories for [qubit1, qubit2, Ising].
    """
    base = jax.random.fold_in(jax.random.PRNGKey(key[0]), key[1])
    ks = jax.random.split(base, 10)
    n_w = jnp.size(noise_mats, 3)
    draw = lambda k: jax.random.normal(k, (n_w, 1))
    comp = lambda i, ka, kb: jnp.matmul(noise_mats[i, 0], draw(ka)) \
        + jnp.matmul(noise_mats[i, 1], draw(kb))
    g0_a = comp(0, ks[0], ks[1])      # shared stream through the e_A filter
    h_a = comp(0, ks[2], ks[3])       # local-A stream, same filter
    g0_b = comp(1, ks[0], ks[1])      # SAME shared stream, shifted e_B filter
    h_b = comp(2, ks[4], ks[5])       # local-B stream, unshifted e_B filter
    n_1 = comp(3, ks[6], ks[7])
    n_2 = comp(4, ks[8], ks[9])
    e_a = _C_SH*g0_a + _C_LOC*h_a
    e_b = _C_SH*g0_b + _C_LOC*h_b
    return jnp.array([jnp.ravel(e_a + n_1), jnp.ravel(e_b + n_2),
                      jnp.ravel(_A_J*e_a - _B_J*e_b)])



def make_init_state(a_sp, c, **kwargs):
    """
    Generate the initial two-qubit state with SPAM errors.

    Parameters
    ----------
    a_sp : array_like
        State preparation errors along the Z axis for [qubit1, qubit2].
    c : array_like
        State preparation errors (coherence) along X/Y axes for [qubit1, qubit2].
    **kwargs
        state : str
            Target state to generate: 'p0', 'p1', '0p', '1p', 'pp', or 'pp_wrung'.
            'pp_wrung' is the wringing partner of 'pp': a high-fidelity Z1Z2
            conjugation applied to the (faulty) 'pp' preparation, used by the
            SPAM-robust protocol to symmetrize transverse SP errors
            (W_pm{E_rho0[O]} = (E_rho0[O] pm E_{Z1Z2 rho0 Z1Z2}[O])/2).

    Returns
    -------
    qutip.Qobj
        Initial 3-qubit density matrix (third qubit is auxiliary/bath).
    """
    zp = qt.basis(2, 0)
    zm = qt.basis(2, 1)
    x_gates = [qt.tensor(qt.sigmax(), qt.identity(2)), qt.tensor(qt.identity(2), qt.sigmax())]
    asp_0 = a_sp[0]
    asp_1 = a_sp[1]
    c_0 = c[0]
    c_1 = c[1]
    rho0_0 = 0.5 * (1. + asp_0) * zp * zp.dag() + 0.5 * (1. - asp_0) * zm * zm.dag() + 0.5 * c_0 * zp * zm.dag() + 0.5 * np.conj(
        c_0) * zm * zp.dag()
    rho0_1 = 0.5 * (1. + asp_1) * zp * zp.dag() + 0.5 * (1. - asp_1) * zm * zm.dag() + 0.5 * c_1 * zp * zm.dag() + 0.5 * np.conj(
        c_1) * zm * zp.dag()
    rho0 = qt.tensor(rho0_0, rho0_1)
    ry = [qt.tensor(np.cos(np.pi/4)*qt.identity(2) - 1j*np.sin(np.pi/4)*qt.sigmay(), qt.identity(2)),
          qt.tensor(qt.identity(2), np.cos(np.pi/4)*qt.identity(2) - 1j*np.sin(np.pi/4)*qt.sigmay())]
    if kwargs.get('state') == 'p0':
        return ry[0] * rho0 * ry[0].dag()
    elif kwargs.get('state') == 'p1':
        return x_gates[1] * ry[0] * rho0 * ry[0].dag() * x_gates[1].dag()
    elif kwargs.get('state') == '0p':
        return ry[1] * rho0 * ry[1].dag()
    elif kwargs.get('state') == '1p':
        return x_gates[0] * ry[1] * rho0 * ry[1].dag() * x_gates[0].dag()
    elif kwargs.get('state') == 'pp':
        return ry[1] * ry[0] * rho0 * ry[0].dag() * ry[1].dag()
    elif kwargs.get('state') == 'pp_wrung':
        zz = qt.tensor(qt.sigmaz(), qt.sigmaz())
        rho_pp = ry[1] * ry[0] * rho0 * ry[0].dag() * ry[1].dag()
        return zz * rho_pp * zz.dag()
    else:
        raise Exception("Invalid state input")


@jax.jit
def make_Hamiltonian(y_uv, b_t):
    """
    Construct the system Hamiltonian at each time step.

    Parameters
    ----------
    y_uv : jax.Array
        Pulse sequence control matrix.
    b_t : jax.Array
        Noise trajectories for [qubit1, qubit2, Ising].

    Returns
    -------
    jax.Array
        Hamiltonian tensor of shape (time_steps, 8, 8).
    """
    paulis = jnp.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]], [[1., 0.], [0., -1.]]])
    z_vec = jnp.array([jnp.kron(paulis[0], paulis[0]), jnp.kron(paulis[3], paulis[0]), jnp.kron(paulis[0], paulis[3]),
                       jnp.kron(paulis[3], paulis[3])])
    h_t = (jnp.tensordot(y_uv[0, 0] * b_t[0] * 0.5, jnp.kron(z_vec[1], paulis[0]), 0)
           + jnp.tensordot(y_uv[1, 1] * b_t[1] * 0.5, jnp.kron(z_vec[2], paulis[0]), 0)
           + jnp.tensordot(y_uv[2, 2] * b_t[2] * 0.5, jnp.kron(z_vec[3], paulis[0]), 0))
    return h_t

# @jax.jit
def f(t, tk):
    """
    Control function generating a sequence of pulse flips.

    Parameters
    ----------
    t : jax.Array
        Time grid.
    tk : array_like
        Pulse switch times.

    Returns
    -------
    jax.Array
        Toggle function values (+1 or -1) at each time step.
    """
    return jnp.sum(jnp.array(
        [((-1) ** i) * jnp.heaviside(t - tk[i], 1) * jnp.heaviside(tk[i + 1] - t, 1) for i in
         range(jnp.size(tk) - 1)]), axis=0)


def cpmg(t, n):
    """
    Generate a CPMG pulse sequence toggle function.

    Parameters
    ----------
    t : array_like
        Time grid.
    n : int
        Number of base-cycle (CDD2) repetitions. The generated sequence has
        2n pi-pulses (matching the paper's [CDD2]_n convention).

    Returns
    -------
    jax.Array
        CPMG control function.
    """
    tk = [(k + 0.50) * t[-1] / (2 * n) for k in range(int(2 * n))]
    tk.append(t[-1])
    tk.insert(0, 0.)
    return f(t, tk)


def cdd1(t, n):
    """
    Generate a CDD1 pulse sequence toggle function.

    Parameters
    ----------
    t : array_like
        Time grid.
    n : int
        Order or pulse factor.

    Returns
    -------
    jax.Array
        CDD1 control function.
    """
    tk = [(k + 1) * (t[-1]) / (2 * n) for k in range(int(2 * n - 1))]
    tk.append(t[-1])
    tk.insert(0, 0.)
    return f(t, tk)



def prim_cycle(ct):
    """
    Generate a primitive cycle toggle function for CDD3.

    Parameters
    ----------
    ct : array_like
        Control time grid.

    Returns
    -------
    jax.Array
        Primitive cycle control function.
    """
    m = 1
    t = ct
    tk1 = [(k + 0.5) * t[-1] / (4 * m) for k in range(int(2))]
    tk1.insert(0, 0.)
    tk1 = np.array(tk1)
    tk2 = tk1 + t[-1] * 0.5
    tk2 = np.concatenate((tk2, [t[-1]]))
    tk = np.concatenate((tk1, tk2))
    return f(t, tk)


def cdd3(t, m):
    """
    Generate a CDD3 pulse sequence toggle function.

    Parameters
    ----------
    t : array_like
        Time grid.
    m : int
        Repetition factor.

    Returns
    -------
    jax.Array
        CDD3 control function.
    """
    if m == 1:
        return prim_cycle(t)
    out = np.tile(prim_cycle(t[:int(t.shape[0] / m)]), m)
    if t.shape[0] > out.shape[0]:
        out = np.concatenate((out, -1 * np.ones(t.shape[0] - out.shape[0])))
    return out


def make_y(t_b : np.ndarray, pulse : list[str], **kwargs):
    """
    Construct the pulse sequence control matrix y_uv.

    Parameters
    ----------
    t_b : np.ndarray
        Time grid for one block.
    pulse : list of str
        Names of the pulse sequences for [qubit1, qubit2].
    **kwargs
        ctime : float
            Total time for one block.
        m : int
            Number of blocks.

    Returns
    -------
    np.ndarray
        Control matrix of shape (3, 3, time_steps).
    """
    ctime = kwargs.get('ctime')
    M = kwargs.get('m')
    n = int((t_b[-1] / ctime).round(0))
    y = np.zeros((3, 3, np.size(t_b)))

    pulse_config = {
        'CPMG': (cpmg, n),
        'CDD1': (cdd1, n),
        'CDD3': (cdd3, n),
        'CPMG-1/2': (cpmg, int((t_b[-1] / (0.5 * ctime)).round(0))),
        'CDD1-1/2': (cdd1, int((t_b[-1] / (0.5 * ctime)).round(0))),
        'CDD1-1/4': (cdd1, int((t_b[-1] / (0.25 * ctime)).round(0))),
    }

    for i in range(2):
        pulse_name = pulse[i]
        if pulse_name in pulse_config:
            pulse_func, pulse_n = pulse_config[pulse_name]
            y[i, i] = pulse_func(t_b, pulse_n)
        elif pulse_name == 'FID':
            y[i, i] = np.ones(np.size(t_b))
        else:
            raise ValueError("The input pulse sequence not recognized.")

    y[2, 2] = y[1, 1] * y[0, 0]
    return np.tile(y, M)


def custom_y(vt, t_b, M):
    """
    Construct a custom pulse sequence control matrix from switch times.

    Parameters
    ----------
    vt : list of jax.Array
        Switch times for [qubit1, qubit2].
    t_b : jax.Array
        Time grid.
    M : int
        Number of blocks.

    Returns
    -------
    jax.Array
        Control matrix.
    """
    y = jnp.zeros((3, 3, np.size(t_b)))
    ftn = f(t_b, vt[0])
    y = y.at[0, 0].set(ftn)
    ftn = f(t_b, vt[1])
    y = y.at[1, 1].set(ftn)
    y = y.at[2, 2].set(jnp.multiply(y[1, 1], y[0, 0]))
    return jnp.tile(y, M)


@jax.jit
def make_propagator(H_t, t_vec):
    """
    Calculate the time-evolution propagator for a given Hamiltonian.

    Parameters
    ----------
    H_t : jax.Array
        Time-dependent Hamiltonian.
    t_vec : jax.Array
        Time vector.

    Returns
    -------
    jax.Array
        Unitary propagator.
    """
    h_diags = jnp.diagonal(H_t, axis1=1, axis2=2)
    phi = -1j * jax.scipy.integrate.trapezoid(h_diags, t_vec, axis=0)
    return jnp.diag(jnp.exp(phi))


@jax.jit
def single_shot_prop(noise_mats, t_vec, y_uv, rho0, key):
    """
    Simulate a single noise trajectory realization of the propagator.

    Parameters
    ----------
    noise_mats : jax.Array
        Precomputed noise matrices.
    t_vec : jax.Array
        Time vector.
    y_uv : jax.Array
        Control matrix.
    rho0 : jax.Array
        Initial state density matrix.
    key : jax.Array
        JAX PRNG key.

    Returns
    -------
    jax.Array
        Final state density matrix after evolution.
    """
    size = jnp.size(t_vec)
    y_uv = y_uv[:, :, :size]
    b_t = make_channel_trajs(noise_mats, key)[:, :size]
    H_t = make_Hamiltonian(y_uv, b_t)
    U = make_propagator(H_t, t_vec)
    rho_MT = jnp.matmul(jnp.matmul(U, rho0), U.conjugate().transpose())
    return rho_MT


# Diagonal phase basis of the dephasing Hamiltonian: H(t) is diagonal in the
# 8-dim (2 qubits + aux) computational basis with diag(H) = sum_a C'_a(t) d_a,
# d_a = diag(Z_a (x) 1). The full propagator is then U = exp(-i sum_a C_a d_a)
# with C_a = int 0.5 y_a(t) b_a(t) dt -- three numbers per shot determine the
# entire evolution. This is what makes the record/replay SPAM pipeline cheap.
_PAULI_Z_DIAG = np.array([1., -1.])
_DIAG_BASIS = jnp.array([
    np.kron(np.kron(_PAULI_Z_DIAG, np.ones(2)), np.ones(2)),   # Z1
    np.kron(np.kron(np.ones(2), _PAULI_Z_DIAG), np.ones(2)),   # Z2
    np.kron(np.kron(_PAULI_Z_DIAG, _PAULI_Z_DIAG), np.ones(2)),  # Z1Z2
])


@jax.jit
def single_shot_phase_coeffs(noise_mats, t_vec, y_uv, key):
    """Per-shot dephasing phase coefficients C_a = int 0.5 y_a b_a dt, a in
    {1, 2, 12}. Identical trajectory draw to `single_shot_prop` (same key ->
    same noise); the propagator is U = exp(-i C . _DIAG_BASIS)."""
    size = jnp.size(t_vec)
    y_uv = y_uv[:, :, :size]
    b_t = make_channel_trajs(noise_mats, key)[:, :size]
    integrand = 0.5*jnp.stack([y_uv[0, 0]*b_t[0], y_uv[1, 1]*b_t[1],
                               y_uv[2, 2]*b_t[2]])
    return jax.scipy.integrate.trapezoid(integrand, t_vec, axis=1)


def solver_phase_coeffs(y_uv, noise_mats, t_vec, n_shots):
    """Phase-coefficient counterpart of `solver_prop`: returns (n_shots, 3).

    Mirrors `solver_prop`'s chunking and np.random key draws exactly, so a run
    that records phases consumes the same RNG stream (and therefore the same
    noise realizations) as a legacy run."""
    y_uv = jnp.array(y_uv)
    output = []
    slice_size = 2000
    n_slices = int(np.ceil(n_shots / slice_size))
    for i in range(n_slices):
        current_slice_size = min(slice_size, n_shots - i * slice_size)
        n_arr = jnp.array(np.random.randint(0, 10000, (current_slice_size, 2)))
        result = jax.vmap(single_shot_phase_coeffs,
                          in_axes=[None, None, None, 0])(noise_mats, t_vec, y_uv, n_arr)
        output.append(result)
    return jnp.concatenate(output, axis=0)


@jax.jit
def _filter_vectors(noise_mats, t_vec, y_uv):
    """Per-call filter vectors F[a, comp, sin/cos, w] = int 0.5 y_a(t) Mat(t, w) dt.

    The phase coefficients are LINEAR in the Gaussian draws, so the entire
    (t x w) trajectory synthesis can be contracted against the control toggles
    ONCE per call; every shot then costs ten dot products of length n_w instead
    of six (t x w) matvecs (~1000x less). The einsum reproduces the trapezoid
    integral exactly on the uniform time grid."""
    size = jnp.size(t_vec)
    y = y_uv[:, :, :size]
    ys = jnp.stack([y[0, 0], y[1, 1], y[2, 2]])          # (3, t)
    mats = noise_mats[:, :, :size, :]                     # (5, 2, t, w)
    dt = t_vec[1] - t_vec[0]
    wt = jnp.full(size, dt).at[0].set(0.5*dt).at[-1].set(0.5*dt)
    return jnp.einsum('at,cstw->acsw', 0.5*ys*wt[None, :], mats)


@jax.jit
def _shot_coeffs_from_filters(F, key):
    """One shot's (3,) phase coefficients from precomputed filter vectors.

    Identical RNG scheme to `make_channel_trajs` (same fold_in/split and the
    same per-stream normal draws), so the same key yields the same noise
    realization as the trajectory-level path, up to float reassociation."""
    base = jax.random.fold_in(jax.random.PRNGKey(key[0]), key[1])
    ks = jax.random.split(base, 10)
    n_w = jnp.size(F, 3)
    draw = lambda k: jax.random.normal(k, (n_w, 1))[:, 0]

    def comp(a, c, ka, kb):
        return jnp.dot(F[a, c, 0], draw(ka)) + jnp.dot(F[a, c, 1], draw(kb))

    # streams: ks[0:2] shared, ks[2:4] local-A, ks[4:6] local-B, ks[6:8] n1,
    # ks[8:10] n2; components: 0 = el_A, 1 = el_B shifted, 2 = el_B, 3 = n1,
    # 4 = n2 (the make_noise_mat_arr ordering).
    def channel_parts(a):
        e_a = _C_SH*comp(a, 0, ks[0], ks[1]) + _C_LOC*comp(a, 0, ks[2], ks[3])
        e_b = _C_SH*comp(a, 1, ks[0], ks[1]) + _C_LOC*comp(a, 2, ks[4], ks[5])
        return e_a, e_b

    e_a1, _ = channel_parts(0)
    _, e_b2 = channel_parts(1)
    e_a12, e_b12 = channel_parts(2)
    c1 = e_a1 + comp(0, 3, ks[6], ks[7])
    c2 = e_b2 + comp(1, 4, ks[8], ks[9])
    c12 = _A_J*e_a12 - _B_J*e_b12
    return jnp.stack([c1, c2, c12])


def solver_phase_coeffs_fast(y_uv, noise_mats, t_vec, n_shots):
    """Filter-vector phase solver: same statistics (and same per-key noise
    realizations, to float reassociation) as `solver_phase_coeffs` at ~1000x
    less per-shot compute. Used by the recording SPAM pipeline; shots are no
    longer the runtime budget."""
    F = _filter_vectors(noise_mats, jnp.asarray(t_vec), jnp.array(y_uv))
    output = []
    slice_size = 20000
    n_slices = int(np.ceil(n_shots / slice_size))
    for i in range(n_slices):
        current_slice_size = min(slice_size, n_shots - i * slice_size)
        n_arr = jnp.array(np.random.randint(0, 10000, (current_slice_size, 2)))
        output.append(jax.vmap(_shot_coeffs_from_filters,
                               in_axes=[None, 0])(F, n_arr))
    return jnp.concatenate(output, axis=0)


class PhasedState:
    """Per-shot diagonal-propagator state: phases u (n_shots, 8) + prep rho (8, 8).

    Equivalent to the dense (n_shots, 8, 8) stack U_s rho U_s^dag (U_s diagonal)
    but ~24x lighter; `observables.compute_probs_jax` consumes it through an
    exact quadratic-form fast path (probs = u^dag [ (G^dag M G)^T o rho ] u)."""

    def __init__(self, u, rho):
        self.u = u
        self.rho = rho

    @property
    def shape(self):
        return (self.u.shape[0], 8, 8)


def phased_state(coeffs, rho):
    """Build the PhasedState for stored per-shot phase coefficients."""
    u = jnp.exp(-1j*jnp.matmul(jnp.asarray(coeffs), _DIAG_BASIS))
    return PhasedState(u, jnp.asarray(rho))


@jax.jit
def apply_phase_coeffs(coeffs, rho):
    """Evolve `rho` through stored per-shot phase coefficients.

    Returns the (n_shots, 8, 8) stack of U_s rho U_s^dag with the diagonal
    U_s = exp(-i coeffs_s . _DIAG_BASIS) -- the exact replay of what
    `solver_prop` would have produced for these noise realizations, for ANY
    initial state (the SPAM-protocol arms differ only in rho and in estimator
    post-processing, so one recorded dataset serves them all)."""
    p = jnp.matmul(coeffs, _DIAG_BASIS)            # (n_shots, 8)
    u = jnp.exp(-1j*p)
    return u[:, :, None]*rho[None, :, :]*jnp.conj(u)[:, None, :]


def solver_prop(y_uv, noise_mats, t_vec, rho, n_shots):
    """
    Solve for the average density matrix across multiple noise shots.

    Parameters
    ----------
    y_uv : array_like
        Control matrix.
    noise_mats : array_like
        Noise matrices.
    t_vec : array_like
        Time vector.
    rho : array_like
        Initial state.
    n_shots : int
        Number of noise realizations (shots).

    Returns
    -------
    jax.Array
        Average final density matrix.
    """
    y_uv = jnp.array(y_uv)
    output = []
    # Memory allocation safety for my laptop with a single GPU
    slice_size = 2000
    n_slices = int(np.ceil(n_shots / slice_size))
    for i in range(n_slices):
        current_slice_size = min(slice_size, n_shots - i * slice_size)
        n_arr = jnp.array(np.random.randint(0, 10000, (current_slice_size, 2)))
        result = jax.vmap(single_shot_prop, in_axes=[None, None, None, None, 0])(noise_mats, t_vec, y_uv, rho, n_arr)
        output.append(result)
    return jnp.concatenate(output, axis=0)

