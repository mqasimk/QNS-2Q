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


def make_noise_mat_arr(act, **kwargs):
    """
    Generate or load matrices for temporally-correlated noise trajectories.

    Parameters
    ----------
    act : str
        Action to perform: 'load', 'make', or 'save'.
    **kwargs
        spec_vec : list of callable
            Spectral density functions.
        t_vec : jax.Array
            Time vector for evolution.
        w_grain : int
            Frequency discretization grain.
        wmax : float
            Maximum frequency cutoff.
        truncate : int
            Number of harmonics to include.
        gamma : float
            Time shift for qubit 2.
        gamma_12 : float
            Time shift for Ising interaction.

    Returns
    -------
    jax.Array
        Array of sine and cosine noise matrices.
    """
    spec_vec = kwargs.get('spec_vec')
    t_vec = kwargs.get('t_vec')
    w_grain = kwargs.get('w_grain')
    wmax = kwargs.get('wmax')
    truncate = kwargs.get('truncate')
    gamma = kwargs.get('gamma')
    gamma_12 = kwargs.get('gamma_12')
    if act == 'load':
        return np.load('noise_mats.npy', allow_pickle=True)
    elif act == 'make':
        S_11, C_11 = make_noise_mat(spec_vec[0], t_vec, w_grain=w_grain, wmax=wmax, trunc_n=truncate, gamma=0.)
        S_22g, C_22g = make_noise_mat(spec_vec[1], t_vec, w_grain=w_grain, wmax=wmax, trunc_n=truncate, gamma=gamma)
        S_1212g, C_1212g = make_noise_mat(spec_vec[2], t_vec, w_grain=w_grain, wmax=wmax, trunc_n=truncate,
                                      gamma=gamma_12)
        return jnp.array([[S_11, C_11], [S_22g, C_22g], [S_1212g, C_1212g]])
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
    size_w = int(2 * w_grain)
    w = jnp.linspace(0, 2 * wmax, size_w)
    dw = wmax / w_grain
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
            Target state to generate: 'p0', 'p1', '0p', '1p', or 'pp'.

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
        Number of pi-pulses.

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
    bvec_1 = make_noise_traj(noise_mats[0, 0], noise_mats[0, 1], key)[:size]
    bvec_2_g = make_noise_traj(noise_mats[1, 0], noise_mats[1, 1], key)[:size]
    bvec_1212g = make_noise_traj(noise_mats[2, 0], noise_mats[2, 1], key)[:size]
    H_t = make_Hamiltonian(y_uv, jnp.array([bvec_1, bvec_2_g, bvec_1212g]))
    U = make_propagator(H_t, t_vec)
    rho_MT = jnp.matmul(jnp.matmul(U, rho0), U.conjugate().transpose())
    return rho_MT


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

