"""
Noise trajectory synthesis and quantum-state propagation: the shared Monte Carlo
simulation engine of the QNS-2Q pipeline.

Physics role
------------
This module answers: "given a classical noise process with a known power spectral
density (PSD) and a chosen dynamical-decoupling pulse sequence, what quantum state
does the two-qubit system end up in after some elapsed time, averaged over many
random noise realizations?" It is the Monte Carlo forward model behind the whole
QNS experiment: synthesize many random noise-trajectory "shots" from a target PSD
(``noise/spectra.py``), build the (dephasing-only) two-qubit-plus-bath-qubit
Hamiltonian for the applied pulse sequence, propagate the initial state through it,
and average the resulting density matrices over shots. The auxiliary "bath" qubit
is bookkeeping only (see CLAUDE.md's "3-Qubit Hilbert Space Convention"): it never
couples to anything physically, it just keeps array shapes (8x8 matrices) uniform
across the codebase.

Pipeline placement
-------------------
This module lives in the shared ``model/`` layer and is used by the
**characterize** arm (QNS experiments -> spectral reconstruction) ONLY -- the
``control/`` arm (gate optimizers ``control/cz.py``, ``control/idle.py``) works
directly from the analytic PSDs in ``noise/spectra.py`` and never runs this Monte
Carlo simulation; it needs closed-form infidelity integrals, not stochastic noise
trajectories. Callers: ``characterize/experiments.py`` (the main QNS experiment
runner), ``characterize/single_qubit.py`` (single-qubit analogue),
``characterize/spam.py`` / ``characterize/systematics.py`` (SPAM calibration and
systematic-error diagnostics), and ``model/observables.py`` (the POVM/measurement
layer one level up, which imports ``make_init_state``, ``make_y``, and
``PhasedState`` from here).

Inputs / outputs
-----------------
Reads the PSD functions and mixing constants from ``qns2q.noise.spectra``
(``S_el_A``, ``S_el_B``, ``S_nuc_1``, ``S_nuc_2`` plus the showcase-regime extras,
and ``C2_SHARE``/``A_J``/``B_J``/``DT_SHIFT``) to synthesize noise. Produces, for
callers: ensemble-averaged 8x8 density matrices (``solver_prop``), or -- on the
faster "phase-coefficient" path used by the SPAM record/replay machinery -- the
much smaller per-shot (3,) dephasing-phase triples that are algebraically
equivalent to those density matrices (``solver_phase_coeffs*``, ``PhasedState``,
``apply_phase_coeffs``). Also defines the pulse-sequence toggle functions (CPMG,
CDD1, CDD3) used to build the control matrix ``y_uv`` consumed throughout.

Two propagation paths, same physics
------------------------------------
Because the model is pure dephasing, every term in the Hamiltonian is diagonal in
the 8-dimensional computational basis at every instant (``make_Hamiltonian``), so
the propagator is *exactly* ``exp(-i * integral of the diagonal)`` -- no ODE solve
is needed (``make_propagator``). This module offers two routes to the same
physics:
    (1) the "dense" path -- ``make_channel_trajs`` -> ``make_Hamiltonian`` ->
        ``make_propagator`` -> ``single_shot_prop`` -> ``solver_prop`` -- builds
        and exponentiates the full 8x8 Hamiltonian for every shot;
    (2) the "phase-coefficient" fast path -- ``single_shot_phase_coeffs`` /
        ``_filter_vectors`` + ``_shot_coeffs_from_filters`` -- exploits that only
        three numbers per shot (the integrated Z1, Z2, Z1Z2 phases) determine the
        whole propagator, and that those phases are linear in the underlying
        Gaussian noise draws, to precompute "filter vectors" once per call instead
        of once per shot (roughly 1000x fewer FLOPs). Both paths draw noise with
        the SAME random-number scheme (see ``make_channel_trajs``'s docstring), so
        they are numerically interchangeable; the fast path is what lets the
        SPAM-robust protocol "record" a noise dataset once and cheaply "replay" it
        against several different initial states / SPAM arms
        (``characterize/experiments.py``).

Uses JAX (``jax.jit``, ``jax.vmap``) throughout for GPU-vectorized Monte Carlo; see
the inline comments at the first use of each JAX idiom below for what it does and
why it is there.
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
        Python idiom: ``**kwargs`` collects any named arguments the caller
        passes into a dict, so this function accepts a loose bag of options
        instead of a long fixed positional-argument list. It is used here
        (and elsewhere in this module) as a lightweight stand-in for a config
        object: callers only need to supply the keys relevant to the ``act``
        they are performing, and new optional knobs (like ``midpoint`` or
        ``zz_extra`` below) can be added without breaking existing call
        sites. The recognized keys are:

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
    # Showcase regime: a SIXTH stream carries the independent coupler-defect
    # process j(t) on the ZZ channel (zeta_12 = A_J e_A - B_J e_B + j). The
    # five-stream regimes are untouched -- same array shape, same key budget,
    # bit-identical draws. Pass zz_extra=None to force the five-stream model.
    zz_extra = kwargs.get('zz_extra', _model_spectra.S_zz_extra)
    # Showcase shared carrier (SHOWCASE-0612): in the showcase noise model, the
    # slow common-mode drift ("carrier") that both qubits pick up is split out
    # of the local-nuclear components into its own shared+local pair of
    # streams, so its cross-qubit correlation can be tuned independently of
    # each qubit's own nuclear noise. Concretely: components 3/4 become the
    # (strictly local) nuclear-Larmor line families, and components 6/7 carry
    # the "carrier" filters for qubits 1/2, driven by one common stream plus
    # one local stream each in make_channel_trajs. Only active when the model
    # declares HAS_QS_SHARED and the caller did not override `components`.
    qs_pair = kwargs.get('qs_pair', '__model__')
    if qs_pair == '__model__':
        qs_pair = ((_model_spectra.S_qs_1, _model_spectra.S_qs_2)
                   if getattr(_model_spectra, 'HAS_QS_SHARED', False)
                   and kwargs.get('components') is None else None)
    if act == 'load':
        return np.load('noise_mats.npy', allow_pickle=True)
    elif act == 'make':
        s_el_a, s_el_b, s_nuc_1, s_nuc_2 = components
        if qs_pair is not None:
            # carrier split: local rows carry the lines only
            s_nuc_1 = _model_spectra.S_lines_1
            s_nuc_2 = _model_spectra.S_lines_2
        mk = lambda spec, shift: make_noise_mat(
            spec, t_vec, w_grain=w_grain, wmax=wmax, trunc_n=truncate,
            gamma=shift, midpoint=midpoint)
        rows = [mk(s_el_a, 0.), mk(s_el_b, dt_shift), mk(s_el_b, 0.),
                mk(s_nuc_1, 0.), mk(s_nuc_2, 0.)]
        if zz_extra is not None:
            rows.append(mk(zz_extra, 0.))
        if qs_pair is not None:
            if zz_extra is None:
                raise ValueError("qs_pair without zz_extra would collide with "
                                 "the 6-stream shape; not a supported regime")
            rows.append(mk(qs_pair[0], 0.))
            rows.append(mk(qs_pair[1], 0.))
        return jnp.array(rows)
    elif act == 'save':
        mats = make_noise_mat_arr('make', **kwargs)
        np.save('noise_mats.npy', mats)
        return mats
    else:
        raise Exception("Invalid action input")


# `@jax.jit` is left commented out (here and at a couple of other functions
# below): `spec` is a plain Python callable, not a JAX array, and jax.jit can
# only trace bare Python objects like that if you mark them "static" (fixed
# for the lifetime of a compiled version); since these two functions are
# themselves vmapped over a (w, t) grid inside `make_noise_mat` (see below),
# they get compiled once as part of that larger vmapped call anyway, so
# jitting them individually would not add anything.
def sinM(spec, w, t, dw, gamma):
    """
    One discretized frequency mode of the noise spectral-synthesis method
    (sine component).

    This implements the "spectral representation" trick used to turn a target
    power spectral density S(w) into a time-domain random process: at each
    discrete frequency w, ``sqrt(dw * S(w) / pi)`` is the standard-deviation
    weight of that Fourier mode, and multiplying by ``sin(w*(t+gamma))``
    produces the mode's time-domain waveform. `make_noise_traj` later sums
    these modes (sine and the matching `cosM` cosine term) against
    independent standard-Gaussian random coefficients to synthesize a
    stationary Gaussian process whose PSD matches ``spec`` in the dw -> 0
    limit; `gamma` is the extra time delay used to build the DT_SHIFT-lagged
    cross-correlated stream in `make_noise_mat_arr`.

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
    One discretized frequency mode of the noise spectral-synthesis method
    (cosine component). See `sinM` above for the shared explanation of the
    method and of why `@jax.jit` is left off.

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
    Precompute the (sine, cosine) synthesis matrices for one noise component.

    This lays out the spectral-synthesis method of `sinM`/`cosM` (see their
    docstrings) over a full frequency grid ``w`` and time grid ``t_vec`` at
    once, producing two ``[n_t, n_w]`` matrices whose columns are the
    frequency modes. `make_noise_traj` then only has to do a matrix-vector
    product against a vector of random Gaussian coefficients (one per
    frequency) to produce a whole noise trajectory -- the expensive
    (spec, w, t) evaluation is done here, ONCE, and reused for every
    Monte Carlo shot.

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
    # jax.vmap turns a scalar function into a vectorized (batched) one without
    # writing an explicit loop: nesting two vmaps here broadcasts sinM/cosM
    # (originally scalar-in, scalar-out) over BOTH the frequency axis `w` and
    # the time axis `t_vec` simultaneously, producing the full [n_t, n_w]
    # matrix in one call. `in_axes` says which argument position holds the
    # batch axis for each of the two vmap layers (`None` = broadcast/shared,
    # not batched); this is standard JAX vectorization, analogous to numpy
    # broadcasting but composable and JIT/GPU-friendly.
    Sf = jax.vmap(jax.vmap(sinM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    Cf = jax.vmap(jax.vmap(cosM, in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None))
    # Build in time-blocks: the fused full-grid build materializes ~3x the
    # [n_t x size_w] output and OOMs a 12 GB GPU at Nyquist-band grids. The
    # matrices are elementwise in (t, w), so block-row concatenation is exact.
    n_t = int(jnp.size(t_vec))
    blk = 2048
    blocks_s, blocks_c = [], []
    for s in range(0, n_t, blk):
        tb = t_vec[s:s + blk]
        blocks_s.append(Sf(spec, w, tb, dw, gamma))
        blocks_c.append(Cf(spec, w, tb, dw, gamma))
    return jnp.concatenate(blocks_s, axis=0), jnp.concatenate(blocks_c, axis=0)


# `@jax.jit` (the first ACTIVE one in this file) traces this function once
# and compiles it to fast XLA code the first time it is called with a given
# set of input shapes/dtypes, then reuses that compiled version on every
# later call -- much faster than re-interpreting the Python each time,
# provided the function is called many times with the same shapes (true here:
# this runs once per Monte Carlo shot).
@jax.jit
def make_noise_traj(S, C, key):
    """
    Synthesize one random noise trajectory from precomputed synthesis matrices.

    Draws two independent vectors of standard-Gaussian random coefficients
    (one per frequency in `S`/`C`'s column count) and forms
    ``traj = S @ A + C @ B``: exactly the weighted sum of sine/cosine
    frequency modes described in `sinM`/`cosM`, so `traj` is one realization
    of a stationary Gaussian process whose PSD is the `spec` that produced
    `S`/`C` in `make_noise_mat`.

    Parameters
    ----------
    S : jax.Array
        Sine matrix from `make_noise_mat`.
    C : jax.Array
        Cosine matrix from `make_noise_mat`.
    key : jax.Array
        A pair of plain integers (NOT a genuine ``jax.random.PRNGKey`` object)
        used as two independent random seeds -- `key[0]` seeds the
        sine-mode coefficients, `key[1]` the cosine-mode coefficients. This
        "key" naming is reused throughout the module for such integer-pair
        seeds; it is turned into real PRNG keys via ``jax.random.PRNGKey``
        inside this function.

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
# C2_SHARE is a POWER fraction (0 to 1: how much of each field's variance
# comes from the common source vs. its own local source), so the amplitude
# weights that add incoherently (as independent Gaussians) need the sqrt:
# e_A = sqrt(C2_SHARE)*g0 + sqrt(1-C2_SHARE)*g_A, matching NOISE_MODEL_SPEC.md.
_C_SH = jnp.sqrt(_model_spectra.C2_SHARE)
_C_LOC = jnp.sqrt(1. - _model_spectra.C2_SHARE)
_A_J = _model_spectra.A_J   # J-coupling weight on e_A in zeta_12 = A_J*e_A - B_J*e_B
_B_J = _model_spectra.B_J   # J-coupling weight on e_B (same difference-coupling story)
# Shared-carrier split (showcase): one common stream + one local stream per
# qubit through the carrier filters (components 6/7). Same power-fraction ->
# amplitude-weight sqrt as C2_SHARE above, just for the showcase-only carrier.
_QS_SH = jnp.sqrt(_model_spectra._SC_C2_QS)
_QS_LOC = jnp.sqrt(1. - _model_spectra._SC_C2_QS)


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
    # The sixth (showcase) stream is the independent coupler defect j(t) on the
    # ZZ channel; streams 7/8 (when present) are the shared-carrier split:
    # ONE common stream through both carrier filters plus one local stream
    # each (SHOWCASE-0612: this is the showcase noise model's extra
    # cross-qubit-correlated slow drift, see NOISE_MODEL_SPEC.md background
    # above). `n_streams` (and hence `has_zz`/`has_qs`) is a plain Python bool
    # computed from a JAX ARRAY SHAPE, which is always static (known at
    # trace/compile time, unlike array VALUES) -- so `if has_zz:` below is an
    # ordinary Python branch resolved once per distinct `noise_mats` shape,
    # not something that needs `jax.lax.cond`. The five-stream path keeps the
    # exact legacy key budget (`split(base, 10)`) -- existing regimes' draws
    # are bit-identical to before these extra streams existed.
    n_streams = jnp.size(noise_mats, 0)
    has_zz = (n_streams >= 6)
    has_qs = (n_streams == 8)
    base = jax.random.fold_in(jax.random.PRNGKey(key[0]), key[1])
    ks = jax.random.split(base, 18 if has_qs else (12 if has_zz else 10))
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
    zeta_1 = e_a + n_1
    zeta_2 = e_b + n_2
    zeta_12 = _A_J*e_a - _B_J*e_b
    if has_zz:
        zeta_12 = zeta_12 + comp(5, ks[10], ks[11])
    if has_qs:
        # qs_c: SAME draws (ks[12:14]) through both qubits' slow-noise filters
        # -> correlated; the zeta_12 difference coupling never sees it.
        zeta_1 = zeta_1 + _QS_SH*comp(6, ks[12], ks[13]) \
            + _QS_LOC*comp(6, ks[14], ks[15])
        zeta_2 = zeta_2 + _QS_SH*comp(7, ks[12], ks[13]) \
            + _QS_LOC*comp(7, ks[16], ks[17])
    return jnp.array([jnp.ravel(zeta_1), jnp.ravel(zeta_2),
                      jnp.ravel(zeta_12)])



def make_init_state(a_sp, c, **kwargs):
    """
    Generate the initial two-qubit state with SPAM (state-preparation-and-
    measurement) errors baked into the preparation.

    In CLAUDE.md's SPAM notation, `a_sp` is alpha_SP^z (the Z-axis state-prep
    visibility per qubit -- 1 means a perfect |0> or |1> prep) and `c` is the
    transverse (X/Y-plane) Bloch component injected by a faulty prep (0 means
    no unwanted coherence). Both are physically per-qubit single-qubit Bloch
    vectors that get combined into single-qubit density matrices `rho0_0`,
    `rho0_1` below and then tensored together (plus rotated to the requested
    basis) to build the full initial state.

    Parameters
    ----------
    a_sp : array_like
        State preparation errors along the Z axis for [qubit1, qubit2]
        (alpha_SP^z per qubit; see CLAUDE.md's SPAM-pipeline section).
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
        Initial 4x4 (two-qubit) density matrix. NOTE: unlike most other
        functions in this module, this does NOT include the auxiliary bath
        qubit -- callers tensor that in themselves (typically as a maximally
        mixed ``0.5 * qt.identity(2)``, e.g.
        ``qt.tensor(make_init_state(...), 0.5*qt.identity(2))``) to build the
        8x8 state that `solver_prop`/`single_shot_prop` expect.
    """
    zp = qt.basis(2, 0)   # computational |0>, the single-qubit "up" Z eigenstate
    zm = qt.basis(2, 1)   # computational |1>, the single-qubit "down" Z eigenstate
    x_gates = [qt.tensor(qt.sigmax(), qt.identity(2)), qt.tensor(qt.identity(2), qt.sigmax())]
    asp_0 = a_sp[0]   # alpha_SP^z for qubit 1
    asp_1 = a_sp[1]   # alpha_SP^z for qubit 2
    c_0 = c[0]        # transverse SP error for qubit 1
    c_1 = c[1]        # transverse SP error for qubit 2
    # Single-qubit density matrix from a Z-visibility (asp) and a transverse
    # coherence (c): (I + asp*Z + Re(c)*X - Im(c)*Y)/2 written out in the
    # |0>,|1> basis (asp=1, c=0 recovers the ideal |0><0| prep).
    rho0_0 = 0.5 * (1. + asp_0) * zp * zp.dag() + 0.5 * (1. - asp_0) * zm * zm.dag() + 0.5 * c_0 * zp * zm.dag() + 0.5 * np.conj(
        c_0) * zm * zp.dag()
    rho0_1 = 0.5 * (1. + asp_1) * zp * zp.dag() + 0.5 * (1. - asp_1) * zm * zm.dag() + 0.5 * c_1 * zp * zm.dag() + 0.5 * np.conj(
        c_1) * zm * zp.dag()
    rho0 = qt.tensor(rho0_0, rho0_1)
    # ry[i]: a +90-degree rotation about Y on qubit i (rotates |0> toward the
    # equator, i.e. prepares a |+>-like state from |0>); used below to turn
    # the Z-basis rho0 into whichever target state `kwargs['state']` asks for.
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
    Construct the pure-dephasing system Hamiltonian at each time step, on the
    full 8-dimensional (2 qubits + 1 bookkeeping bath qubit) Hilbert space.

    Physically this builds
        H(t) = 0.5 * [ y_1(t) b_1(t) Z1 + y_2(t) b_2(t) Z2
                       + y_12(t) b_12(t) Z1 Z2 ] (x) I_bath ,
    i.e. each noise channel `b_t[i]` (qubit-1 dephasing, qubit-2 dephasing,
    Ising/ZZ dephasing) couples to its matching Pauli-Z combination, with
    strength modulated in time by the pulse-sequence toggle function
    `y_uv[i, i]` (+-1, from `make_y`/`custom_y`) -- this is what makes
    dynamical decoupling work: flipping the qubit inverts the sign of the
    noise coupling, so noise accumulated before a pulse can be cancelled by
    noise accumulated after it. Every term here is diagonal (built only from
    Z-type Pauli matrices), which is what lets `make_propagator` exponentiate
    the Hamiltonian exactly and cheaply instead of solving a Schrodinger
    equation. The trailing ``kron(_, paulis[0])`` in each term tensors on the
    identity for the 3rd (bath) qubit, expanding every 4x4 two-qubit operator
    to 8x8 (see CLAUDE.md's "3-Qubit Hilbert Space Convention").

    Parameters
    ----------
    y_uv : jax.Array
        Pulse sequence control matrix (see `make_y`); only the diagonal
        entries `y_uv[0,0]`, `y_uv[1,1]`, `y_uv[2,2]` (qubit 1, qubit 2,
        Ising toggle functions) are used here.
    b_t : jax.Array
        Noise trajectories for [qubit1, qubit2, Ising], as produced by
        `make_channel_trajs`.

    Returns
    -------
    jax.Array
        Hamiltonian tensor of shape (time_steps, 8, 8).
    """
    # paulis[0..3] = I, X, Y, Z (2x2) in the usual physics ordering.
    paulis = jnp.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]], [[1., 0.], [0., -1.]]])
    # z_vec[1] = Z (x) I = Z1, z_vec[2] = I (x) Z = Z2, z_vec[3] = Z (x) Z = Z1 Z2
    # (z_vec[0] = I (x) I is built for symmetry but not used below).
    z_vec = jnp.array([jnp.kron(paulis[0], paulis[0]), jnp.kron(paulis[3], paulis[0]), jnp.kron(paulis[0], paulis[3]),
                       jnp.kron(paulis[3], paulis[3])])
    h_t = (jnp.tensordot(y_uv[0, 0] * b_t[0] * 0.5, jnp.kron(z_vec[1], paulis[0]), 0)
           + jnp.tensordot(y_uv[1, 1] * b_t[1] * 0.5, jnp.kron(z_vec[2], paulis[0]), 0)
           + jnp.tensordot(y_uv[2, 2] * b_t[2] * 0.5, jnp.kron(z_vec[3], paulis[0]), 0))
    return h_t

# `@jax.jit` is left off here: `tk` (the list of pulse switch times) has a
# different LENGTH for every different pulse sequence/repetition count, and
# jax.jit recompiles from scratch whenever an input shape changes -- so
# jitting `f` would mean paying a fresh compilation cost for almost every
# call instead of reusing one compiled version, with no net speedup.
def f(t, tk):
    """
    Toggle (control) function for a sequence of instantaneous pi pulses.

    Between consecutive switch times `tk[i]` and `tk[i+1]`, the returned
    function is a constant +1 or -1, alternating sign at each pulse: this is
    the "y(t)" that multiplies the noise coupling in `make_Hamiltonian`, so a
    -1 half-interval represents the system having been flipped by a pi pulse
    relative to the previous interval. Implemented as a sum of boxcar
    (heaviside-minus-heaviside) windows, one per interval between switch
    times, each carrying its alternating sign `(-1)**i`.

    Parameters
    ----------
    t : jax.Array
        Time grid.
    tk : array_like
        Pulse switch times, INCLUDING the start (0) and end (t[-1]) of the
        block as the first/last entries -- so there are ``len(tk) - 1``
        intervals and ``len(tk) - 2`` actual pulses.

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
    Generate a CPMG (Carr-Purcell-Meiboom-Gill) dynamical-decoupling toggle
    function: pi pulses spaced EVENLY across the block, at the midpoints of
    ``2n`` equal sub-intervals. CPMG is the simplest decoupling sequence --
    good at suppressing slow (low-frequency) dephasing noise, and the
    baseline every other sequence in this file is compared against.

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
    Generate a CDD1 (first-order Concatenated Dynamical Decoupling) toggle
    function. Unlike CPMG's evenly-spaced pulses, CDD nests decoupling
    sequences inside each other (concatenation) to cancel noise to higher
    order in the pulse-interval time; CDD1 uses pulses PACKED toward one end
    of each period rather than evenly spaced, which is what gives it a
    different (and, for some noise spectra, better) noise-suppression
    profile than CPMG.

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
    Build the primitive (order-1) cycle of a CDD3 (3rd-order Concatenated
    Dynamical Decoupling) sequence: two CPMG-like pulses in the first half of
    the block, then the same pair again in the second half. `cdd3` builds the
    full higher-order sequence by tiling copies of this primitive cycle, the
    "concatenation" step that gives CDD its name and its improved noise
    suppression relative to plain CPMG.

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
    Generate a CDD3 pulse sequence toggle function by tiling `m` copies of
    the `prim_cycle` primitive back-to-back across the time grid (padding the
    tail with -1, i.e. "still flipped", if `t`'s length is not an exact
    multiple of `m`'s block length). See `prim_cycle` for what makes this a
    3rd-order *concatenated* sequence rather than a plain repeated CPMG-like
    pattern.

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
    Construct the pulse sequence control matrix ``y_uv`` -- the by-name,
    library-sequence counterpart of `custom_y` -- for one measurement block,
    then tile it across `m` blocks.

    ``y_uv`` is a (3, 3, time_steps) array but, as used elsewhere in this
    module (`make_Hamiltonian`, `single_shot_prop`, ...), only its DIAGONAL
    entries `y_uv[0,0]`, `y_uv[1,1]`, `y_uv[2,2]` matter -- the qubit-1,
    qubit-2, and Ising toggle functions respectively. The Ising toggle
    `y[2,2]` is set to the PRODUCT of the two qubit toggles: physically, the
    ZZ coupling only flips sign when exactly one of the two qubits has been
    pulsed (both-or-neither pulsed leaves the coupling's effective sign
    unchanged).

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

    # `pulse_config` is a name -> (generator function, repetition count) table
    # -- a common Python idiom for turning a big if/elif chain of string
    # comparisons into a single dict lookup. Each value pairs one of the pulse
    # generators above (`cpmg`, `cdd1`, `cdd3`) with the repetition count that
    # sequence needs for the requested total block time `ctime` (or a
    # fraction of it, for the "-1/2"/"-1/4" faster-pulsing variants).
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
    Construct a pulse sequence control matrix directly from explicit switch
    times, rather than by naming a library sequence (see `make_y`). This is
    the building block an optimizer would use to evaluate an arbitrary,
    numerically-searched pulse timing rather than one of the fixed CPMG/CDD
    families; as of this writing it is exercised only by
    ``tests/test_trajectories.py`` (no pipeline script currently calls it),
    but it is kept as the general-purpose entry point `f`'s switch-time
    interface is built around.

    Parameters
    ----------
    vt : list of jax.Array
        Switch times for [qubit1, qubit2], in the same "include the block's
        start and end time" format `f` expects for `tk`.
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
    # JAX arrays are immutable (no in-place `y[0, 0] = ftn` like numpy); the
    # `.at[idx].set(value)` idiom instead returns a NEW array equal to `y`
    # except at `idx`, which is why the result is reassigned back to `y`.
    y = y.at[0, 0].set(ftn)
    ftn = f(t_b, vt[1])
    y = y.at[1, 1].set(ftn)
    y = y.at[2, 2].set(jnp.multiply(y[1, 1], y[0, 0]))
    return jnp.tile(y, M)


@jax.jit
def make_propagator(H_t, t_vec):
    """
    Calculate the time-evolution propagator for the (time-dependent)
    dephasing Hamiltonian.

    Because `make_Hamiltonian` only ever builds Z-type (diagonal) terms, H(t)
    at any two different times commutes with itself ([H(t1), H(t2)] = 0), so
    the time-ordered Schrodinger-equation propagator has the closed form
    ``U = exp(-i * integral H(t) dt)`` with no ODE solve or matrix
    exponential of a non-diagonal matrix needed: it is enough to integrate
    each diagonal entry of H(t) separately (trapezoid rule) and exponentiate
    the resulting phases. This is the key simplification that makes the
    dephasing-only model in this file cheap to simulate at scale.

    Parameters
    ----------
    H_t : jax.Array
        Time-dependent Hamiltonian, shape (time_steps, 8, 8), as built by
        `make_Hamiltonian`.
    t_vec : jax.Array
        Time vector.

    Returns
    -------
    jax.Array
        Unitary propagator (8x8, diagonal).
    """
    h_diags = jnp.diagonal(H_t, axis1=1, axis2=2)
    phi = -1j * jax.scipy.integrate.trapezoid(h_diags, t_vec, axis=0)
    return jnp.diag(jnp.exp(phi))


@jax.jit
def single_shot_prop(noise_mats, t_vec, y_uv, rho0, key):
    """
    Simulate one Monte Carlo "shot": draw one noise trajectory realization,
    build the resulting Hamiltonian and propagator, and evolve the initial
    state through it. `solver_prop` calls this (via `jax.vmap`) once per shot
    and averages the results over many shots to get the ensemble-averaged
    final state that mimics what a real (noisy, repeated) experiment would
    measure.

    Parameters
    ----------
    noise_mats : jax.Array
        Precomputed noise matrices (from `make_noise_mat_arr`).
    t_vec : jax.Array
        Time vector.
    y_uv : jax.Array
        Control matrix (from `make_y`/`custom_y`).
    rho0 : jax.Array
        Initial state density matrix (8x8: 2 qubits + bath).
    key : jax.Array
        A pair of integers seeding this shot's noise draw (see
        `make_channel_trajs`) -- not a literal `jax.random.PRNGKey` object.

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
        # Ordinary (non-JAX) `np.random` draws the two integer seeds that
        # become each shot's `key` pair (see `make_channel_trajs`); this is
        # deliberately plain numpy, not jax.random splitting, so that calling
        # this in a loop consumes numpy's global RNG stream exactly once per
        # shot in shot order -- which is what makes it possible for a
        # recorded run and a live run to draw bit-identical noise (both just
        # advance the same global stream the same number of times).
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
    # jnp.einsum names each array's axes with a letter and says which letters
    # survive in the output: here 'at,cstw->acsw' takes ys (axes a=channel,
    # t=time) and mats (axes c=component, s=sin/cos, t=time, w=frequency),
    # multiplies them elementwise over the SHARED `t` axis, sums it away (it
    # does not appear on the right of '->'), and keeps a, c, s, w -- i.e. it
    # is a compact way to write "integrate over time for every (channel,
    # component, sin/cos, frequency) combination at once" without an explicit
    # Python loop.
    return jnp.einsum('at,cstw->acsw', 0.5*ys*wt[None, :], mats)


@jax.jit
def _shot_coeffs_from_filters(F, key):
    """One shot's (3,) phase coefficients from precomputed filter vectors.

    Identical RNG scheme to `make_channel_trajs` (same fold_in/split and the
    same per-stream normal draws), so the same key yields the same noise
    realization as the trajectory-level path, up to float reassociation."""
    # Same static-shape branch as make_channel_trajs: component axis 1 of F is
    # the stream axis, so a 6-stream (showcase) run carries the coupler-defect
    # filter at index 5 and an 8-stream run adds the shared-carrier filters at
    # 6/7; the 5-stream path keeps the legacy key budget.
    n_comp = jnp.size(F, 1)
    has_zz = (n_comp >= 6)
    has_qs = (n_comp == 8)
    base = jax.random.fold_in(jax.random.PRNGKey(key[0]), key[1])
    ks = jax.random.split(base, 18 if has_qs else (12 if has_zz else 10))
    n_w = jnp.size(F, 3)
    draw = lambda k: jax.random.normal(k, (n_w, 1))[:, 0]

    def comp(a, c, ka, kb):
        return jnp.dot(F[a, c, 0], draw(ka)) + jnp.dot(F[a, c, 1], draw(kb))

    # streams: ks[0:2] shared, ks[2:4] local-A, ks[4:6] local-B, ks[6:8] n1,
    # ks[8:10] n2 (+ ks[10:12] coupler defect; + ks[12:14] common carrier,
    # ks[14:16] local carrier 1, ks[16:18] local carrier 2 when present);
    # components: 0 = el_A, 1 = el_B shifted, 2 = el_B, 3 = n1, 4 = n2
    # (+ 5 = zz_extra; + 6/7 = carrier filters) -- make_noise_mat_arr order.
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
    if has_zz:
        c12 = c12 + comp(2, 5, ks[10], ks[11])
    if has_qs:
        c1 = c1 + _QS_SH*comp(0, 6, ks[12], ks[13]) \
            + _QS_LOC*comp(0, 6, ks[14], ks[15])
        c2 = c2 + _QS_SH*comp(1, 7, ks[12], ks[13]) \
            + _QS_LOC*comp(1, 7, ks[16], ks[17])
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
    """Lightweight stand-in for a (n_shots, 8, 8) density-matrix stack: stores
    only the per-shot diagonal-propagator phases ``u`` (n_shots, 8) plus the
    single shared preparation ``rho`` (8, 8), instead of materializing the
    full ``U_s rho U_s^dag`` matrix (U_s diagonal) for every shot.

    This works only because the model here is pure dephasing: since every
    propagator U_s is diagonal (see the ``_DIAG_BASIS`` comment above), the
    evolved state ``U_s rho U_s^dag`` is fully determined by `rho` and the 8
    phase entries of `U_s` -- so storing `u` and `rho` separately is EXACT,
    not an approximation, and is roughly 24x lighter in memory than the dense
    stack. Consumers never reconstruct the dense stack either:
    `observables.compute_probs_jax` (in `model/observables.py`) consumes a
    `PhasedState` through an exact quadratic-form fast path,
    ``probs = u^dag [ (G^dag M G)^T o rho ] u``, where `G`/`M` are the POVM
    and pulse-rotation operators defined in that module.
    """

    def __init__(self, u, rho):
        self.u = u
        self.rho = rho

    @property
    def shape(self):
        return (self.u.shape[0], 8, 8)


def phased_state(coeffs, rho):
    """Build a `PhasedState` from stored per-shot phase coefficients `coeffs`
    (n_shots, 3) and a shared preparation `rho` (8, 8), by exponentiating the
    phase coefficients into the diagonal `_DIAG_BASIS` (Z1, Z2, Z1Z2) to get
    each shot's diagonal propagator phases `u`."""
    u = jnp.exp(-1j*jnp.matmul(jnp.asarray(coeffs), _DIAG_BASIS))
    return PhasedState(u, jnp.asarray(rho))


def fast_solver(y_uv, noise_mats, t_vec, rho, n_shots):
    """Drop-in for ``solver_prop`` on the filter-vector fast path: returns a
    ``PhasedState`` (exact for this diagonal-propagator dephasing model, for ANY
    ``rho``) at ~1000x less per-shot compute. Unlike ``PhaseRecorder``
    (``characterize/experiments.py``, the object that saves a noise dataset to
    disk for later replay) it stores nothing to disk -- it lets an arm whose
    suite cannot replay the recorded non-robust dataset (the SPAM-robust D^+-
    estimators) run on the fast path WITHOUT the record/replay machinery, so
    the dense ``solver_prop`` is not needed there.
    (Downstream estimators consume the output only through
    ``observables.compute_probs_jax``, which has an exact PhasedState fast path.)"""
    coeffs = solver_phase_coeffs_fast(y_uv, noise_mats, t_vec, n_shots)
    return phased_state(coeffs, rho)


@jax.jit
def apply_phase_coeffs(coeffs, rho):
    """Replay a previously-recorded noise dataset against a NEW initial
    state `rho`, by evolving `rho` through the stored per-shot phase
    coefficients `coeffs`.

    Returns the (n_shots, 8, 8) stack of U_s rho U_s^dag with the diagonal
    U_s = exp(-i coeffs_s . _DIAG_BASIS) -- the exact replay of what
    `solver_prop` would have produced for these noise realizations, for ANY
    initial state (the SPAM-protocol arms differ only in rho and in estimator
    post-processing, so one recorded dataset serves them all). The
    broadcasting pattern below -- ``u[:, :, None] * rho[None, :, :] *
    conj(u)[:, None, :]`` -- computes U_s rho U_s^dag WITHOUT ever forming the
    (8, 8) matrix U_s explicitly: because U_s is diagonal, matrix
    multiplication reduces to the elementwise product
    ``(U_s rho U_s^dag)[i, j] = u[i] * rho[i, j] * conj(u[j])``, which is what
    the three broadcast axes above implement for every shot at once."""
    p = jnp.matmul(coeffs, _DIAG_BASIS)            # (n_shots, 8)
    u = jnp.exp(-1j*p)
    return u[:, :, None]*rho[None, :, :]*jnp.conj(u)[:, None, :]


def solver_prop(y_uv, noise_mats, t_vec, rho, n_shots):
    """
    Run the Monte Carlo dephasing simulation over many noise shots and return
    the FULL per-shot batch of final density matrices (NOT yet averaged --
    despite the name, this is the ensemble of individual noisy outcomes, one
    per random noise realization). Averaging over shots happens later,
    downstream, at the point where an observable is computed from this batch
    (e.g. ``observables.compute_probs_jax`` takes the per-shot outcome
    probabilities and only then averages them over the shot axis with
    ``jnp.mean(..., axis=0)``) -- keeping the per-shot batch around lets
    different observables/estimators combine the same shots differently
    without re-running the simulation.

    Internally this is just `single_shot_prop` called once per shot via
    `jax.vmap` (vectorized over a fresh random `key` per shot), chunked into
    slices to bound GPU memory (see the `slice_size` note below).

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
        Batch of final density matrices, shape (n_shots, 8, 8) -- one per
        noise realization, not averaged.
    """
    y_uv = jnp.array(y_uv)
    output = []
    # Memory allocation safety for my laptop with a single GPU: `jax.vmap`
    # below runs all `n_shots` shots as one batched (parallel) computation,
    # which for large `n_shots` would try to allocate the entire
    # (n_shots, 8, 8) result (plus intermediates) on the GPU at once and run
    # out of memory. Splitting `n_shots` into chunks of `slice_size` and
    # vmapping (then concatenating) each chunk separately bounds the peak
    # memory to one chunk's worth, at the cost of a Python-level loop over
    # chunks. CLAUDE.md notes this value may need adjusting on different
    # hardware (more GPU memory -> larger slice_size -> less loop overhead).
    slice_size = 2000
    n_slices = int(np.ceil(n_shots / slice_size))
    for i in range(n_slices):
        current_slice_size = min(slice_size, n_shots - i * slice_size)
        n_arr = jnp.array(np.random.randint(0, 10000, (current_slice_size, 2)))
        result = jax.vmap(single_shot_prop, in_axes=[None, None, None, None, 0])(noise_mats, t_vec, y_uv, rho, n_arr)
        output.append(result)
    return jnp.concatenate(output, axis=0)

