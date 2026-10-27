"""Exact verification of the spectator-trace / locality formulas in the paper.

Self-contained (NumPy only; does not import the QNS-2Q gate model) brute-force
check of the reduced-process-fidelity results in App. "Embedding the active pair
in an N-qubit register" of main_v10.tex.

Model: N-qubit register, classical zero-mean Gaussian dephasing with generators
Z_a (all single Z_q and two-body Z_iZ_j); accumulated phases varphi_a jointly
Gaussian with <varphi_a varphi_b> = 2 I_ab.  The averaged channel acts elementwise
in the computational basis,

    E(rho)_xy = rho_xy * W_xy,   W_xy = exp[-(z(x)-z(y))^T I (z(x)-z(y))],

with z_a(x) = +-1/2 the eigenvalue of Z_a/2 on basis state x (the Z_a/2 coupling
fixes the single-Pauli decay to 1-E ~ I_aa, the paper's overlap normalization).

Pair = qubits {0,1} (l,l'); spectators = qubits {2..N-1}.

What is asserted:
  * exact reduction      D_sigma = Tr_spec[E(. (x) sigma)]  ==  conditional-channel mixture
                         sum_z p(z) D_z (Eq. convex_mix; per-config cond_gens/cond_cumulant), and ==
                         PTM char-sum  sum_Q s_Q E_{(.,1),(.,Q)}  (Eq. pop_only).    [L1-L3]
  * coherence independence: F_pro depends only on the Z-diagonal part of sigma.   [L1]
  * Jensen split         1-F = G(Ibar) + Delta   (auxiliary 2nd-order decomposition; the
                         paper's exact form is Eq. infid_exact) reproduces the cond. average.
  * order of accuracy    O(I) locality (Eq. locality) residual ~ lam^2;
                         O(I^2) Jensen residual ~ lam^3  ->  this is the regression
                         guard on the cross-overlap coefficient: it is -1/2
                         (Eq. secondorder_infid / infid_exact).  The earlier -3/4
                         leaves an O(lam^2) error and fails the lam^3 test.
  * closed form          N=2 bare pair matches Eq. infid_exact to machine precision.

Run:  pytest tests/test_spectator_locality.py
"""
import numpy as np
import itertools as it
import pytest

# Cross-overlap coefficient in the second-order infidelity, Eq. (secondorder_infid).
# CORRECT value is -1/2 (verified below); -3/4 was an erratum corrected 2026-06-19.
XCOEF = -0.5

s1 = {'I': np.eye(2, dtype=complex),
      'X': np.array([[0, 1], [1, 0]], dtype=complex),
      'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
      'Z': np.array([[1, 0], [0, -1]], dtype=complex)}


def kron(ms):
    o = np.array([[1]], dtype=complex)
    for m in ms:
        o = np.kron(o, m)
    return o


def bit(x, q, N):
    return (x >> (N - 1 - q)) & 1


PAIR = [kron([s1[a], s1[b]]) for a in 'IXYZ' for b in 'IXYZ']   # 16 pair Paulis
SPEC1 = ['I', 'X', 'Y', 'Z']


def channels(N):
    ch = [frozenset([q]) for q in range(N)]
    ch += [frozenset([i, j]) for i in range(N) for j in range(i + 1, N)]
    return ch


def zeta_tab(ch, N):
    dim = 1 << N
    Z = np.zeros((len(ch), dim))
    for a, sup in enumerate(ch):
        for x in range(dim):
            p = 0
            for q in sup:
                p ^= bit(x, q, N)
            Z[a, x] = -0.5 if p else 0.5
    return Z


def decay_matrix(ch, Iov, N):
    dim = 1 << N
    Z = zeta_tab(ch, N)
    W = np.zeros((dim, dim))
    for x in range(dim):
        for y in range(dim):
            d = Z[:, x] - Z[:, y]
            W[x, y] = np.exp(-(d @ Iov @ d))
    return W


def ptrace_spec(M, N):
    D = 1 << (N - 2)
    return np.einsum('psqs->pq', M.reshape(4, D, 4, D))


def reduced_ptm_exact(W, sigma, N):
    Dm = np.zeros((16, 16), dtype=complex)
    for nu, Pn in enumerate(PAIR):
        ro = W * np.kron(Pn, sigma)
        rp = ptrace_spec(ro, N)
        for mu, Pm in enumerate(PAIR):
            Dm[mu, nu] = 0.25 * np.trace(Pm @ rp)
    return Dm


# ---- spectator state (z-diagonal: sigma = diag(p)) ----
def sgn(c, k, N):
    j = k - 2
    b = (c >> (N - 2 - 1 - j)) & 1
    return -1.0 if b else 1.0


def moment(p, S, N):
    m = 0.0
    for c in range(len(p)):
        v = 1.0
        for k in S:
            v *= sgn(c, k, N)
        m += p[c] * v
    return m


def make_idx(ch):
    return {frozenset(x): i for i, x in enumerate(ch)}


def IV(Iov, idx, A, B):
    a, b = frozenset(A), frozenset(B)
    if a not in idx or b not in idx:
        return 0.0
    return Iov[idx[a], idx[b]]


def dvec(lab, c, ch, idx, N):
    v = np.zeros(len(ch))
    spec = list(range(2, N))
    if lab == 'l':
        v[idx[frozenset([0])]] = 1.0
        for k in spec:
            v[idx[frozenset([0, k])]] = sgn(c, k, N)
    elif lab == 'lp':
        v[idx[frozenset([1])]] = 1.0
        for k in spec:
            v[idx[frozenset([1, k])]] = sgn(c, k, N)
    elif lab == 'llp':
        v[idx[frozenset([0, 1])]] = 1.0
    return v


def tI(a, b, c, Iov, ch, idx, N):
    return dvec(a, c, ch, idx, N) @ Iov @ dvec(b, c, ch, idx, N)


def Gfun(Ill, Ilplp, Iz, Illp, Ilz, Ilpz):
    """Second-order process infidelity, Eq. (secondorder_infid)."""
    return (0.5 * (Ill + Ilplp + Iz)
            - 0.125 * ((Ill + Ilplp) ** 2 + (Ilplp + Iz) ** 2 + (Ill + Iz) ** 2)
            + XCOEF * (Illp ** 2 + Ilz ** 2 + Ilpz ** 2))


def reduced_ptm_cond(Iov, ch, idx, p, N):
    """Conditional-channel mixture:  sum_z p(z) D_z  (Eq. convex_mix)."""
    labs = ['l', 'lp', 'llp']
    Dred = np.zeros((16, 16), dtype=complex)

    def xi(lab, x):
        b0, b1 = (x >> 1) & 1, x & 1
        return {'l': -0.5 if b0 else 0.5, 'lp': -0.5 if b1 else 0.5,
                'llp': -0.5 if (b0 ^ b1) else 0.5}[lab]

    for c in range(len(p)):
        t = {(A, B): tI(A, B, c, Iov, ch, idx, N) for A in labs for B in labs}
        Wz = np.zeros((4, 4))
        for x in range(4):
            for y in range(4):
                Wz[x, y] = np.exp(-sum((xi(A, x) - xi(A, y)) * (xi(B, x) - xi(B, y)) * t[(A, B)]
                                       for A in labs for B in labs))
        for nu, Pn in enumerate(PAIR):
            ro = Wz * Pn
            for mu, Pm in enumerate(PAIR):
                Dred[mu, nu] += p[c] * 0.25 * np.trace(Pm @ ro)
    return Dred


def reduced_ptm_eq1(W, sigma, N):
    """PTM char-sum reduction:  [D]_{mu,nu} = sum_Q s_Q E_{(mu,1),(nu,Q)}  (Eq. pop_only)."""
    spec = N - 2
    Dm = np.zeros((16, 16), dtype=complex)
    Qlist = ([(lab, kron([s1[ch] for ch in lab])) for lab in it.product(SPEC1, repeat=spec)]
             if spec > 0 else [((), np.array([[1]], dtype=complex))])
    Id_spec = np.eye(1 << spec, dtype=complex)
    sQ = {lab: np.trace(Q @ sigma) for lab, Q in Qlist}
    for nu, Pn in enumerate(PAIR):
        for mu, Pm in enumerate(PAIR):
            acc = 0.0
            for lab, Q in Qlist:
                out = W * np.kron(Pn, Q)
                acc += sQ[lab] * np.trace(np.kron(Pm, Id_spec) @ out) / (1 << N)
            Dm[mu, nu] = acc
    return Dm


def cov_master(ab, cd, Iov, ch, idx, p, N):
    """Cov_z(tilde I_ab, tilde I_cd) via the master formula (Jensen split, auxiliary; paper exact form Eq. infid_exact)."""
    spec = list(range(2, N))

    def W(lab, S):
        a, b = lab
        sup = {'l': [0], 'lp': [1], 'llp': [0, 1]}
        A, B = sup[a], sup[b]
        if len(S) == 1:
            k = S[0]
            return IV(Iov, idx, A, B + [k]) + IV(Iov, idx, B, A + [k])
        k, kp = S
        return IV(Iov, idx, A + [k], B + [kp]) + IV(Iov, idx, A + [kp], B + [k])

    Ss = [(k,) for k in spec] + [(k, kp) for k in spec for kp in spec if k < kp]
    tot = 0.0
    for S in Ss:
        for Sp in Ss:
            sym = tuple(sorted(set(S) ^ set(Sp)))
            tot += W(ab, S) * W(cd, Sp) * (moment(p, sym, N) - moment(p, S, N) * moment(p, Sp, N))
    return tot


def o2_jensen(Iov, ch, idx, p, N):
    """G(Ibar) + Delta (Jensen split, auxiliary; paper exact form Eq. infid_exact at Eq. dressed_decomp)."""
    labs = {'ll': ('l', 'l'), 'lplp': ('lp', 'lp'), 'llp': ('l', 'lp'),
            'l_z': ('l', 'llp'), 'lp_z': ('lp', 'llp'), 'z': ('llp', 'llp')}
    bar = {k: sum(p[c] * tI(*lab, c, Iov, ch, idx, N) for c in range(len(p)))
           for k, lab in labs.items()}
    Gmean = Gfun(bar['ll'], bar['lplp'], bar['z'], bar['llp'], bar['l_z'], bar['lp_z'])
    Vll = cov_master(('l', 'l'), ('l', 'l'), Iov, ch, idx, p, N)
    Vlp = cov_master(('lp', 'lp'), ('lp', 'lp'), Iov, ch, idx, p, N)
    Cllp = cov_master(('l', 'l'), ('lp', 'lp'), Iov, ch, idx, p, N)
    Vx = cov_master(('l', 'lp'), ('l', 'lp'), Iov, ch, idx, p, N)
    Vlz = cov_master(('l', 'llp'), ('l', 'llp'), Iov, ch, idx, p, N)
    Vlpz = cov_master(('lp', 'llp'), ('lp', 'llp'), Iov, ch, idx, p, N)
    # Hessian of G: self-block -> -1/4; cross-block -> XCOEF (= -1/2).
    Delta = -0.25 * (Vll + Vlp + Cllp) + XCOEF * (Vx + Vlz + Vlpz)
    return Gmean + Delta


def o2_conditional(Iov, ch, idx, p, N):
    labs = ['l', 'lp', 'llp']
    val = 0.0
    for c in range(len(p)):
        t = {a: {b: tI(a, b, c, Iov, ch, idx, N) for b in labs} for a in labs}
        val += p[c] * Gfun(t['l']['l'], t['lp']['lp'], t['llp']['llp'],
                           t['l']['lp'], t['l']['llp'], t['lp']['llp'])
    return val


def o1_value(Iov, ch, idx, p, N):
    """Leading-order locality, Eq. (locality)/(infid_diagonal)."""
    barll = sum(p[c] * tI('l', 'l', c, Iov, ch, idx, N) for c in range(len(p)))
    barlp = sum(p[c] * tI('lp', 'lp', c, Iov, ch, idx, N) for c in range(len(p)))
    return 0.5 * (barll + barlp + IV(Iov, idx, [0, 1], [0, 1]))


def _setup(N, seed=1, state='correlated'):
    rng = np.random.default_rng(seed)
    ch = channels(N)
    idx = make_idx(ch)
    nc = len(ch)
    A = rng.standard_normal((nc, nc))
    base = A @ A.T / nc                      # PSD overlap matrix
    Dsp = 1 << (N - 2)
    if N == 2:
        p = np.array([1.0])
    elif state == 'product':
        qs = rng.random(N - 2)
        p = np.ones(Dsp)
        for c in range(Dsp):
            for j in range(N - 2):
                b = (c >> (N - 2 - 1 - j)) & 1
                p[c] *= qs[j] if b == 0 else 1 - qs[j]
    else:
        p = rng.dirichlet(np.ones(Dsp))      # generic correlated spectator state
    return ch, idx, p, np.diag(p).astype(complex), base


# ----------------------------- tests -----------------------------
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_exact_reduction(N):
    """L3 (mixture, Eq. convex_mix) and PTM char-sum (Eq. pop_only) reductions equal the brute-force trace."""
    ch, idx, p, sigma, base = _setup(N)
    W = decay_matrix(ch, base, N)
    Dex = reduced_ptm_exact(W, sigma, N)
    assert np.abs(Dex - reduced_ptm_cond(base, ch, idx, p, N)).max() < 1e-10
    assert np.abs(Dex - reduced_ptm_eq1(W, sigma, N)).max() < 1e-10


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_coherence_independence(N):
    """L1: F_pro depends only on the Z-diagonal part of the spectator state."""
    ch, idx, p, sigma, base = _setup(N)
    W = decay_matrix(ch, base, N)
    Dsp = 1 << (N - 2)
    rng = np.random.default_rng(7)
    H = rng.standard_normal((Dsp, Dsp)) + 1j * rng.standard_normal((Dsp, Dsp))
    H = H + H.conj().T
    np.fill_diagonal(H, 0.0)               # pure coherences, same populations
    sig_coh = sigma + 1e-3 * H
    F_diag = np.trace(reduced_ptm_exact(W, sigma, N)).real / 16
    F_coh = np.trace(reduced_ptm_exact(W, sig_coh, N)).real / 16
    assert abs(F_coh - F_diag) < 1e-10


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_jensen_identity(N):
    """Jensen split G(Ibar)+Delta equals the conditional average sum_z p(z) G(z)."""
    ch, idx, p, sigma, base = _setup(N)
    assert abs(o2_jensen(base, ch, idx, p, N) - o2_conditional(base, ch, idx, p, N)) < 1e-12


@pytest.mark.parametrize("N,state", [(2, 'correlated'), (3, 'correlated'), (4, 'correlated'),
                                     (5, 'correlated'), (6, 'correlated'), (4, 'product')])
def test_perturbative_order(N, state):
    """O(I) residual ~ lam^2 (ratio ~4); O(I^2) residual ~ lam^3 (ratio ~8).

    The lam^3 scaling is the regression guard on XCOEF = -1/2: the erratum -3/4
    leaves an O(lam^2) error and drives the O(I^2) ratio toward 4.
    """
    ch, idx, p, sigma, base = _setup(N, state=state)

    def residuals(lam):
        Iov = base * lam
        W = decay_matrix(ch, Iov, N)
        infid = 1 - np.trace(reduced_ptm_exact(W, sigma, N)).real / 16
        return (abs(infid - o1_value(Iov, ch, idx, p, N)),
                abs(infid - o2_conditional(Iov, ch, idx, p, N)))
    r1a, r2a = residuals(0.01)
    r1b, r2b = residuals(0.005)
    assert 3.5 < r1a / r1b < 4.5          # O(I) error is O(lam^2): ratio ~4
    assert r2a / r2b > 6.0                # O(I^2) error is O(lam^3): ratio ~8 (erratum -3/4 -> ~4)


def test_closed_form_bare_pair():
    """N=2 bare pair matches the all-orders closed form, Eq. (infid_exact)."""
    ch, idx, p, sigma, base = _setup(2)
    for lam in (0.05, 0.2, 0.6):
        Iov = base * lam
        infid = 1 - np.trace(reduced_ptm_exact(decay_matrix(ch, Iov, 2), sigma, 2)).real / 16
        I11, I22, I33 = IV(Iov, idx, [0], [0]), IV(Iov, idx, [1], [1]), IV(Iov, idx, [0, 1], [0, 1])
        I1_12, I2_12, I12 = (IV(Iov, idx, [0], [0, 1]), IV(Iov, idx, [1], [0, 1]), IV(Iov, idx, [0], [1]))
        closed = 0.25 * (3 - np.exp(-(I11 + I33)) * np.cosh(2 * I1_12)
                         - np.exp(-(I22 + I33)) * np.cosh(2 * I2_12)
                         - np.exp(-(I11 + I22)) * np.cosh(2 * I12))
        assert abs(infid - closed) < 1e-12


def test_cross_coefficient_is_one_half():
    """Single-cross limit: 1-F = (1/4)(1 - cosh 2 I_{1,2}) = -1/2 I_{1,2}^2 + O(I^4).

    Pins the cross-overlap coefficient to -1/2 (Eq. secondorder_infid); -3/4 fails.
    """
    I12 = 1e-3
    infid = 0.25 * (1 - np.cosh(2 * I12))          # exact bare-pair, only I_{1,2} != 0
    assert abs(infid - XCOEF * I12 ** 2) < 1e-12   # XCOEF = -1/2
    assert abs(infid - (-0.75) * I12 ** 2) > 1e-8  # the erratum would be wrong here


# ------- worst-case-over-spectator-states certificate (Eq. worstcase) -------
def _reduced_infid(W, sigma, N):
    D = reduced_ptm_exact(W, np.asarray(sigma, dtype=complex), N)
    return 1.0 - np.trace(D).real / 16.0


@pytest.mark.parametrize("N", [3, 4])
def test_infidelity_linear_in_populations(N):
    """1-F(sigma) = sum_z p(z)(1-F_z): the reduced infidelity is linear in the
    spectator computational populations (coherences drop). This convex
    decomposition is what makes the worst spectator state a computational config.
    """
    ch, idx, p, sigma, base = _setup(N)
    W = decay_matrix(ch, base, N)
    Dsp = 1 << (N - 2)
    rng = np.random.default_rng(11)
    H = rng.standard_normal((Dsp, Dsp)) + 1j * rng.standard_normal((Dsp, Dsp))
    H = H + H.conj().T
    np.fill_diagonal(H, 0.0)                          # add coherences, keep populations p
    sig = np.diag(p).astype(complex) + 1e-2 * H
    lhs = _reduced_infid(W, sig, N)
    rhs = sum(p[z] * _reduced_infid(W, np.diag(np.eye(Dsp)[z]).astype(complex), N)
              for z in range(Dsp))
    assert abs(lhs - rhs) < 1e-10


@pytest.mark.parametrize("N", [3, 4])
def test_worst_state_is_a_config(N):
    """Eq. (worstcase): the maximum reduced infidelity over all spectator states is
    attained at a computational configuration. No sampled state -- pure, mixed,
    correlated, or coherent -- exceeds the worst config.
    """
    ch, idx, p, sigma, base = _setup(N)
    W = decay_matrix(ch, base, N)
    Dsp = 1 << (N - 2)
    config_max = max(_reduced_infid(W, np.diag(np.eye(Dsp)[z]).astype(complex), N)
                     for z in range(Dsp))
    rng = np.random.default_rng(5)
    sampled_max = 0.0
    for _ in range(300):
        A = rng.standard_normal((Dsp, Dsp)) + 1j * rng.standard_normal((Dsp, Dsp))
        rho = A @ A.conj().T
        rho /= np.trace(rho).real                     # random mixed spectator state
        sampled_max = max(sampled_max, _reduced_infid(W, rho, N))
    assert sampled_max <= config_max + 1e-10
