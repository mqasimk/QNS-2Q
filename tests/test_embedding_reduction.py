"""Exact verification of the App E.5 register->pair reduction cascade (V10-EMBED-REDUX-0629).

Self-contained (NumPy only) regression guards for main_v10.tex App "Embedding the
active pair in an N-qubit register" (App. E.5), complementing the locality
formulas already pinned in test_spectator_locality.py.  Adds the four checks that
test lacks:

  (A) NON-GAUSSIAN generality of the reduction + PTM character sum (cf. eq::pop_only):
      [D_z]_mm = sum_R (prod_{k in R} s_k) D_{(m,1),(m,R)} holds for an arbitrary
      classical dephasing channel (a finite ensemble of diagonal unitaries), since
      it uses only the computational-diagonal structure -- exact at all orders.
  (B) the operational-overlap FOLD  I~_{l,l} = I_{l,l} + sum_k I_{lk,lk}  (eq::operationalI).
  (C) the SINGLE-ELEMENT DEFECT: the bare element D_{(m,1),(m,1)} is only the R=0
      (unpolarized) term of the conditional diagonal [D_z]_mm; the polarization
      defect is nonzero exactly when the pair-spectator cross I_{l,lk} != 0.  This
      guards the imprecision the E.5 rewrite corrected (the old line asserted
      [D_z]_mm = D_{(m,1),(m,1)} unconditionally).
  (D) the same identities at the REAL featured-model pair overlaps (skipped if jax
      / DraftRun_NoSPAM_featured data are unavailable).

Convention matches test_spectator_locality.py: generators Z_a (all single Z_q and
two-body Z_iZ_j); the channel acts elementwise, E(rho)_xy = rho_xy * W_xy, with the
Gaussian factor W_xy = exp[-(z(x)-z(y))^T I (z(x)-z(y))], z_a = +-1/2 the eigenvalue
of Z_a/2 (so the single-Pauli decay is 1 - E ~ I_aa).  Pair = {0,1}; spectators = {2..}.
"""
import itertools

import numpy as np
import pytest

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]


def _kron(ops):
    out = np.array([[1]], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def _gens(N):
    return [(i,) for i in range(N)] + [(i, j) for i in range(N) for j in range(i + 1, N)]


def _z_table(N):
    gens = _gens(N)
    Zt = np.zeros((2 ** N, len(gens)))
    for x in range(2 ** N):
        s = [1 - 2 * ((x >> (N - 1 - q)) & 1) for q in range(N)]
        for a, g in enumerate(gens):
            Zt[x, a] = 0.5 * np.prod([s[q] for q in g])
    return Zt


def _gaussian_factor(Imat, Zt):
    d = Zt[:, None, :] - Zt[None, :, :]
    return np.exp(-np.einsum("uva,ab,uvb->uv", d, Imat, d))


def _ensemble_factor(N, seed, K=6):
    """Non-Gaussian: W_xy = <exp(-i (z(x)-z(y)).eta)> over arbitrary phase draws."""
    rng = np.random.default_rng(seed)
    eta = rng.uniform(-0.9, 0.9, size=(K, len(_gens(N))))
    w = rng.uniform(0.2, 1.0, size=K)
    w /= w.sum()
    d = _z_table(N)[:, None, :] - _z_table(N)[None, :, :]
    return np.einsum("r,ruv->uv", w, np.exp(-1j * np.einsum("uva,ra->ruv", d, eta)))


def _two_qubit_paulis():
    return [_kron([PAULI[a], PAULI[b]]) for a in range(4) for b in range(4)]


def _random_overlap(N, seed, scale=0.18, cross=True):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((len(_gens(N)), len(_gens(N))))
    S = A @ A.T
    S *= scale / np.mean(np.diag(S))
    return np.diag(np.diag(S)) if not cross else S


def _layout(N, pair=(0, 1)):
    l, lp = pair
    spec = [q for q in range(N) if q not in pair]
    pair_states = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def full_index(xbits, sbits):
        b = [0] * N
        b[l], b[lp] = xbits
        for i, k in enumerate(spec):
            b[k] = sbits[i]
        u = 0
        for i in range(N):
            u = (u << 1) | b[i]
        return u

    def pair_op_full(mu, spec_op):
        ops = [I2] * N
        ops[l], ops[lp] = PAULI[mu[0]], PAULI[mu[1]]
        for k in spec:
            ops[k] = spec_op[k]
        return _kron(ops)

    return l, lp, spec, pair_states, full_index, pair_op_full


def _reduced_block(W, bits, layout):
    _, _, _, pair_states, full_index, _ = layout
    D = np.zeros((4, 4), dtype=complex)
    for x, xb in enumerate(pair_states):
        for xp, xpb in enumerate(pair_states):
            D[x, xp] = W[full_index(xb, bits), full_index(xpb, bits)]
    return D


def _conditional_diag(W, bits, P2, mi, layout):
    D = _reduced_block(W, bits, layout)
    return np.trace(P2[mi] @ (D * P2[mi])) / 4.0


def _char_sum(W, N, bits, mu, layout, subsets):
    _, _, spec, _, _, pair_op_full = layout
    s = {spec[i]: 1 - 2 * bits[i] for i in range(len(spec))}
    out = pair_op_full(mu, {k: I2 for k in spec})
    total = 0.0
    for R in subsets:
        sop = {k: (Z if k in R else I2) for k in spec}
        Dfull = np.trace(out @ (W * pair_op_full(mu, sop))) / 2 ** N
        total += np.prod([s[k] for k in R]) * Dfull
    return total


def _reduction_charsum_err(W, N):
    """max | [D_z]_mm - sum_R (prod s) D_{(m,1),(m,R)} | over all Paulis & configs."""
    layout = _layout(N)
    spec = layout[2]
    P2 = _two_qubit_paulis()
    mu_list = [(a, b) for a in range(4) for b in range(4)]
    subsets = [c for r in range(len(spec) + 1) for c in itertools.combinations(spec, r)]
    err = 0.0
    for bits in itertools.product([0, 1], repeat=len(spec)):
        for mi, mu in enumerate(mu_list):
            err = max(err, abs(_conditional_diag(W, bits, P2, mi, layout)
                              - _char_sum(W, N, bits, mu, layout, subsets)))
    return err


def _fold_err(Imat, N, pair=(0, 1)):
    l, lp = pair
    spec = [q for q in range(N) if q not in pair]
    gens = _gens(N)
    gi = {g: a for a, g in enumerate(gens)}

    def fold(a):
        return Imat[gi[(a,)], gi[(a,)]] + sum(
            Imat[gi[tuple(sorted((a, k)))], gi[tuple(sorted((a, k)))]] for k in spec)

    def Itil(a, s):
        c = np.zeros(len(gens))
        c[gi[(a,)]] = 1.0
        for i, k in enumerate(spec):
            c[gi[tuple(sorted((a, k)))]] = s[i]
        return c @ Imat @ c

    cfgs = list(itertools.product([1, -1], repeat=len(spec)))
    return max(abs(np.mean([Itil(a, s) for s in cfgs]) - fold(a)) for a in (l, lp))


def _single_element_defect(W, N):
    """max_mu max_z | [D_z]_mm - mean_z [D_z]_mm |  (the bare D_{(m,1),(m,1)})."""
    layout = _layout(N)
    spec = layout[2]
    P2 = _two_qubit_paulis()
    cfgs = list(itertools.product([0, 1], repeat=len(spec)))
    defect = 0.0
    for mi in range(16):
        vals = [_conditional_diag(W, bits, P2, mi, layout).real for bits in cfgs]
        defect = max(defect, max(abs(v - np.mean(vals)) for v in vals))
    return defect


def _gram_embed(M_real, N, pair=(0, 1),
                c=0.7, d=0.4, e=1.0, f=0.12, g=0.6, h=0.10, eta=0.30):
    l, lp = pair
    spec = [q for q in range(N) if q not in pair]
    gens = _gens(N)
    Lp = np.linalg.cholesky(M_real)
    width = 3 + 3 * len(spec) + N * N
    V = {(l,): np.r_[Lp[0], np.zeros(width - 3)],
         (lp,): np.r_[Lp[1], np.zeros(width - 3)],
         (l, lp): np.r_[Lp[2], np.zeros(width - 3)]}
    nxt = [3]

    def fresh():
        v = np.zeros(width)
        v[nxt[0]] = 1.0
        nxt[0] += 1
        return v

    for s in spec:
        V[(s,)] = c * V[(lp,)] + d * fresh()
        V[tuple(sorted((l, s)))] = e * V[(l, lp)] + f * fresh()
        V[tuple(sorted((lp, s)))] = g * V[(l, lp)] + h * fresh()
    for i, s in enumerate(spec):
        for s2 in spec[i + 1:]:
            V[tuple(sorted((s, s2)))] = eta * fresh()
    Vmat = np.array([V[gg] for gg in gens])
    return Vmat @ Vmat.T


# ----------------------------------------------------------------- tests ---- #

@pytest.mark.parametrize("N", [3, 4])
def test_reduction_charsum_gaussian(N):
    W = _gaussian_factor(_random_overlap(N, seed=N), _z_table(N))
    assert _reduction_charsum_err(W, N) < 1e-10


@pytest.mark.parametrize("N", [3, 4])
def test_reduction_charsum_nongaussian(N):
    # arbitrary diagonal-unitary ensemble: reduction + PTM char-sum (cf. eq::pop_only) are structure-only
    assert _reduction_charsum_err(_ensemble_factor(N, seed=10 + N), N) < 1e-10


@pytest.mark.parametrize("N", [3, 4])
def test_operational_overlap_fold(N):
    assert _fold_err(_random_overlap(N, seed=N), N) < 1e-12


def test_single_element_defect_is_polarization():
    # with the pair-spectator cross, the bare element is only the spectator average
    W = _gaussian_factor(_random_overlap(4, seed=4, cross=True), _z_table(4))
    assert _single_element_defect(W, 4) > 1e-3
    # remove every cross-overlap -> the single-element identity becomes exact
    W0 = _gaussian_factor(_random_overlap(4, seed=4, cross=False), _z_table(4))
    assert _single_element_defect(W0, 4) < 1e-12


def test_featured_model_overlaps_satisfy_identities():
    """Identities at the real featured-model pair overlaps (live from qns2q)."""
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    try:
        from qns2q.control.idle import (Config, prepare_time_domain_overlap,
                                        evaluate_overlap_folded)
        cfg = Config(fname="DraftRun_NoSPAM_featured", use_simulated=True, M=1)
        RMat, dt, n_base = prepare_time_domain_overlap(cfg.SMat_ideal, cfg.w_ideal,
                                                       cfg.tau, 320.0, 1)
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"featured-model data unavailable: {type(exc).__name__}: {exc}")
    pt0 = jnp.array([0.0, 320.0])
    M = np.array([[float(np.real(evaluate_overlap_folded(pt0, pt0, RMat[i, j], dt, n_base)))
                   for j in range(1, 4)] for i in range(1, 4)])
    for N in (3, 4):
        Imat = _gram_embed(M, N)
        W = _gaussian_factor(Imat, _z_table(N))
        assert _reduction_charsum_err(W, N) < 1e-10
        assert _fold_err(Imat, N) < 1e-12
        assert _single_element_defect(W, N) > 1e-3      # real pair-spectator cross present
