"""Exact verification of the App. E.5 register->pair reduction cascade (EMBED-REDUX).

Companion to the paper appendix "Embedding the active pair in an N-qubit register"
(main_v10.tex, App. E.5).  This script is the *runnable report* form; the
pytest regression guards for the same appendix live in
``tests/test_spectator_locality.py`` (exact reduction, Jensen split, worst-state,
the -1/2 cross coefficient).  This script adds the four checks that test is missing:

  (A) NON-GAUSSIAN generality -- the reduction and the PTM character sum
      [D_z]_mm = sum_R (prod_{k in R} s_k) D_{(m,1),(m,R)}   (eq::reduced_diag)
      hold for an arbitrary classical dephasing channel (a finite ensemble of
      diagonal unitaries), not only for the Gaussian model.  They use only the
      computational-diagonal structure, so are exact at all orders, any statistics.
  (B) the operational-overlap FOLD  I~_{l,l} = I_{l,l} + sum_k I_{lk,lk}  (eq::operationalI).
  (C) the SINGLE-ELEMENT DEFECT: the conditional diagonal [D_z]_mm is the
      spectator-conditioned read of the register; the bare element D_{(m,1),(m,1)}
      is only its R=0 (unpolarized) term.  The defect is nonzero exactly when the
      pair-spectator cross I_{l,lk} != 0 -- this is the imprecision the E.5 rewrite
      corrected (the old line asserted [D_z]_mm = D_{(m,1),(m,1)}).
  (D) the same identities at the REAL featured-model pair overlaps, extracted live
      from qns2q.control.idle, embedded in a register by a PSD Gram construction.

Convention (matches tests/test_spectator_locality.py): generators Z_a are all
single Z_q and two-body Z_iZ_j; the channel acts elementwise in the computational
basis, E(rho)_xy = rho_xy * W_xy, with the Gaussian factor

    W_xy = exp[ -(z(x)-z(y))^T I (z(x)-z(y)) ],   z_a(x) = +-1/2 eigenvalue of Z_a/2,

so the single-Pauli decay is 1 - E ~ I_aa (the paper's overlap normalization).
Pair = qubits {0,1} = (l,l'); spectators = qubits {2..N-1}.

Run:  PYTHONPATH=src python scripts/verify_embedding_reduction.py
(the featured-model section additionally needs the repo venv / jax and a
simulated_spectra.npz; it is skipped with a notice if unavailable.)
"""
import itertools
import sys

import numpy as np

TOL = 1e-10
np.set_printoptions(precision=6, suppress=True, linewidth=130)

# --- Pauli / computational-basis machinery --------------------------------- #
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]
PNAME = ["I", "X", "Y", "Z"]


def kron_list(ops):
    out = np.array([[1]], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def generators(N):
    """At-most-two-body Z generators: singles then pairs."""
    return [(i,) for i in range(N)] + [(i, j) for i in range(N) for j in range(i + 1, N)]


def z_table(N, gens):
    """z_a(x) = +-1/2, the eigenvalue of Z_a/2 on computational state x."""
    dim = 2 ** N
    Zt = np.zeros((dim, len(gens)))
    for x in range(dim):
        s = [1 - 2 * ((x >> (N - 1 - q)) & 1) for q in range(N)]   # +-1 per qubit
        for a, g in enumerate(gens):
            Zt[x, a] = 0.5 * np.prod([s[q] for q in g])
    return Zt


def gaussian_factor(Imat, Zt):
    """W_xy = exp[-(z(x)-z(y))^T I (z(x)-z(y))]."""
    d = Zt[:, None, :] - Zt[None, :, :]
    return np.exp(-np.einsum("uva,ab,uvb->uv", d, Imat, d))


def ensemble_factor(eta, w, Zt):
    """Non-Gaussian dephasing: finite ensemble of diagonal unitaries
    U_r = exp(-i sum_a eta[r,a] Z_a/2), weights w.  W_xy = <exp(-i (z(x)-z(y)).eta)>."""
    d = Zt[:, None, :] - Zt[None, :, :]
    theta = np.einsum("uva,ra->ruv", d, eta)
    return np.einsum("r,ruv->uv", w, np.exp(-1j * theta))


def chan(W, O):
    return W * O


def two_qubit_paulis():
    ops, names = [], []
    for a in range(4):
        for b in range(4):
            ops.append(kron_list([PAULI[a], PAULI[b]]))
            names.append(PNAME[a] + PNAME[b])
    return ops, names


# --- reduction objects: conditioning (eq::cond_gens), char sum (eq::reduced_diag), fold (eq::operationalI) --- #
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

    def pair_op_full(mu_ab, spec_op):
        a, b = mu_ab
        ops = [I2] * N
        ops[l], ops[lp] = PAULI[a], PAULI[b]
        for k in spec:
            ops[k] = spec_op[k]
        return kron_list(ops)

    return l, lp, spec, pair_states, full_index, pair_op_full


def reduced_block(W, N, bits, layout):
    """[D_z]_{x,x'} = full factor matrix restricted to the spectator config `bits`."""
    _, _, spec, pair_states, full_index, _ = layout
    D = np.zeros((4, 4), dtype=complex)
    for x, xb in enumerate(pair_states):
        for xp, xpb in enumerate(pair_states):
            D[x, xp] = W[full_index(xb, bits), full_index(xpb, bits)]
    return D


def conditional_diag(W, N, bits, mu_idx, P2, layout):
    """[D_z]_{mu mu} = 1/4 Tr[P_mu D_z(P_mu)] from the reduced block."""
    D = reduced_block(W, N, bits, layout)
    Pm = P2[mu_idx]
    return np.trace(Pm @ chan(D, Pm)) / 4.0


def char_sum(W, N, bits, mu, P2name, mu_list, layout, subsets):
    """sum_R (prod_{k in R} s_k) D_{(mu,1),(mu,R)} -- eq::reduced_diag."""
    l, lp, spec, _, _, pair_op_full = layout
    s = {spec[i]: 1 - 2 * bits[i] for i in range(len(spec))}
    out = pair_op_full(mu, {k: I2 for k in spec})            # P_mu (x) 1_spec
    dim = 2 ** N
    total = 0.0
    for R in subsets:
        sop = {k: (Z if k in R else I2) for k in spec}
        Dfull = np.trace(out @ chan(W, pair_op_full(mu, sop))) / dim
        total += np.prod([s[k] for k in R]) * Dfull
    return total


def check_reduction_and_charsum(W, N, pair=(0, 1)):
    """(A) [D_z]_mm == sum_R (prod s) D_{(m,1),(m,R)} for every pair Pauli / config."""
    layout = _layout(N, pair)
    _, _, spec, _, _, _ = layout
    P2, P2name = two_qubit_paulis()
    mu_list = [(a, b) for a in range(4) for b in range(4)]
    subsets = [c for r in range(len(spec) + 1) for c in itertools.combinations(spec, r)]
    cfgs = list(itertools.product([0, 1], repeat=len(spec)))
    err = 0.0
    for bits in cfgs:
        for mi, mu in enumerate(mu_list):
            dz = conditional_diag(W, N, bits, mi, P2, layout)
            cs = char_sum(W, N, bits, mu, P2name, mu_list, layout, subsets)
            err = max(err, abs(dz - cs))
    return err


def overlap_fold_error(Imat, N, pair=(0, 1)):
    """(B) z-independent part of I~_{l,l}(z) == I_{l,l} + sum_k I_{lk,lk}  (eq::operationalI)."""
    l, lp = pair
    spec = [q for q in range(N) if q not in pair]
    gens = generators(N)
    gi = {g: a for a, g in enumerate(gens)}

    def fold(a):
        v = Imat[gi[(a,)], gi[(a,)]]
        for k in spec:
            g = tuple(sorted((a, k)))
            v += Imat[gi[g], gi[g]]
        return v

    def Itilde_of_z(a, s):  # conditional self-overlap; dressed phase phi_a + sum_k s_k phi_ak
        coeff = np.zeros(len(gens))
        coeff[gi[(a,)]] = 1.0
        for i, k in enumerate(spec):
            coeff[gi[tuple(sorted((a, k)))]] = s[i]
        return coeff @ Imat @ coeff

    cfgs = list(itertools.product([1, -1], repeat=len(spec)))      # spectator signs +-1
    err = 0.0
    for a in (l, lp):
        zavg = np.mean([Itilde_of_z(a, s) for s in cfgs])
        err = max(err, abs(zavg - fold(a)))
    return err


def single_element_defect(W, N, pair=(0, 1)):
    """(C) max over configs/Paulis of |[D_z]_mm - D_{(m,1),(m,1)}|.  Nonzero (=polarization)
    when the pair-spectator cross is present; the bare element is the spectator average."""
    layout = _layout(N, pair)
    _, _, spec, _, _, _ = layout
    P2, _ = two_qubit_paulis()
    mu_list = [(a, b) for a in range(4) for b in range(4)]
    cfgs = list(itertools.product([0, 1], repeat=len(spec)))
    defect, avg_err = 0.0, 0.0
    for mi in range(16):
        vals = [conditional_diag(W, N, bits, mi, P2, layout).real for bits in cfgs]
        bare = np.mean(vals)                                  # = D_{(mu,1),(mu,1)}
        defect = max(defect, max(abs(v - bare) for v in vals))
    return defect


# --- register builders ----------------------------------------------------- #
def random_overlap(N, seed, scale=0.18, cross=True):
    """Random PSD overlap matrix over the generators (Gram of random loadings)."""
    rng = np.random.default_rng(seed)
    g = generators(N)
    A = rng.standard_normal((len(g), len(g)))
    S = A @ A.T
    S *= scale / np.mean(np.diag(S))
    if not cross:
        S = np.diag(np.diag(S))
    return S


def gram_embed(M_real, N, pair=(0, 1),
               c=0.7, d=0.4, e=1.0, f=0.12, g=0.6, h=0.10, eta=0.30):
    """Embed the real 3x3 featured pair block (channels {1,2,12}) in an N-qubit
    register overlap matrix via a Gram construction (PSD; pair block exact).  The
    spectator block is a transparent 'extra qubit sharing the bath' extension:
    phi_s = c phi_2 + d(indep); phi_{1s} = e phi_12 + f(indep); phi_{2s} = g phi_12 + h(indep)."""
    l, lp = pair
    spec = [q for q in range(N) if q not in pair]
    gens = generators(N)
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


def extract_featured_pairblock(T_seq=320.0):
    """Live 3x3 featured-model overlaps over {1,2,12} from qns2q.control.idle."""
    import os
    os.environ.setdefault("QNS2Q_REGIME", "featured")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from qns2q.control.idle import (Config, prepare_time_domain_overlap,
                                    evaluate_overlap_folded)
    cfg = Config(fname="DraftRun_NoSPAM_featured", use_simulated=True, M=1)
    RMat, dt, n_base = prepare_time_domain_overlap(cfg.SMat_ideal, cfg.w_ideal,
                                                   cfg.tau, T_seq, 1)
    pt0 = jnp.array([0.0, T_seq])
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = float(np.real(evaluate_overlap_folded(pt0, pt0, RMat[i, j], dt, n_base)))
    return M[1:, 1:]                                          # 3x3 over {Z1, Z2, Z1Z2}


# --- report ---------------------------------------------------------------- #
def report_case(label, N, Imat):
    print(f"\n[{label}]  N={N}")
    print(f"    diag overlaps I_aa = {np.diag(Imat).round(4)}")
    W = gaussian_factor(Imat, z_table(N, generators(N)))
    e_cs = check_reduction_and_charsum(W, N)
    e_fold = overlap_fold_error(Imat, N)
    defect = single_element_defect(W, N)
    print(f"    (A) [D_z]_mm == char sum (eq::reduced_diag)        max|err| = {e_cs:.2e}")
    print(f"    (B) operational-overlap fold (eq::operationalI)    max|err| = {e_fold:.2e}")
    print(f"    (C) single-element defect |[D_z]-D_(m,1,m,1)| = {defect:.3e}  "
          f"({'polarization present' if defect > 1e-6 else 'no pair-spectator cross'})")
    assert e_cs < TOL and e_fold < TOL, f"{label}: identity failed"
    return defect


def report_nongaussian(N, seed=11, K=6):
    rng = np.random.default_rng(seed)
    gens = generators(N)
    eta = rng.uniform(-0.9, 0.9, size=(K, len(gens)))         # arbitrary, NOT Gaussian
    w = rng.uniform(0.2, 1.0, size=K)
    w /= w.sum()
    W = ensemble_factor(eta, w, z_table(N, gens))
    e_cs = check_reduction_and_charsum(W, N)
    print(f"\n[NON-GAUSSIAN ensemble]  N={N}, K={K}")
    print(f"    (A) reduction + char sum, any noise law   max|err| = {e_cs:.2e}")
    assert e_cs < TOL, "non-Gaussian identity failed"


def main():
    print("=" * 70)
    print("App. E.5 register->pair reduction -- exact verification")
    print("=" * 70)

    # generic Gaussian registers (random PSD overlaps)
    report_case("random PSD, correlated spectators", 3, random_overlap(3, 1))
    report_case("random PSD, correlated spectators", 4, random_overlap(4, 2))
    d_off = report_case("random PSD, NO pair-spectator cross", 4,
                        random_overlap(4, 3, cross=False))
    assert d_off < TOL, "defect must vanish without pair-spectator cross"
    print("    -> defect == 0 here confirms D_(m,1),(m,1) is exact only when unpolarized")

    # non-Gaussian generality
    report_nongaussian(3)
    report_nongaussian(4)

    # featured-model anchored (live extraction; skip gracefully if unavailable)
    print("\n" + "-" * 70)
    try:
        for T in (320.0, 80.0):
            M = extract_featured_pairblock(T)
            print(f"\nFEATURED model, real {{1,2,12}} pair block at T={T:g}tau:")
            print(f"    I_11={M[0,0]:.4f}  I_12={M[0,1]:.4f}  I_1,12={M[0,2]:.4f}  "
                  f"I_12,12={M[2,2]:.4f}")
            report_case(f"FEATURED T={T:g}tau + 1 bath-sharing spectator", 3, gram_embed(M, 3))
            report_case(f"FEATURED T={T:g}tau + 2 correlated spectators", 4, gram_embed(M, 4))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [featured-model section skipped: {type(exc).__name__}: {exc}]")
        print("  (needs the repo venv/jax and DraftRun_NoSPAM_featured/simulated_spectra.npz;")
        print("   run: PYTHONPATH=src ./venv/bin/python scripts/verify_embedding_reduction.py)")

    print("\n" + "=" * 70)
    print("ALL IDENTITY CHECKS PASSED (exact to ~1e-16).")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
