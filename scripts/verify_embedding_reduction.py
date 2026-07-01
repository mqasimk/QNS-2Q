"""Exact verification of the App. E.5 register->pair reduction cascade.

ORIENTATION -- what this file is and why it exists:
This is a standalone, self-contained math check, not a stage of the QNS
pipeline described in CLAUDE.md. It runs no experiments, reconstructs no
spectra, and optimizes no pulses; it never writes any output file (only
prints a report to stdout). Its job is to defend one specific claim used in
the paper's App. E.5 ("Embedding the active pair in an N-qubit register"):
that when the two qubits we actually care about (the "pair") sit inside a
bigger device with other qubits ("spectators") coupled to the same noise
bath, the paper is nonetheless allowed to compute everything (infidelities,
overlaps, spectra) as if only the pair existed, with the spectators' effect
folded into a few extra, computable overlap terms. If that reduction were
wrong, every gate-optimization result in the paper (built from a 2-qubit +
1 bath-qubit model, per the "3-Qubit Hilbert Space Convention" in CLAUDE.md)
would not actually generalize to a real, larger device. This script proves
the reduction algebraically, on toy noise models, standing in for a real
device register, so nobody has to trust the appendix's algebra on faith.

It reads no run-folder data by default; (D) below optionally reads the
already-fitted `simulated_spectra.npz` for the `featured` regime indirectly,
by importing `qns2q.control.idle` and constructing its `Config` (the same
object `control/idle.py`'s optimizer uses) -- so that one part of the check
runs on the actual numbers behind the paper's figures, not just invented
matrices. Nothing else in the repository imports this script or calls into
it; it is meant to be read and re-run by a human, not by another stage.

Companion pytest coverage: `tests/test_spectator_locality.py` guards a
different, earlier slice of the same appendix (the exact reduction, the
Jensen-inequality split, the worst-case state, and the sign of the -1/2
cross-overlap coefficient). `tests/test_embedding_reduction.py` is the
direct pytest port of the four checks (A)-(D) below -- same numerics, CI-
checked automatically on every run of `pytest tests/`. This script is the
older, narrative "runnable report" version of those same four checks: it
prints intermediate numbers (the overlap values, the defect size) so a
human debugging a mismatch with the manuscript can see *where* an identity
would fail, which a pass/fail pytest assertion does not show.

The four checks, each on a synthetic or real noise-overlap matrix:

  (A) NON-GAUSSIAN generality -- the reduction and the PTM (Pauli-transfer-
      matrix) character sum
      [D_z]_mm = sum_R (prod_{k in R} s_k) D_{(m,1),(m,R)}   (cf. eq::pop_only
      in the manuscript: same object, indexed there by "Q" instead of "R")
      hold for an arbitrary classical dephasing channel (a finite ensemble of
      diagonal unitaries), not only for the Gaussian model.  They use only the
      computational-diagonal structure, so are exact at all orders, any
      statistics -- i.e. this is not a small-noise approximation; it is exact
      algebra, so it must match to numerical roundoff, not just "close".
  (B) the operational-overlap FOLD  I~_{l,l} = I_{l,l} + sum_k I_{lk,lk}
      (eq::operationalI in the manuscript). In words: the effective decay
      rate a pair qubit `l` experiences, once you trace out (average over)
      the spectators, equals its own bare decay rate `I_{l,l}` PLUS one
      additive term `I_{lk,lk}` per spectator `k` it is correlated with --
      spectators never enter any other way (no new cross-terms appear).
  (C) the SINGLE-ELEMENT DEFECT: the conditional diagonal [D_z]_mm (the
      pair's population read, GIVEN a specific frozen classical spectator
      configuration z) is the spectator-conditioned read of the register;
      the bare element D_{(m,1),(m,1)} (obtained by simply ignoring the
      spectators, i.e. embedding the pair's own overlap block with no
      spectators present) is only its R=0 (unpolarized, spectator-averaged)
      term. The two differ by a "defect" that is nonzero exactly when the
      pair-spectator cross term I_{l,lk} != 0 -- i.e. exactly when the pair's
      decay is correlated with which classical state the spectators are in.
      This check guards against a subtly wrong version of the appendix that
      would have asserted [D_z]_mm = D_{(m,1),(m,1)} unconditionally (true
      only when there is no pair-spectator correlation at all).
  (D) the same three identities, but evaluated on the REAL featured-regime
      pair overlaps (see NOISE_MODEL_SPEC.md / CLAUDE.md for what "featured"
      means), extracted live from `qns2q.control.idle`, then embedded into a
      larger synthetic register via a PSD ("positive semi-definite", i.e.
      guaranteed to be a valid covariance/overlap matrix) Gram construction
      -- so (A)-(C) are checked against actual physics numbers, not only
      hand-picked random matrices.

Convention (matches tests/test_spectator_locality.py): generators Z_a are all
single Z_q and two-body Z_iZ_j (i.e. every possible single- or two-qubit
dephasing coupling in the register); the channel acts elementwise in the
computational basis, E(rho)_xy = rho_xy * W_xy, with the Gaussian factor

    W_xy = exp[ -(z(x)-z(y))^T I (z(x)-z(y)) ],   z_a(x) = +-1/2 eigenvalue of Z_a/2,

so the single-Pauli decay is 1 - E ~ I_aa (the paper's overlap normalization,
i.e. `Imat` below plays the role of the manuscript's overlap/covariance
matrix I_ab, one row/column per generator Z_a).
Pair = qubits {0,1} = (l,l'); spectators = qubits {2..N-1}.

Run:  PYTHONPATH=src python scripts/verify_embedding_reduction.py
(the featured-model section, check (D), additionally needs the repo venv /
jax and a `simulated_spectra.npz`; it prints a notice and is skipped, rather
than failing, if those are unavailable -- see the try/except in `main()`.)
"""
import itertools
import sys

import numpy as np

TOL = 1e-10  # pass/fail cutoff for the identities below. These are EXACT algebraic
# identities (not small-noise approximations), so in principle the error should be
# at machine precision (~1e-16); 1e-10 just leaves headroom for the roundoff that
# accumulates over the many chained matrix products/sums in the checks -- if an
# assertion below ever fails, treat it as a real bug, not noise.
np.set_printoptions(precision=6, suppress=True, linewidth=130)

# --- Pauli / computational-basis machinery --------------------------------- #
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]
PNAME = ["I", "X", "Y", "Z"]


def kron_list(ops):
    """Tensor (Kronecker) product of a list of single-qubit 2x2 operators into
    one 2^N x 2^N operator on the full register -- the standard way to build a
    multi-qubit operator (e.g. Z_1 (x) I (x) I) out of single-qubit pieces."""
    out = np.array([[1]], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def generators(N):
    """At-most-two-body Z generators: singles then pairs.

    Each entry is a tuple of qubit indices: `(q,)` means the single-qubit
    dephasing coupling Z_q (qubit q's own noise), `(i, j)` means the two-body
    coupling Z_i Z_j (a correlated/crosstalk noise term between qubits i, j).
    These are the noise "channels" the paper's overlap matrix I_ab is indexed
    by; for the real two-qubit pair, channels {1,2,12} in `CLAUDE.md`'s
    S_11/S_22/S_1212 correspond to generators `(l,)`, `(lp,)`, `(l, lp)` here.
    """
    return [(i,) for i in range(N)] + [(i, j) for i in range(N) for j in range(i + 1, N)]


def z_table(N, gens):
    """z_a(x) = +-1/2, the eigenvalue of Z_a/2 on computational state x.

    Precomputes, for every computational basis state x of the N-qubit
    register and every generator a from `generators`, the classical +-1/2
    "which way is this Z_a pointing" value -- e.g. for a two-body generator
    (i, j) it is +1/2 if qubits i and j agree, else -1/2. This table is the
    only place basis-state bit patterns are touched; everything downstream
    works with these +-1/2 numbers instead.
    """
    dim = 2 ** N
    Zt = np.zeros((dim, len(gens)))
    for x in range(dim):
        s = [1 - 2 * ((x >> (N - 1 - q)) & 1) for q in range(N)]   # +-1 per qubit
        for a, g in enumerate(gens):
            Zt[x, a] = 0.5 * np.prod([s[q] for q in g])
    return Zt


def gaussian_factor(Imat, Zt):
    """W_xy = exp[-(z(x)-z(y))^T I (z(x)-z(y))].

    This literally *is* the noise-averaged quantum channel, written as the
    elementwise multiplier applied to each density-matrix entry rho_xy (see
    `chan` below) -- exact for classical, zero-mean Gaussian dephasing with
    covariance 2*Imat over the generators. `Imat` (paper symbol I_ab) is the
    overlap/covariance matrix indexed by `generators(N)`; `Zt` is the
    `z_table` above. (`np.einsum` below is NumPy's Einstein-summation
    helper: it is just a compact way to write the batched
    "for every pair of basis states u,v: d[u,v] . Imat . d[u,v]" contraction
    without an explicit Python loop over the 2^N x 2^N grid.)
    """
    d = Zt[:, None, :] - Zt[None, :, :]
    return np.exp(-np.einsum("uva,ab,uvb->uv", d, Imat, d))


def ensemble_factor(eta, w, Zt):
    """Non-Gaussian dephasing: finite ensemble of diagonal unitaries
    U_r = exp(-i sum_a eta[r,a] Z_a/2), weights w.  W_xy = <exp(-i (z(x)-z(y)).eta)>.

    Stands in for `gaussian_factor` but for a channel that is NOT Gaussian:
    instead of a continuous Gaussian phase, the noise is modeled as picking,
    with probability w[r], one of a handful (K) of fixed classical
    "realizations" r, each applying a definite phase eta[r, a] through
    generator a. Used only by check (A) (`report_nongaussian`) to prove the
    reduction identity does not secretly depend on the noise being Gaussian.
    """
    d = Zt[:, None, :] - Zt[None, :, :]
    theta = np.einsum("uva,ra->ruv", d, eta)
    return np.einsum("r,ruv->uv", w, np.exp(-1j * theta))


def chan(W, O):
    """Apply the elementwise (Schur-product) dephasing channel: E(O)_xy =
    W_xy * O_xy. This is the computational-basis definition of the channel
    used throughout -- multiplying every operator entry by the corresponding
    decay factor from `gaussian_factor`/`ensemble_factor`."""
    return W * O


def two_qubit_paulis():
    """All 16 two-qubit Pauli operators P_a (x) P_b (a,b in {I,X,Y,Z}), plus
    their string labels (e.g. "XZ"). Used to test the reduction identity
    against every possible operator the pair's reduced density matrix could
    have a nonzero component along -- i.e. an exhaustive check, not a
    spot-check on one operator."""
    ops, names = [], []
    for a in range(4):
        for b in range(4):
            ops.append(kron_list([PAULI[a], PAULI[b]]))
            names.append(PNAME[a] + PNAME[b])
    return ops, names


# --- reduction objects: conditioning, character sum, and the overlap fold ---
# These three correspond to the manuscript's App. E.5 equations for the
# conditional-generator reduction (eq::cond_gens), the PTM character-sum
# rewrite of it (cf. eq::pop_only), and the operational-overlap fold
# (eq::operationalI) -- see the module docstring above for what each means
# in words.
def _layout(N, pair=(0, 1)):
    """Index bookkeeping shared by the functions below (private helper, not
    meant to be used outside this file -- the leading underscore is Python's
    convention for "internal, not part of this module's public interface").

    Splits the N-qubit register into the `pair` (qubits l, l' -- the two
    qubits whose reduced dynamics we actually care about) and the remaining
    `spec` (spectator) qubits, and returns the pieces needed to translate
    back and forth between "pair state x + spectator bit-string" and a
    single index into the full 2^N-dimensional Hilbert space:

      full_index(xbits, sbits) -- basis-state index for pair state `xbits`
        (a (0/1, 0/1) pair) together with a specific spectator bit-string.
      pair_op_full(mu_ab, spec_op) -- embeds a 2-qubit Pauli `mu_ab` = (a, b)
        acting on the pair, tensored with per-spectator single-qubit
        operators `spec_op`, into a full N-qubit operator.
    """
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
    """[D_z]_{x,x'} = full factor matrix restricted to the spectator config `bits`.

    `bits` is one specific classical configuration (a tuple of 0/1, one per
    spectator qubit) that the spectators are frozen in -- physically, "what
    if we somehow knew/fixed the spectators' state". `D_z` is then the
    pair's own 4x4 reduced dephasing-channel matrix (over the 4 pair basis
    states) GIVEN that spectator configuration z = `bits`; this is the
    building block both (A) and (C) below compare against.
    """
    _, _, spec, pair_states, full_index, _ = layout
    D = np.zeros((4, 4), dtype=complex)
    for x, xb in enumerate(pair_states):
        for xp, xpb in enumerate(pair_states):
            D[x, xp] = W[full_index(xb, bits), full_index(xpb, bits)]
    return D


def conditional_diag(W, N, bits, mu_idx, P2, layout):
    """[D_z]_{mu mu} = 1/4 Tr[P_mu D_z(P_mu)] from the reduced block.

    This is the "population" (diagonal, in the Pauli-transfer-matrix sense)
    read of pair-Pauli `P2[mu_idx]` -- i.e. <P_mu> after the noise channel,
    computed directly from the spectator-conditioned block `reduced_block`
    -- to be compared against the algebraically different-looking but
    provably equal `char_sum` formula below. The 1/4 is the normalization
    for a two-qubit Pauli expectation value, Tr[P rho]/dim with dim=4.
    """
    D = reduced_block(W, N, bits, layout)
    Pm = P2[mu_idx]
    return np.trace(Pm @ chan(D, Pm)) / 4.0


def char_sum(W, N, bits, mu, P2name, mu_list, layout, subsets):
    """sum_R (prod_{k in R} s_k) D_{(mu,1),(mu,R)} -- PTM char-sum form, cf. eq::pop_only.

    Same physical quantity as `conditional_diag`, but computed by the
    manuscript's alternative formula: a sum over every subset `R` of
    spectator qubits (`subsets`, built by `itertools.combinations` in
    `check_reduction_and_charsum` below -- the standard way to enumerate
    "every way to choose a subset" in Python), each term weighted by the
    product of that subset's classical spectator signs `s_k` (+-1, from the
    frozen configuration `bits`). Agreement between this and
    `conditional_diag` for every config and every pair-Pauli is exactly
    identity (A).
    """
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
    """(A) [D_z]_mm == sum_R (prod s) D_{(m,1),(m,R)} for every pair Pauli / config.

    Loops over every spectator configuration (`cfgs`, all 0/1 bit-strings,
    via `itertools.product`) and every one of the 16 two-qubit Paulis, and
    checks the two formulas for the same conditional population
    (`conditional_diag` vs `char_sum`) agree to within `TOL`. Returns the
    worst (max) mismatch found; `main()`/`report_case` asserts it is ~0.
    """
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
    """(B) z-independent part of I~_{l,l}(z) == I_{l,l} + sum_k I_{lk,lk}  (eq::operationalI).

    `I~_{l,l}(z)` is the pair qubit `l`'s own "dressed" overlap once you
    account for its correlation with a specific frozen spectator
    configuration z (computed below as `Itilde_of_z`); it fluctuates with z
    in general. This checks that its AVERAGE over every spectator sign
    configuration (`cfgs`, +-1 per spectator) equals the simple additive
    `fold(a)`: the pair qubit's bare overlap `I_{l,l}` plus one term
    `I_{lk,lk}` per spectator it shares a two-body coupling with -- i.e. once
    you average over what the spectators are doing, no new cross-terms
    survive beyond this fixed additive shift.
    """
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
    when the pair-spectator cross is present; the bare element is the spectator average.

    For every pair-Pauli, computes the conditional population `[D_z]_mm` for
    every spectator config (`vals`), and its spectator average `bare` --
    which is exactly the bare (spectator-ignored) element D_{(m,1),(m,1)},
    since averaging uniformly over +-1 spectator signs kills every term that
    depends on them (matching `fold`/`overlap_fold_error` above). `defect`
    is how far any single config's value strays from that average: zero
    means the pair truly behaves as if isolated regardless of the
    spectators' state; nonzero means the pair's population is genuinely
    correlated with ("polarized by") the spectators, and the bare element
    alone would be a biased stand-in for any particular real config.
    """
    layout = _layout(N, pair)
    _, _, spec, _, _, _ = layout
    P2, _ = two_qubit_paulis()
    cfgs = list(itertools.product([0, 1], repeat=len(spec)))
    defect = 0.0
    for mi in range(16):
        vals = [conditional_diag(W, N, bits, mi, P2, layout).real for bits in cfgs]
        bare = np.mean(vals)                                  # = D_{(mu,1),(mu,1)}
        defect = max(defect, max(abs(v - bare) for v in vals))
    return defect


# --- register builders ----------------------------------------------------- #
# The two functions below build *synthetic test-fixture* overlap matrices --
# they are numerical scaffolding to stress-test the identities above on
# plausible-looking noise-correlation structures; the specific numbers are
# not derived from the paper (only `extract_featured_pairblock` below pulls
# real, paper-anchored numbers).
def random_overlap(N, seed, scale=0.18, cross=True):
    """Random PSD (positive semi-definite, i.e. a mathematically valid
    covariance/overlap matrix) overlap matrix over the generators, built as
    a Gram matrix `A @ A.T` of random loadings -- guaranteed PSD by
    construction. `cross=False` zeroes the off-diagonal (no pair-spectator
    correlation at all), used as a control case where check (C)'s defect
    must vanish exactly (see the `assert d_off < TOL` in `main()`)."""
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
    phi_s = c phi_2 + d(indep); phi_{1s} = e phi_12 + f(indep); phi_{2s} = g phi_12 + h(indep).

    Concretely: takes the REAL 3x3 pair overlap block `M_real` (measured/
    reconstructed from the featured noise model, see
    `extract_featured_pairblock`) and manufactures extra "spectator" qubits
    around it whose noise loadings are partly copies of the pair's own
    loadings (so the pair and spectators are correlated, as they would be
    if physically sharing the same noise bath) and partly independent
    ("fresh") directions. Every generator gets a fixed-length loading vector
    `V[...]`; stacking them and forming `Vmat @ Vmat.T` (a Gram matrix) is
    what GUARANTEES the resulting overlap matrix is PSD (a real,
    physically-allowed covariance matrix) while reproducing `M_real`
    exactly in the pair-only block. The coefficients c, d, e, f, g, h, eta
    are arbitrary "reasonable-looking" mixing strengths chosen for this
    test, not manuscript-derived constants -- only their PSD-ness and their
    exact reproduction of `M_real` matter for what is being checked.
    """
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
    """Live 3x3 featured-model overlaps over {1,2,12} from qns2q.control.idle.

    Pulls the REAL pair overlap matrix (over channels {Z_1, Z_2, Z_1 Z_2},
    i.e. qubit 1's noise, qubit 2's noise, and their Ising-coupling noise --
    the same three channels reconstructed in Stage 2 of the pipeline, per
    CLAUDE.md) for a given idling-sequence duration `T_seq` (units of tau,
    the repo-wide time unit), computed exactly the way `control/idle.py`'s
    optimizer computes it for the `featured` noise regime. This is what
    lets check (D) test the appendix identities against the actual overlap
    numbers behind the paper's idling-gate figures, not just invented ones.

    All the imports below are deliberately LOCAL to this function (not at
    the top of the file) so that the pure-NumPy identity checks (A)-(C)
    above can run -- and this file can even be imported/py_compiled --
    without needing jax or the rest of the repo's environment installed;
    only this one, optional, real-data check needs them. `os.environ.
    setdefault("QNS2Q_REGIME", "featured")` must run BEFORE `qns2q` is
    imported, because (per CLAUDE.md) the active noise model is chosen once,
    at import time, by reading this environment variable -- setting it any
    later would silently have no effect.
    """
    import os
    os.environ.setdefault("QNS2Q_REGIME", "featured")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from qns2q.control.idle import (Config, prepare_time_domain_overlap,
                                    evaluate_overlap_folded)
    # Constructing `Config` here does real disk I/O (it loads the featured
    # regime's saved spectra) -- this is the dataclass-with-I/O-in-__init__
    # pattern used throughout the repo's pipeline stages (see CLAUDE.md);
    # it is why this whole function is wrapped in a try/except in `main()`.
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
    """Run all three identities (A)-(C) on one overlap matrix `Imat` (an
    N-qubit register, pair = qubits {0,1}) and print a human-readable
    summary line per check, so a mismatch with the manuscript would show up
    with actual numbers rather than a bare assertion failure. Returns the
    check-(C) defect (see `single_element_defect`) for the caller to
    inspect further (e.g. `main()` asserts it is exactly 0 in the
    no-pair-spectator-cross control case)."""
    print(f"\n[{label}]  N={N}")
    print(f"    diag overlaps I_aa = {np.diag(Imat).round(4)}")
    W = gaussian_factor(Imat, z_table(N, generators(N)))
    e_cs = check_reduction_and_charsum(W, N)
    e_fold = overlap_fold_error(Imat, N)
    defect = single_element_defect(W, N)
    print(f"    (A) [D_z]_mm == char sum (cf. eq::pop_only)        max|err| = {e_cs:.2e}")
    print(f"    (B) operational-overlap fold (eq::operationalI)    max|err| = {e_fold:.2e}")
    print(f"    (C) single-element defect |[D_z]-D_(m,1,m,1)| = {defect:.3e}  "
          f"({'polarization present' if defect > 1e-6 else 'no pair-spectator cross'})")
    assert e_cs < TOL and e_fold < TOL, f"{label}: identity failed"
    return defect


def report_nongaussian(N, seed=11, K=6):
    """Check (A) only, on a NON-Gaussian ensemble dephasing channel (see
    `ensemble_factor`) instead of a Gaussian one -- `eta`/`w` are just
    arbitrary random numbers (deliberately NOT drawn to look Gaussian; the
    comment "arbitrary, NOT Gaussian" is the point of the test), to confirm
    the reduction/char-sum identity does not secretly rely on Gaussianity."""
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
    """Run the full report: (A)-(C) on synthetic registers (random PSD
    overlaps, with and without pair-spectator correlation, plus a
    non-Gaussian ensemble), then (D) the same checks anchored to the real
    featured-model pair overlaps. Raises `AssertionError` (caught by
    `__main__` below, turning it into a nonzero process exit code) if any
    identity fails outside tolerance."""
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

    # featured-model anchored (live extraction; skip gracefully if unavailable).
    # The broad `except Exception` (silencing the "blind except" lint warning,
    # `noqa: BLE001`) is intentional here, not sloppy error handling: this
    # section needs jax and a pre-existing run folder that may not exist in
    # every environment (e.g. a lightweight checkout without the venv), and
    # the script should still finish and report the rest of its results
    # rather than crash just because this optional extra is unavailable.
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


# Standard Python "only run this when executed directly, not when imported"
# guard. Converts a failed identity (an `AssertionError` raised inside
# `main()`) into a printed message plus a nonzero process exit code, rather
# than an unhandled traceback -- useful if this script is ever wired into an
# automated check that inspects the exit code.
if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
