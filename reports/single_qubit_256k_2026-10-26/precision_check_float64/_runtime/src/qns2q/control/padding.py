"""Pad pulse-sequence delay vectors to a common length without changing the
physics (OPT-SPEEDUPS shape unification): this is a small, self-contained
software-engineering helper, not noise physics. Both gate optimizers --
``control/cz.py`` (CZ gate) and ``control/idle.py`` (identity / dynamical-
decoupling gate) -- import ``pad_targets``, ``pad_count``, and ``pad_delays``
from here (see their "Shape-unify the library with exact-identity padding"
call sites) to batch many candidate pulse sequences of DIFFERENT pulse counts
into JAX-jitted cost/gradient evaluators that expect ONE fixed array shape.

Where this sits in the pipeline: it is called from inside the CZ/idle
optimization stage (Stage 3a/3b in CLAUDE.md), specifically from the library-
evaluation and restart loops that score many candidate pulse sequences
(CPMG/CDD/mqCDD library entries, or SLSQP restart timings) against the
reconstructed noise spectra. It has no direct dependence on the noise model or
on any earlier pipeline stage -- it only manipulates arrays of pulse-to-pulse
time DELAYS (durations between consecutive pi pulses in a sequence, which sum
to the total gate/sequence time T_seq).

Why padding is needed at all (a JAX/XLA idiom, not a physics one): JAX
``jit``-compiles one machine program PER distinct input shape it sees, and
recompiles (slow) whenever the shape changes. The optimizers need to evaluate
many candidate sequences with different numbers of pulses (different delay-
vector lengths) against the same jitted cost function. Rather than pay a
fresh compile for every distinct pulse count, we pad every sequence's delay
vector up to a single target length per PARITY class (even/odd pulse count,
see below), so at most two compiled programs are needed for an entire batch
of candidates instead of one per distinct length.

The padding trick that makes this exact (not an approximation): a pair of
coincident pi pulses is the identity operation, so appending ZERO-length
delays to a delay vector places extra "pulses" exactly ON TOP OF the last
real pulse (or at t = 0 for an empty vector, i.e. no real pulses at all). As
long as an EVEN number of these zero-delay pulses are appended, this leaves
every physical quantity the pipeline computes from the sequence unchanged:

* the sampled switching function y(t): ``searchsorted(side='right')`` jumps
  by 2 across the coincident cluster, so (-1)^(index-1) is unchanged at every
  sample;
* the spectral amplitudes A(w): zero-length intervals contribute exactly 0
  and the interval-sign parity after the cluster is preserved;
* the DC term sum(diff * signs): zero-length diffs contribute 0 at preserved
  parity;
* the combined pt12 sequence: ``make_tk12`` keeps duplicates, so the cluster
  cancels in y_12 the same way;
* all gradients w.r.t. the REAL delays: the pad cluster rides on the last
  real pulse, so perturbing any real delay moves the cluster rigidly with it.

Concretely: if a padded delay vector's real (non-padded) entries are
perturbed, the extra zero-delay pulses at the end just ride along on the
last real pulse's position, so they never introduce a spurious dependence
on the real delays and never bias the optimizer's gradient.

Because ``sum(delays)`` is unchanged by appending zeros, the total sequence
duration T_seq -- and therefore the optimizer's bounds and linear timing
constraints -- are untouched: SLSQP keeps operating on the original
(n1 + n2)-dimensional problem (n1, n2 = number of REAL delays on qubit 1 and
qubit 2) and the padding happens only inside the jitted-cost wrapper, which
sees the padded (longer) arrays. The payoff: every (n1, n2) restart and every
library entry at one gate time shares a single compiled program per parity
class (pad counts must be even, so even/odd real pulse counts each map to
one target shape) instead of one compile per distinct pulse count.
"""
import numpy as np


def pad_targets(ns):
    """Compute the per-parity padding target shape for a batch of sequences.

    Given the pulse counts ``ns`` of every sequence about to be evaluated
    together, return the length every EVEN-count sequence and every ODD-count
    sequence should be padded up to (the max within each parity group), so
    the whole batch collapses to at most two distinct array shapes -- see
    the module docstring for why only two shapes (not one per length) are
    needed. Returns ``{0: max even count or None, 1: max odd or None}``;
    a value of ``None`` means no sequence of that parity is present in ``ns``.
    """
    evens = [int(n) for n in ns if n % 2 == 0]
    odds = [int(n) for n in ns if n % 2 == 1]
    return {0: (max(evens) if evens else None),
            1: (max(odds) if odds else None)}


def pad_count(n, targets):
    """Look up the padded length for a single sequence of length ``n``.

    ``targets`` is the dict returned by :func:`pad_targets`. Looks up the
    target for ``n``'s parity (``n % 2``) and returns it, unless that target
    is missing or would be a shrink (defensive guard only -- in normal use
    ``targets`` is built from the same batch ``n`` belongs to, so the target
    is always >= n); a sequence is never padded shorter than it already is.
    """
    n = int(n)
    t = targets[n % 2]
    return n if t is None or t < n else t


def pad_delays(delays, n_pad):
    """Pad one sequence's delay vector to length ``n_pad`` with an identity
    pulse cluster.

    Appends ``(n_pad - n)`` zero-length delays, which is exactly the same as
    inserting that many extra pi pulses coincident with the last real pulse
    (an identity operation in pairs -- see the module docstring for why this
    leaves every downstream physical quantity unchanged). ``n_pad`` must be
    even relative to the current length (the padded cluster must contain an
    even number of extra pulses) and must not be smaller than the current
    length; a bare :class:`ValueError` guards against silently corrupting a
    sequence rather than padding it. If already at the target length
    (``n_pad == n``), returns the (possibly float-cast) array as-is with no
    padding appended -- note ``np.asarray`` only copies if a dtype cast is
    needed, so this may be the same object as ``delays`` when it was already
    a float array; callers should treat the return value as read-only.
    """
    d = np.asarray(delays, dtype=float)
    k = n_pad - d.shape[0]
    if k == 0:
        return d
    if k < 0 or k % 2:
        raise ValueError(f"identity padding needs an even, non-negative pad "
                         f"count; got n={d.shape[0]} -> n_pad={n_pad}")
    return np.concatenate([d, np.zeros(k)])
