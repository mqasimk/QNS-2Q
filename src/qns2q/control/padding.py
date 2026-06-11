"""Identity padding of delay vectors (OPT-SPEEDUPS shape unification).

A pair of coincident pi pulses is the identity. Appending ZERO delays to a
delay vector places extra pulses exactly ON the last real pulse (or at t = 0
for an empty vector); an EVEN number of them is therefore exact in every
quantity the pipeline computes:

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

``sum(delays)`` is unchanged, so the optimizer's bounds and linear timing
constraints are untouched -- SLSQP keeps operating on the original
(n1 + n2)-dimensional problem and the padding happens inside the jitted-cost
wrapper. The payoff: every (n1, n2) restart and every library entry at one
gate time shares a single compiled program per PARITY class (pad counts must
be even, so even/odd pulse counts each get one target shape) instead of one
compile per pulse count.
"""
import numpy as np


def pad_targets(ns):
    """Per-parity shape targets: {0: max even count or None, 1: max odd or None}."""
    evens = [int(n) for n in ns if n % 2 == 0]
    odds = [int(n) for n in ns if n % 2 == 1]
    return {0: (max(evens) if evens else None),
            1: (max(odds) if odds else None)}


def pad_count(n, targets):
    """The padded count for n under per-parity targets (never shrinks)."""
    n = int(n)
    t = targets[n % 2]
    return n if t is None or t < n else t


def pad_delays(delays, n_pad):
    """Append (n_pad - n) zero delays -- an identity cluster of coincident
    pulses on the last real pulse. The pad count must be even."""
    d = np.asarray(delays, dtype=float)
    k = n_pad - d.shape[0]
    if k == 0:
        return d
    if k < 0 or k % 2:
        raise ValueError(f"identity padding needs an even, non-negative pad "
                         f"count; got n={d.shape[0]} -> n_pad={n_pad}")
    return np.concatenate([d, np.zeros(k)])
