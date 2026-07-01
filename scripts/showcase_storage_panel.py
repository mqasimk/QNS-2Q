"""Bell-pair storage-time panel: why the measured qubit-qubit cross-spectrum matters.

**Physics role.** This script answers a concrete design question that no
single-qubit characterization can answer: if the idle ("do-nothing"/
identity) gate is used to STORE one half of an entangled Bell pair
Phi+ = (|00> + |11>)/sqrt(2) for a time Tg, does it matter whether the two
qubits' decoupling pulses fire in sync, or with one qubit's toggling frame
flipped? The two implementations look identical from either qubit's own
point of view (same single-qubit marginals, same average gate fidelity to
leading order) -- the only thing that tells them apart is the measured
inter-qubit cross-spectrum S_1_2, which is exactly what the two-qubit QNS
protocol in this repo is built to reconstruct. This file builds the numbers
behind that argument for the paper's storage-split figure
(`showcase_storage.pdf` / `fig_storage.pdf`).

**Pipeline position.** A "control"-arm consumer that sits AFTER Stage 3b
(`qns2q.control.idle`, the identity/DD-gate optimizer): instead of
re-deriving the overlap-integral machinery, it imports that module as
`idmod` and calls its building blocks directly (`idmod.Config`,
`idmod.prepare_time_domain_overlap`, `idmod.evaluate_overlap_folded`,
`idmod.calculate_idling_fidelity`, `idmod.make_tk12` -- see
DEPENDENCY_MAP.md, where this counts as a protected "attribute contract"
rather than a plain import, since renaming any of those in `control/idle.py`
would silently break this script). The optional Monte-Carlo cross-check
additionally reaches into `qns2q.model.trajectories`, the noise-synthesis
layer the characterization side of the pipeline uses, to validate the
analytic formulas against the same random noise-generation code the QNS
experiments themselves run.

**Inputs.** `idmod.Config(fname=FOLDER, ...)` loads the showcase run's
reconstructed spectra (`specs.npz`/`params.npz`) exactly like the idle-gate
optimizer does; `optimization_data_all_M_cap.npz` (written by
`control.idle`'s `__main__` M-sweep) supplies the "NT winner" reference
curve below.

**Output.** `storage_panel.npz` in the same run folder (or `--out`): one row
per gate time `Tg`, with the FID/sync/anti/NT-winner infidelities for both
Bell states, plus their blindly-predicted counterparts. Consumed by
`scripts/report_showcase_figs.py` to draw the paper's storage panel (see
FIGURE_PROVENANCE.md).

**The physics in one paragraph.** The Bell coherences are exactly
ZZ-immune: |00> and |11> share the Z1 Z2 eigenvalue, and so do |01> and
|10>, so a ZZ-only ("Ising") dephasing channel alone could never
distinguish the Bell-basis states -- only S_11, S_22, and the inter-qubit
cross-spectrum S_1_2 can leak information out of the Phi+/Psi+ subspaces:

    1 - F_{Phi+} = (1 - exp(-v_DQ/2))/2,   v_DQ = I_11 + I_22 + 2 Re I_12
    1 - F_{Psi+} = (1 - exp(-v_ZQ/2))/2,   v_ZQ = I_11 + I_22 - 2 Re I_12

with I_ab the same overlap integrals (pulse-sequence switching function
folded against the noise autocorrelation) the gate optimizer uses
(`control.idle`). For IDENTICAL pulse trains on the two qubits the only
remaining design choice is the relative toggling-frame parity: pulse both
qubits at the same instants ("sync", the choice an engineer would default
to), or bracket qubit 2's train with extra X frame flips so its switching
function is the negative of qubit 1's ("anti"). In-phase gives switching
function y_2 = +y_1 and pays the (S_11 + S_22 + 2 Re S_12) combination on
Phi+; anti-phase gives y_2 = -y_1 and pays (S_11 + S_22 - 2 Re S_12)
instead. With the slow shared-carrier noise's cross-qubit correlation at
c = 0.85, these two numbers differ by up to (1+c)/(1-c) ~ 12x in how fast
the stored Bell pair decoheres -- picking the wrong parity throws away over
an order of magnitude of storage time -- and yet, as noted above, nothing
in either qubit's own single-qubit data would tell an experimenter which
parity is which; only the measured two-qubit cross-spectrum can.

The panel evaluates, per idle gate time:
  FID         -- no decoupling pulses at all, Phi+ / Psi+ infidelity (the
                 bare noise hit, plus the "decoherence-free subspace" (DFS)
                 splitting between the two Bell manifolds that survives
                 even with zero control);
  sync        -- best symmetric CPMG (predicted-optimal pulse count n),
                 simultaneous pulses on both qubits (the implementation an
                 engineer would build by default);
  anti        -- the SAME pulse train, frame-flipped on qubit 2 (the
                 implementation the measured Re S_1_2 > 0 says to use
                 instead);
  NT winner   -- the blind average-fidelity-optimal idle sequence (best
                 over repetition count M) from the Stage-3b optimizer,
                 included for reference: it has the best F_pro
                 (average-fidelity figure of merit) by construction, but
                 because that figure of merit is parity-blind, its Phi+
                 storage performance sits between the sync and anti
                 extremes.
Each is evaluated on the analytic ground-truth noise model, plus (for the
sync/anti pair) the values PREDICTED from the blind reconstruction -- so the
panel also shows how well the reconstructed cross-spectrum would have let an
experimenter make this call without ever seeing the true model. A
Monte-Carlo cross-check (`--mc-check`, using the same record/replay phase
solver the QNS experiments themselves use) independently validates the
analytic formulas above AND the shared-carrier noise synthesis end to end,
at one representative gate time.

Usage:
    QNS2Q_REGIME=showcase PYTHONPATH=src ./venv/bin/python \
        scripts/showcase_storage_panel.py [--mc-check] [--out FILE]
"""
import argparse
import os
import sys

# This repo's scripts are run directly from the repo root (not `pip install`ed),
# so `qns2q` is not on the default import path; prepend `src/` by hand before
# importing it. This is the same one-line idiom every `scripts/*.py` entry
# point in this repo uses.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax.numpy as jnp

from qns2q.paths import project_root
from qns2q.control import idle as idmod

# Showcase's characterization+optimization run folder (see FIGURE_PROVENANCE.md);
# the vestigial "_cap" filename tag on its .npz files predates this script and
# is kept only so this reads the same files the other showcase figures do.
FOLDER = "DraftRun_NoSPAM_showcase_cap"
# Idle gate times to sweep, in units of tau (CLAUDE.md "Units: tau = 1"), doubling
# geometrically to sample storage performance across roughly a decade and a half.
GATE_TIMES = [320., 640., 1280., 2560., 5120., 10240.]


def cpmg_pt(T, n):
    """CPMG-n pulse-time array [0, t_1, ..., t_{2n}, T] (2n equally-spaced pi
    pulses over the interval [0, T], each at the midpoint of one of 2n equal
    sub-intervals). This is the "switching-function" representation used
    throughout `control.idle`: the boundary points 0 and T are included so a
    train can be turned directly into a +-1 toggling function (see
    `idmod.evaluate_overlap_folded`'s `get_y_samples`)."""
    tk = [(k + 0.5) * T / (2 * n) for k in range(2 * n)]
    return jnp.array([0.] + tk + [T])


def overlap_elements(cfg, pt1, pt2, T_seq, M, use_ideal):
    """Compute (I_11, I_22, Re I_12) -- the overlap integrals that feed the
    Bell-state infidelity formulas in `bell_infs` -- for one pair of pulse
    trains (`pt1` on qubit 1, `pt2` on qubit 2).

    `use_ideal` selects which spectral model the noise autocorrelation
    (`RMat` below) is built from: `cfg.SMat_ideal`/`cfg.w_ideal` is the
    analytic ground truth (used to report the TRUE infidelity), while
    `cfg.SMat`/`cfg.w` is what the (possibly finite-sample, possibly
    SPAM-limited) characterization actually reconstructed (used to report
    what a blind experimenter would have PREDICTED). `idmod.
    prepare_time_domain_overlap` does the actual FFT-based construction of
    the folded noise autocorrelation and caches it by (spectrum, T_seq, M),
    so repeated calls at the same gate time are cheap."""
    SMat = cfg.SMat_ideal if use_ideal else cfg.SMat
    w_grid = cfg.w_ideal if use_ideal else cfg.w
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_grid, cfg.tau,
                                                      T_seq, M)
    i11 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt1, RMat[1, 1], dt, nbs)))
    i22 = float(np.real(idmod.evaluate_overlap_folded(pt2, pt2, RMat[2, 2], dt, nbs)))
    i12 = float(np.real(idmod.evaluate_overlap_folded(pt1, pt2, RMat[1, 2], dt, nbs)))
    return i11, i22, i12


def bell_infs(i11, i22, i12):
    """(1 - F_Phi+, 1 - F_Psi+): the two Bell-pair storage infidelities from
    the overlap integrals (see the "physics in one paragraph" section of the
    module docstring for the v_DQ/v_ZQ formulas). Pass ``-i12`` (or
    ``-abs(i12)``) instead of ``i12`` to evaluate the anti-phase parity --
    this function itself is agnostic to which spectral model or parity the
    caller used to produce the three numbers."""
    v_dq = i11 + i22 + 2 * i12
    v_zq = i11 + i22 - 2 * i12
    return (0.5 * (1 - np.exp(-0.5 * v_dq)),
            0.5 * (1 - np.exp(-0.5 * v_zq)))


def fpro_for_parity(cfg, pt1, pt2, T_seq, M, flip):
    """Average (process) infidelity of the idling gate on the pair -- the
    parity-BLIND figure of merit the Stage-3b optimizer actually optimizes,
    as opposed to the Bell-state-specific numbers `bell_infs` reports.

    Builds the 4x4 overlap-integral matrix ``I_matrix`` that
    `idmod.calculate_idling_fidelity` expects: index 0 is the trivial
    "no pulses" reference train, 1 and 2 are qubit 1's/qubit 2's actual
    switching functions, and 3 is their combined two-qubit ("12"/ZZ) channel
    built by `idmod.make_tk12`. Always scores against the analytic
    `cfg.SMat_ideal` (this reports true infidelity, never a blind
    prediction).

    `flip=True` reproduces the anti-phase parity from the module docstring
    by negating y_2 (equivalently, the (1,2)/(2,1) and (1,3)/(3,1) entries
    of `I_matrix`) without recomputing any overlap integral -- both entries
    involve exactly one factor of qubit 2's switching function, so negating
    y_2 flips their sign. The (2,3) entry involves qubit 2's switching
    function TWICE (once directly, once inside the "12" channel) so the two
    sign flips cancel and it is left untouched."""
    SMat = cfg.SMat_ideal
    w_grid = cfg.w_ideal
    RMat, dt, nbs = idmod.prepare_time_domain_overlap(SMat, w_grid, cfg.tau,
                                                      T_seq, M)
    pt12 = idmod.make_tk12(pt1, pt2)
    pt0 = jnp.array([0., T_seq])
    pts = [pt0, pt1, pt2, pt12]
    I = np.zeros((4, 4), dtype=complex)
    for r in range(4):
        for c in range(4):
            I[r, c] = complex(idmod.evaluate_overlap_folded(
                pts[r], pts[c], RMat[r, c], dt, nbs))
    if flip:
        for r, c in ((1, 2), (2, 1), (1, 3), (3, 1)):
            I[r, c] = -I[r, c]
    fid = float(idmod.calculate_idling_fidelity(jnp.asarray(I)))
    return 1.0 - fid / 16.0


def pick_n(cfg, Tg, min_sep):
    """Predicted-optimal symmetric CPMG order (a power of two, respecting the
    hardware pulse-spacing floor `min_sep`) from the CHARACTERIZED spectra --
    i.e. the choice a real, blind experimenter would make, using only what
    the reconstruction actually measured (`use_ideal=False`), never the true
    model.

    Scores each candidate n by its anti-phase Phi+ infidelity (`-abs(i12)`
    rather than `+i12`): anti is the parity this dataset's positive Re S_1_2
    favors (see the module docstring), so it is the metric that matters for
    the row this panel ultimately highlights. `abs()` makes the search
    itself insensitive to any sign error in the characterized i12 estimate
    -- only the actually-reported anti/sync infidelities in `main()` use the
    signed value."""
    best = None
    n = 4
    while (Tg / (2 * n)) >= min_sep:
        pt = cpmg_pt(Tg, n)
        i11, i22, i12 = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=False)
        inf_anti = bell_infs(i11, i22, -abs(i12))[0]
        if best is None or inf_anti < best[1]:
            best = (n, inf_anti)
        n *= 2
    return best[0]


def nt_winner(data, Tg):
    """Best-over-M blind NT ("noise-tailored", i.e. free-optimized rather
    than a textbook CDD/CPMG family -- see `control.idle`'s module
    docstring) idle sequence at gate time Tg.

    `data` is the loaded `optimization_data_all_M_cap.npz` produced by
    `control.idle`'s `__main__` M-sweep: for each repetition count M it
    tried, that file stores parallel arrays keyed
    ``M{m}_gate_times``/``M{m}_infs_opt``/``M{m}_sequences_opt`` (average
    infidelity and winning (qubit-1, qubit-2) pulse-time sequence at each
    gate time it optimized for that M). This scans every M for the row
    matching `Tg` and keeps whichever M gave the lowest average-fidelity
    infidelity -- i.e. it reproduces the same "best over M" selection the
    paper's gate-comparison figure uses, so the reference curve here is the
    real winning sequence, not a re-optimization."""
    best = None
    for m in (int(x) for x in data["M_values"]):
        gts = np.asarray(data[f"M{m}_gate_times"], dtype=float)
        idx = np.where(np.isclose(gts, Tg))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        inf = float(data[f"M{m}_infs_opt"][k])
        seq = data[f"M{m}_sequences_opt"][k]
        if seq is not None and (best is None or inf < best[0]):
            best = (inf, m, seq)
    return best


def mc_check(cfg, Tg, n, n_shots=20000):
    """Independent Monte-Carlo validation of the analytic Phi+/Psi+ formulas
    (`bell_infs`) at one gate time, for the SYNC (simultaneous-pulse) parity
    only -- compare its return value to the `sync_phi`/`sync_psi` row printed
    by `main()` for the same `Tg`.

    Rather than trusting the cumulant-expansion overlap-integral formula
    that the rest of this script relies on, this function goes back to first
    principles: it draws actual noise trajectories with the SAME
    random-number machinery the real QNS Monte-Carlo experiments use
    (`qns2q.model.trajectories`, the noise-synthesis layer behind
    `characterize/experiments.py`), accumulates each shot's noise-induced
    phase on qubit 1 and qubit 2, and averages the resulting Bell-state
    coherence directly. Agreement between this and `bell_infs` end to end
    confirms both the overlap-integral formula AND the shared-carrier noise
    synthesis are self-consistent, independent of the fully analytic
    fast path used everywhere else in the pipeline.

    `_filter_vectors`/`_shot_coeffs_from_filters` are a JAX-vmapped speed
    trick, not new physics: because each shot's accumulated phase is LINEAR
    in its underlying Gaussian noise draws, the (time x frequency) synthesis
    matrices can be contracted against the pulse sequence's switching
    function ONCE (`_filter_vectors`), and then every individual shot costs
    only a handful of dot products (`_shot_coeffs_from_filters`) instead of
    redoing the full trajectory integral per shot -- see those functions'
    own docstrings in `model/trajectories.py` for the ~1000x speedup this
    buys. Shots are still drawn in chunks of 2000 (mirroring
    `trajectories.solver_prop`'s `slice_size`) so the per-shot Gaussian
    draws for all `n_shots` never have to live in GPU memory at once."""
    import jax
    from qns2q.model.trajectories import make_noise_mat_arr, _filter_vectors, \
        _shot_coeffs_from_filters
    t_vec = jnp.linspace(0, Tg, 4096)
    wmax = float(cfg.w_max) / 8.   # synthesis band: the comb band is enough
    mats = make_noise_mat_arr('make', t_vec=t_vec, w_grain=600, wmax=wmax,
                              truncate=int(cfg.mc), midpoint=True)
    pt = np.asarray(cpmg_pt(Tg, n))
    y = np.zeros((3, 3, t_vec.size))
    # y_t is the +-1 CPMG switching function sampled on t_vec (same
    # searchsorted-based construction as `evaluate_overlap_folded`'s
    # `get_y_samples`, done here in plain NumPy on a dense time grid instead
    # of JAX on the folded lag grid). Diagonal (0,0)/(1,1) are qubit 1's and
    # qubit 2's own switching functions (sync parity: identical); (2,2) is
    # the combined ZZ/"12" channel, whose square is identically +1 because
    # simultaneous pulsing leaves the ZZ term undisturbed (see the module
    # docstring: this is exactly the effect the sync/anti comparison is
    # about).
    idx = np.searchsorted(pt, np.asarray(t_vec), side='right')
    y_t = (-1.0) ** (idx - 1)
    y[0, 0] = y_t
    y[1, 1] = y_t
    y[2, 2] = y_t * y_t
    F = _filter_vectors(jnp.asarray(mats), t_vec, jnp.asarray(y))
    rng = np.random.default_rng(20260612)
    chunks = []
    for s in range(0, n_shots, 2000):
        keys = jnp.asarray(rng.integers(0, 10000, (min(2000, n_shots - s), 2)))
        # jax.vmap batches `_shot_coeffs_from_filters` over the whole chunk of
        # per-shot RNG keys at once (no explicit Python loop over shots) --
        # the JAX/GPU idiom for "map this function over an array of inputs".
        chunks.append(jax.vmap(_shot_coeffs_from_filters,
                               in_axes=[None, 0])(F, keys))
    coeffs = jnp.concatenate(chunks, axis=0)
    # c1, c2: each shot's accumulated noise-induced phase on qubit 1 / qubit 2
    # (third column, unused here, would be the "12"/ZZ channel's phase).
    c1, c2 = np.asarray(coeffs[:, 0]), np.asarray(coeffs[:, 1])
    # Phi+ dephases via the SUM of the two qubits' phases (double-quantum,
    # v_DQ above), Psi+ via their DIFFERENCE (zero-quantum, v_ZQ above); the
    # sample mean of cos(2*phase) is the Monte-Carlo estimate of that Bell
    # state's surviving coherence, i.e. its fidelity.
    f_phi = float(np.mean(0.5 * (1 + np.cos(2 * (c1 + c2)))))
    f_psi = float(np.mean(0.5 * (1 + np.cos(2 * (c1 - c2)))))
    return 1 - f_phi, 1 - f_psi


def main():
    """Build the storage panel: for every gate time in `GATE_TIMES`, compute
    the FID/sync/anti/NT-winner Bell-state infidelities described in the
    module docstring, print a one-line summary, and save everything to
    `storage_panel.npz` (or `--out`). `--mc-check` additionally runs the
    first-principles Monte-Carlo cross-check (`mc_check`) at one
    representative gate time. Both `--mc-check` and `--out`, and the default
    `storage_panel.npz` output filename, are relied on by
    FIGURE_PROVENANCE.md / other tooling -- do not rename them."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc-check', action='store_true')
    ap.add_argument('--out', default=None,
                    help="npz output (default <folder>/storage_panel.npz)")
    args = ap.parse_args()

    # M=1 (no repetition folding needed for the FID/sync/anti rows below --
    # they each use a single base sequence spanning the whole gate time);
    # max_pulses/min_sep_factor are effectively "no pulse-count cap, but obey
    # a realistic minimum pulse spacing" for this panel's own CPMG search.
    cfg = idmod.Config(fname=FOLDER, M=1, max_pulses=10**9, min_sep_factor=8.0)
    opt_path = os.path.join(project_root(), FOLDER,
                            "optimization_data_all_M_cap.npz")
    opt = np.load(opt_path, allow_pickle=True)

    rows = []
    for Tg in GATE_TIMES:
        # FID: the "do nothing" baseline -- a single [0, Tg] interval with no
        # pulses at all, on the analytic ground truth.
        pt_fid = jnp.array([0., Tg])
        i = overlap_elements(cfg, pt_fid, pt_fid, Tg, 1, use_ideal=True)
        fid_phi, fid_psi = bell_infs(*i)

        # sync/anti: the blind-optimal symmetric CPMG order, then both parities
        # evaluated on the true model (it) and, for comparison, on what the
        # reconstruction alone would have predicted (ip).
        n = pick_n(cfg, Tg, cfg.min_sep)
        pt = cpmg_pt(Tg, n)
        it = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=True)
        sync_phi, sync_psi = bell_infs(*it)
        anti_phi, anti_psi = bell_infs(it[0], it[1], -it[2])
        ip = overlap_elements(cfg, pt, pt, Tg, 1, use_ideal=False)
        sync_phi_pred = bell_infs(*ip)[0]
        anti_phi_pred = bell_infs(ip[0], ip[1], -ip[2])[0]

        # The parity-blind average-fidelity figure of merit for the same
        # sync/anti pulse train, for the "identical average fidelity" claim
        # in the module docstring.
        fpro_sync = fpro_for_parity(cfg, pt, pt, Tg, 1, flip=False)
        fpro_anti = fpro_for_parity(cfg, pt, pt, Tg, 1, flip=True)

        # NT winner: the Stage-3b optimizer's actual best-over-M sequence at
        # this gate time, re-evaluated here for its Bell-state (rather than
        # average-fidelity) performance. A different M means a different
        # base-sequence duration (Tg / m_nt) repeated m_nt times, so a fresh
        # Config is only needed when m_nt != 1 (cfg above was built for M=1).
        win = nt_winner(opt, Tg)
        if win is not None:
            inf_nt, m_nt, seq_nt = win
            cfg_m = (cfg if m_nt == 1 else
                     idmod.Config(fname=FOLDER, M=m_nt, max_pulses=10**9,
                                  min_sep_factor=8.0))
            pt1, pt2 = jnp.asarray(seq_nt[0]), jnp.asarray(seq_nt[1])
            iw = overlap_elements(cfg_m, pt1, pt2, Tg / m_nt, m_nt,
                                  use_ideal=True)
            nt_phi, nt_psi = bell_infs(*iw)
        else:
            # No optimized sequence recorded at this Tg for any M (e.g. a
            # gate time outside the optimizer's sweep) -- fall back to NaN so
            # this row is visibly missing rather than silently wrong in the
            # saved npz/plot.
            inf_nt = m_nt = nt_phi = nt_psi = np.nan

        rows.append(dict(Tg=Tg, n_cpmg=n,
                         fid_phi=fid_phi, fid_psi=fid_psi,
                         sync_phi=sync_phi, anti_phi=anti_phi,
                         sync_psi=sync_psi, anti_psi=anti_psi,
                         sync_phi_pred=sync_phi_pred,
                         anti_phi_pred=anti_phi_pred,
                         fpro_sync=fpro_sync, fpro_anti=fpro_anti,
                         nt_fpro=inf_nt, nt_M=m_nt, nt_phi=nt_phi,
                         nt_psi=nt_psi))
        r = rows[-1]
        print(f"Tg={Tg:7.0f}  CPMG-{n:<3d} | FID Phi+ {fid_phi:.3e}  "
              f"sync Phi+ {sync_phi:.3e} (pred {sync_phi_pred:.3e})  "
              f"anti Phi+ {anti_phi:.3e} (pred {anti_phi_pred:.3e})  "
              f"split {sync_phi/anti_phi:5.1f}x | F_pro sync/anti "
              f"{fpro_sync:.3e}/{fpro_anti:.3e} (ratio "
              f"{fpro_sync/fpro_anti:.3f}) | NT(M={m_nt}) F_pro "
              f"{inf_nt:.3e} Phi+ {nt_phi:.3e}")

    # One array per dict key, stacked across gate times -- this is the
    # `storage_panel.npz` schema `report_showcase_figs.py` reads.
    out = args.out or os.path.join(project_root(), FOLDER, "storage_panel.npz")
    keys = rows[0].keys()
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in keys})
    print(f"\nSaved {out}")

    if args.mc_check:
        # Re-use whichever CPMG order was picked for Tg=2560 above so the
        # Monte-Carlo check is validating the exact same sync-row pulse
        # train reported in the printed summary, not a fresh choice.
        Tg, n = 2560., None
        for r in rows:
            if r['Tg'] == Tg:
                n = r['n_cpmg']
        mc_phi, mc_psi = mc_check(cfg, Tg, n)
        print(f"\n[MC check @ Tg={Tg:.0f}, CPMG-{n}] Phi+ {mc_phi:.3e}, "
              f"Psi+ {mc_psi:.3e} (sync parity; compare the sync row above)")


if __name__ == "__main__":
    main()
