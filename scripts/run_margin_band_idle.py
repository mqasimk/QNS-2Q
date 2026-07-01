"""Idle/DD-gate twin of run_margin_band.py: propagate the noise-spectrum
reconstruction's error bars into predicted idle (identity/dynamical-
decoupling, "DD") gate infidelities and into the "does the noise-tailored
(NT) sequence actually beat the known/CDD sequence" margin band.

Physics role
------------
This script asks the same referee-facing question as run_margin_band.py, but
for the idle gate instead of CZ: Stage 2 (`characterize/reconstruct.py`)
quotes a statistical (shot-noise) and a systematic (unfold/tail-model) error
bar on every reconstructed spectral point, but Stage 3b (`control/idle.py`)
picks its winning pulse sequences using only the *central* spectrum. Given
how uncertain the reconstruction really is, how much could the winners'
predicted infidelities -- and, more importantly, the known-vs-NT margin --
move around? This script Monte-Carlo-resamples the reconstructed spectrum
within its quoted bars (reusing `run_margin_band.draw_specs`, so both gates'
bands come from the exact same perturbation model) and re-scores each fixed
winner on every resampled spectrum, without re-running the pulse-sequence
search itself.

The idle gate adds one wrinkle CZ does not have: `control/idle.py` sweeps a
grid of repetition counts M (M=1,2,4,...) for every gate time Tg and keeps
whichever M gives the lowest infidelity, so a "winner" here is a (sequence,
M) pair, not just a sequence. `load_best_over_M` below reproduces that same
best-over-M reduction (on the same stored "ideal/true-benchmark" infidelities
`control/idle.py` itself uses for the reduction, not the cheaper comb
approximation used only to speed up the search) so the fixed pair being
re-scored here is identical to what the paper's report table quotes.

Pipeline position
------------------
Post-processing / diagnostics script, downstream of both pipeline arms: it
reads Stage 2's error bars (`specs.npz`, via `qns2q.control.idle.Config`,
the same config class `control/idle.py` itself uses, so the frequency grid /
tail model / DC handling match the winner search exactly) and Stage 3b's
`optimization_data_all_M*.npz` (written by `control/idle.py`'s M-sweep
`__main__`), which holds every (Tg, M, kind) infidelity/sequence/label.
Unlike the CZ script, this file does not render its own PDF (no `--replot` /
`render_figure` here) -- its `margin_band_id*.npz` output is read directly by
`scripts/report_showcase_figs.py`, which draws the paper's combined
CZ+idle "gates" panel from both gates' npz files together.

Inputs
------
  - The run folder (`--folder`, default: the active regime's NoSPAM folder)
    supplies BOTH `specs.npz` (central spectra + `<key>_err`/`<key>_sys`/
    `<key>_dc_ok` bars, loaded through `idle.Config`) and
    `optimization_data_all_M<tag>.npz` (the fixed winner sequences), unlike
    the CZ script where the specs folder and winners folder can differ.

Outputs
-------
  - `<folder>/margin_band_id<tag>.npz` -- per-draw infidelities for every
    fixed (sequence, M) winner plus per-gate-time NT/known infidelity-ratio
    arrays (`main()`'s `margin_out` dict), plus an `entry_M` array recording
    which M each winner used (the CZ file has no such column: CZ always
    uses M=1).

Method: the resampling model
-----------------------------
Identical to run_margin_band.py's (see that module's docstring for the full
derivation): a shared-per-channel systematic offset plus independent-per-
tooth statistical noise, self-spectra floored at 0, flagged DC points held
fixed, tail fits re-run on every draw. The same draw evaluates every fixed
winner, so the margin band is the percentile of the per-draw RATIO
(known/opt), which keeps the correlation between a "quiet" or "noisy" draw
shared across both sequences rather than combining two independent bands.

Usage:
    python scripts/run_margin_band_idle.py [--folder DraftRun_NoSPAM_featured]
        [--n-draws 200]
"""
import argparse
import os
import sys

# Two separate sys.path insertions, for two separate reasons. Python only
# looks in directories already on sys.path, so both are needed for the
# imports below to resolve when this file is run directly as
# `python scripts/run_margin_band_idle.py` (not installed as a package):
#   1. this file's own directory (scripts/), so `from run_margin_band import
#      draw_specs` below finds its *sibling script* -- run_margin_band.py is
#      a plain module, not part of the qns2q package, so it is only
#      importable if scripts/ itself is searched.
#   2. src/, so `from qns2q...` finds the installed-package tree. Every
#      pipeline script under scripts/ does (2); this one additionally needs
#      (1) because it reuses the CZ script's resampling code instead of
#      duplicating it (see module docstring).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.paths import run_folder, project_root
# draw_specs is imported, not reimplemented, so both gates' margin bands are
# always perturbed by the identical model -- if run_margin_band.py's
# resampling recipe changes, this script picks the change up automatically.
from run_margin_band import draw_specs


def load_best_over_M(opt_path):
    """Reduce optimization_data_all_M*.npz down to one fixed winner per
    (gate time Tg, kind), the same "best-over-M" reduction control/idle.py's
    own report table performs.

    control/idle.py sweeps M = 1, 2, 4, ... for every Tg and, for each kind
    ('known' = best pulse-library/CDD sequence, 'opt' = best free-timing
    noise-tailored sequence), keeps whichever M gave the lowest infidelity.
    This function re-does that same reduction on the saved per-M arrays so
    the sequence this script re-scores under spectrum perturbation is
    identical to the one the paper actually reports.

    Returns a list of (Tg, kind, label, M, (pt1, pt2)) tuples -- pt1/pt2 are
    the per-qubit pulse-timing arrays (`jax.numpy` arrays, the type
    `idle.calculate_infidelity` expects).
    """
    # Deferred import: jax is a heavy (GPU-aware) dependency, so it is only
    # imported once we actually need to build jnp arrays, not at module load
    # (keeps `--help` and argument-parsing errors fast).
    import jax.numpy as jnp
    d = np.load(opt_path, allow_pickle=True)
    M_values = [int(m) for m in d['M_values']]
    # Every M's gate-time grid should coincide (control/idle.py builds each
    # from the same gate_time_factors), but collecting the union defensively
    # means a partial/differently-configured M-sweep still degrades
    # gracefully instead of raising a KeyError below.
    gts = sorted({round(float(g), 10) for m in M_values
                  for g in d[f'M{m}_gate_times']})
    entries = []
    for Tg in gts:
        for kind in ('known', 'opt'):
            best = None
            for m in M_values:
                m_gts = np.asarray(d[f'M{m}_gate_times'], dtype=float)
                # Tolerance-based match rather than `==`: Tg was rounded
                # above but the per-M arrays store the raw float, so an
                # equality test could miss a bit-identical-in-theory value.
                idx = np.where(np.abs(m_gts - Tg) < 1e-9)[0]
                if idx.size == 0:
                    continue
                i = int(idx[0])
                inf = float(d[f'M{m}_infs_{kind}'][i])
                seq = d[f'M{m}_sequences_{kind}'][i]
                # inf >= 1.0 is control/idle.py's sentinel for "no valid
                # sequence was found at this (Tg, M)" (e.g. the pulse
                # library was empty or every candidate failed), not a
                # genuinely bad-but-real infidelity -- skip it so it can
                # never be picked as the "best" M.
                if seq is None or inf >= 1.0:
                    continue
                if best is None or inf < best[0]:
                    lab = f"{str(d[f'M{m}_labels_{kind}'][i])}"
                    best = (inf, m, lab, seq)
            if best is None:
                continue
            inf, m, lab, seq = best
            entries.append((float(Tg), kind, lab, m,
                            (jnp.array(seq[0]), jnp.array(seq[1]))))
    return entries


def main():
    ap = argparse.ArgumentParser(
        description="Recon-uncertainty band on predicted idle infidelities "
                    "and the NT-vs-CDD margin (fixed best-over-M winners)")
    ap.add_argument('--folder', default=None,
                    help="run-folder name under the repo root supplying both "
                         "the specs/bars and the idle winners "
                         "(default: the active regime's NoSPAM folder)")
    ap.add_argument('--n-draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260611)
    ap.add_argument('--spectral-model', choices=('interp', 'selfconsistent'),
                    default='interp')
    ap.add_argument('--tag', type=str, default="",
                    help="suffix: read winners from optimization_data_all_M_"
                         "<tag>.npz and write margin_band_id_<tag>.npz "
                         # UNCAP-0611 tags the 2026-06-11 change that let
                         # control/idle.py's search consider sequences bounded
                         # only by the minimum pulse separation, instead of a
                         # small fixed pulse-count cap -- --tag exists so a
                         # rerun (e.g. with a different flag/regime) reads and
                         # writes its own filename suffix instead of
                         # clobbering a previous, possibly published, run's
                         # optimization_data_all_M<tag>.npz / margin_band_id
                         # <tag>.npz files.
                         "(UNCAP-0611)")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""

    # jax (and this project's control/idle module, which imports jax) is a
    # heavy, GPU-touching import; deferring it until after argparse has
    # already validated the command line keeps `--help`/bad-flag errors fast.
    from qns2q.control import idle  # heavy import (JAX) after arg parsing

    fname = a.folder or run_folder()
    # idle.Config is a plain class (not a @dataclass) whose __init__ does
    # real disk I/O right here -- it loads specs.npz/params.npz from
    # `fname` immediately, so this line raises FileNotFoundError right away
    # if Stage 2 hasn't been run for this folder yet, rather than failing
    # later inside the draw loop. gate_time_factors=[] is deliberate: we
    # only want the Config's loaded specs/SMat machinery here, not its own
    # (expensive) gate-time sweep.
    cfg = idle.Config(fname=fname, use_simulated=False,
                      spectral_model=a.spectral_model,
                      gate_time_factors=[])

    opt_path = os.path.join(project_root(), fname,
                            f'optimization_data_all_M{sfx}.npz')
    entries = load_best_over_M(opt_path)
    if not entries:
        raise ValueError(f"no winner sequences found in {opt_path}")

    print(f"[margin-band-id] specs+bars+winners: {fname} "
          f"({len(entries)} sequences) | {a.n_draws} draws, seed={a.seed}")
    for Tg, kind, lab, m, _ in entries:
        print(f"  Tg={Tg:8.1f} tau  {kind:5s} M={m:<3d} {lab}")

    # Snapshot the central (unperturbed) reconstructed spectra so they can be
    # restored after the draw loop below overwrites cfg.specs/cfg.SMat.
    central_specs = cfg.specs

    def eval_entries():
        # idle._OVERLAP_SETUP_CACHE memoizes per-(sequence, spectrum) overlap
        # setups; since cfg.SMat is mutated in place for each draw (not
        # replaced by a new Config object), a stale cache entry could
        # otherwise silently reuse a PREVIOUS draw's spectrum -- clearing it
        # here guarantees every call actually scores the current cfg.SMat.
        idle._OVERLAP_SETUP_CACHE.clear()
        # Idle convention: a sequence of M repeated blocks tiles the gate
        # time Tg, so each block lasts T_seq = Tg/M; calculate_infidelity's
        # third argument is that per-block time, not Tg itself (see the
        # module docstring's "M is part of the winner" note -- M is fixed
        # here, never re-optimized). use_ideal=False (matching how
        # control/idle.py itself grades candidates during the search, not
        # the always-exact use_ideal=True path it uses only for the single
        # published number) trades a small, previously-quantified accuracy
        # loss at large M for the speed needed to re-score every winner on
        # hundreds of perturbed spectra (see `use_comb_approximation`'s
        # docstring in control/idle.py for the bound).
        return np.array([
            float(idle.calculate_infidelity(seq, cfg, m, Tg / m, use_ideal=False))
            for Tg, _, _, m, seq in entries])

    central = eval_entries()

    rng = np.random.default_rng(a.seed)
    draws = np.empty((a.n_draws, len(entries)))
    for n in range(a.n_draws):
        # draw_specs (from run_margin_band.py) returns ONE perturbed-spectra
        # dict per call; _build_interpolated_spectra() re-derives the 3x3
        # frequency-domain SMat tensor from it (re-running the tail fit too),
        # so every draw gets its own consistently-perturbed spectral model.
        cfg.specs = draw_specs(central_specs, rng)
        cfg.SMat = cfg._build_interpolated_spectra()
        draws[n] = eval_entries()
        if (n + 1) % 25 == 0:
            print(f"  draw {n + 1}/{a.n_draws}")
    # Restore the central (unperturbed) state so cfg is left consistent for
    # any caller that inspects it after main() returns.
    cfg.specs = central_specs
    cfg.SMat = cfg._build_interpolated_spectra()
    idle._OVERLAP_SETUP_CACHE.clear()

    # 95% CI convention: 2.5th/97.5th percentile of the draw ensemble
    # brackets the central 95% of the distribution; the median is reported
    # alongside the (unperturbed) central value for comparison.
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    med = np.median(draws, axis=0)
    print("\nPredicted infidelity on the reconstructed spectra "
          "(central | median [95% CI]):")
    for j, (Tg, kind, lab, m, _) in enumerate(entries):
        print(f"  Tg={Tg:8.1f} tau  {kind:5s} M={m:<3d} {lab:18s} "
              f"{central[j]:.3e} | {med[j]:.3e} [{lo[j]:.3e}, {hi[j]:.3e}]")

    # Margin = known/opt (>1 means the noise-tailored "opt" sequence is
    # better) computed per draw, THEN percentiled -- not the percentile of
    # each side divided separately -- because a given draw's spectrum
    # realization is shared by both the 'known' and 'opt' entries at this
    # Tg, so genuinely correlated swings in the underlying noise cancel out
    # of the ratio instead of inflating the quoted uncertainty.
    print("\nNT-vs-known margin (known / opt, central | median [95% CI]):")
    tgs = sorted({Tg for Tg, _, _, _, _ in entries})
    margin_out = {}
    for Tg in tgs:
        jk = [j for j, e in enumerate(entries) if e[0] == Tg and e[1] == 'known']
        jo = [j for j, e in enumerate(entries) if e[0] == Tg and e[1] == 'opt']
        if not jk or not jo:
            continue
        ratio = draws[:, jk[0]] / draws[:, jo[0]]
        c = central[jk[0]] / central[jo[0]]
        rlo, rmed, rhi = np.percentile(ratio, [2.5, 50, 97.5])
        print(f"  Tg={Tg:8.1f} tau  {c:.2f}x | {rmed:.2f}x [{rlo:.2f}x, {rhi:.2f}x]"
              f"   ({entries[jk[0]][2]} vs {entries[jo[0]][2]})")
        margin_out[f'margin_{Tg:g}'] = ratio

    # Note: unlike run_margin_band.py's CZ version, this script does not call
    # a render_figure()/--replot step -- there is no standalone PDF here.
    # scripts/report_showcase_figs.py reads this npz directly (together with
    # the CZ script's margin_band_cz*.npz) to draw the paper's combined
    # "gates" panel, so a separate diagnostic plot for this file alone was
    # never needed.
    out_path = os.path.join(cfg.path, f"margin_band_id{sfx}.npz")
    np.savez(out_path,
             draws=draws, central=central,
             entry_Tg=np.array([e[0] for e in entries]),
             entry_kind=np.array([e[1] for e in entries]),
             entry_label=np.array([e[2] for e in entries]),
             entry_M=np.array([e[3] for e in entries]),
             n_draws=a.n_draws, seed=a.seed,
             specs_folder=fname, winners_from=fname,
             model_version=cfg.model_version,
             spectral_model=cfg.spectral_model,
             **margin_out)
    print(f"\nSaved ensemble to {out_path}")


if __name__ == '__main__':
    main()
