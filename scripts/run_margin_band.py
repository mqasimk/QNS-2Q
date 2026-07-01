"""Propagate the noise-spectrum reconstruction's error bars into a CZ-gate
predicted-infidelity uncertainty band, and into the "does NT actually beat
the known/CDD sequence" margin band.

Physics role
------------
Stage 2 (`characterize/reconstruct.py`) does not just report a best-estimate
spectrum at each measured comb frequency (tooth) -- it also quotes, per
channel and per tooth, a statistical error bar (shot noise from the finite
number of QNS repetitions) and a systematic error bar (model bias from the
unfold/tail-extrapolation procedure). Stage 3a (`control/cz.py`) then picks
"winner" pulse sequences (the best known-library CDD/mqCDD sequence and the
best free-timing noise-tailored, "NT", sequence) using only the *central*
reconstructed spectrum, and reports each winner's predicted infidelity
1-F_pro on that central spectrum. That single number hides a real physics
question a referee will ask: given how uncertain the reconstruction is, how
much could the predicted infidelities -- and, more importantly, the NT-vs-
known margin (is NT actually better, or is that within the noise?) -- move
around? This script answers that by Monte-Carlo-resampling the reconstructed
spectrum within its quoted bars, many times, and re-computing each fixed
winner sequence's infidelity on every resampled spectrum, without re-running
the pulse-sequence search itself (the winners are frozen; only the noise
model they are graded against is perturbed).

Pipeline position
------------------
This is a **post-processing / diagnostics script**, downstream of both
pipeline arms: it reads Stage 2's error bars (`specs.npz`, `characterize/`)
and Stage 3a's winner sequences (`plotting_data/plotting_data_cz_v2*.npz`,
written by `control/cz.py::run_optimization`, `control/` arm), and its own
output (`margin_band_cz*.npz` + a companion PDF) feeds the "gates" panel of
`scripts/report_showcase_figs.py` (the paper's margin-band error bars) and
`scripts/run_margin_band_idle.py`, the identity/idling-gate twin of this
script (which reuses `draw_specs` below -- do not change its name or
argument order, it is imported by that file).

Inputs
------
  - The specs run folder (`--folder`, or the SPAM arm via `--protocol`,
    or the active regime's default NoSPAM folder): `specs.npz`'s central
    spectra plus their `<key>_err` (stat) / `<key>_sys` (sys) / `<key>_dc_ok`
    arrays, loaded through `qns2q.control.cz.CZOptConfig` (the same config
    class the optimizer itself uses, so the frequency grid / tail model /
    DC handling are identical between the winner search and this recheck).
  - The winners folder (`--winners-from`, default: same as the specs
    folder)'s `plotting_data/plotting_data_cz_v2*.npz`, i.e. the fixed
    pulse-timing sequences whose infidelity gets re-evaluated (this script
    never re-optimizes a sequence, only re-scores the ones Stage 3a already
    chose).

Outputs
-------
  - `<specs folder>/margin_band_cz<tag>.npz` -- per-draw infidelities for
    every winner sequence plus per-gate-time NT/known infidelity-ratio
    arrays (see `main()`'s `margin_out` dict).
  - `<specs folder>/margin_band_cz<tag>.pdf` -- the two-panel figure
    (`render_figure`): predicted-infidelity histograms per sequence, and
    the NT-vs-known margin histograms with their 95% CI.

Method: the resampling model
-----------------------------
Fixed winner sequences (saved by the CZ run's plotting_data) are re-evaluated
on an ensemble of spectra drawn within the reconstruction's stat + sys bars:

    S_c(w_k) -> S_c(w_k) + alpha_c * sys_c(w_k) + eps_{c,k} * stat_c(w_k)

with alpha_c ~ N(0,1) SHARED across all teeth of channel c -- the unfold-model
systematic is a coherent broad envelope (a single mis-estimated tail slope or
mis-modeled line shape shifts every tooth together in the same direction), and
drawing an INDEPENDENT sys value per tooth would let the errors cancel out
under the broadband overlap integral (central-limit averaging), which would
under-report how uncertain the gate's infidelity really is -- and eps_{c,k}
~ N(0,1) independent per tooth (shot noise: each tooth's measurement really is
statistically independent). Cross spectra draw Re/Im stat components
independently and scale the complex sys envelope by one real alpha. Self-
spectra are floored at 0 after the draw (a PSD cannot be negative, but a
noisy draw near zero could dip below it); flagged (`*_dc_ok = False`) DC
points are held at their stored floor value (that DC sample was itself
flagged unreliable at reconstruction time, so perturbing it further would
just add noise without adding information); all-NaN channels (robust SPAM
protocol, which cannot reconstruct S_1_12/S_2_12) pass through unperturbed
and are dropped by the SMat builder exactly as they are for the central
spectrum. Tail fits re-run on every draw (via
`CZOptConfig._build_interpolated_spectra`), so the tail-extrapolation
sensitivity to scatter in the top (highest-frequency) teeth is propagated
into the band too, not just the tooth-by-tooth error bars.

The same draw evaluates every sequence, so the margin band is the percentile
of the per-draw RATIO (common-spectrum correlation retained) -- i.e. on a
given draw the "quiet" or "noisy" spectrum realization is shared between the
NT and the known sequence, so genuinely correlated swings cancel in the
ratio -- rather than a quadrature combination of two independently-computed
bands, which would overstate the margin's uncertainty.

Usage (winners from the 6/11 probe, uncertainties from the reference arm):
    python scripts/run_margin_band.py --protocol reference \
        --winners-from DraftRun_NoSPAM_featured [--n-draws 200]
"""
import argparse
import os
import sys

# Make the `src/` package tree importable when this script is run directly
# (e.g. `python scripts/run_margin_band.py`) rather than installed -- Python
# only searches directories already on `sys.path`, so without this line
# `import qns2q...` below would fail. Every pipeline script in `scripts/`
# does this same insert; CLAUDE.md's "CWD-independent" guarantee relies on
# it (paths are still resolved from `qns2q.paths.project_root()`, not CWD).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.paths import run_folder, project_root

# specs.npz channel-name convention (matches Stage 2's on-disk keys, NOT the
# paper's S_11/S_1_2-style math notation): SELF_KEYS are the three diagonal
# spectra (qubit 1, qubit 2, Ising/"ZZ" coupling); CROSS_KEYS are the three
# off-diagonal cross-spectra. Every key in both tuples has companion
# `<key>_err` (stat) and `<key>_sys` (sys) arrays in specs.npz.
SELF_KEYS = ("S11", "S22", "S1212")
CROSS_KEYS = ("S12", "S112", "S212")


def draw_specs(specs, rng):
    """Draw ONE ensemble member: the central reconstructed spectra, each
    perturbed by one shared systematic shift plus independent per-tooth
    statistical noise (see the module docstring's "Method" section for the
    physics reasoning behind this specific draw model).

    NOTE for maintainers: this function is imported by
    `scripts/run_margin_band_idle.py` (`from run_margin_band import
    draw_specs`) so the CZ and idle-gate margin bands use IDENTICAL
    resampling -- do not rename it or change its argument order/meaning.

    Parameters
    ----------
    specs : Mapping[str, array-like]
        A `specs.npz`-shaped mapping (or the dict this function itself
        returns, for chaining): central spectra under `SELF_KEYS` +
        `CROSS_KEYS`, each with `<key>_err` (stat bar), `<key>_sys`
        (sys bar), and optionally `<key>_dc_ok` (bool: whether the w=0
        sample is trustworthy) siblings, plus the frequency grid `'wk'`.
    rng : numpy.random.Generator
        Source of the draw's random numbers (caller owns the seed so the
        whole ensemble is reproducible run-to-run).

    Returns
    -------
    dict
        Same keys/shapes as the input specs, with the perturbed spectra;
        suitable for assigning directly to `CZOptConfig.specs`.
    """
    out = {'wk': np.asarray(specs['wk'])}
    for key in SELF_KEYS + CROSS_KEYS:
        base = np.asarray(specs[key])
        if np.all(np.isnan(base)):  # robust arm: channel not reconstructed
            out[key] = base
            continue
        stat = np.asarray(specs[key + '_err'])
        sysv = np.asarray(specs[key + '_sys'])
        hold_dc = (f'{key}_dc_ok' in specs and not bool(specs[f'{key}_dc_ok']))
        # One shared alpha per channel per draw (not one per tooth): this is
        # what makes the systematic envelope move the whole comb together
        # instead of averaging away -- see the module docstring.
        alpha = rng.standard_normal()
        if key in SELF_KEYS:
            eps = rng.standard_normal(base.shape)
            d = base + alpha * sysv + eps * stat
            if hold_dc:
                # This channel's DC (w=0) tooth was flagged unreliable at
                # reconstruction time; hold it at its stored value instead
                # of perturbing it further.
                d[0] = base[0]
            out[key] = np.maximum(d, 0.0)  # PSD floor: a spectral density can't go negative
        else:
            # Cross-spectra are complex: perturb the real and imaginary
            # statistical parts independently, but still scale the (complex)
            # systematic envelope by the one shared real alpha above.
            eps_r = rng.standard_normal(base.shape)
            eps_i = rng.standard_normal(base.shape)
            d = (base + alpha * sysv
                 + eps_r * np.real(stat) + 1j * eps_i * np.imag(stat))
            if hold_dc:
                d[0] = base[0]
            out[key] = d
    return out


def load_winners(pdat):
    """Extract the fixed winner pulse sequences to re-evaluate from a CZ
    `plotting_data_cz_v2*.npz` (written by `control/cz.py::run_optimization`).

    Each returned entry is `(Tg, kind, label, (pt1, pt2))`:
      - `Tg` -- the gate time (in tau) this winner was optimized for.
      - `kind` -- `'known'` (best CDD/mqCDD library sequence) or `'opt'`
        (best free-timing noise-tailored, "NT", sequence); the whole point
        of the margin band is comparing these two per Tg.
      - `label` -- the human-readable sequence name used in plot legends.
      - `(pt1, pt2)` -- the qubit-1/qubit-2 pulse-time arrays (each
        includes the 0 and Tg endpoints, so the pulse COUNT is
        `pt.size - 2`), as `jax.numpy` arrays ready for
        `control.cz.calculate_infidelity`.

    Two on-disk layouts are supported: newer runs save one winner per gate
    time scanned (`'sequences_known' in pdat.files`, "per-Tg arrays");
    older runs only kept the single overall-best pair across all scanned
    gate times (the `best_known_seq_pt1`-style keys) -- this function
    normalizes both into the same list of entries.
    """
    # Deferred import: jax.numpy is only needed to box the raw pulse-time
    # arrays for control.cz's JAX-jitted infidelity evaluator, and JAX takes
    # a moment to import/initialize its backend -- doing this lazily here
    # (rather than at module load) keeps `--help`/argument-parsing errors
    # fast, matching the `from qns2q.control import cz` deferred import in
    # `main()` below.
    import jax.numpy as jnp
    entries = []
    if 'sequences_known' in pdat.files:
        for i, Tg in enumerate(np.asarray(pdat['taxis'], dtype=float)):
            for kind in ('known', 'opt'):
                s = pdat[f'sequences_{kind}'][i]
                if s is None:
                    continue
                lab = str(pdat[f'labels_{kind}'][i])
                entries.append((float(Tg), kind, lab,
                                (jnp.array(s[0]), jnp.array(s[1]))))
    else:
        for kind in ('known', 'opt'):
            k1, k2 = f'best_{kind}_seq_pt1', f'best_{kind}_seq_pt2'
            if k1 not in pdat.files:
                continue
            pt1, pt2 = np.asarray(pdat[k1]), np.asarray(pdat[k2])
            Tg = float(pdat[f'T_seq_best_{kind}'])
            lab = (f"NT({pt1.size - 2},{pt2.size - 2})" if kind == 'opt'
                   else f"known({pt1.size - 2},{pt2.size - 2})")
            entries.append((Tg, kind, lab, (jnp.array(pt1), jnp.array(pt2))))
    return entries


def render_figure(npz_path):
    """Render a saved ensemble `.npz` (from `main()`, below) into a
    two-panel PDF next to it, and return the PDF's path.

    Left panel: one histogram per fixed winner sequence, of its predicted
    infidelity 1-F_pro across the resampled-spectrum ensemble (the dashed
    line is the infidelity on the un-perturbed, central reconstructed
    spectrum -- i.e. the number Stage 3a actually reports). Right panel:
    one histogram per gate time Tg that has BOTH a known and an NT winner,
    of the per-draw ratio known/NT (>1 means NT wins on that draw); the
    shaded band is the 95% CI of that ratio, which is the actual "is NT
    robustly better" margin band this script exists to compute.
    """
    import matplotlib
    # Force the non-interactive "Agg" (Anti-Grain Geometry) renderer: this
    # script runs headless (no display), so the default interactive
    # backend would either fail to import or block waiting for a window
    # manager -- Agg rasterizes straight to the PDF file below instead.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    draws, central = d['draws'], d['central']
    labels = [str(x) for x in d['entry_label']]
    kinds = [str(x) for x in d['entry_kind']]
    tgs = np.asarray(d['entry_Tg'], dtype=float)
    margin_keys = sorted(k for k in d.files if k.startswith('margin_'))

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = {'known': 'C0', 'opt': 'C1'}
    for j in range(draws.shape[1]):
        c = colors.get(kinds[j], f'C{j}')
        axs[0].hist(draws[:, j], bins=30, alpha=0.55, color=c,
                    label=f"{labels[j]}, $T_G$={tgs[j]:g}$\\tau$")
        axs[0].axvline(central[j], color=c, ls='--', lw=1.2)
    axs[0].set_xlabel(r"predicted $1-F_\mathrm{pro}$ on reconstructed spectra")
    axs[0].set_ylabel("draws")
    axs[0].legend(fontsize=8)
    axs[0].set_title("recon-uncertainty ensemble (dashed: central)", fontsize=9)

    for i, k in enumerate(margin_keys):
        ratio = d[k]
        lo, med, hi = np.percentile(ratio, [2.5, 50, 97.5])
        axs[1].hist(ratio, bins=30, alpha=0.6, color=f'C{2 + i}',
                    label=f"$T_G$={k.split('_')[1]}$\\tau$: "
                          f"{med:.2f}x [{lo:.2f}, {hi:.2f}]")
        axs[1].axvspan(lo, hi, color=f'C{2 + i}', alpha=0.12)
    axs[1].axvline(1.0, color='k', lw=1, ls=':')
    axs[1].set_xlabel("margin: known / NT (>1 = NT wins)")
    axs[1].legend(fontsize=8)
    axs[1].set_title("NT-vs-known margin band (95% CI shaded)", fontsize=9)

    fig.tight_layout()
    out = os.path.splitext(npz_path)[0] + ".pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved to {out}")
    return out


def main():
    """CLI entry point: parse arguments, build the resampled-spectrum
    ensemble (or just re-plot a previously-saved one with `--replot`),
    evaluate every fixed winner sequence on every draw, print a summary
    table, save the ensemble `.npz`, and render its figure.

    PROTECTED CLI surface: `--folder`/`--protocol`/`--winners-from`/
    `--n-draws`/`--seed`/`--spectral-model`/`--replot`/`--tag` and the
    `margin_band_cz<tag>.npz` output filename pattern are relied on by
    other scripts/paper-figure commands (see `FIGURE_PROVENANCE.md`) --
    do not rename or repurpose them.
    """
    ap = argparse.ArgumentParser(
        description="Recon-uncertainty band on predicted CZ infidelities "
                    "and the NT-vs-CDD margin (fixed winner sequences)")
    # A mutually-exclusive group: argparse enforces that the user passes AT
    # MOST one of --folder/--protocol (erroring out otherwise), since they
    # are two different ways of picking the same one thing -- which specs
    # run folder to read.
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--folder', help="specs run-folder name under the repo "
                     "root (default: the active regime's NoSPAM folder)")
    src.add_argument('--protocol',
                     choices=('reference', 'raw', 'mitigated', 'robust'),
                     help="read the SPAM arm DraftRun_SPAM_<regime>_<protocol>")
    ap.add_argument('--winners-from', default=None,
                    help="folder whose plotting_data/plotting_data_cz_v2.npz "
                         "supplies the fixed winner sequences (default: the "
                         "specs folder)")
    ap.add_argument('--n-draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260611)
    ap.add_argument('--spectral-model', choices=('interp', 'selfconsistent'),
                    default='interp',
                    # OPT-SPECTRAL-MODEL tags every place in cz.py/idle.py
                    # (and here) that builds/selects between the two ways of
                    # turning the discrete measured comb into a continuous
                    # spectrum: 'interp' (straight line through the teeth)
                    # vs 'selfconsistent' (the same line/tail/head-aware
                    # physics-informed fit the bias-correction step uses).
                    # This flag must match whatever the winners were
                    # actually optimized under, or the re-evaluated
                    # infidelities/margins would be inconsistent with the
                    # sequences being scored.
                    help="characterized-SMat construction (OPT-SPECTRAL-MODEL)")
    ap.add_argument('--replot', action='store_true',
                    help="skip the ensemble; re-render the figure from the "
                         "existing margin_band_cz.npz in the specs folder")
    ap.add_argument('--tag', type=str, default="",
                    # UNCAP-0611 tags every place (here and in control/cz.py,
                    # control/idle.py) that threads a filename suffix through
                    # for a specific labeled run variant (e.g. the showcase
                    # run's "_cap" pulse-count-cap tag) -- grep for it to
                    # find every producer/consumer of a given tagged file.
                    help="suffix: read winners from plotting_data_cz_v2_<tag>"
                         ".npz and write margin_band_cz_<tag>.npz (UNCAP-0611)")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""

    if a.replot:
        specs_fname = a.folder or (run_folder(spam=True, protocol=a.protocol)
                                   if a.protocol else run_folder())
        render_figure(os.path.join(project_root(), specs_fname,
                                   f"margin_band_cz{sfx}.npz"))
        return

    from qns2q.control import cz  # heavy import (JAX) after arg parsing

    specs_fname = a.folder or (run_folder(spam=True, protocol=a.protocol)
                               if a.protocol else run_folder())
    # Constructing a CZOptConfig does real disk I/O (loads specs.npz/params.npz
    # and builds the frequency-grid spectral matrices in __post_init__) --
    # see the "dataclass __post_init__ doing disk I/O" note in cz.py's own
    # docstring for why that pattern is used instead of a plain data class.
    cfg = cz.CZOptConfig(fname=specs_fname, use_simulated=False,
                         spectral_model=a.spectral_model,
                         gate_time_factors=[])

    winners_fname = a.winners_from or specs_fname
    pd_path = os.path.join(project_root(), winners_fname,
                           'plotting_data', f'plotting_data_cz_v2{sfx}.npz')
    pdat = np.load(pd_path, allow_pickle=True)
    if str(pdat['gate_type']) != 'cz':
        raise ValueError(f"{pd_path} is not a CZ plotting-data file")
    entries = load_winners(pdat)
    if not entries:
        raise ValueError(f"no winner sequences found in {pd_path}")

    print(f"[margin-band] specs+bars: {specs_fname} | winners: {winners_fname} "
          f"({len(entries)} sequences) | {a.n_draws} draws, seed={a.seed}")

    # M = number of times the base pulse block repeats within the gate (the
    # `idle`/DD optimizer sweeps this to reach longer idle durations); a CZ
    # gate is evaluated as a single, one-shot application of its sequence,
    # so M is fixed at 1 here rather than being another axis to scan.
    M = 1
    central_specs = cfg.specs

    # A closure (a function defined inside main() that captures its
    # enclosing variables -- here cfg, M, entries): reused unchanged for
    # both the one central evaluation and every resampled draw below, so
    # the exact same evaluation code path is guaranteed to run every time.
    def eval_entries():
        # cz.calculate_infidelity memoizes expensive per-(SMat, w_grid, Tg,
        # M) setup work in the module-level _OVERLAP_SETUP_CACHE, keyed
        # partly by Python object id(). Each draw below assigns cfg.SMat to
        # a brand-new array, but Python can and does reuse a freed array's
        # memory address for the next one -- so an id()-based key could
        # collide with a STALE entry from a previous, different-content
        # SMat if we didn't clear the cache first, silently reusing setup
        # data for the wrong spectrum. Clearing before every evaluation
        # closes that hole.
        cz._OVERLAP_SETUP_CACHE.clear()
        return np.array([
            float(cz.calculate_infidelity(seq, cfg, M, Tg, use_ideal=False))
            for Tg, _, _, seq in entries])

    central = eval_entries()

    rng = np.random.default_rng(a.seed)
    draws = np.empty((a.n_draws, len(entries)))
    for n in range(a.n_draws):
        # Perturb the spectra (draw_specs), rebuild the interpolated/tail-
        # extended SMat from that perturbed data, then re-score every fixed
        # winner sequence against it -- repeated a.n_draws times to build up
        # the ensemble.
        cfg.specs = draw_specs(central_specs, rng)
        cfg.SMat = cfg._build_interpolated_spectra()
        draws[n] = eval_entries()
        if (n + 1) % 25 == 0:
            print(f"  draw {n + 1}/{a.n_draws}")
    # Restore the central (un-perturbed) state so cfg is left in the same
    # condition it would be in without this script's resampling loop, in
    # case anything downstream inspects it further.
    cfg.specs = central_specs
    cfg.SMat = cfg._build_interpolated_spectra()
    cz._OVERLAP_SETUP_CACHE.clear()

    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    med = np.median(draws, axis=0)
    print("\nPredicted infidelity on the reconstructed spectra "
          "(central | median [95% CI]):")
    for j, (Tg, kind, lab, _) in enumerate(entries):
        print(f"  Tg={Tg:6.1f} tau  {kind:5s} {lab:14s} "
              f"{central[j]:.3e} | {med[j]:.3e} [{lo[j]:.3e}, {hi[j]:.3e}]")

    # Margin = known/opt per draw at each Tg with both present. Because the
    # SAME draw's ratio is used (not independently-drawn known and opt
    # samples), correlated swings in the shared perturbed spectrum cancel
    # in this ratio -- see the module docstring's note on why this is a
    # tighter, more honest band than combining two separate CI's would be.
    print("\nNT-vs-known margin (known / opt, central | median [95% CI]):")
    tgs = sorted({Tg for Tg, _, _, _ in entries})
    margin_out = {}
    for Tg in tgs:
        # entries can have more than one 'known'/'opt' winner at the same Tg
        # only in unusual inputs; take the first of each (matching how the
        # rest of the pipeline treats "the" winner at a given gate time).
        jk = [j for j, e in enumerate(entries) if e[0] == Tg and e[1] == 'known']
        jo = [j for j, e in enumerate(entries) if e[0] == Tg and e[1] == 'opt']
        if not jk or not jo:
            continue
        ratio = draws[:, jk[0]] / draws[:, jo[0]]
        c = central[jk[0]] / central[jo[0]]
        rlo, rmed, rhi = np.percentile(ratio, [2.5, 50, 97.5])
        print(f"  Tg={Tg:6.1f} tau  {c:.2f}x | {rmed:.2f}x [{rlo:.2f}x, {rhi:.2f}x]"
              f"   ({entries[jk[0]][2]} vs {entries[jo[0]][2]})")
        margin_out[f'margin_{Tg:g}'] = ratio

    # This is the file `render_figure` (above) and the paper's showcase
    # panel (scripts/report_showcase_figs.py) read back: the per-draw
    # infidelity table (`draws`/`central`), which entry each column is
    # (`entry_Tg`/`entry_kind`/`entry_label`), the per-Tg margin ratio
    # arrays (`margin_out`, expanded via `**`), and provenance metadata
    # (RNG seed, which folders/model version were used) so a saved figure
    # can always be traced back to exactly how it was produced.
    out_path = os.path.join(cfg.path, f"margin_band_cz{sfx}.npz")
    np.savez(out_path,
             draws=draws, central=central,
             entry_Tg=np.array([e[0] for e in entries]),
             entry_kind=np.array([e[1] for e in entries]),
             entry_label=np.array([e[2] for e in entries]),
             n_draws=a.n_draws, seed=a.seed,
             specs_folder=specs_fname, winners_from=winners_fname,
             model_version=cfg.model_version,
             spectral_model=cfg.spectral_model,
             **margin_out)
    print(f"\nSaved ensemble to {out_path}")
    render_figure(out_path)


if __name__ == '__main__':
    main()
