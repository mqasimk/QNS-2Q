"""Stage 4 of the pipeline: assemble the paper's showcase-regime figures.

Physics/pipeline role
----------------------
This is the FINAL step of the two-arm pipeline described in CLAUDE.md: Stage 1
(`characterize/experiments.py` via `scripts/run_capture_arm.py`) simulates the
QNS experiments, Stage 2 (`characterize/reconstruct.py`) inverts them into
reconstructed noise power spectral densities (`specs.npz`), Stage 3
(`control/cz.py`, `control/idle.py`) optimizes control pulses against those
spectra. This script does NOT run any new physics simulation or optimization
-- it only *reads* the small summary `.npz` files those earlier stages already
wrote to disk (all committed to the repo; see the "Run folders" table in
FIGURE_PROVENANCE.md) and turns them into the eight PDF panels used in the
manuscript (`main_v10.tex`). Six of the eight are produced here; the other two
("standalone" figures) are produced directly by `scripts/run_single_qubit.py`
and `scripts/run_cz_pulse_plot.py` -- see FIGURE_PROVENANCE.md for the
authoritative figure-to-data map instead of re-deriving it by reading the code.

What each figure needs (inputs) and what comes out (outputs) is documented on
each `fig_*` function below. At a high level: this script reads reconstructed
spectra (`specs.npz`) from the no-SPAM run and from the four SPAM-protocol run
folders, the gate-optimization outputs (`plotting_data_cz_v2*.npz`,
`optimization_data_all_M*.npz`, `margin_band_*.npz`), the pre-harvested design
ladder (`design_numbers.npz`, built by `scripts/harvest_design_numbers.py`),
the storage-protocol panel (`storage_panel.npz`, built by
`scripts/showcase_storage_panel.py`), and the analytic noise-model functions
in `qns2q.noise.spectra` directly (for the "ground truth" dashed curves) --
and writes one PDF per `fig_*` function to `SHOWCASE_FIGS_DIR` (see below).
Nothing in the rest of the package imports this file; it is a leaf script run
directly from the command line (`python scripts/report_showcase_figs.py`).

Historical note on the styling (kept only because a reader may find the tag
in old commit messages): this reuses the same look -- `setup_pub_rcparams
('compact')`, Okabe-Ito colorblind-safe palette, dashed-black analytic-truth
curves, top-center figure legends, grid alpha 0.25 -- as an earlier "2026-06-11
report" script, `report_lorenza_figs.py`, which was deleted in the `CLEANUP-0616`
repo cleanup (see CLAUDE.md) once this showcase-specific version superseded it.
Must be run with `QNS2Q_REGIME=showcase` (enforced by an assert in `__main__`)
so that `qns2q.noise.spectra` exports the showcase noise landscape rather than
the `bland`/`featured` regimes used elsewhere in development.

Outputs -> reports/showcase_0612/figs/ by default, but in practice always
overridden via `SHOWCASE_FIGS_DIR` (see CLAUDE.md / FIGURE_PROVENANCE.md,
which use `reports/showcase_0613/figs/`) so a new report revision doesn't
clobber an older one that a figure may still point at:
  fig_model_spectra.pdf       six showcase spectra, comb teeth, DC points,
                              trap-line family + coupler resonance, NT window
  fig_spam_comparison.pdf     3x3 SPAM-arm reconstructions vs truth
  fig_gates.pdf               2x2: infidelity vs Tg + NT-margin band (CZ, idle)
  fig_design_experiments.pdf  the ablation ladder + SPAM-arm designs
  fig_recon_capture.pdf       single-arm reconstruction vs truth (blind 256k run)
  fig_storage.pdf             Bell-pair (entanglement) storage infidelity panel
"""
import os
import sys

# This script lives in scripts/, not inside the installed src/qns2q package,
# so Python would not find `qns2q` on its own; this line manually adds
# ../src to the import search path (sys.path) before any `qns2q...` import
# below can succeed. This is what lets every scripts/*.py be run directly
# ("python scripts/foo.py") from the repo root without a pip install step.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend: render straight to PDF, no display needed (safe on a headless cluster/CI)
import matplotlib.pyplot as plt
import numpy as np

from qns2q.characterize.reconstruct import setup_pub_rcparams
from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 line_priors_per_channel)
from qns2q.paths import current_regime, project_root

# Okabe-Ito colorblind-safe palette, reused consistently across every panel so
# a given color always means the same experimental condition (e.g. C_REF is
# always "no SPAM / reference / NT-optimized" in every figure in this file).
C_REF = '#0072B2'   # blue (reference / NT)
C_RAW = '#999999'   # grey (raw / FID)
C_MIT = '#D55E00'   # vermillion (mitigated / CDD)
C_ROB = '#009E73'   # green (robust / Im parts)
C_BLD = '#CC79A7'   # magenta (line-blind smooth fit)

ROOT = project_root()
# Output dir overridable so a new showcase revision writes to its own folder
# (e.g. SHOWCASE_FIGS_DIR=reports/showcase_0613/figs) without clobbering 0612.
# Part of this script's env-var contract (see CLAUDE.md / FIGURE_PROVENANCE.md):
# always set this explicitly when regenerating the paper's figures.
OUT = os.path.join(ROOT, os.environ.get("SHOWCASE_FIGS_DIR",
                                        "reports/showcase_0612/figs"))
os.makedirs(OUT, exist_ok=True)

# Capture-grade arm: the run folder with enough shots/repetitions (128k shots,
# M=16 CPMG repetitions swept) that the reconstructed spectra sit clearly above
# the shot-noise floor everywhere the paper needs them to (as opposed to an
# earlier, noisier "v1-landscape" run whose outputs are kept under different
# filename tags so they aren't silently mixed with this one).
# Gate-data folder overridable (e.g. SHOWCASE_RUN_GATES=DraftRun_NoSPAM_showcase_cap_backup0612
# to render figures from a backed-up run without disturbing the live folder).
# Second half of this script's env-var contract (with SHOWCASE_FIGS_DIR above);
# see CLAUDE.md / FIGURE_PROVENANCE.md for the canonical value used for the paper.
RUN_GATES = os.path.join(ROOT, os.environ.get("SHOWCASE_RUN_GATES",
                                              "DraftRun_NoSPAM_showcase_cap"))
# "_cap" ("capture-grade") is baked into every filename this script reads out of
# RUN_GATES below (e.g. plotting_data_cz_v2_cap.npz). It is a naming CONTRACT
# shared with scripts/harvest_design_numbers.py and scripts/run_carrier_battery_0616.sh
# -- do not change it without updating those other two scripts to match, or the
# files this script expects to find simply won't exist under the new name.
GATE_TAG = "_cap"
SPAM_FMT = os.path.join(ROOT, "DraftRun_SPAM_showcase_{arm}")

T2_FID = 3500.0       # showcase T2* (tau units; chi(T2*) = 1 by calibration).
                      # T2* is the free-induction-decay coherence time of the
                      # noise model itself, used below only to draw a vertical
                      # reference line on the infidelity-vs-gate-time plots.
NT_WINDOW = (0.258, 0.312)   # between the 4w0 line's +3sig and the top -3sig:
                             # a frequency band with no noise lines in it, i.e.
                             # a "quiet" spectral window a noise-tailored (NT)
                             # pulse sequence can be designed to exploit.
JMAX_TAU = 0.05       # max Ising coupling J_max*tau (cz.CZOptConfig.Jmax);
                      # sets the CZ floor T_G >= pi/(4 J_max) ~ 16 tau -- the
                      # shortest possible CZ gate time given the hardware's max
                      # coupling strength, annotated on the CZ panel below.


def fig_model_spectra():
    """Panel 1/6: the six showcase noise-model spectra themselves (no QNS data
    involved -- this plots the *analytic* model functions from
    qns2q.noise.spectra directly), annotated with the noise-line locations and
    the quiet window a noise-tailored pulse exploits.

    Reads: nothing from disk -- calls the showcase-regime S_11/S_22/S_1212/
    S_1_2/S_1_12/S_2_12 spectral functions and line_priors_per_channel()
    directly (both defined in qns2q.noise.spectra; the showcase regime is
    selected at import time by QNS2Q_REGIME, asserted in __main__ below).
    Writes: fig_model_spectra.pdf.
    """
    w = np.linspace(1e-3, np.pi / 4, 4000)   # fine grid for the smooth analytic curve
    # wk: the actual discrete frequencies (harmonics of a period-160tau comb) that
    # a QNS comb experiment samples -- overlaid as points on top of the smooth
    # curve so the reader can see which frequencies are directly measured.
    wk = 2 * np.pi * np.arange(1, 21) / 160.0
    panels = [(r"$S_{1,1}$ (qubit 1)", S_11, 'self'),
              (r"$S_{2,2}$ (qubit 2)", S_22, 'self'),
              (r"$S_{12,12}$ ($ZZ$)", S_1212, 'self'),
              (r"$S_{1,2}$", S_1_2, 'cross'),
              (r"$S_{1,12}$", S_1_12, 'cross'),
              (r"$S_{2,12}$", S_2_12, 'cross')]
    # line_priors_per_channel(): the known center frequencies of each noise
    # "line" (defect/two-level-fluctuator resonance) built into the showcase
    # model, per channel -- used only to draw the vertical marker lines below,
    # not for any reconstruction.
    pri = line_priors_per_channel()
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.0), sharex=True)
    for ax, (title, fn, kind) in zip(axs.ravel(), panels):
        yc = np.asarray(fn(w))
        yk = np.asarray(fn(wk))
        ydc = complex(np.asarray(fn(np.array([0.0]))).ravel()[0])
        ax.axvspan(*NT_WINDOW, color=C_REF, alpha=0.08, lw=0,
                   label='quiet band between the lines\n(exploited by the tailored gates)')
        if kind == 'self':
            ax.plot(w, np.real(yc), 'k--', lw=1.0, label='featured model')
            ax.plot(wk, np.real(yk), 'o', ms=3.5, color=C_REF,
                    label='comb harmonics (QNS samples)')
            ax.plot([0.0], [np.real(ydc)], 'D', ms=4.5, mfc='none',
                    color=C_MIT, label='dedicated DC experiment')
            ax.set_yscale('log')
        else:
            ax.plot(w, np.real(yc), 'k--', lw=1.0)
            ax.plot(w, np.imag(yc), '-.', lw=1.0, color=C_ROB,
                    label='model, $\\mathrm{Im}$ part')
            ax.plot(wk, np.real(yk), 'o', ms=3.5, color=C_REF)
            ax.plot(wk, np.imag(yk), 'o', ms=3.0, mfc='none', color=C_ROB)
            ax.plot([0.0], [np.real(ydc)], 'D', ms=4.5, mfc='none', color=C_MIT)
            ax.set_yscale('symlog', linthresh=1e-9, linscale=0.5)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
    # trap-line markers: defect-harmonic family on the qubit-local channels,
    # coupler resonance on the ZZ channel ONLY
    c11 = pri['S11'][0]
    for j, c in enumerate(c11):
        for ax in (axs[0, 0], axs[0, 1]):
            ax.axvline(c, color=C_MIT, lw=1.1, alpha=0.9,
                       label=('defect line + harmonics'
                              if ax is axs[0, 0] and j == 0 else None))
    zz_c = pri['S1212'][0][0]
    axs[0, 2].axvline(zz_c, color=C_ROB, lw=1.1, alpha=0.9,
                      label='line on the exchange channel only')
    axs[0, 2].annotate('visible only to\ntwo-qubit QNS',
                       xy=(0.42, 0.78), xycoords='axes fraction', fontsize=8,
                       color=C_ROB)
    # the shared-TLF line is correlated: it sits on S11/S22 (already marked) AND
    # in Re S_1_2 -- the cross-channel peak the decoupling train passes.
    sh_c = pri['S12'][0][0]
    axs[1, 0].axvline(sh_c, color=C_MIT, lw=1.1, alpha=0.9,
                      label='shared-TLF line (correlated: on $S_{1,1},S_{2,2}$\nand $\\mathrm{Re}\\,S_{1,2}$)')
    for ax in axs[1]:
        ax.set_xlabel(r"$\omega\tau$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$S(\omega)\,/\,\tau\quad[\mathrm{dimensionless}]$")
    handles, labels = [], []
    for ax in (axs[0, 0], axs[0, 2], axs[1, 0]):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_model_spectra.pdf"), bbox_inches='tight')
    plt.close(fig)


# The four SPAM ("state preparation and measurement" error) handling protocols
# from CLAUDE.md's SPAM pipeline section -- 'reference' is the no-SPAM-error
# control arm, 'raw' applies no error mitigation at all, 'mitigated' and
# 'robust' are the two mitigation strategies under test. Each maps to one
# `DraftRun_SPAM_showcase_<arm>/` run folder (see SPAM_FMT above) and gets a
# fixed color/marker so it reads the same way in every panel of this file.
ARM_STYLE = {
    'reference': dict(color=C_REF, marker='o', label='no SPAM (reference)'),
    'raw':       dict(color=C_RAW, marker='v', label='SPAM, unmitigated'),
    'mitigated': dict(color=C_MIT, marker='^', label='SPAM-mitigated'),
    'robust':    dict(color=C_ROB, marker='D', label='SPAM-robust (4 spectra)'),
}


def _sig_parts(err):
    """Split a reconstruction-error array into (real-part error bar, imaginary-
    part error bar) magnitudes, so the same call works whether `err` is complex
    (cross-spectra, which have both parts) or real (self-spectra, where both
    "parts" are just the one error bar)."""
    err = np.asarray(err)
    if np.iscomplexobj(err):
        return np.abs(np.real(err)), np.abs(np.imag(err))
    return np.abs(err), np.abs(err)


def fig_spam_comparison():
    """Panel 3/6: overlay all four SPAM-protocol reconstructions (each its own
    marker/color, per ARM_STYLE) against the analytic ground truth, one 3x3
    grid of spectra (self-spectra top row, cross-spectra real/imag below).

    Reads: DraftRun_SPAM_showcase_{reference,raw,mitigated,robust}/specs.npz
    (whichever exist on disk -- an arm is silently skipped if its folder/file
    is missing) plus the analytic truth via
    qns2q.characterize.systematics.analytic_spectra(). Writes: fig_spam_comparison.pdf.
    """
    from qns2q.characterize.systematics import analytic_spectra

    # robust included when present: its self-spectra carry the C_l,12=0 leakage
    # bias and its S_1,12/S_2,12 are NaN (not reconstructible -- the SPAM-robust
    # protocol simply cannot estimate those two cross-spectra, per CLAUDE.md) --
    # the panels show both effects directly rather than hiding them.
    arms = {a: np.load(SPAM_FMT.format(arm=a) + "/specs.npz", allow_pickle=True)
            for a in ('reference', 'raw', 'mitigated', 'robust')
            if os.path.exists(SPAM_FMT.format(arm=a) + "/specs.npz")}
    truth = analytic_spectra()
    wk = np.asarray(next(iter(arms.values()))['wk'])
    w_fine = np.linspace(0, wk.max() * 1.02, 2000)
    dw = wk[1] - wk[0]
    offsets = np.linspace(-0.22, 0.22, len(arms)) * dw

    SELF = [('S11', r"$S_{1,1}$"), ('S22', r"$S_{2,2}$"),
            ('S1212', r"$S_{12,12}$")]
    CROSS = [('S12', r"$S_{1,2}$"), ('S112', r"$S_{1,12}$"),
             ('S212', r"$S_{2,12}$")]

    fig, axs = plt.subplots(3, 3, figsize=(9.6, 6.9), sharex=True)

    def panel(ax, key, part, title, yscale):
        tr_f = np.asarray(truth[key](w_fine))
        tr_f = np.real(tr_f) if part == 're' else np.imag(tr_f)
        ax.plot(w_fine, tr_f, 'k--', lw=1.0, label='analytic truth', zorder=1)
        for off, (a, d) in zip(offsets, arms.items()):
            rec = np.asarray(d[key])
            rec_p = np.real(rec) if part == 're' else np.imag(rec)
            sig_re, sig_im = _sig_parts(d[f'{key}_errtot'])
            sig = sig_re if part == 're' else sig_im
            st = ARM_STYLE[a]
            ax.errorbar(wk + off, rec_p, yerr=sig, fmt=st['marker'], ms=3.2,
                        color=st['color'], ecolor=st['color'], elinewidth=0.8,
                        capsize=1.6, lw=0, label=st['label'], zorder=3,
                        alpha=0.9)
        if yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog', linthresh=1e-7, linscale=0.6)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)

    for j, (key, lab) in enumerate(SELF):
        panel(axs[0, j], key, 're', lab, 'log' if key != 'S1212' else 'symlog')
    for j, (key, lab) in enumerate(CROSS):
        panel(axs[1, j], key, 're', r'$\mathrm{Re}\,$' + lab, 'symlog')
        panel(axs[2, j], key, 'im', r'$\mathrm{Im}\,$' + lab, 'symlog')
    for j in range(3):
        axs[2, j].set_xlabel(r'$\omega\tau$')
    for i in range(3):
        axs[i, 0].set_ylabel(r"$S(\omega)\,/\,\tau$")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_spam_comparison.pdf"),
                bbox_inches='tight')
    plt.close(fig)


def fig_recon_capture():
    """Panel 2/6: single-arm capture overlay: the reconstruction (with error
    bars) tracking all six spectra of the engineered landscape -- the 'QNS
    actually captures the spectrum' figure (no-SPAM run, all six channels,
    unlike fig_spam_comparison which is SPAM-arm-focused).

    Reads: <RUN_GATES>/specs.npz (the no-SPAM showcase reconstruction) plus
    the analytic truth via qns2q.characterize.systematics.analytic_spectra().
    Writes: fig_recon_capture.pdf.
    """
    from qns2q.characterize.systematics import analytic_spectra

    d = np.load(os.path.join(RUN_GATES, "specs.npz"), allow_pickle=True)
    truth = analytic_spectra()
    wk = np.asarray(d['wk'])
    w_fine = np.linspace(0, wk.max() * 1.02, 2000)

    SELF = [('S11', r"$S_{1,1}$"), ('S22', r"$S_{2,2}$"),
            ('S1212', r"$S_{12,12}$")]
    CROSS = [('S12', r"$S_{1,2}$"), ('S112', r"$S_{1,12}$"),
             ('S212', r"$S_{2,12}$")]

    fig, axs = plt.subplots(3, 3, figsize=(9.6, 6.9), sharex=True)

    def panel(ax, key, part, title, yscale):
        tr_f = np.asarray(truth[key](w_fine))
        tr_f = np.real(tr_f) if part == 're' else np.imag(tr_f)
        ax.plot(w_fine, tr_f, 'k--', lw=1.0, label='analytic truth',
                zorder=1)
        rec = np.asarray(d[key])
        rec_p = np.real(rec) if part == 're' else np.imag(rec)
        sig_re, sig_im = _sig_parts(d[f'{key}_errtot'])
        sig = sig_re if part == 're' else sig_im
        ax.errorbar(wk, rec_p, yerr=sig, fmt='o', ms=3.2, color=C_REF,
                    ecolor=C_REF, elinewidth=0.8, capsize=1.6, lw=0,
                    label='QNS reconstruction', zorder=3, alpha=0.9)
        if yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog', linthresh=1e-8, linscale=0.5)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)

    for j, (key, lab) in enumerate(SELF):
        panel(axs[0, j], key, 're', lab, 'log')
    for j, (key, lab) in enumerate(CROSS):
        panel(axs[1, j], key, 're', r'$\mathrm{Re}\,$' + lab, 'symlog')
        panel(axs[2, j], key, 'im', r'$\mathrm{Im}\,$' + lab, 'symlog')
    for j in range(3):
        axs[2, j].set_xlabel(r'$\omega\tau$')
    for i in range(3):
        axs[i, 0].set_ylabel(r"$S(\omega)\,/\,\tau$")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_recon_capture.pdf"),
                bbox_inches='tight')
    plt.close(fig)


def _margin_quantiles(*npzs):
    """Merge one or more margin-band files (e.g. the cap + cap_short CZ runs)
    into a single sorted (Tg, lo, med, hi) quantile table.

    A "margin" here is the ratio (best known/CDD-sequence infidelity) /
    (best noise-tailored, NT, sequence infidelity) at a fixed gate time Tg --
    >1 means the NT-optimized pulse genuinely beats the best sequence found
    without using the reconstructed spectra. Each margin_band_*.npz file
    (written upstream by scripts/run_margin_band.py, not this script) stores
    many Monte Carlo samples of that ratio per gate time, drawn by propagating
    the reconstruction's statistical+systematic error bars through to the
    predicted infidelity; `lo`/`med`/`hi` are the 2.5/50/97.5 percentiles of
    those samples, i.e. a 95% confidence band on the margin, not on the raw
    infidelity itself.
    """
    tgs, lo, med, hi = [], [], [], []
    for npz in npzs:
        if npz is None:
            continue
        for key in sorted(k for k in npz.files if k.startswith('margin_')):
            q = np.percentile(npz[key], [2.5, 50, 97.5])
            tgs.append(float(key.split('_')[1]))
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    o = np.argsort(tgs)
    return (np.asarray(tgs)[o],) + tuple(np.asarray(v)[o] for v in (lo, med, hi))


def _idle_best_over_M():
    """For the idle (identity/dynamical-decoupling) gate, the same total gate
    time Tg can be built from different numbers of CPMG repetitions M (e.g.
    M=1 long block vs M=4 shorter blocks concatenated); the optimizer in
    control/idle.py sweeps M independently. This helper collapses that sweep
    down to, for every gate time actually run at some M, the BEST (lowest)
    infidelity achieved at ANY M -- 'known' (best fixed/reference sequence)
    and 'opt' (best noise-tailored sequence) -- so the gates figure below can
    show a single "best over M" curve per Tg instead of one curve per M.

    Reads: <RUN_GATES>/optimization_data_all_M<GATE_TAG>.npz.
    Returns: dict of Tg / fid (no-pulse baseline) / known / opt arrays.
    """
    d = np.load(os.path.join(RUN_GATES, f"optimization_data_all_M{GATE_TAG}.npz"),
                allow_pickle=True)
    Ms = [int(m) for m in d['M_values']]
    gts = sorted({round(float(g), 10) for m in Ms
                  for g in d[f'M{m}_gate_times']})
    out = {'Tg': [], 'fid': [], 'known': [], 'opt': []}
    m1g = np.asarray(d['M1_gate_times'], dtype=float)
    for gt in gts:
        best = {}
        for kind in ('known', 'opt'):
            vals = []
            for m in Ms:
                mg = np.asarray(d[f'M{m}_gate_times'], dtype=float)
                ix = np.where(np.abs(mg - gt) < 1e-9)[0]
                if ix.size:
                    vals.append(float(d[f'M{m}_infs_{kind}'][int(ix[0])]))
            best[kind] = min(vals)
        ix = np.where(np.abs(m1g - gt) < 1e-9)[0]
        out['Tg'].append(gt)
        out['fid'].append(float(d['M1_infs_nopulse'][int(ix[0])]))
        out['known'].append(best['known'])
        out['opt'].append(best['opt'])
    return {k: np.asarray(v) for k, v in out.items()}


def fig_gates():
    """Panel 4/6: the headline gate-performance figure -- 2x2 grid, entangling
    (CZ) gate on top row, idle (dynamical-decoupling) gate on bottom row (only
    if idle data is present). Left column: true infidelity vs total gate time
    Tg for three cases -- free induction ('FID', no pulses at all), the best
    *known* fixed sequence family (CPMG/CDD -- concatenated dynamical
    decoupling, a standard sequence NOT informed by the reconstructed
    spectrum), and the best noise-tailored ('NT') sequence found by the
    optimizer using the reconstructed spectra. Right column: the resulting
    "NT margin" (best-CDD / best-NT infidelity ratio) with its reconstruction-
    uncertainty confidence band from _margin_quantiles.

    Reads: <RUN_GATES>/plotting_data/plotting_data_cz_v2<GATE_TAG>[_short].npz,
    <RUN_GATES>/margin_band_{cz,id}<GATE_TAG>[_short].npz,
    <RUN_GATES>/optimization_data_all_M<GATE_TAG>.npz (idle row, if present).
    Writes: fig_gates.pdf.
    """
    def _load_cz(tag):
        return np.load(os.path.join(RUN_GATES, "plotting_data",
                                    f"plotting_data_cz_v2{tag}.npz"),
                       allow_pickle=True)

    pd_parts = [_load_cz(GATE_TAG)]
    # The optional "_short" file is a supplementary CZ sweep at additional,
    # typically shorter, gate times (e.g. filling in near the J_max-limited
    # floor) computed in a separate optimizer run; when present it is
    # concatenated onto the main sweep below rather than replacing it.
    short_path = os.path.join(RUN_GATES, "plotting_data",
                              f"plotting_data_cz_v2{GATE_TAG}_short.npz")
    if os.path.exists(short_path):
        pd_parts.append(np.load(short_path, allow_pickle=True))

    def _mb(name):
        p = os.path.join(RUN_GATES, name)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    mb_cz = [_mb(f"margin_band_cz{GATE_TAG}.npz"),
             _mb(f"margin_band_cz{GATE_TAG}_short.npz")]
    mb_id = [_mb(f"margin_band_id{GATE_TAG}.npz")]
    idle_path = os.path.join(RUN_GATES, f"optimization_data_all_M{GATE_TAG}.npz")
    idl = _idle_best_over_M() if os.path.exists(idle_path) else None

    tg = np.concatenate([np.asarray(p['taxis'], dtype=float) for p in pd_parts])
    cz_k = np.concatenate([np.asarray(p['infs_known'], dtype=float)
                           for p in pd_parts])
    cz_o = np.concatenate([np.asarray(p['infs_opt'], dtype=float)
                           for p in pd_parts])
    cz_np = np.concatenate([np.asarray(p['infs_nopulse'], dtype=float)
                            for p in pd_parts])
    order = np.argsort(tg)
    tg, cz_k, cz_o, cz_np = tg[order], cz_k[order], cz_o[order], cz_np[order]

    n_rows = 2 if idl is not None else 1
    fig, axs = plt.subplots(n_rows, 2, figsize=(9.6, 2.9 * n_rows + 0.2),
                            squeeze=False)

    def curve_panel(ax, x, fid, known, opt, title):
        ax.plot(x, fid, ':', color=C_RAW, marker='v', ms=4, label='FID (no pulse)')
        ax.plot(x, known, '-', color=C_MIT, marker='o', ms=4, label='best CDD')
        ax.plot(x, opt, '--', color=C_REF, marker='s', ms=4, label='best NT')
        ax.axvline(T2_FID, color='k', lw=0.7, ls='-', alpha=0.35)
        ax.text(T2_FID * 1.08, 0.04, r'$T_2^*$', fontsize=8, alpha=0.6,
                transform=ax.get_xaxis_transform())
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r"$T_G/\tau$")
        ax.set_ylabel(r"$1-F_\mathrm{pro}$ (true)")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    def margin_panel(ax, mb, title):
        mb = [m for m in mb if m is not None]
        if not mb:
            ax.set_axis_off(); return
        tgs_m, lo, med, hi = _margin_quantiles(*mb)
        ax.fill_between(tgs_m, lo, hi, color=C_REF, alpha=0.18,
                        label=r'95\% CI under recon.\ uncertainty')
        ax.plot(tgs_m, med, '-', color=C_REF, marker='s', ms=4,
                label='median margin')
        ax.axhline(1.0, color='k', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$T_G/\tau$")
        ax.set_ylabel(r"margin: best CDD\,/\,best NT")
        ax.grid(True, alpha=0.25, which='both')
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    curve_panel(axs[0, 0], tg, cz_np, cz_k, cz_o,
                "(a) entangling (CZ) gate: infidelity vs gate time")
    axs[0, 0].text(0.965, 0.04, rf'$J_{{\max}}\tau = {JMAX_TAU:g}$',
                   transform=axs[0, 0].transAxes, fontsize=8.5,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7',
                             lw=0.5))
    margin_panel(axs[0, 1], mb_cz, "(b) entangling gate: NT margin over best CDD")
    if idl is not None:
        curve_panel(axs[1, 0], idl['Tg'], idl['fid'], idl['known'], idl['opt'],
                    "(c) idle gate (best over repetition number $M$)")
        margin_panel(axs[1, 1], mb_id, "(d) idle gate: NT margin over best CDD")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gates.pdf"), bbox_inches='tight')
    plt.close(fig)


def _arms_bars(ax, arms, title, ylabel=r"$1-F_\mathrm{pro}$ at $T_G=320\tau$"):
    """Draw one 'SPAM-arm design' bar panel: for each SPAM protocol arm, a
    dark bar (the gate design's TRUE infidelity, measured against the
    analytic model) next to a lighter bar (that same gate's infidelity as
    PREDICTED from the arm's own -- possibly SPAM-biased -- reconstruction),
    with the percentage gap between the two annotated above each light bar.

    `arms` maps arm name -> a 3-element array/tuple whose index [1] is the
    true infidelity and [2] is the predicted/certified infidelity (index [0],
    unused here, is a third quantity harvest_design_numbers.py also stores).

    The SPAM story in one panel: the DESIGN is SPAM-invariant (best-NT TRUE
    infidelity, dark bars, sits on the dashed line for every arm, since the
    physical gate performance cannot depend on how you estimated the noise)
    while the CERTIFICATION is not (best-NT PREDICTED on each arm's own
    reconstruction, light bars: raw over-certifies -- i.e. claims a better
    gate than it actually built -- while mitigation/robust recover the truth).
    """
    order = [a for a in ('reference', 'mitigated', 'robust', 'raw') if a in arms]
    pretty = {'reference': 'reference\n(SPAM-free recon.)',
              'mitigated': 'SPAM-mitigated\nrecon.',
              'robust': 'SPAM-robust\nrecon. (4 spectra)',
              'raw': 'raw\n(SPAM-biased recon.)'}
    x = np.arange(len(order))
    ntv = [arms[a][1] for a in order]
    ncv = [arms[a][2] for a in order]
    true_val = float(np.mean(ntv))            # SPAM-invariant design value
    ax.axhline(true_val, ls='--', lw=1.0, color='0.35', zorder=1,
               label='true infidelity (SPAM-invariant)')
    ax.bar(x - 0.12, ntv, width=0.22, color=C_REF, alpha=0.85, zorder=2,
           label='best NT (true)')
    ax.bar(x + 0.12, ncv, width=0.22, color=C_REF, alpha=0.40, zorder=2,
           label='best NT (predicted/certified)')
    # annotate the certification error (predicted vs the SPAM-invariant truth)
    for xi, v in zip(x + 0.12, ncv):
        pct = 100.0 * (v - true_val) / true_val
        ax.text(xi, v * 1.05, f"{pct:+.0f}\\%", ha='center', fontsize=6.5,
                color=(C_RAW if abs(pct) > 3 else '0.3'))
    ax.set_xticks(x, [pretty[a] for a in order], fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    # Headroom above the (near-equal-height) predicted bars and their +x%%
    # annotations, so the legend sits clear; loc='best' otherwise lands on top
    # of the right-column bars.
    ax.set_ylim(top=max(ncv) * 1.7)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.25, axis='y')


# The "knowledge ladder": four increasingly-complete sets of reconstructed
# spectra that a gate could be (re-)optimized against, from "only the two
# single-qubit self-spectra" up to "the full six-spectrum model" -- this shows
# how much the gate design improves as more of the noise environment is
# characterized. The string keys ('rung_c', 'diag3', 'robust4', 'full') are a
# NAMING CONTRACT: they must match, verbatim, both the '<prefix>_<key>[_<Tg>]'
# fields harvest_design_numbers.py writes into design_numbers.npz (read below
# in fig_design/_ladder_grouped) and the run-folder suffixes
# (`_diag3`, `_robust4`) that scripts/run_carrier_battery_0616.sh produces
# upstream -- do not rename any of the four keys without updating both of
# those other scripts.
LADDER_LABELS = [
    ('rung_c', "1Q only (2):\n$S_{1,1},S_{2,2}$", C_ROB),
    ('diag3', "selfs (3):\n$+\\,S_{12,12}$", C_BLD),
    ('robust4', "robust (4):\n$+\\,S_{1,2}$", C_MIT),
    ('full', "all six\nspectra", C_REF),
]


def _ladder_grouped(ax, dn, prefix, tgs, title):
    """Draw one knowledge-ladder bar panel: one bar group per LADDER_LABELS
    rung, x-axis = how much of the reconstructed spectrum the optimizer was
    allowed to see, y-axis = the resulting gate's TRUE infidelity (log scale)
    -- reads `dn[f'{prefix}_{rung_key}[_{int(tg)}]']` for each rung. If two
    gate times `tgs` are given (used for the idle-gate panel, which is
    compared at two different total gate times), each rung gets a pair of
    bars (lighter = first Tg, darker = second) side by side instead of one.
    """
    x = np.arange(len(LADDER_LABELS))
    width = 0.34 if len(tgs) == 2 else 0.62
    all_vals = []
    for j, tg in enumerate(tgs):
        sfx = f"_{int(tg)}" if len(tgs) == 2 else ""
        vals = [float(dn[f'{prefix}_{k}{sfx}']) for k, _, _ in LADDER_LABELS]
        all_vals += vals
        off = (j - (len(tgs) - 1) / 2) * (width + 0.04)
        ax.bar(x + off, vals, width=width,
               color=[c for _, _, c in LADDER_LABELS],
               alpha=0.55 if (len(tgs) == 2 and j == 0) else 0.95,
               label=f"$T_G={int(tg)}\\tau$")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v * 1.12, f"{v:.1e}", ha='center', fontsize=5.8,
                    rotation=0)
    ax.set_yscale('log')
    ax.set_ylim(min(all_vals) / 2.5, max(all_vals) * 4.0)
    ax.set_xticks(x, [l for _, l, _ in LADDER_LABELS], fontsize=7)
    ax.set_ylabel(r"true $1-F_\mathrm{pro}$")
    ax.set_title(title, fontsize=9.5)
    if len(tgs) == 2:
        ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, alpha=0.25, axis='y')


def fig_design():
    """Panel 5/6: the "what does the gate design actually need to know about
    the noise" figure -- 2x2 grid, entangling (CZ) gate top row, idle gate
    bottom row. Left column: the knowledge ladder (see LADDER_LABELS /
    _ladder_grouped) -- how gate infidelity improves as the optimizer is given
    more of the reconstructed spectrum. Right column: the SPAM-arm comparison
    (see _arms_bars) -- how SPAM errors bias the CERTIFIED infidelity even
    though the underlying DESIGN stays SPAM-invariant. (Style/layout inherited
    from the same earlier "2026-06-11" report generator mentioned in this
    file's module docstring; this is functionally its own showcase-regime
    figure now.)

    Reads: <RUN_GATES>/design_numbers.npz, which is produced offline (not by
    this script) by scripts/harvest_design_numbers.py -- see that script /
    FIGURE_PROVENANCE.md if the numbers here need to be re-derived from the
    underlying optimization runs. Writes: fig_design_experiments.pdf.
    """
    dn = np.load(os.path.join(RUN_GATES, "design_numbers.npz"))
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.8))
    _ladder_grouped(axs[0, 0], dn, 'cz_ladder', [320.0],
                    "(a) entangling (CZ), $T_G=320\\tau$: blind NT design\n"
                    "vs the optimizer's spectral knowledge")
    _arms_bars(axs[0, 1], {a: dn[f'cz_arm_{a}']
                           for a in ('reference', 'mitigated', 'robust', 'raw')
                           if f'cz_arm_{a}' in dn.files},
               "(b) entangling (CZ): gates designed on the\n"
               "SPAM arms' reconstructions ($T_G=320\\tau$)")
    _ladder_grouped(axs[1, 0], dn, 'id_ladder', [640.0, 10240.0],
                    "(c) idle (best over $M$): the same ladder")
    _arms_bars(axs[1, 1], {a: dn[f'id_arm_{a}']
                           for a in ('reference', 'mitigated', 'robust', 'raw')
                           if f'id_arm_{a}' in dn.files},
               "(d) idle: SPAM-arm designs ($T_G=640\\tau$)",
               ylabel=r"$1-F_\mathrm{pro}$ at $T_G=640\tau$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_design_experiments.pdf"),
                bbox_inches='tight')
    plt.close(fig)


def fig_storage():
    """Panel 6/6: entanglement-storage panel -- holding one of the two
    "even-parity" Bell pairs, Phi+ = (|00>+|11>)/sqrt(2) or
    Psi+ = (|01>+|10>)/sqrt(2), alive through an idling period Tg, and asking
    how well different pulse-timing choices protect it.

    The physics point: `aligned` (y2=+y1) and `opposed` (y2=-y1) are two ways
    to apply the SAME per-qubit CPMG decoupling train to both qubits, just
    with the second qubit's pulses either in phase or antiphase with the
    first's. Measuring either qubit alone (single-qubit QNS), or even the
    average two-qubit gate fidelity, cannot tell these two choices apart --
    but because the two qubits' noise is CORRELATED (a nonzero real part of
    the cross-spectrum Re S_1_2), the two choices protect the TWO Bell states
    differently: only two-qubit QNS (which measures Re S_1_2) can predict, in
    advance, that the opposed train is the one that protects Phi+.

    Reads: <RUN_GATES>/storage_panel.npz, produced offline (not by this
    script) by scripts/showcase_storage_panel.py. Writes: fig_storage.pdf.
    """
    d = np.load(os.path.join(RUN_GATES, "storage_panel.npz"))
    tg = np.asarray(d['Tg'], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.6))
    ax.plot(tg, d['fid_phi'], ':', color=C_RAW, marker='v', ms=4,
            label='free induction, $\\Phi^+$')
    ax.plot(tg, d['fid_psi'], '-.', color=C_RAW, marker='^', ms=4, mfc='none',
            label='free induction, $\\Psi^+$ (protected)')
    ax.plot(tg, d['sync_phi'], '-', color=C_MIT, marker='o', ms=4,
            label='aligned pulses, $\\Phi^+$\n($y_2=+y_1$)')
    ax.plot(tg, d['anti_phi'], '-', color=C_REF, marker='s', ms=4,
            label='opposed pulses, $\\Phi^+$\n($y_2=-y_1$)')
    ax.plot(tg, d['nt_phi'], '--', color=C_ROB, marker='D', ms=3.5,
            label='blind NT idle (avg.-fidelity winner)')
    ax.plot(tg, d['sync_phi_pred'], 'o', ms=7, mfc='none', color=C_MIT,
            alpha=0.7, label='predicted (blind recon.), aligned')
    ax.plot(tg, d['anti_phi_pred'], 's', ms=7, mfc='none', color=C_REF,
            alpha=0.7, label='predicted (blind recon.), opposed')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"storage time $T_G/\tau$")
    ax.set_ylabel(r"Bell-pair infidelity $1-F$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
              handlelength=1.8, labelspacing=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_storage.pdf"), bbox_inches='tight')
    plt.close(fig)


# `if __name__ == "__main__":` is the standard Python idiom for "only run this
# block when the file is executed directly (`python report_showcase_figs.py`),
# not when some other file merely imports names from it" -- harmless here
# since nothing else imports this file, but it is the pattern that lets a
# script double as an importable module.
if __name__ == "__main__":
    # Fail fast rather than silently plotting the wrong noise model: this
    # entire file assumes the showcase-regime spectra/run folders (see the
    # module docstring), so refuse to run under bland/featured.
    assert current_regime() == "showcase", "run with QNS2Q_REGIME=showcase"
    setup_pub_rcparams('compact')
    # Optional CLI arg selects a single figure to (re)build (e.g.
    # `python report_showcase_figs.py gates`) instead of the full set --
    # convenient when only one panel changed and the others are slow/unneeded
    # to regenerate. Defaults to 'all'.
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'spectra'):
        fig_model_spectra()
        print("fig_model_spectra done")
    if which in ('all', 'capture'):
        fig_recon_capture()
        print("fig_recon_capture done")
    if which in ('all', 'spam'):
        # The SPAM-arm run folders are the most likely to be a scratch/backup
        # copy that isn't present in every checkout (see FIGURE_PROVENANCE.md);
        # skip gracefully rather than aborting the whole figure batch.
        try:
            fig_spam_comparison()
            print("fig_spam_comparison done")
        except FileNotFoundError as e:
            print(f"fig_spam_comparison SKIPPED ({e})")
    if which in ('all', 'gates'):
        fig_gates()
        print("fig_gates done")
    if which in ('all', 'design'):
        fig_design()
        print("fig_design_experiments done")
    if which in ('all', 'storage'):
        fig_storage()
        print("fig_storage done")
    print(f"figures -> {OUT}")
