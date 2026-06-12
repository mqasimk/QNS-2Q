"""Figures for the SHOWCASE-0612 report (the engineered trap landscape).

Same four-figure set and styling as report_lorenza_figs.py (the 2026-06-11
report): setup_pub_rcparams('compact'), Okabe-Ito palette, dashed-black
analytic truth, top-center figure legends, grid alpha 0.25. Run under
QNS2Q_REGIME=showcase so the spectra module exports the showcase landscape.

Outputs -> reports/showcase_0612/figs/
  fig_model_spectra.pdf       six showcase spectra, comb teeth, DC points,
                              trap-line family + coupler resonance, NT window
  fig_spam_comparison.pdf     3x3 SPAM-arm reconstructions vs truth
  fig_gates.pdf               2x2: infidelity vs Tg + NT-margin band (CZ, idle)
  fig_design_experiments.pdf  the ablation ladder + SPAM-arm designs
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qns2q.characterize.reconstruct import setup_pub_rcparams
from qns2q.noise.spectra import (S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12,
                                 line_priors_per_channel)
from qns2q.paths import current_regime, project_root

C_REF = '#0072B2'   # blue (reference / NT)
C_RAW = '#999999'   # grey (raw / FID)
C_MIT = '#D55E00'   # vermillion (mitigated / CDD)
C_ROB = '#009E73'   # green (robust / Im parts)
C_BLD = '#CC79A7'   # magenta (line-blind smooth fit)

ROOT = project_root()
OUT = os.path.join(ROOT, "reports", "showcase_0612", "figs")
os.makedirs(OUT, exist_ok=True)

# Capture-grade arm (2026-06-12 evening: measurable-floor landscape, 128k
# shots / M=16 sweeps). The earlier 64k/v1-landscape outputs keep their tags.
RUN_GATES = os.path.join(ROOT, "DraftRun_NoSPAM_showcase_cap")
GATE_TAG = "_cap"
SPAM_FMT = os.path.join(ROOT, "DraftRun_SPAM_showcase_{arm}")

T2_FID = 3500.0       # showcase T2* (tau units; chi(T2*) = 1 by calibration)
NT_WINDOW = (0.258, 0.312)   # between the 4w0 line's +3sig and the top -3sig


def fig_model_spectra():
    w = np.linspace(1e-3, np.pi / 4, 4000)
    wk = 2 * np.pi * np.arange(1, 21) / 160.0
    panels = [(r"$S_{1,1}$ (qubit 1)", S_11, 'self'),
              (r"$S_{2,2}$ (qubit 2)", S_22, 'self'),
              (r"$S_{12,12}$ ($ZZ$)", S_1212, 'self'),
              (r"$S_{1,2}$", S_1_2, 'cross'),
              (r"$S_{1,12}$", S_1_12, 'cross'),
              (r"$S_{2,12}$", S_2_12, 'cross')]
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


ARM_STYLE = {
    'reference': dict(color=C_REF, marker='o', label='no SPAM (reference)'),
    'raw':       dict(color=C_RAW, marker='v', label='SPAM, unmitigated'),
    'mitigated': dict(color=C_MIT, marker='^', label='SPAM-mitigated'),
}


def _sig_parts(err):
    err = np.asarray(err)
    if np.iscomplexobj(err):
        return np.abs(np.real(err)), np.abs(np.imag(err))
    return np.abs(err), np.abs(err)


def fig_spam_comparison():
    from qns2q.characterize.systematics import analytic_spectra

    arms = {a: np.load(SPAM_FMT.format(arm=a) + "/specs.npz",
                       allow_pickle=True)
            for a in ('reference', 'raw', 'mitigated')}
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
    """Single-arm capture overlay: the reconstruction (with bars) tracking all
    six spectra of the engineered landscape -- the 'QNS actually captures the
    spectrum' figure."""
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


def _margin_quantiles(npz):
    tgs, lo, med, hi = [], [], [], []
    for key in sorted(k for k in npz.files if k.startswith('margin_')):
        q = np.percentile(npz[key], [2.5, 50, 97.5])
        tgs.append(float(key.split('_')[1]))
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    o = np.argsort(tgs)
    return (np.asarray(tgs)[o],) + tuple(np.asarray(v)[o] for v in (lo, med, hi))


def _idle_best_over_M():
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
    pd_cz = np.load(os.path.join(RUN_GATES, "plotting_data",
                                 f"plotting_data_cz_v2{GATE_TAG}.npz"),
                    allow_pickle=True)
    mb_cz_path = os.path.join(RUN_GATES, f"margin_band_cz{GATE_TAG}.npz")
    mb_cz = (np.load(mb_cz_path, allow_pickle=True)
             if os.path.exists(mb_cz_path) else None)
    mb_id_path = os.path.join(RUN_GATES, f"margin_band_id{GATE_TAG}.npz")
    mb_id = np.load(mb_id_path, allow_pickle=True) if os.path.exists(mb_id_path) else None
    idle_path = os.path.join(RUN_GATES, f"optimization_data_all_M{GATE_TAG}.npz")
    idl = _idle_best_over_M() if os.path.exists(idle_path) else None

    tg = np.asarray(pd_cz['taxis'], dtype=float)
    order = np.argsort(tg)
    tg = tg[order]
    cz_k = np.asarray(pd_cz['infs_known'], dtype=float)[order]
    cz_o = np.asarray(pd_cz['infs_opt'], dtype=float)[order]
    cz_np = np.asarray(pd_cz['infs_nopulse'], dtype=float)[order]

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
        if mb is None:
            ax.set_axis_off(); return
        tgs_m, lo, med, hi = _margin_quantiles(mb)
        ax.fill_between(tgs_m, lo, hi, color=C_REF, alpha=0.18,
                        label=r'95\% CI under recon.\ uncertainty')
        ax.plot(tgs_m, med, '-', color=C_REF, marker='s', ms=4,
                label='median margin')
        ax.axhline(1.0, color='k', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel(r"$T_G/\tau$")
        ax.set_ylabel(r"margin: best CDD\,/\,best NT")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    curve_panel(axs[0, 0], tg, cz_np, cz_k, cz_o,
                "(a) entangling (CZ) gate: infidelity vs gate time")
    margin_panel(axs[0, 1], mb_cz, "(b) entangling gate: NT margin over best CDD")
    if idl is not None:
        curve_panel(axs[1, 0], idl['Tg'], idl['fid'], idl['known'], idl['opt'],
                    "(c) idle gate (best over repetition number $M$)")
        margin_panel(axs[1, 1], mb_id, "(d) idle gate: NT margin over best CDD")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gates.pdf"), bbox_inches='tight')
    plt.close(fig)


def _ladder_panel(ax, entries, title):
    """The ablation ladder: every bar is the TRUE infidelity of the gate a
    designer ends up with, given what they know about the noise."""
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    cols = [e[2] for e in entries]
    x = np.arange(len(entries))
    ax.bar(x, vals, width=0.62, color=cols, alpha=0.88)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.12, f"{v:.2e}", ha='center', fontsize=6.5)
    ax.set_yscale('log')
    ax.set_ylim(min(vals) / 2.2, max(vals) * 3.5)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(r"true $1-F_\mathrm{pro}$")
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, alpha=0.25, axis='y')


def fig_design_experiments(ladder_cz, ladder_id=None, arms_cz=None,
                           arms_id=None):
    """ladder_*: list of (label, true_infidelity, color); arms_*: dict with
    keys reference/raw/mitigated -> (known_true, nt_true, nt_char)."""
    n_rows = 2 if ladder_id is not None else 1
    fig, axs = plt.subplots(n_rows, 2, figsize=(9.6, 3.4 * n_rows),
                            squeeze=False)
    _ladder_panel(axs[0, 0], ladder_cz,
                  "(a) entangling (CZ), $T_G=320\\tau$: the same device,\n"
                  "five levels of noise knowledge")
    if arms_cz:
        _arms_bars(axs[0, 1], arms_cz,
                   "(b) entangling (CZ): gates designed on the\n"
                   "SPAM arms' reconstructions")
    else:
        axs[0, 1].set_axis_off()
    if ladder_id is not None:
        _ladder_panel(axs[1, 0], ladder_id,
                      "(c) idle: the knowledge ladder")
        if arms_id:
            _arms_bars(axs[1, 1], arms_id, "(d) idle: SPAM-arm designs")
        else:
            axs[1, 1].set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_design_experiments.pdf"),
                bbox_inches='tight')
    plt.close(fig)


def _arms_bars(ax, arms, title):
    order = [a for a in ('reference', 'mitigated', 'raw') if a in arms]
    pretty = {'reference': 'reference\n(SPAM-free recon.)',
              'mitigated': 'SPAM-mitigated\nrecon.',
              'raw': 'raw\n(SPAM-biased recon.)'}
    x = np.arange(len(order))
    kv = [arms[a][0] for a in order]
    ntv = [arms[a][1] for a in order]
    ncv = [arms[a][2] for a in order]
    ax.bar(x - 0.22, kv, width=0.2, color=C_MIT, alpha=0.85,
           label='best CDD (true)')
    ax.bar(x, ntv, width=0.2, color=C_REF, alpha=0.85, label='best NT (true)')
    ax.bar(x + 0.22, ncv, width=0.2, color=C_REF, alpha=0.45,
           label='best NT (predicted)')
    for xi, v in zip(x, ntv):
        ax.text(xi, v * 1.04, f"{v:.2e}", ha='center', fontsize=6.5)
    ax.set_xticks(x, [pretty[a] for a in order], fontsize=8)
    ax.set_ylabel(r"$1-F_\mathrm{pro}$ at $T_G=320\tau$")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')


def _idle_penalty_panel(ax, title):
    """1Q-only design penalty vs Tg for the idle gate (best over M per Tg)."""
    def best_table(tag):
        d = np.load(os.path.join(
            RUN_GATES, f"optimization_data_all_M_{tag}.npz"), allow_pickle=True)
        Ms = [int(m) for m in d['M_values']]
        gts = sorted({round(float(g), 6) for m in Ms
                      for g in d[f'M{m}_gate_times']})
        out = {}
        for gt in gts:
            vals = []
            for m in Ms:
                mg = np.asarray(d[f'M{m}_gate_times'], dtype=float)
                ix = np.where(np.abs(mg - gt) < 1e-6)[0]
                if ix.size:
                    vals.append(float(d[f'M{m}_infs_opt'][int(ix[0])]))
            out[gt] = min(vals)
        return out
    full = best_table(GATE_TAG.lstrip('_'))
    solo = best_table('rung_c_idle')
    gts = sorted(full)
    pen = [solo[g] / full[g] for g in gts]
    ax.plot(gts, pen, '-', color=C_ROB, marker='o', ms=4,
            label='idle: 1Q-only design / full design')
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r"$T_G/\tau$")
    ax.set_ylabel(r"true-infidelity penalty")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)


def _load_design_data():
    """Assemble the ladder data from the capture-landscape rung outputs.

    The smoothfit/1Q rungs ran on the truth comb (--simulated; on this
    landscape the blind-on-recon and truth-comb variants agreed to ~2% in the
    v1 battery); CDD + full come from the blind capture-arm curve. The SPAM
    arms rerun overnight at capture grade -- panel omitted until then."""
    def opt_at(fname, tag):
        d = np.load(os.path.join(ROOT, fname, "plotting_data",
                                 f"plotting_data_cz_v2_{tag}.npz"),
                    allow_pickle=True)
        return float(np.asarray(d['infs_opt'], dtype=float)[0])

    pd = np.load(os.path.join(RUN_GATES, "plotting_data",
                              f"plotting_data_cz_v2{GATE_TAG}.npz"),
                 allow_pickle=True)
    tg = np.asarray(pd['taxis'], dtype=float)
    i320 = int(np.argmin(np.abs(tg - 320.0)))
    cdd_320 = float(np.asarray(pd['infs_known'], dtype=float)[i320])
    full_320 = float(np.asarray(pd['infs_opt'], dtype=float)[i320])

    ladder_cz = [
        ("best CDD\n(no spectral knowledge)", cdd_320, C_MIT),
        ("smooth fit of the\nsame comb (no lines)",
         opt_at("DraftRun_NoSPAM_showcase", "rung_b_cap"), C_BLD),
        ("1Q (2):\n$S_{1,1},S_{2,2}$ only",
         opt_at("DraftRun_NoSPAM_showcase", "rung_c_cap"), C_ROB),
        ("all 6", full_320, C_REF),
    ]
    return ladder_cz, None


def fig_design():
    ladder_cz, arms_cz = _load_design_data()
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
    _ladder_panel(ax, ladder_cz,
                  "entangling (CZ): blind NT gate vs spectral\n"
                  "knowledge given to the optimizer ($T_G=320\\tau$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_design_experiments.pdf"),
                bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    assert current_regime() == "showcase", "run with QNS2Q_REGIME=showcase"
    setup_pub_rcparams('compact')
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'spectra'):
        fig_model_spectra()
        print("fig_model_spectra done")
    if which in ('all', 'capture'):
        fig_recon_capture()
        print("fig_recon_capture done")
    if which in ('all', 'spam'):
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
    print(f"figures -> {OUT}")
