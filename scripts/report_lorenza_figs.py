"""Figures for the 2026-06-11 Lorenza report (TALK-0609 + CA-REPRO numbers).

Styling matches DraftRun_SPAM_featured_mitigated/figures/
spam_protocol_comparison.pdf: setup_pub_rcparams('compact') (usetex, Computer
Modern, inward ticks) + the Okabe-Ito palette, dashed-black analytic truth,
top-center figure legends, grid alpha 0.25.

Outputs -> reports/lorenza_0611/figs/
  fig_model_spectra.pdf       six anchored spectra, comb teeth, DC points,
                              line triplet, measured-vs-extrapolated band
  fig_gates.pdf               2x2: infidelity vs Tg + NT-margin band for the
                              entangling (CZ) and idle gates
  fig_design_experiments.pdf  spectral-knowledge subsets + SPAM-arm designs
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qns2q.characterize.reconstruct import setup_pub_rcparams
from qns2q.noise.spectra import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
from qns2q.paths import project_root

C_REF = '#0072B2'   # blue (reference / NT)
C_RAW = '#999999'   # grey (raw / FID)
C_MIT = '#D55E00'   # vermillion (mitigated / CDD)
C_ROB = '#009E73'   # green (robust / Im parts)

ROOT = project_root()
OUT = os.path.join(ROOT, "reports", "lorenza_0611", "figs")
os.makedirs(OUT, exist_ok=True)
# 2026-06-11 (user): all plots on the 64k runs for equal footing -- gate
# curves, margin bands, and design experiments read the reference arm.
RUN = os.path.join(ROOT, "DraftRun_SPAM_featured_reference")

T2_FID = 771.0          # Gaussian FID T2 (tau units), model output
W_MEASURED = 0.157      # 1 MHz at tau = 25 ns: top of the measured PSD window


def fig_model_spectra():
    w = np.linspace(1e-3, np.pi / 4, 3000)
    wk = 2 * np.pi * np.arange(1, 21) / 160.0
    panels = [(r"$S_{1,1}$ (qubit 1)", S_11, 'self'),
              (r"$S_{2,2}$ (qubit 2)", S_22, 'self'),
              (r"$S_{12,12}$ ($ZZ$)", S_1212, 'self'),
              (r"$S_{1,2}$", S_1_2, 'cross'),
              (r"$S_{1,12}$", S_1_12, 'cross'),
              (r"$S_{2,12}$", S_2_12, 'cross')]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.0), sharex=True)
    for ax, (title, fn, kind) in zip(axs.ravel(), panels):
        yc = np.asarray(fn(w))
        yk = np.asarray(fn(wk))
        ydc = complex(np.asarray(fn(np.array([0.0]))).ravel()[0])
        ax.axvspan(W_MEASURED, w[-1], color='k', alpha=0.05, lw=0,
                   label='exponents extrapolated\n(above measured window)')
        if kind == 'self':
            ax.plot(w, np.real(yc), 'k--', lw=1.0, label='anchored model')
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
            ax.set_yscale('symlog', linthresh=1e-5, linscale=0.6)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
    # Nuclear-difference triplet markers on the qubit-local channels only
    for j, c in enumerate((0.261, 0.273, 0.534)):
        for ax in (axs[0, 0], axs[0, 1]):
            ax.axvline(c, color=C_MIT, lw=1.0, alpha=0.8,
                       label=('nuclear Larmor-difference lines'
                              if ax is axs[0, 0] and j == 0 else None))
    axs[0, 2].annotate('no lines: $J$-noise\nis electrical', xy=(0.40, 0.55),
                       xycoords='axes fraction', fontsize=8, color=C_MIT)
    for ax in axs[1]:
        ax.set_xlabel(r"$\omega\tau$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$S(\omega)\,/\,\tau\quad[\mathrm{dimensionless}]$")
    handles, labels = [], []
    for ax in (axs[0, 0], axs[1, 0]):
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
    """3x3 arm-vs-truth overlay in the fig_model_spectra style: log/symlog
    axes with decade ticks (no asinh), top legend strip, labeled y axes."""
    from qns2q.characterize.systematics import analytic_spectra

    arms = {a: np.load(os.path.join(ROOT, f"DraftRun_SPAM_featured_{a}",
                                    "specs.npz"), allow_pickle=True)
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
            ax.set_yscale('symlog', linthresh=1e-5, linscale=0.6)
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


def _margin_quantiles(npz):
    tgs, lo, med, hi = [], [], [], []
    for key in sorted(k for k in npz.files if k.startswith('margin_')):
        q = np.percentile(npz[key], [2.5, 50, 97.5])
        tgs.append(float(key.split('_')[1]))
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    o = np.argsort(tgs)
    return (np.asarray(tgs)[o],) + tuple(np.asarray(v)[o] for v in (lo, med, hi))


# UNCAP-0611 (2026-06-12): the gate curves and margin bands now read the
# separation-limited (uncapped pulse budget) reruns; the published-cap files
# remain untagged alongside.
GATE_TAG = "_uncapped"


def _idle_best_over_M():
    d = np.load(os.path.join(RUN, f"optimization_data_all_M{GATE_TAG}.npz"),
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
    pd_cz = np.load(os.path.join(RUN, "plotting_data",
                                 f"plotting_data_cz_v2{GATE_TAG}.npz"),
                    allow_pickle=True)
    mb_cz = np.load(os.path.join(RUN, f"margin_band_cz{GATE_TAG}.npz"),
                    allow_pickle=True)
    mb_id = np.load(os.path.join(RUN, f"margin_band_id{GATE_TAG}.npz"),
                    allow_pickle=True)
    idl = _idle_best_over_M()

    tg = np.asarray(pd_cz['taxis'], dtype=float)
    order = np.argsort(tg)
    tg = tg[order]
    cz_k = np.asarray(pd_cz['infs_known'], dtype=float)[order]
    cz_o = np.asarray(pd_cz['infs_opt'], dtype=float)[order]
    cz_np = np.asarray(pd_cz['infs_nopulse'], dtype=float)[order]

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.8))

    def curve_panel(ax, x, fid, known, opt, title):
        ax.plot(x, fid, ':', color=C_RAW, marker='v', ms=4, label='FID (no pulse)')
        ax.plot(x, known, '-', color=C_MIT, marker='o', ms=4, label='best CDD')
        ax.plot(x, opt, '--', color=C_REF, marker='s', ms=4, label='best NT')
        ax.axvline(T2_FID, color='k', lw=0.7, ls='-', alpha=0.35)
        ax.text(T2_FID * 1.08, 0.04, r'$T_2$', fontsize=8, alpha=0.6,
                transform=ax.get_xaxis_transform())
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r"$T_G/\tau$")
        ax.set_ylabel(r"$1-F_\mathrm{pro}$ (true)")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    def margin_panel(ax, mb, title):
        tgs_m, lo, med, hi = _margin_quantiles(mb)
        ax.fill_between(tgs_m, lo, hi, color=C_REF, alpha=0.18,
                        label=r'95\% CI under recon.\ uncertainty')
        ax.plot(tgs_m, med, '-', color=C_REF, marker='s', ms=4,
                label='median margin')
        ax.axhline(1.0, color='k', lw=0.8, ls=':')
        ax.axvline(T2_FID, color='k', lw=0.7, ls='-', alpha=0.35)
        ax.text(T2_FID * 1.08, 0.04, r'$T_2$', fontsize=8, alpha=0.6,
                transform=ax.get_xaxis_transform())
        ax.set_xscale('log')
        ax.set_xlabel(r"$T_G/\tau$")
        ax.set_ylabel(r"margin: best CDD\,/\,best NT")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    curve_panel(axs[0, 0], tg, cz_np, cz_k, cz_o,
                "(a) entangling (CZ) gate: infidelity vs gate time")
    margin_panel(axs[0, 1], mb_cz, "(b) entangling gate: NT margin over best CDD")
    curve_panel(axs[1, 0], idl['Tg'], idl['fid'], idl['known'], idl['opt'],
                "(c) idle gate (best over repetition number $M$)")
    margin_panel(axs[1, 1], mb_id, "(d) idle gate: NT margin over best CDD")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gates.pdf"), bbox_inches='tight')
    plt.close(fig)


def _subset_panel(ax, res, tg_entries, title):
    """Absolute true NT infidelity of the blind winner per spectral-knowledge
    subset (log scale: the two gate times sit a factor ~6 apart)."""
    labels = ['all-6', 'robust-4', 'diag-3', '1Q-2']
    pretty = ['all 6', r'robust (4):' '\n' r'no $S_{1,12},S_{2,12}$',
              'diag (3):\nno crosses', r'1Q (2):' '\n' r'$S_{1,1},S_{2,2}$ only']
    shades = (0.55, 0.95)
    x = np.arange(len(labels))
    allv = []
    for k, ((Tg, leg), a) in enumerate(zip(tg_entries, shades)):
        vals = [res[(l, Tg)]['nt_true'] for l in labels]
        allv += vals
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.32, color=C_REF, alpha=a,
               label=leg)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v * 1.07, f"{v:.4f}", ha='center', fontsize=6)
    ax.set_yscale('log')
    ax.set_ylim(min(allv) / 1.6, max(allv) * 2.6)
    ax.set_xticks(x, pretty, fontsize=7)
    ax.set_ylabel(r"true $1-F_\mathrm{pro}$")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25, axis='y')


def _arms_panel(ax, res, Tg, title):
    """Gates designed on the SPAM-biased vs SPAM-free reconstruction."""
    labels2 = ['arm-reference', 'arm-raw']
    pretty2 = ['reference\n(SPAM-free recon.)', 'raw\n(SPAM-biased recon.)']
    ntv = [res[(l, Tg)]['nt_true'] for l in labels2]
    ncv = [res[(l, Tg)]['nt_char'] for l in labels2]
    kv = [res[(l, Tg)]['known_true'] for l in labels2]
    x = np.arange(2)
    ax.bar(x - 0.22, kv, width=0.2, color=C_MIT, alpha=0.85, label='best CDD (true)')
    ax.bar(x, ntv, width=0.2, color=C_REF, alpha=0.85, label='best NT (true)')
    ax.bar(x + 0.22, ncv, width=0.2, color=C_REF, alpha=0.45,
           label='best NT (predicted)')
    for xi, v in zip(x, ntv):
        ax.text(xi, v * 1.03, f"{v:.4f}", ha='center', fontsize=6.5)
    ax.set_xticks(x, pretty2, fontsize=8)
    ax.set_ylim(0, max(kv) * 1.25)
    ax.set_ylabel(rf"$1-F_\mathrm{{pro}}$ at $T_G={Tg:.0f}\tau$")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')


def fig_design_experiments():
    d_cz = np.load(os.path.join(RUN, "report_ask_experiments.npz"),
                   allow_pickle=True)
    res_cz = {(r['label'], r.get('Tg', 320.0)): r for r in d_cz['results']}
    d_id = np.load(os.path.join(RUN, "report_ask_experiments_idle.npz"),
                   allow_pickle=True)
    res_id = {(r['label'], r['Tg']): r for r in d_id['results']}

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.2))
    _subset_panel(axs[0, 0], res_cz,
                  ((80.0, r"$T_G=80\tau$"), (320.0, r"$T_G=320\tau$")),
                  "(a) entangling (CZ): blind NT gate vs\nspectral knowledge "
                  "given to the optimizer")
    _arms_panel(axs[0, 1], res_cz, 320.0,
                "(b) entangling (CZ): gates designed on\nSPAM-biased vs "
                "unbiased spectra")
    id_tgs = sorted({r['Tg'] for r in d_id['results']
                     if not str(r['label']).startswith('arm-')})
    id_entries = tuple(
        (Tg, rf"$T_G={Tg:.0f}\tau$ ($M{{=}}{int(res_id[('all-6', Tg)]['M'])}$)")
        for Tg in id_tgs)
    _subset_panel(axs[1, 0], res_id, id_entries,
                  "(c) idle: blind NT gate vs\nspectral knowledge given to "
                  "the optimizer")
    _arms_panel(axs[1, 1], res_id, 640.0,
                "(d) idle: gates designed on\nSPAM-biased vs unbiased spectra")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_design_experiments.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    setup_pub_rcparams('compact')
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'spectra'):
        fig_model_spectra()
    if which in ('all', 'spam'):
        fig_spam_comparison()
    if which in ('all', 'gates'):
        fig_gates()
    if which in ('all', 'design'):
        fig_design_experiments()
    print(f"figures -> {OUT}")
