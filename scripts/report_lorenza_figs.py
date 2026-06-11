"""Figures for the 2026-06-11 Lorenza report (TALK-0609 + CA-REPRO numbers).

Styling matches DraftRun_SPAM_featured_mitigated/figures/
spam_protocol_comparison.pdf: setup_pub_rcparams('compact') (usetex, Computer
Modern, inward ticks) + the Okabe-Ito palette, dashed-black analytic truth,
top-center figure legends, grid alpha 0.25.

Outputs -> reports/lorenza_0611/figs/
  fig_model_spectra.pdf   six anchored spectra + comb teeth (ask 1)
  fig_cz_curve_band.pdf   CZ infidelity vs Tg + NT-margin band (CA-REPRO)
  fig_asks.pdf            knowledge subsets (ask 2) + SPAM arms (ask 3)
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
C_ROB = '#009E73'   # green (robust)

ROOT = project_root()
OUT = os.path.join(ROOT, "reports", "lorenza_0611", "figs")
os.makedirs(OUT, exist_ok=True)
RUN = os.path.join(ROOT, "DraftRun_NoSPAM_featured")


def fig_model_spectra():
    w = np.linspace(1e-3, np.pi / 4, 3000)
    wk = 2 * np.pi * np.arange(1, 21) / 160.0
    panels = [(r"$S_{1,1}$", S_11, False), (r"$S_{2,2}$", S_22, False),
              (r"$S_{12,12}$", S_1212, False),
              (r"$\mathrm{Re}\,S_{1,2}$", lambda x: np.real(np.asarray(S_1_2(x))), False),
              (r"$\mathrm{Re}\,S_{1,12}$", lambda x: np.real(np.asarray(S_1_12(x))), False),
              (r"$\mathrm{Re}\,S_{2,12}$", lambda x: np.real(np.asarray(S_2_12(x))), False)]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=True)
    for ax, (title, fn, _) in zip(axs.ravel(), panels):
        y = np.asarray(fn(w), dtype=float)
        yk = np.asarray(fn(wk), dtype=float)
        ax.plot(w, y, 'k--', lw=1.0, label='anchored model')
        ax.plot(wk, yk, 'o', ms=3.5, color=C_REF, label='comb teeth (QNS samples)')
        ax.set_yscale('log' if np.all(y > 0) else 'symlog')
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for j, (c, lab) in enumerate(zip((0.261, 0.273, 0.534),
                                     ('nuclear-difference lines', None, None))):
        for ax in (axs[0, 0], axs[0, 1]):
            ax.axvline(c, color=C_MIT, lw=0.7, alpha=0.6,
                       label=lab if ax is axs[0, 0] and j == 0 else None)
    for ax in axs[1]:
        ax.set_xlabel(r"$\omega\tau$")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle("Anchored two-qubit dephasing model (featured regime); "
                 r"$T_2^{*}=800\tau=20\,\mu\mathrm{s}$ at $\tau=25$ ns",
                 y=1.06, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_model_spectra.pdf"), bbox_inches='tight')
    plt.close(fig)


def fig_cz_curve_band():
    pd = np.load(os.path.join(RUN, "plotting_data", "plotting_data_cz_v2.npz"),
                 allow_pickle=True)
    mb = np.load(os.path.join(RUN, "margin_band_cz.npz"), allow_pickle=True)
    tg = np.asarray(pd['taxis'], dtype=float)
    order = np.argsort(tg)
    tg = tg[order]
    inf_k = np.asarray(pd['infs_known'], dtype=float)[order]
    inf_o = np.asarray(pd['infs_opt'], dtype=float)[order]
    inf_np = np.asarray(pd['infs_nopulse'], dtype=float)[order]

    tgs_m, med, lo, hi = [], [], [], []
    for key in sorted(k for k in mb.files if k.startswith('margin_')):
        ratio = mb[key]
        tgs_m.append(float(key.split('_')[1]))
        q = np.percentile(ratio, [2.5, 50, 97.5])
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    o2 = np.argsort(tgs_m)
    tgs_m = np.asarray(tgs_m)[o2]
    lo, med, hi = (np.asarray(v)[o2] for v in (lo, med, hi))

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axs[0]
    ax.plot(tg, inf_np, ':', color=C_RAW, marker='v', ms=4, label='FID (no pulse)')
    ax.plot(tg, inf_k, '-', color=C_MIT, marker='o', ms=4, label='best CDD')
    ax.plot(tg, inf_o, '--', color=C_REF, marker='s', ms=4, label='best NT')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$T_G/\tau$")
    ax.set_ylabel(r"$1-F_\mathrm{pro}$ (true)")
    ax.grid(True, alpha=0.25)
    ax.set_title("CZ infidelity vs gate time (anchored model)")

    ax = axs[1]
    ax.fill_between(tgs_m, lo, hi, color=C_REF, alpha=0.18,
                    label=r'95\% CI (recon ensemble, 200 draws)')
    ax.plot(tgs_m, med, '-', color=C_REF, marker='s', ms=4, label='median margin')
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r"$T_G/\tau$")
    ax.set_ylabel(r"margin: best CDD / best NT")
    ax.grid(True, alpha=0.25)
    ax.set_title("NT advantage under reconstruction uncertainty")
    for a in axs:
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cz_curve_band.pdf"), bbox_inches='tight')
    plt.close(fig)


def fig_asks():
    d = np.load(os.path.join(RUN, "report_ask_experiments.npz"), allow_pickle=True)
    res = {(r['label'], r.get('Tg', 320.0)): r for r in d['results']}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5))

    # Left: true NT infidelity of the blind winner, relative to full knowledge,
    # per spectral-knowledge subset, at both gate times.
    ax = axs[0]
    labels = ['all-6', 'robust-4', 'diag-3', '1Q-2']
    pretty = ['all 6', r'robust (4):' '\n' r'no $S_{1,12},S_{2,12}$',
              'diag (3):\nno crosses', r'1Q (2):' '\n' r'$S_{1,1},S_{2,2}$ only']
    tgs = (80.0, 320.0)
    shades = (0.55, 0.95)
    x = np.arange(len(labels))
    for k, (Tg, a) in enumerate(zip(tgs, shades)):
        ref = res[('all-6', Tg)]['nt_true']
        vals = [res[(l, Tg)]['nt_true'] / ref for l in labels]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.32, color=C_REF, alpha=a,
               label=rf"$T_G={Tg:.0f}\tau$")
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha='center', fontsize=7)
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xticks(x, pretty, fontsize=7)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel(r"true $1-F_\mathrm{pro}$, relative to all-6 knowledge")
    ax.set_title("Ask 2: blind NT gate vs spectral knowledge")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')

    # Right: reference vs raw arm (ask 3), absolute true infidelities.
    ax = axs[1]
    labels2 = ['arm-reference', 'arm-raw']
    pretty2 = ['reference\n(SPAM-free)', 'raw\n(SPAM-biased recon)']
    ntv = [res[(l, 320.0)]['nt_true'] for l in labels2]
    ncv = [res[(l, 320.0)]['nt_char'] for l in labels2]
    kv = [res[(l, 320.0)]['known_true'] for l in labels2]
    x = np.arange(2)
    ax.bar(x - 0.22, kv, width=0.2, color=C_MIT, alpha=0.85, label='best CDD (true)')
    ax.bar(x, ntv, width=0.2, color=C_REF, alpha=0.85, label='best NT (true)')
    ax.bar(x + 0.22, ncv, width=0.2, color=C_REF, alpha=0.45,
           label='best NT (predicted)')
    for xi, v in zip(x, ntv):
        ax.text(xi, v * 1.03, f"{v:.3e}", ha='center', fontsize=6.5)
    ax.set_xticks(x, pretty2, fontsize=8)
    ax.set_ylim(0, max(kv) * 1.25)
    ax.set_ylabel(r"$1-F_\mathrm{pro}$ at $T_G=320\tau$")
    ax.set_title("Ask 3: gates from biased vs unbiased spectra")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_asks.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    setup_pub_rcparams('compact')
    fig_model_spectra()
    fig_cz_curve_band()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which == 'all' and os.path.exists(os.path.join(RUN, "report_ask_experiments.npz")):
        fig_asks()
    print(f"figures -> {OUT}")
