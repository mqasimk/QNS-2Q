"""Compare the SPAM-protocol reconstruction arms against the analytic truth.

Usage (after running the arms with run_spam_experiments/run_spam_reconstruct):

    python scripts/compare_spam_protocols.py [--arms reference,raw,mitigated]

Loads ``specs.npz`` from DraftRun_<regime>_<arm> for the requested arms
(default: all four), prints per-spectrum accuracy metrics (median relative
deviation from the analytic truth, median pull |rec - truth| / sigma_tot, and
the within-2-sigma fraction), and writes two figures into
DraftRun_SPAM_<regime>_mitigated/figures/:

  spam_protocol_comparison.pdf -- 3x3 overlay: row 1 the self-spectra, rows
      2/3 the Re/Im cross-spectra; analytic truth as a curve, one errorbar
      series per arm (slightly x-offset so the bars stay legible).
  spam_protocol_pulls.pdf      -- same grid showing (rec - truth)/sigma_tot
      per arm with the +/-2 sigma band shaded; this is the "is any arm biased
      beyond its own error bars?" view.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qns2q.characterize.systematics import analytic_spectra
from qns2q.paths import run_folder, project_root, current_regime

ARM_STYLE = {
    'reference': dict(color='#0072B2', marker='o', label='no SPAM (reference)'),
    'raw':       dict(color='#999999', marker='v', label='SPAM, unmitigated'),
    'mitigated': dict(color='#D55E00', marker='^', label='SPAM-mitigated'),
    'robust':    dict(color='#009E73', marker='s', label='SPAM-robust'),
}
SELF_KEYS = ['S11', 'S22', 'S1212']
CROSS_KEYS = ['S12', 'S112', 'S212']
LABELS = {'S11': r'$S_{1,1}$', 'S22': r'$S_{2,2}$', 'S1212': r'$S_{12,12}$',
          'S12': r'$S_{1,2}$', 'S112': r'$S_{1,12}$', 'S212': r'$S_{2,12}$'}


def load_arm(protocol):
    folder = os.path.join(project_root(), run_folder(spam=True, protocol=protocol))
    spec_path = os.path.join(folder, 'specs.npz')
    if not os.path.exists(spec_path):
        return None
    return np.load(spec_path)


def _sig_parts(err):
    """Split an error array into (Re-channel, Im-channel) sigmas.

    Cross-spectrum errors may be stored complex (separate Re/Im inversions);
    real arrays apply to both parts."""
    err = np.asarray(err)
    if np.iscomplexobj(err):
        return np.abs(np.real(err)), np.abs(np.imag(err))
    return np.abs(err), np.abs(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='reference,raw,mitigated,robust',
                    help='comma-separated arm list to compare')
    args = ap.parse_args()
    arm_names = [a.strip() for a in args.arms.split(',') if a.strip()]

    arms = {a: load_arm(a) for a in arm_names}
    missing = [a for a, d in arms.items() if d is None]
    arms = {a: d for a, d in arms.items() if d is not None}
    if missing:
        print(f"(skipping arms with no specs.npz: {', '.join(missing)})")
    if not arms:
        raise SystemExit("no arms found")

    truth = analytic_spectra()
    first = next(iter(arms.values()))
    wk = np.asarray(first['wk'])
    w_fine = np.linspace(0, wk.max()*1.02, 2000)

    # ---- table -------------------------------------------------------------------
    keys = SELF_KEYS + CROSS_KEYS
    header = f"{'spectrum':>8} |" + "".join(f" {ARM_STYLE[a]['label'][:22]:>26} |" for a in arms)
    print(header)
    print(f"{'':>8} |" + "".join(f" {'rel.dev / pull / <2sig':>26} |" for _ in arms))
    print("-" * len(header))
    for key in keys:
        tr = np.asarray(truth[key](wk))
        row = f"{key:>8} |"
        for a, d in arms.items():
            rec_arr = np.asarray(d[key]) if key in d.files else None
            if rec_arr is None or np.all(rec_arr == 0) or np.all(np.isnan(rec_arr)):
                row += f" {'n/a (not accessible)':>26} |"
                continue
            rec = np.asarray(d[key])
            scale = np.median(np.abs(tr)) + 1e-30
            rel = np.abs(rec - tr) / (np.abs(tr) + 0.05*scale)
            sig = np.abs(np.asarray(d[f'{key}_errtot'])) + 1e-30
            pull = np.abs(rec - tr) / sig
            row += f"  {np.median(rel):6.2%} / {np.median(pull):4.2f} / {np.mean(pull <= 2):4.0%} |"
        print(row)

    # ---- DC (w=0) table ------------------------------------------------------------
    # The medians above dilute the w=0 point ~1:25; SPAM bias concentrates there
    # (the harmonic estimators self-cancel static SPAM), so quote DC explicitly.
    print("\nDC (w=0) points -- fitted S(0) / pull vs truth ('!' = flagged not-determined):")
    print(header)
    for key in keys:
        tr0 = float(np.real(truth[key](np.array([0.0]))[0]))
        row = f"{key:>8} |"
        for a, d in arms.items():
            rec_arr = np.asarray(d[key]) if key in d.files else None
            if rec_arr is None or np.all(rec_arr == 0) or np.all(np.isnan(rec_arr)):
                row += f" {'n/a (not accessible)':>26} |"
                continue
            v0 = float(np.real(rec_arr[0]))
            if not np.isfinite(v0):
                row += f" {'n/a (no DC observable)':>26} |"
                continue
            sig_re, _ = _sig_parts(d[f'{key}_errtot'])
            pull0 = abs(v0 - tr0) / (float(sig_re[0]) + 1e-30)
            ok = bool(np.asarray(d[f'{key}_dc_ok'])) if f'{key}_dc_ok' in d.files else True
            cell = f"{v0:9.2e} / pull {pull0:6.2f}{'!' if not ok else ' '}"
            row += f" {cell:>26} |"
        print(row + f"   truth {tr0:9.2e}")

    # ---- figure 1: overlay with error bars ----------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
    dw = (wk[1] - wk[0]) if len(wk) > 1 else 0.02
    offsets = np.linspace(-0.22, 0.22, len(arms)) * dw

    def panel(ax, key, part, title):
        tr_f = np.asarray(truth[key](w_fine))
        tr_f = np.real(tr_f) if part == 're' else np.imag(tr_f)
        ax.plot(w_fine, tr_f, 'k--', lw=1.1, label='analytic truth', zorder=1)
        for off, (a, d) in zip(offsets, arms.items()):
            rec = np.asarray(d[key]) if key in d.files else None
            if rec is None or np.all(rec == 0) or np.all(np.isnan(rec)):
                continue
            rec_p = np.real(rec) if part == 're' else np.imag(rec)
            sig_re, sig_im = _sig_parts(d[f'{key}_errtot'])
            sig = sig_re if part == 're' else sig_im
            st = ARM_STYLE[a]
            ax.errorbar(wk + off, rec_p, yerr=sig, fmt=st['marker'], ms=4,
                        color=st['color'], ecolor=st['color'], elinewidth=0.9,
                        capsize=2, lw=0, label=st['label'], zorder=3, alpha=0.9)
        ax.set_yscale('asinh', linear_width=float(np.median(np.abs(tr_f)) + 1e-12))
        ax.set_title(title, fontsize=11)
        ax.tick_params(direction='in', labelsize=9)
        ax.grid(True, alpha=0.25)

    for j, key in enumerate(SELF_KEYS):
        panel(axs[0, j], key, 're', LABELS[key])
    for j, key in enumerate(CROSS_KEYS):
        panel(axs[1, j], key, 're', r'Re ' + LABELS[key])
        panel(axs[2, j], key, 'im', r'Im ' + LABELS[key])
    for j in range(3):
        axs[2, j].set_xlabel(r'$\omega\tau$', fontsize=11)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(arms) + 1,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"SPAM-protocol comparison vs analytic truth "
                 f"(regime: {current_regime()})", y=1.035, fontsize=12)
    fig.tight_layout()

    out_dir = os.path.join(project_root(), run_folder(spam=True, protocol='mitigated'),
                           'figures')
    os.makedirs(out_dir, exist_ok=True)
    f1 = os.path.join(out_dir, 'spam_protocol_comparison.pdf')
    fig.savefig(f1, bbox_inches='tight')
    print(f"\nSaved comparison figure to {f1}")

    # ---- figure 2: pulls -----------------------------------------------------------
    fig2, axs2 = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True, sharey=True)

    def pull_panel(ax, key, part, title):
        ax.axhspan(-2, 2, color='0.92', zorder=0)
        ax.axhline(0, color='k', lw=0.8)
        tr = np.asarray(truth[key](wk))
        tr_p = np.real(tr) if part == 're' else np.imag(tr)
        for off, (a, d) in zip(offsets, arms.items()):
            rec = np.asarray(d[key]) if key in d.files else None
            if rec is None or np.all(rec == 0) or np.all(np.isnan(rec)):
                continue
            rec_p = np.real(rec) if part == 're' else np.imag(rec)
            sig_re, sig_im = _sig_parts(d[f'{key}_errtot'])
            sig = (sig_re if part == 're' else sig_im) + 1e-30
            st = ARM_STYLE[a]
            ax.plot(wk + off, np.clip((rec_p - tr_p)/sig, -5, 5), st['marker'],
                    ms=4, color=st['color'], label=st['label'], alpha=0.9)
        ax.set_ylim(-5, 5)
        ax.set_title(title, fontsize=11)
        ax.tick_params(direction='in', labelsize=9)
        ax.grid(True, alpha=0.25)

    for j, key in enumerate(SELF_KEYS):
        pull_panel(axs2[0, j], key, 're', LABELS[key])
    for j, key in enumerate(CROSS_KEYS):
        pull_panel(axs2[1, j], key, 're', r'Re ' + LABELS[key])
        pull_panel(axs2[2, j], key, 'im', r'Im ' + LABELS[key])
    for j in range(3):
        axs2[2, j].set_xlabel(r'$\omega\tau$', fontsize=11)
    axs2[0, 0].set_ylabel(r'pull $(\hat{S}-S)/\sigma_{\rm tot}$', fontsize=10)
    axs2[1, 0].set_ylabel(r'pull', fontsize=10)
    axs2[2, 0].set_ylabel(r'pull', fontsize=10)
    handles, labels = axs2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', ncol=max(len(arms), 1),
                frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig2.suptitle(f"Reconstruction pulls (shaded: |pull| <= 2; points clipped at 5) "
                  f"(regime: {current_regime()})", y=1.035, fontsize=12)
    fig2.tight_layout()
    f2 = os.path.join(out_dir, 'spam_protocol_pulls.pdf')
    fig2.savefig(f2, bbox_inches='tight')
    print(f"Saved pulls figure to {f2}")


if __name__ == "__main__":
    main()
