"""Compare the SPAM-protocol reconstruction arms against the analytic truth.

Usage (after running the four arms with run_spam_experiments/run_spam_reconstruct):

    python scripts/compare_spam_protocols.py

Loads ``specs.npz`` from DraftRun_SPAM_<regime>_{reference,raw,mitigated,robust},
prints per-spectrum accuracy metrics (median relative deviation from the analytic
truth, and the median pull |recon - truth| / sigma_tot), and writes a 3x2
comparison figure into DraftRun_SPAM_<regime>_mitigated/figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qns2q.noise.spectra import S_11, S_22, S_1212, S_1_2, S_1_12, S_2_12
from qns2q.paths import run_folder, project_root

ARMS = ['reference', 'raw', 'mitigated', 'robust']
ARM_STYLE = {
    'reference': dict(color='#0072B2', marker='o', label='no SPAM (reference)'),
    'raw':       dict(color='#999999', marker='v', label='SPAM, unmitigated'),
    'mitigated': dict(color='#D55E00', marker='^', label='SPAM-mitigated'),
    'robust':    dict(color='#009E73', marker='s', label='SPAM-robust'),
}
SPEC_KEYS = ['S11', 'S22', 'S1212', 'S12', 'S112', 'S212']
SPEC_LABELS = [r'$S_{1,1}$', r'$S_{2,2}$', r'$S_{12,12}$',
               r'$S_{1,2}$', r'$S_{1,12}$', r'$S_{2,12}$']


def load_arm(protocol):
    folder = os.path.join(project_root(), run_folder(spam=True, protocol=protocol))
    spec_path = os.path.join(folder, 'specs.npz')
    par_path = os.path.join(folder, 'params.npz')
    if not (os.path.exists(spec_path) and os.path.exists(par_path)):
        return None
    return dict(specs=np.load(spec_path), params=np.load(par_path))


def truth_fns(gamma, gamma_12):
    return {
        'S11': lambda w: np.real(np.asarray(S_11(w))),
        'S22': lambda w: np.real(np.asarray(S_22(w))),
        'S1212': lambda w: np.real(np.asarray(S_1212(w))),
        'S12': lambda w: np.asarray(S_1_2(w)),
        'S112': lambda w: np.asarray(S_1_12(w)),
        'S212': lambda w: np.asarray(S_2_12(w)),
    }


def main():
    arms = {p: load_arm(p) for p in ARMS}
    arms = {p: d for p, d in arms.items() if d is not None}
    if not arms:
        raise SystemExit("No SPAM-arm specs.npz found; run the pipeline first.")
    ref = next(iter(arms.values()))
    gamma = float(ref['params']['gamma'])
    gamma_12 = float(ref['params']['gamma_12'])
    truths = truth_fns(gamma, gamma_12)

    print(f"{'spectrum':>8} | " + " | ".join(f"{p:>22}" for p in arms))
    print(f"{'':>8} | " + " | ".join(f"{'med rel.dev / pull':>22}" for _ in arms))
    print("-" * (10 + 25 * len(arms)))
    for sk in SPEC_KEYS:
        row = []
        for p, d in arms.items():
            specs = d['specs']
            wk = specs['wk']
            val = specs[sk]
            errk = sk + '_errtot' if sk + '_errtot' in specs else sk + '_err'
            err = specs[errk] if errk in specs else np.full_like(val, np.nan)
            tr = truths[sk](wk)
            finite = np.isfinite(val)
            if not np.any(finite):
                row.append(f"{'n/a (not accessible)':>22}")
                continue
            scale = np.max(np.abs(tr))
            dev = np.abs(val - tr)[finite]
            rel = np.median(dev) / scale
            sig = np.abs(np.real(err)) + 1j * np.abs(np.imag(err)) \
                if np.iscomplexobj(err) else np.abs(err)
            if np.iscomplexobj(val):
                pulls = np.concatenate([
                    np.abs(np.real(val - tr))[finite] / np.maximum(np.real(sig)[finite], 1e-30),
                    np.abs(np.imag(val - tr))[finite] / np.maximum(np.imag(sig)[finite], 1e-30)])
            else:
                pulls = dev / np.maximum(np.asarray(sig)[finite], 1e-30)
            row.append(f"{100*rel:8.2f}% / {np.median(pulls):7.2f}")
        print(f"{sk:>8} | " + " | ".join(f"{r:>22}" for r in row))

    # --- Overlay figure -------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(11, 9))
    wmax = max(float(np.max(d['specs']['wk'])) for d in arms.values())
    w = np.linspace(0, 1.05 * wmax, 800)
    xu = 1.0   # tau units: plot the dimensionless w*tau directly
    for i, (sk, lab) in enumerate(zip(SPEC_KEYS, SPEC_LABELS)):
        ax = axs[i // 2, i % 2]
        tr = truths[sk](w)
        ax.plot(w / xu, np.real(tr), 'k-', lw=1.2, label='truth (Re)')
        if np.iscomplexobj(tr) and np.any(np.imag(tr) != 0):
            ax.plot(w / xu, np.imag(tr), 'k--', lw=1.0, label='truth (Im)')
        for p, d in arms.items():
            specs = d['specs']
            wk, val = specs['wk'], specs[sk]
            st = ARM_STYLE[p]
            ax.plot(wk / xu, np.real(val), ls='none', marker=st['marker'],
                    ms=4, color=st['color'], label=st['label'])
            if np.iscomplexobj(val) and np.any(np.isfinite(np.imag(val))):
                ax.plot(wk / xu, np.imag(val), ls='none', marker=st['marker'],
                        ms=4, mfc='none', color=st['color'])
        ax.set_ylabel(lab)
        ax.set_yscale('asinh')
        if i >= 4:
            ax.set_xlabel(r'$\omega\tau$')
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle('SPAM-protocol comparison (filled: Re, open: Im)')
    fig.tight_layout()
    out_dir = os.path.join(project_root(),
                           run_folder(spam=True, protocol='mitigated'), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'spam_protocol_comparison.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparison figure to {out}")


if __name__ == "__main__":
    main()
