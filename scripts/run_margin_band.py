"""OPT-MARGIN-BAND: propagate the reconstruction's quoted uncertainties into
the predicted CZ infidelities and the NT-vs-CDD margin.

Fixed winner sequences (saved by the CZ run's plotting_data) are re-evaluated
on an ensemble of spectra drawn within the reconstruction's stat + sys bars:

    S_c(w_k) -> S_c(w_k) + alpha_c * sys_c(w_k) + eps_{c,k} * stat_c(w_k)

with alpha_c ~ N(0,1) SHARED across all teeth of channel c -- the unfold-model
systematic is a coherent broad envelope, and independent per-tooth sys draws
would CLT-average away in the broadband overlap integral, under-quoting the
band -- and eps_{c,k} ~ N(0,1) independent per tooth (shot noise). Cross
spectra draw Re/Im stat components independently and scale the complex sys
envelope by one real alpha. Self-spectra are floored at 0 after the draw;
flagged (*_dc_ok = False) DC points are held at their stored floor value;
all-NaN channels (robust protocol) pass through and are dropped by the SMat
builder as usual. Tail fits re-run on every draw, so the tail-extrapolation
sensitivity to top-tooth scatter is propagated too.

The same draw evaluates every sequence, so the margin band is the percentile
of the per-draw RATIO (common-spectrum correlation retained), not a
quadrature of two independent bands.

Usage (winners from the 6/11 probe, uncertainties from the reference arm):
    python scripts/run_margin_band.py --protocol reference \
        --winners-from DraftRun_NoSPAM_featured [--n-draws 200]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.paths import run_folder, project_root

SELF_KEYS = ("S11", "S22", "S1212")
CROSS_KEYS = ("S12", "S112", "S212")


def draw_specs(specs, rng):
    """One ensemble member: central spectra perturbed within stat + sys."""
    out = {'wk': np.asarray(specs['wk'])}
    for key in SELF_KEYS + CROSS_KEYS:
        base = np.asarray(specs[key])
        if np.all(np.isnan(base)):  # robust arm: channel not reconstructed
            out[key] = base
            continue
        stat = np.asarray(specs[key + '_err'])
        sysv = np.asarray(specs[key + '_sys'])
        hold_dc = (f'{key}_dc_ok' in specs and not bool(specs[f'{key}_dc_ok']))
        alpha = rng.standard_normal()
        if key in SELF_KEYS:
            eps = rng.standard_normal(base.shape)
            d = base + alpha * sysv + eps * stat
            if hold_dc:
                d[0] = base[0]
            out[key] = np.maximum(d, 0.0)
        else:
            eps_r = rng.standard_normal(base.shape)
            eps_i = rng.standard_normal(base.shape)
            d = (base + alpha * sysv
                 + eps_r * np.real(stat) + 1j * eps_i * np.imag(stat))
            if hold_dc:
                d[0] = base[0]
            out[key] = d
    return out


def load_winners(pdat):
    """(Tg, kind, label, (pt1, pt2)) entries from a plotting_data npz --
    per-Tg arrays when present (new format), else the overall-best pair."""
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
    """Render the ensemble npz into a two-panel figure next to it."""
    import matplotlib
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
    ap = argparse.ArgumentParser(
        description="Recon-uncertainty band on predicted CZ infidelities "
                    "and the NT-vs-CDD margin (fixed winner sequences)")
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
                    help="characterized-SMat construction (OPT-SPECTRAL-MODEL)")
    ap.add_argument('--replot', action='store_true',
                    help="skip the ensemble; re-render the figure from the "
                         "existing margin_band_cz.npz in the specs folder")
    ap.add_argument('--tag', type=str, default="",
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

    M = 1
    central_specs = cfg.specs

    def eval_entries():
        cz._OVERLAP_SETUP_CACHE.clear()
        return np.array([
            float(cz.calculate_infidelity(seq, cfg, M, Tg, use_ideal=False))
            for Tg, _, _, seq in entries])

    central = eval_entries()

    rng = np.random.default_rng(a.seed)
    draws = np.empty((a.n_draws, len(entries)))
    for n in range(a.n_draws):
        cfg.specs = draw_specs(central_specs, rng)
        cfg.SMat = cfg._build_interpolated_spectra()
        draws[n] = eval_entries()
        if (n + 1) % 25 == 0:
            print(f"  draw {n + 1}/{a.n_draws}")
    # Restore the central state
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

    # Margin = known/opt per draw at each Tg with both present
    print("\nNT-vs-known margin (known / opt, central | median [95% CI]):")
    tgs = sorted({Tg for Tg, _, _, _ in entries})
    margin_out = {}
    for Tg in tgs:
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
