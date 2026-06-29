"""Idle-gate twin of run_margin_band.py: propagate the reconstruction's
quoted uncertainties into the predicted idle infidelities and the NT-vs-CDD
margin.

The idle winners live in optimization_data_all_M.npz (per-M arrays). At each
gate time the fixed winner is the best-over-M entry for each kind -- the same
selection the best-M curve and the report table use -- and that fixed
(sequence, M) pair is re-evaluated on every ensemble draw. Draw model,
correlation structure, and margin definition are identical to the CZ band
(see run_margin_band.py's docstring); the same draw evaluates every sequence,
so the margin band is the percentile of the per-draw RATIO.

Usage:
    python scripts/run_margin_band_idle.py [--folder DraftRun_NoSPAM_featured]
        [--n-draws 200]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from qns2q.paths import run_folder, project_root
from run_margin_band import draw_specs


def load_best_over_M(opt_path):
    """(Tg, kind, label, M, (pt1, pt2)) for the best-over-M winner at each
    gate time, selected on the stored (true-benchmark) infidelities, like
    the report table."""
    import jax.numpy as jnp
    d = np.load(opt_path, allow_pickle=True)
    M_values = [int(m) for m in d['M_values']]
    gts = sorted({round(float(g), 10) for m in M_values
                  for g in d[f'M{m}_gate_times']})
    entries = []
    for Tg in gts:
        for kind in ('known', 'opt'):
            best = None
            for m in M_values:
                m_gts = np.asarray(d[f'M{m}_gate_times'], dtype=float)
                idx = np.where(np.abs(m_gts - Tg) < 1e-9)[0]
                if idx.size == 0:
                    continue
                i = int(idx[0])
                inf = float(d[f'M{m}_infs_{kind}'][i])
                seq = d[f'M{m}_sequences_{kind}'][i]
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
                         "(UNCAP-0611)")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""

    from qns2q.control import idle  # heavy import (JAX) after arg parsing

    fname = a.folder or run_folder()
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

    central_specs = cfg.specs

    def eval_entries():
        idle._OVERLAP_SETUP_CACHE.clear()
        # Idle convention: sequences live in one block of duration T_seq =
        # Tg/M; calculate_infidelity takes the per-block time (idle.py:1188).
        return np.array([
            float(idle.calculate_infidelity(seq, cfg, m, Tg / m, use_ideal=False))
            for Tg, _, _, m, seq in entries])

    central = eval_entries()

    rng = np.random.default_rng(a.seed)
    draws = np.empty((a.n_draws, len(entries)))
    for n in range(a.n_draws):
        cfg.specs = draw_specs(central_specs, rng)
        cfg.SMat = cfg._build_interpolated_spectra()
        draws[n] = eval_entries()
        if (n + 1) % 25 == 0:
            print(f"  draw {n + 1}/{a.n_draws}")
    cfg.specs = central_specs
    cfg.SMat = cfg._build_interpolated_spectra()
    idle._OVERLAP_SETUP_CACHE.clear()

    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    med = np.median(draws, axis=0)
    print("\nPredicted infidelity on the reconstructed spectra "
          "(central | median [95% CI]):")
    for j, (Tg, kind, lab, m, _) in enumerate(entries):
        print(f"  Tg={Tg:8.1f} tau  {kind:5s} M={m:<3d} {lab:18s} "
              f"{central[j]:.3e} | {med[j]:.3e} [{lo[j]:.3e}, {hi[j]:.3e}]")

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
