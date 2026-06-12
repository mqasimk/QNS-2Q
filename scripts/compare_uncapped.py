"""UNCAP-0611: compare the published capped runs against the separation-limited
(uncapped) reruns, for both gates.

Reads, from DraftRun_NoSPAM_featured/:
  CZ   capped:   plotting_data/plotting_data_cz_v2.npz            (max_pulses=150)
  CZ   uncapped: plotting_data/plotting_data_cz_v2_uncapped.npz   (Tg = 40..1280)
                 plotting_data/plotting_data_cz_v2_uncapped2560.npz (Tg = 2560)
  idle capped:   optimization_data_all_M.npz                       (max_pulses=1000)
  idle uncapped: optimization_data_all_M_uncapped.npz

Prints, per gate time: best CDD / best NT / NT winner label under each cap,
and the NT-over-CDD margin then vs now.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qns2q.paths import project_root

_FOLDER_NAME = sys.argv[1] if len(sys.argv) > 1 else "DraftRun_NoSPAM_featured"
FOLDER = os.path.join(project_root(), _FOLDER_NAME)
PD = os.path.join(FOLDER, "plotting_data")


def load_cz(name):
    d = np.load(os.path.join(PD, name), allow_pickle=True)
    return {
        'Tg': np.asarray(d['taxis'], dtype=float),
        'known': np.asarray(d['infs_known'], dtype=float),
        'opt': np.asarray(d['infs_opt'], dtype=float),
        'nopulse': np.asarray(d['infs_nopulse'], dtype=float),
        'labels_known': [str(x) for x in d['labels_known']],
        'labels_opt': [str(x) for x in d['labels_opt']],
        'max_pulses': int(d['max_pulses']) if 'max_pulses' in d.files else None,
    }


def merge(a, b):
    order = np.argsort(np.concatenate([a['Tg'], b['Tg']]))
    out = {}
    for k in ('Tg', 'known', 'opt', 'nopulse'):
        out[k] = np.concatenate([a[k], b[k]])[order]
    for k in ('labels_known', 'labels_opt'):
        out[k] = [(a[k] + b[k])[i] for i in order]
    return out


def cz_table():
    capped = load_cz("plotting_data_cz_v2.npz")
    unc = load_cz("plotting_data_cz_v2_uncapped.npz")
    p2560 = os.path.join(PD, "plotting_data_cz_v2_uncapped2560.npz")
    if os.path.exists(p2560):
        unc = merge(unc, load_cz("plotting_data_cz_v2_uncapped2560.npz"))
    else:
        print("[note] uncapped Tg=2560 CZ block not finished yet; table covers 40-1280\n")

    print("ENTANGLING (CZ) -- capped (max_pulses=150) vs separation-limited")
    hdr = (f"{'Tg':>6} | {'CDD inf':>10} {'(seq)':<12}| "
           f"{'NT150 inf':>10} {'(seq)':<14}| {'NTfree inf':>10} {'(seq)':<14}| "
           f"{'margin150':>9} | {'marginfree':>10} | {'NTfree/NT150':>12}")
    print(hdr); print('-' * len(hdr))
    for i, Tg in enumerate(unc['Tg']):
        j = np.where(np.isclose(capped['Tg'], Tg))[0]
        if len(j) == 0:
            continue
        j = int(j[0])
        cdd_c, nt_c = capped['known'][j], capped['opt'][j]
        cdd_u, nt_u = unc['known'][i], unc['opt'][i]
        cdd_best = min(cdd_c, cdd_u)  # library only grows when uncapped
        print(f"{Tg:>6.0f} | {cdd_u:>10.3e} {unc['labels_known'][i]:<12}| "
              f"{nt_c:>10.3e} {capped['labels_opt'][j]:<14}| "
              f"{nt_u:>10.3e} {unc['labels_opt'][i]:<14}| "
              f"{cdd_c/nt_c:>9.2f} | {cdd_best/nt_u:>10.2f} | {nt_c/nt_u:>12.2f}")
    print()


def load_idle(name):
    path = os.path.join(FOLDER, name)
    d = np.load(path, allow_pickle=True)
    Ms = [int(m) for m in d['M_values']]
    rows = {}   # Tg -> list of (M, inf_known, lab_known, inf_opt, lab_opt)
    for m in Ms:
        p = f"M{m}_"
        if p + 'gate_times' not in d.files:
            continue
        for Tg, ik, lk, io, lo in zip(d[p + 'gate_times'], d[p + 'infs_known'],
                                      d[p + 'labels_known'], d[p + 'infs_opt'],
                                      d[p + 'labels_opt']):
            rows.setdefault(float(Tg), []).append((m, float(ik), str(lk),
                                                   float(io), str(lo)))
    # winner over M per gate time
    out = {}
    for Tg, entries in rows.items():
        bk = min(entries, key=lambda e: e[1])
        bo = min(entries, key=lambda e: e[3])
        out[Tg] = {'cdd': bk[1], 'cdd_lab': f"{bk[2]}", 'nt': bo[3],
                   'nt_lab': f"{bo[4]}"}
    return out


def idle_table():
    capped = load_idle("optimization_data_all_M.npz")
    try:
        unc = load_idle("optimization_data_all_M_uncapped.npz")
    except FileNotFoundError:
        print("[note] uncapped idle sweep not finished yet; no idle table")
        return
    print("IDLE -- capped (max_pulses=1000 total) vs separation-limited "
          "(winner over M per gate time)")
    hdr = (f"{'Tg':>6} | {'CDD inf':>10} {'(seq)':<14}| "
           f"{'NT1k inf':>10} {'(seq)':<16}| {'NTfree inf':>10} {'(seq)':<16}| "
           f"{'margin1k':>8} | {'marginfree':>10} | {'NTfree/NT1k':>11}")
    print(hdr); print('-' * len(hdr))
    for Tg in sorted(unc):
        if Tg not in capped:
            continue
        c, u = capped[Tg], unc[Tg]
        cdd_best = min(c['cdd'], u['cdd'])
        print(f"{Tg:>6.0f} | {u['cdd']:>10.3e} {u['cdd_lab']:<14}| "
              f"{c['nt']:>10.3e} {c['nt_lab']:<16}| "
              f"{u['nt']:>10.3e} {u['nt_lab']:<16}| "
              f"{c['cdd']/c['nt']:>8.2f} | {cdd_best/u['nt']:>10.2f} | {c['nt']/u['nt']:>11.2f}")
    print()


if __name__ == '__main__':
    cz_table()
    idle_table()
