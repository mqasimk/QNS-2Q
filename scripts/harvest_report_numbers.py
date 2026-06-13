"""Harvest all numbers needed to propagate the strong-ZZ model through the
showcase report: CZ/idle true margins (plotting + best-over-M), predicted
margin CIs (margin bands), and FID/CDD/NT infidelities for the table.
"""
import numpy as np, os
ROOT = '/home/mqasimk/IdeaProjects/QNS-2Q'
CAP = 'DraftRun_NoSPAM_showcase_cap'


def cz_rows():
    out = {}
    for tag in ('cap', 'cap_short'):
        d = np.load(f'{ROOT}/{CAP}/plotting_data/plotting_data_cz_v2_{tag}.npz', allow_pickle=True)
        tg = np.asarray(d['taxis'], float)
        op = np.asarray(d['infs_opt'], float)
        kn = np.asarray(d['infs_known'], float)
        fidk = next((k for k in ('infs_nopulse', 'infs_fid', 'infs_free') if k in d.files), None)
        fid = np.asarray(d[fidk], float) if fidk else None
        for i in range(len(tg)):
            out[float(tg[i])] = (op[i], kn[i], (fid[i] if fid is not None else np.nan))
    return out


print("=== CZ true (FID / CDD / NT / margin) ===")
for tg, (nt, cdd, fid) in sorted(cz_rows().items()):
    print(f"  Tg={tg:6.0f}: FID {fid:.3e}  CDD {cdd:.3e}  NT {nt:.3e}  margin {cdd/nt:5.1f}x")

print("=== CZ predicted-margin CIs (95%) ===")
for tag in ('cap', 'cap_short'):
    d = np.load(f'{ROOT}/{CAP}/margin_band_cz_{tag}.npz', allow_pickle=True)
    for k in sorted(d.files):
        if k.startswith('margin_'):
            m = np.asarray(d[k], float)
            print(f"  CZ {k}: [{np.percentile(m,2.5):.2f}, {np.percentile(m,97.5):.2f}]")

print("=== idle true best-over-M (CDD / NT / margin) ===")
d = np.load(f'{ROOT}/{CAP}/optimization_data_all_M_cap.npz', allow_pickle=True)
fid_id = {}
for Tg in (320, 640, 1280, 2560, 5120, 10240):
    bn = bc = bf = None
    for m in (int(x) for x in d['M_values']):
        gts = np.asarray(d[f'M{m}_gate_times'], float)
        ix = np.where(np.isclose(gts, Tg))[0]
        if not ix.size:
            continue
        k = int(ix[0])
        nt = float(d[f'M{m}_infs_opt'][k]); cdd = float(d[f'M{m}_infs_known'][k])
        npl = float(d[f'M{m}_infs_nopulse'][k]) if f'M{m}_infs_nopulse' in d.files else np.nan
        if bn is None or nt < bn:
            bn = nt
        if bc is None or cdd < bc:
            bc = cdd
        bf = npl if (bf is None or not np.isnan(npl)) else bf
    if bn:
        fid_id[Tg] = (bf, bc, bn)
        print(f"  Tg={Tg:6d}: FID {bf:.3e}  CDD {bc:.3e}  NT {bn:.3e}  margin {bc/bn:5.1f}x")

print("=== idle predicted-margin CIs (95%) ===")
d = np.load(f'{ROOT}/{CAP}/margin_band_id_cap.npz', allow_pickle=True)
for k in sorted(d.files):
    if k.startswith('margin_'):
        m = np.asarray(d[k], float)
        print(f"  ID {k}: [{np.percentile(m,2.5):.2f}, {np.percentile(m,97.5):.2f}]")
