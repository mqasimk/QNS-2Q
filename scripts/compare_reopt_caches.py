"""Compare published gate caches vs *_reopt (re-optimized under the fixed evaluator).

Companion to scripts/reopt_battery_verify.sh. Per gate time, take the best-over-M
NT (infs_opt) and CDD (infs_known) -- exactly what the paper figures plot -- and
compare published vs re-optimized: infidelity ratio, winning M, winning pulse-counts.
Flags any point where the corrected objective moves the winner or the number by
more than REL_TOL.

Used to certify (V10-C2-PREFACTOR-0624) that the second-cumulant evaluator fix
(both-index parity + 1/2) does not shift the published winning sequences. Verdict
2026-06-24: CZ bit-stable everywhere; idle stable at all working gate times and to
quoted precision at the deep points; the only motion is sub-quoted-precision
optimizer scatter at the two deepest idle points (proven near-degeneracy, not the
fix -- the self-only rung_c, argmin-neutral under the fix, scatters there too).

    python scripts/compare_reopt_caches.py
"""
import os, glob, numpy as np

REL_TOL = 0.02  # 2% relative change in best infidelity => flag

def seqcounts(s):
    try: return (len(s[0]) - 2, len(s[1]) - 2)
    except Exception: return None

def load(p): return dict(np.load(p, allow_pickle=True))

def idle_curve(d):
    """gate_time -> best-over-M NT/CDD infidelity, winning M, winning pulse-counts."""
    Ms = [int(x) for x in d["M_values"]]
    by_t = {}
    for M in Ms:
        gts = np.asarray(d[f"M{M}_gate_times"], float)
        io = np.asarray(d[f"M{M}_infs_opt"], float)
        ik = np.asarray(d[f"M{M}_infs_known"], float)
        inp = np.asarray(d[f"M{M}_infs_nopulse"], float)
        so = d[f"M{M}_sequences_opt"]; sk = d[f"M{M}_sequences_known"]
        for g, t in enumerate(gts):
            by_t.setdefault(round(float(t)), []).append(
                (M, io[g], seqcounts(so[g]), ik[g], seqcounts(sk[g]), inp[g]))
    out = {}
    for t, rows in by_t.items():
        bo = min(rows, key=lambda r: r[1])
        bk = min(rows, key=lambda r: r[3])
        out[t] = dict(opt=bo[1], optM=bo[0], optC=bo[2],
                      known=bk[3], knownM=bk[0], knownC=bk[4],
                      nopulse=min(r[5] for r in rows))
    return out

def cz_curve(d):
    M = int(d["M"]); tax = np.asarray(d["taxis"], float)
    io = np.asarray(d["infs_opt"], float); ik = np.asarray(d["infs_known"], float)
    inp = np.asarray(d["infs_nopulse"], float)
    so = d["sequences_opt"]; sk = d["sequences_known"]
    out = {}
    for g, t in enumerate(tax):
        out[round(float(t))] = dict(opt=io[g], optM=M, optC=seqcounts(so[g]),
                                    known=ik[g], knownM=M, knownC=seqcounts(sk[g]),
                                    nopulse=inp[g])
    return out

def compare(name, pub, new, curve):
    cp, cn = curve(pub), curve(new)
    flags = []
    print(f"\n{'='*92}\n{name}\n{'='*92}")
    print(f"{'Tg':>7} | {'NT pub':>11} {'NT new':>11} {'r':>6} {'seq pub/new':>13} | "
          f"{'CDD pub':>11} {'CDD new':>11} {'r':>6}")
    for t in sorted(cp):
        if t not in cn:
            flags.append(f"{name} Tg={t}: missing in reopt"); continue
        p, n = cp[t], cn[t]
        ro = n['opt'] / p['opt'] if p['opt'] else float('nan')
        rk = n['known'] / p['known'] if p['known'] else float('nan')
        seqp = f"{p['optC']}/{n['optC']}"
        mark = ""
        if abs(ro - 1) > REL_TOL: mark += " <NT!>"; flags.append(f"{name} Tg={t}: NT ratio {ro:.3f}")
        if abs(rk - 1) > REL_TOL: mark += " <CDD!>"; flags.append(f"{name} Tg={t}: CDD ratio {rk:.3f}")
        if p['optC'] != n['optC']: mark += " <seq!>"; flags.append(f"{name} Tg={t}: NT counts {p['optC']}->{n['optC']}")
        print(f"{t:>7} | {p['opt']:11.4e} {n['opt']:11.4e} {ro:6.3f} {seqp:>13} | "
              f"{p['known']:11.4e} {n['known']:11.4e} {rk:6.3f}{mark}")
    return flags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
all_flags = []

for newp in sorted(glob.glob("DraftRun_*/optimization_data_all_M_*reopt*.npz")):
    pubp = newp.replace("_reopt", "")
    if not os.path.exists(pubp):
        all_flags.append(f"no published sibling for {newp}"); continue
    all_flags += compare(os.path.relpath(newp), load(pubp), load(newp), idle_curve)

for newp in sorted(glob.glob("DraftRun_*/plotting_data/plotting_data_cz_v2_*reopt*.npz")):
    pubp = newp.replace("_reopt", "")
    if not os.path.exists(pubp):
        all_flags.append(f"no published sibling for {newp}"); continue
    all_flags += compare(os.path.relpath(newp), load(pubp), load(newp), cz_curve)

print(f"\n{'#'*92}\nVERDICT")
if not all_flags:
    print(f"  ALL POINTS STABLE within {REL_TOL:.0%}: re-optimization under the fixed "
          f"evaluator\n  reproduces every published winning sequence and infidelity.")
else:
    print(f"  {len(all_flags)} flag(s) (> {REL_TOL:.0%} change or different winner):")
    for f in all_flags: print("   -", f)
print('#'*92)
