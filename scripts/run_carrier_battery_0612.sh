#!/usr/bin/env bash
# SHOWCASE-0612 shared-carrier battery (everything downstream of the capture
# arm). Run AFTER scripts/run_capture_arm.py has refreshed
# DraftRun_NoSPAM_showcase_cap/{results,specs,phases}.npz on the new model.
#
#   QNS2Q_REGIME=showcase bash scripts/run_carrier_battery_0612.sh
#
# Stages: SPAM arms (record/replay + reconstruct) -> knowledge-ladder folders
# (NaN-dropped channels) -> CZ + idle gate batteries incl. ladder rungs and
# SPAM-arm design blocks -> margin bands -> entanglement-storage panel.
set -euo pipefail
cd "$(dirname "$0")/.."
export QNS2Q_REGIME=showcase
PY=./venv/bin/python
CAP=DraftRun_NoSPAM_showcase_cap

echo "=== [1/6] SPAM arms (64k fine, record/replay) ==="
$PY scripts/run_spam_experiments.py reference --fine --record
$PY scripts/run_spam_experiments.py raw --fine --replay
$PY scripts/run_spam_experiments.py mitigated --fine --replay
for p in reference raw mitigated; do
  $PY scripts/run_spam_reconstruct.py $p
done

echo "=== [2/6] knowledge-ladder folders (diag3 / robust4) ==="
$PY - <<'EOF'
import numpy as np, os, shutil
src = 'DraftRun_NoSPAM_showcase_cap'
for variant, drop in (('diag3', ('S12', 'S112', 'S212')),
                      ('robust4', ('S112', 'S212'))):
    dst = f'{src}_{variant}'
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, 'params.npz'), dst)
    d = dict(np.load(os.path.join(src, 'specs.npz'), allow_pickle=True))
    for k in drop:
        d[k] = np.full(np.asarray(d[k]).shape, np.nan + 0j)
    np.savez(os.path.join(dst, 'specs.npz'), **d)
    print(f'{dst}: dropped {drop}')
EOF

echo "=== [3/6] CZ battery ==="
CZF="--min-sep 8 --max-pulses 0 --pair-gap 4 --informed-counts"
PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 0,-1,-3 --tag cap
PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 3,2,1,-2 --tag cap_short
PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 0 --self-only --tag rung_c_cap
PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_diag3 $CZF --factors 0 --tag diag3_cap
PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_robust4 $CZF --factors 0 --tag robust4_cap
for p in reference raw mitigated; do
  PYTHONPATH=src $PY -m qns2q.control.cz --protocol $p $CZF --factors 0 --tag rung_d_$p
done

echo "=== [4/6] idle battery ==="
IDF="--min-sep 8 --max-pulses 0 --max-dim 2600 --informed-counts"
PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --tag cap
PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --self-only --tag rung_c_idle_cap
PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_diag3 $IDF --tag diag3_idle_cap
PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_robust4 $IDF --tag robust4_idle_cap
for p in reference raw mitigated; do
  PYTHONPATH=src $PY -m qns2q.control.idle --protocol $p $IDF --tag rung_d_idle_$p
done

echo "=== [5/6] margin bands ==="
$PY scripts/run_margin_band.py --folder $CAP --tag cap
$PY scripts/run_margin_band.py --folder $CAP --tag cap_short
$PY scripts/run_margin_band_idle.py --folder $CAP --tag cap

echo "=== [6/6] storage panel ==="
PYTHONPATH=src $PY scripts/showcase_storage_panel.py --mc-check

echo "CARRIER-BATTERY-DONE"
