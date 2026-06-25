#!/usr/bin/env bash
# V10-C2-PREFACTOR verification: re-run the SHOWCASE-0616 gate-optimization
# battery (step [3/8] of run_carrier_battery_0616.sh) under the FIXED cumulant
# evaluator, writing to non-destructive *_reopt tags. Same args + same pinned
# RNG seed => identical pulse-count candidates explored; only the corrected
# objective differs. A diff of *_reopt vs the committed caches then certifies
# whether the published winning sequences are still the argmin under the fix.
set -uo pipefail
cd "$(dirname "$0")/.."
export QNS2Q_REGIME=showcase
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
PY=./venv/bin/python
CAP=DraftRun_NoSPAM_showcase_cap
GATE_JOBS="${GATE_JOBS:-3}"
LOGD=/tmp/reopt_battery_logs
mkdir -p "$LOGD"

CZF="--min-sep 8 --max-pulses 0 --pair-gap 4 --informed-counts"
IDF="--min-sep 8 --max-pulses 0 --max-dim 2600 --informed-counts"

prun() {  # prun N <stage>   (commands on stdin, one per line)
  local n="$1" stage="$2"
  xargs -P "$n" -I {} bash -c \
    'log="'"$LOGD/${stage}"'_$$.log"; echo "+ $1" >"$log"; eval "$1" >>"$log" 2>&1' \
    _ {}
}

echo "=== re-opt battery (GATE_JOBS=$GATE_JOBS, MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION) ==="
date
# rung_c_cap CZ already re-run by hand (identical); the rest below.
{
  # CZ (M=1; cheap) -- mirror step [3/8] tags + _reopt
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 0,-1,-3 --tag cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 3,2,1,-2 --tag cap_short_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_diag3 $CZF --factors 0 --tag diag3_cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_robust4 $CZF --factors 0 --tag robust4_cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --protocol reference $CZF --factors 0 --tag rung_d_reference_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --protocol raw       $CZF --factors 0 --tag rung_d_raw_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --protocol mitigated $CZF --factors 0 --tag rung_d_mitigated_reopt"
  # Idle (8 M x 6 Tg; the long pole) -- heaviest first
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --tag cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_diag3 $IDF --tag diag3_idle_cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_robust4 $IDF --tag robust4_idle_cap_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --protocol reference $IDF --tag rung_d_idle_reference_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --protocol raw       $IDF --tag rung_d_idle_raw_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --protocol mitigated $IDF --tag rung_d_idle_mitigated_reopt"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --self-only --tag rung_c_idle_cap_reopt"
} | prun "$GATE_JOBS" gate

echo "=== REOPT-BATTERY-DONE ==="
date
