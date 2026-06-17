#!/usr/bin/env bash
# SHOWCASE-0616 battery (PARALLEL): everything downstream of the capture arm,
# for the DT_SHIFT=4.0 (larger Im S_1,2) + 512k-cap model. Restores+extends the
# 0612 carrier battery with the SPAM-ROBUST arm (fork 3), the design-number
# harvest, the figure regeneration, AND concurrency: independent gate
# optimizations / SPAM replay arms / margin bands / figures run in parallel,
# with JAX set to grow GPU memory on demand so several processes share the GPU.
#
#   QNS2Q_REGIME=showcase bash scripts/run_carrier_battery_0616.sh
#
# Tunables (env): GATE_JOBS (parallel gate opts, default 3), SPAM_JOBS (default 2),
# AUX_JOBS (margins/figures, default 3). Run AFTER run_capture_arm.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export QNS2Q_REGIME=showcase
export PYTHONUNBUFFERED=1   # live progress in the per-stage logs (no block-buffering)
# Share the GPU across concurrent processes: grow on demand, cap each process.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
PY=./venv/bin/python
CAP=DraftRun_NoSPAM_showcase_cap
# Arm set is overridable: ARMS="reference raw mitigated" runs the core (3-arm)
# battery and skips the robust arm (intractable on the dense solver; see the
# robust fast-solver work). Default includes robust for once it is fast.
ARMS="${ARMS:-reference raw mitigated robust}"
GATE_JOBS="${GATE_JOBS:-3}"
SPAM_JOBS="${SPAM_JOBS:-2}"
AUX_JOBS="${AUX_JOBS:-3}"
LOGD=/tmp/battery_0616_logs
mkdir -p "$LOGD"

# run a newline-separated list of shell commands (stdin) with <= N concurrent;
# each command + its stdout/stderr -> $LOGD/<stage>_<pid>.log. xargs returns
# 123 if ANY command fails, which (set -e + pipefail) aborts the battery.
prun() {  # prun N <stage-label>   (commands on stdin, one per line)
  local n="$1" stage="$2"
  xargs -P "$n" -I {} bash -c \
    'log="'"$LOGD/${stage}"'_$$.log"; echo "+ $1" >"$log"; eval "$1" >>"$log" 2>&1' \
    _ {}
}

echo "=== [1/8] SPAM arms (64k fine; reference records, raw/mitigated replay, robust synthesizes) ==="
# The reference --record and robust arms are the synthesis-heavy stages (they
# build the noise matrices via make_noise_mat_arr); each runs alone, so hand it
# (nearly) the whole GPU. The 0.30 global cap is sized for GATE_JOBS parallel gate
# opts, which do NOT synthesize -- at 0.30 (3.6 GB) the synthesis OOMs. Both heavy
# steps skip when their output is already newer than the truth model, so a re-run
# after a downstream-stage fix doesn't redo the synthesis.
MODEL="$CAP/simulated_spectra.npz"
REF_DS="DraftRun_SPAM_showcase_reference/phases.npz"
if [ -f "$REF_DS" ] && [ "$REF_DS" -nt "$MODEL" ]; then
  echo "    reference phase dataset current -- skipping record"
else
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $PY scripts/run_spam_experiments.py reference --fine --record > "$LOGD/spam_reference.log" 2>&1
fi
# raw + mitigated share the reference's recorded non-robust suite -> cheap replay.
printf '%s\n' \
  "$PY scripts/run_spam_experiments.py raw       --fine --replay" \
  "$PY scripts/run_spam_experiments.py mitigated --fine --replay" \
  | prun "$SPAM_JOBS" spam_replay
# robust runs a DIFFERENT experiment suite (its D+- estimators) and CANNOT replay
# the reference recording (experiments.py raises) -- it synthesizes its own.
ROB_RES="DraftRun_SPAM_showcase_robust/results.npz"
if [[ " $ARMS " != *" robust "* ]]; then
  echo "    robust not in ARMS -- skipping its synthesis"
elif [ -f "$ROB_RES" ] && [ "$ROB_RES" -nt "$MODEL" ]; then
  echo "    robust arm current -- skipping synthesis"
else
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $PY scripts/run_spam_experiments.py robust --fine > "$LOGD/spam_robust.log" 2>&1
fi
# NB: pipe the loop straight to prun -- `printf '%s\n' $(...)` would word-split
# each command into separate args (python / script / arg run as 3 commands).
for p in $ARMS; do echo "$PY scripts/run_spam_reconstruct.py $p"; done \
  | prun "$SPAM_JOBS" spam_recon

echo "=== [2/8] knowledge-ladder folders (diag3 / robust4) ==="
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
        for suf in ('_err', '_sys', '_errtot'):
            if k + suf in d:
                d[k + suf] = np.full(np.asarray(d[k + suf]).shape, np.nan + 0j)
    np.savez(os.path.join(dst, 'specs.npz'), **d)
    print(f'{dst}: dropped {drop}')
EOF

echo "=== [3/8] CZ + idle gate batteries (parallel, GATE_JOBS=$GATE_JOBS) ==="
CZF="--min-sep 8 --max-pulses 0 --pair-gap 4 --informed-counts"
IDF="--min-sep 8 --max-pulses 0 --max-dim 2600 --informed-counts"
{
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 0,-1,-3 --tag cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 3,2,1,-2 --tag cap_short"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder $CAP $CZF --factors 0 --self-only --tag rung_c_cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_diag3 $CZF --factors 0 --tag diag3_cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.cz --folder ${CAP}_robust4 $CZF --factors 0 --tag robust4_cap"
  for p in $ARMS; do echo "PYTHONPATH=src $PY -m qns2q.control.cz --protocol $p $CZF --factors 0 --tag rung_d_$p"; done
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --tag cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder $CAP $IDF --self-only --tag rung_c_idle_cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_diag3 $IDF --tag diag3_idle_cap"
  echo "PYTHONPATH=src $PY -m qns2q.control.idle --folder ${CAP}_robust4 $IDF --tag robust4_idle_cap"
  for p in $ARMS; do echo "PYTHONPATH=src $PY -m qns2q.control.idle --protocol $p $IDF --tag rung_d_idle_$p"; done
} | prun "$GATE_JOBS" gates

echo "=== [4/8] margin bands (parallel) ==="
printf '%s\n' \
  "$PY scripts/run_margin_band.py --folder $CAP --tag cap" \
  "$PY scripts/run_margin_band.py --folder $CAP --tag cap_short" \
  "$PY scripts/run_margin_band_idle.py --folder $CAP --tag cap" \
  | prun "$AUX_JOBS" margins

echo "=== [5/8] storage panel ==="
PYTHONPATH=src $PY scripts/showcase_storage_panel.py --mc-check > "$LOGD/storage.log" 2>&1

echo "=== [6/8] design-number harvest (incl. robust arm) ==="
PYTHONPATH=src $PY scripts/harvest_design_numbers.py > "$LOGD/harvest.log" 2>&1

echo "=== [7/8] figures (parallel) ==="
printf '%s\n' \
  "SHOWCASE_FIGS_DIR=reports/showcase_0613/figs PYTHONPATH=src $PY scripts/report_showcase_figs.py" \
  "PYTHONPATH=src $PY scripts/run_reconstruct.py   --folder $CAP" \
  "PYTHONPATH=src $PY scripts/run_cz_pulse_plot.py --folder $CAP --tag _cap" \
  | prun "$AUX_JOBS" figures
# run_single_qubit.py is a synthesis-heavy single-qubit M-scaling diagnostic
# (writes DraftRun_MScaling via make_noise_mat_arr) -- NOT one of the showcase
# report figures, and it OOMs at the parallel figures budget. Run it alone with
# the full GPU.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src $PY scripts/run_single_qubit.py > "$LOGD/single_qubit.log" 2>&1

echo "=== [8/8] done ==="
echo "CARRIER-BATTERY-0616-DONE"
