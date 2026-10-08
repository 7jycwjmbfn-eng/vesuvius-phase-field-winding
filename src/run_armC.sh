#!/bin/bash
# After run_spiral_arms.sh has printed "all done": arm C (all auto ladders) and Cs (strict) with seeds 1 and 2, then slips scoring.
#   run_armC.sh [STEPS]
set -u
STEPS="${1:-12000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="D:/vesuvius_downstream/p4/spiral_runs_1008"
PY="${SPIRAL_FITTING_DIR:-spiral-fitting}/.venv/Scripts/python.exe"
DRV="$RUNS/driver_C.log"
until grep -q "all done" "$RUNS/driver.log" 2>/dev/null; do sleep 30; done
for job in "Cs 1" "C 1" "Cs 2" "C 2"; do
  set -- $job; ARM=$1; SEED=$2; TAG="${ARM}_s${SEED}"
  echo "[$(date +%H:%M:%S)] begin $TAG" | tee -a "$DRV"
  bash "$HERE/run_spiral_one_v2.sh" "$ARM" "$SEED" "$STEPS" "$TAG"
  rc=$?
  echo "[$(date +%H:%M:%S)] fit $TAG exit $rc" | tee -a "$DRV"
  if [ $rc -ne 0 ]; then echo "stop: $TAG failed" | tee -a "$DRV"; exit $rc; fi
  "$PY" "$HERE/slips.py" score "$RUNS/$TAG" --out "$RUNS/$TAG.score.json" --constraints D:/vesuvius_downstream/p4/ds_boxA_$ARM/relative_windings.json 2>&1 | tee -a "$DRV"
done
echo "[$(date +%H:%M:%S)] all done" | tee -a "$DRV"
