#!/bin/bash
# Serial driver: A seed1, H seed1, A seed2, H seed2 (12000 steps each), then slips scoring per run.
#   run_spiral_arms.sh [STEPS]      (default 12000)
# Arm C (our ladders) is run separately, see docs/RESULTS.md:
#   python src/make_arm_C.py ARMC_JSON && bash src/run_spiral_one.sh C 1 12000 && ...
set -u
STEPS="${1:-12000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="D:/vesuvius_downstream/p4/spiral_runs_1008"
PY="${SPIRAL_FITTING_DIR:-spiral-fitting}/.venv/Scripts/python.exe"
mkdir -p "$RUNS"
DRV="$RUNS/driver.log"
for job in "A 1" "H 1" "A 2" "H 2"; do
  set -- $job; ARM=$1; SEED=$2; TAG="${ARM}_s${SEED}"
  echo "[$(date +%H:%M:%S)] begin $TAG" | tee -a "$DRV"
  bash "$HERE/run_spiral_one.sh" "$ARM" "$SEED" "$STEPS" "$TAG"
  rc=$?
  echo "[$(date +%H:%M:%S)] fit $TAG exit $rc" | tee -a "$DRV"
  if [ $rc -ne 0 ]; then echo "stop: $TAG failed" | tee -a "$DRV"; exit $rc; fi
  CON=""; [ "$ARM" != "A" ] && CON="--constraints D:/vesuvius_downstream/p4/ds_boxA_$ARM/relative_windings.json"
  "$PY" "$HERE/slips.py" score "$RUNS/$TAG" --out "$RUNS/$TAG.score.json" $CON 2>&1 | tee -a "$DRV"
done
echo "[$(date +%H:%M:%S)] all done" | tee -a "$DRV"
