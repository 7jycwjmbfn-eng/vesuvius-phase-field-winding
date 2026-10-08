#!/bin/bash
# One official-spiral fit on box A (L2 z 10496-11008), fitter wt/rebase-1696 (read-only).
#   run_spiral_one.sh ARM SEED STEPS [TAG]
#   ARM = A | H | C (all auto ladders) | Cs (strict auto ladders)
# Output: D:/vesuvius_downstream/p4/spiral_runs_1008/<TAG or ARM_sSEED>/ , log next to it.
# Waits while free virtual memory < 10 GB; records GPU memory every 5 s to <name>.gpu.csv.
set -u
ARM="$1"; SEED="$2"; STEPS="$3"; TAG="${4:-${ARM}_s${SEED}}"
SF="${SPIRAL_FITTING_DIR:-spiral-fitting}"
PY="$SF/.venv/Scripts/python.exe"
BASE="D:/vesuvius_downstream/p4"
RUNS="$BASE/spiral_runs_1008"
DS="$BASE/ds_boxA_$ARM"
mkdir -p "$RUNS" "$BASE/cache"
export OMP_NUM_THREADS=2 FIT_SPIRAL_TRITON=1 PYTHONUNBUFFERED=1
export FIT_SPIRAL_CACHE_DIR="$BASE/cache"
export FIT_SPIRAL_OUT_DIR="$RUNS/$TAG"
LOG="$RUNS/$TAG.log"

freevm() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)" | tr -d '\r'; }
freed() { powershell -NoProfile -Command "[math]::Round((Get-PSDrive D).Free/1GB,1)" | tr -d '\r'; }

COMMON="\"z_begin\":10496,\"z_end\":11008,\"optimizer_num_training_steps\":$STEPS,\"optimizer_random_seed\":$SEED,\"input_use_verified_patches\":false,\"input_use_fibers\":false,\"input_use_fiber_directions\":false,\"input_use_tracks\":false,\"input_use_winding_inference\":false,\"dense_spacing_mode\":\"grad_mag\",\"loss_weight_dense_spacing\":0,\"input_use_pcl_absolute\":false,\"input_use_pcl_same_winding\":false,\"input_use_pcl_drawn_control_points\":false"
case "$ARM" in
  A) EXTRA='"input_use_pcl_relative":false' ;;
  H|C|Cs) EXTRA='"input_use_pcl_relative":true,"sample_count_unattached_pcls_per_step":840,"loss_weight_unattached_pcl_radius":10.0,"loss_weight_unattached_pcl_dt":10.0' ;;
  *) echo "bad ARM $ARM"; exit 2 ;;
esac
export FIT_SPIRAL_CONFIG_OVERRIDES="{$COMMON,$EXTRA}"

# guards: memory, disk, GPU
for i in $(seq 1 90); do
  fv=$(freevm); fd=$(freed)
  if awk "BEGIN{exit !($fv >= 10 && $fd >= 10)}"; then break; fi
  echo "[$(date +%H:%M:%S)] waiting: free virtual memory ${fv} GB, D: free ${fd} GB" | tee -a "$LOG"
  sleep 120
done
fv=$(freevm); fd=$(freed)
if ! awk "BEGIN{exit !($fv >= 10 && $fd >= 10)}"; then echo "ABORT: resources still low (vm $fv GB, D: $fd GB)" | tee -a "$LOG"; exit 3; fi
nfit=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*fit_spiral.py*' -and \$_.Name -like 'python*' }).Count" | tr -d '')
if [ "$nfit" != "0" ]; then echo "ABORT: another fit_spiral.py process is running ($nfit)" | tee -a "$LOG"; exit 4; fi
gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "[$(date +%H:%M:%S)] GPU memory in use by others before start: ${gmem} MiB" | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] START ARM=$ARM SEED=$SEED STEPS=$STEPS DS=$DS free_vm=${fv}GB D_free=${fd}GB" | tee "$LOG"
echo "$FIT_SPIRAL_CONFIG_OVERRIDES" | tee -a "$LOG"
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" >> "$RUNS/$TAG.gpu.csv"; sleep 5; done ) &
MON=$!
T0=$(date +%s)
cd "$SF" && "$PY" fit_spiral.py --dataset "$DS" --scroll-spec "$DS/spiral-scroll.json" 2>&1 | tr '\r' '\n' >> "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
kill $MON 2>/dev/null
echo "FIT_EXIT=$RC ELAPSED_S=$((T1-T0)) $(date +%H:%M:%S) free_vm_after=$(freevm)GB" | tee -a "$LOG"
exit $RC
