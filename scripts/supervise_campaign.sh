#!/bin/bash
# Relaunches the campaign until it reports fully complete (exit 0). Each relaunch
# RESUMES interrupted training from the last checkpoint, so a process-level death
# (NCCL/SIGABRT/OOM-kill) never loses progress and never skips the 2nd model.
cd /home/jovyan/shared/s0598584
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export YOLO_CONFIG_DIR=/home/jovyan/shared/s0598584/ultralytics_cfg
export NCCL_ASYNC_ERROR_HANDLING=1
PY=/home/jovyan/shared/s0598584/mcp-venv/bin/python
for i in $(seq 1 200); do
  echo "===== supervisor: launch attempt $i ($(date '+%F %T')) ====="
  "$PY" scripts/run_campaign.py && { echo "===== supervisor: CAMPAIGN COMPLETE ====="; exit 0; }
  echo "===== supervisor: campaign exited non-zero; relaunch in 30s ====="
  sleep 30
done
echo "===== supervisor: gave up after 200 relaunches ====="
exit 1
