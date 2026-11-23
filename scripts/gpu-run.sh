#!/usr/bin/env bash
# Match the full song on a CUDA machine you can run commands on directly, such as an
# ICRN notebook terminal. Runs in the background and survives a closed browser tab;
# follow it with: tail -f output/full-song/run.log
#
#   scripts/gpu-run.sh                     # 30 fps, 6,960 frames
#   scripts/gpu-run.sh 10                  # 10 fps, 2,320 frames
#   scripts/gpu-run.sh 30 --samples 65536  # extra options go to match
#
# The retrieval is saved to output/full-song/retrieval.json.gz, so rerunning the same
# command after an interruption repeats only the CPU refinement. Render locally afterwards:
#   .venv/bin/python -m pathapple render --output output/full-song
set -euo pipefail
cd "$(dirname "$0")/.."
fps=${1:-30}
shift || true
out=output/full-song
mkdir -p "$out"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo "CPUs for refinement: ${CPU_LIMIT:-$(nproc)}"
nohup .venv/bin/python -m pathapple match --start 0 --seconds 232 --fps "$fps" --output "$out" "$@" \
  >> "$out/run.log" 2>&1 &
echo "Started match (pid $!). Follow with: tail -f $out/run.log"
