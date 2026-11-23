#!/bin/bash
# Run on any machine with an NVIDIA GPU and CUDA: a GHC machine (e.g. ghc73),
# or a Bridges-2 GPU node once Bridges-2 is back. Records the environment,
# builds, runs the litmus tests, and replays the MLX trace if one is given.
#   bash build_and_test.sh [trace_dir]
set -eu
cd "$(dirname "$0")"
TRACE=${1:-}
OUT=results_$(hostname)_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

# Bridges-2 uses modules; GHC machines keep CUDA in /usr/local/cuda
module load cuda 2>/dev/null || true
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "nvcc not found: is this a GPU machine with CUDA installed?"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi failed: no NVIDIA GPU visible on $(hostname)"; exit 1; }
{
  echo "== host"; hostname; date
  echo "== modules"; module list 2>&1 || true
  echo "== nvcc"; nvcc --version
  echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
} | tee "$OUT/environment.txt"

make clean >/dev/null
make ARCH=native 2>&1 | tee "$OUT/build.log"

echo "== litmus tests"
./litmus --csv "$OUT/litmus.csv" | tee "$OUT/litmus.txt" || echo "(litmus reported model violations; see $OUT/litmus.txt)"

if [ -n "$TRACE" ]; then
  echo "== trace replay: $TRACE"
  ./replay "$TRACE" | tee "$OUT/replay.txt"
fi
echo "results in $OUT/"
