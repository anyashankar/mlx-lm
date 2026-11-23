#!/bin/bash
# Run on the Bridges-2 LOGIN node (ssh <psc-username>@bridges2.psc.edu).
# Confirms the GPU allocation and CUDA modules, and prints the commands for
# getting a GPU. Building and running happen on a GPU node (build_and_test.sh).
set -u

echo "== 1. Allocations on this account"
if command -v projects >/dev/null 2>&1; then
  projects
  echo
  echo "Look for a charge ID whose resource is Bridges-2 GPU with hours left."
  echo "That ID is what -A takes below (and in run_bridges2.sbatch)."
else
  echo "'projects' not found: are you on bridges2.psc.edu?"
fi

echo
echo "== 2. CUDA modules"
module avail cuda 2>&1 | grep -i cuda || echo "no cuda modules listed"

echo
echo "== 3. Get a GPU (pick one):"
echo "  interactive V100: interact -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00 -A <chargeID>"
echo "  interactive H100: interact -p GPU-shared --gres=gpu:h100-80:1 -t 01:00:00 -A <chargeID>"
echo "  batch (both tests): sbatch -A <chargeID> run_bridges2.sbatch"
echo
echo "Then on the GPU node:  cd $(pwd) && bash build_and_test.sh <trace_dir>"
echo
echo "== 4. Copy the trace from the Mac (run this ON THE MAC):"
echo "  rsync -av runs/<run>/trace/ <psc-username>@data.bridges2.psc.edu:$(pwd)/trace/"
