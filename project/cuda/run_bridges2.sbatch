#!/bin/bash
# Batch version: sbatch -A <chargeID> run_bridges2.sbatch [trace_dir]
# For an H100 instead of a V100, change the --gpus line to h100-80:1.
#SBATCH -J moe-litmus-replay
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:45:00
#SBATCH -o %x_%j.out

cd "$SLURM_SUBMIT_DIR"
bash build_and_test.sh "${1:-}"
