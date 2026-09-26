#!/bin/bash
#SBATCH -A m4790
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.06
#SBATCH -J era5_tendency_stats
#SBATCH -o slurm_logs/%x-%j.out
#
# Tendency (increment) stats for the euler step + decorrelation times; see
# scripts/compute_tendency_stats.py. Submit from the repo root:
#   sbatch scripts/compute_tendency_stats.sh
# Outputs go to $OUT_DIR (default $SCRATCH/era5_stats).

set -euo pipefail

OUT_DIR="${OUT_DIR:-${SCRATCH}/era5_stats}"
N_ANCHORS="${N_ANCHORS:-512}"
mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/partials/part_*.npz

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

srun shifter python -u scripts/compute_tendency_stats.py accumulate \
  --out-dir "${OUT_DIR}" --n-anchors "${N_ANCHORS}"

export OMP_NUM_THREADS=64
srun -N 1 -n 1 --cpus-per-task=128 shifter python -u scripts/compute_tendency_stats.py finalize \
  --out-dir "${OUT_DIR}" --n-anchors "${N_ANCHORS}"
