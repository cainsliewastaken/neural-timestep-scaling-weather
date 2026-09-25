#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH -C gpu
#SBATCH -A m4790
#SBATCH --qos regular
#SBATCH --cpus-per-task=32
#SBATCH --mem=224GB
#SBATCH --image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.06
#SBATCH --module=gpu,nccl-plugin
#
# Slurm array launcher for isoflop ERA5 configs.
# Resource flags from sbatch (e.g. --nodes from per-config slurm: in YAML) override #SBATCH above.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRIPT_DIR}/wandb_cache}"
mkdir -p "${WANDB_CACHE_DIR}"

# Map array task to sweep config unless TRAIN_CONFIG is set explicitly.
if [[ -z "${TRAIN_CONFIG:-}" && -n "${ARRAY_CONFIG_MANIFEST:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TRAIN_CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ARRAY_CONFIG_MANIFEST}")"
fi

if [[ -z "${TRAIN_CONFIG:-}" ]]; then
  echo "error: TRAIN_CONFIG not set and no ARRAY_CONFIG_MANIFEST/SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

# Mount data and experiment directories for the weather codebase defaults (/data, /expts, /registry).
DATAROOT="${DATAROOT:-/pscratch/sd/s/shas1693/data/weather/era5}"
REGISTRY="${REGISTRY:-${SCRIPT_DIR}/registry}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/expts}"
mkdir -p "${REGISTRY}" "${OUTPUT}"

if [[ ! -d "${DATAROOT}" ]]; then
  echo "warning: DATAROOT=${DATAROOT} does not exist; set DATAROOT before submitting" >&2
fi

echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "nodes=${SLURM_JOB_NUM_NODES} gpus_per_node=${SLURM_GPUS_PER_NODE} ntasks=${SLURM_NTASKS}"

srun --ntasks-per-node="${SLURM_GPUS_PER_NODE}" --gpus-per-node="${SLURM_GPUS_PER_NODE}" shifter \
  -V "${DATAROOT}:/data;${OUTPUT}:/expts;${REGISTRY}:/registry" \
  bash -lc "
    export HDF5_USE_FILE_LOCKING=FALSE
    export WANDB_CACHE_DIR='${WANDB_CACHE_DIR}'
    cd '${SCRIPT_DIR}'
    python scripts/launch_train.py --config '${TRAIN_CONFIG}'
  "
