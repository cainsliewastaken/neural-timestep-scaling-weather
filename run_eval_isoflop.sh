#!/bin/bash -l
#SBATCH -C gpu&hbm40g
#SBATCH -A m4790
#SBATCH --qos regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=224GB
#SBATCH --image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.06
#SBATCH --module=gpu,nccl-plugin
#
# IsoFLOP rollout evaluation (eval_isoflop.py) for one or more runs, one after another on this allocation.
#   EVAL_RUNS="<sweep_main.yaml> ..."   sweep YAMLs of the runs to evaluate (run tag read from each YAML)
#   EVAL_ARGS="--max-ics 2 ..."         extra eval_isoflop.py arguments (optional)
# Submit from the repo root, e.g.
#   EVAL_RUNS="$(ls scripts_6T_aspect_weighted_rollout_flops_flat/*_main.yaml)" sbatch -t 02:00:00 run_eval_isoflop.sh
# or run inside an existing allocation with `bash run_eval_isoflop.sh` (needs SLURM_JOB_ID set).

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_isoflop.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

IMAGE="registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.06"
DATAROOT="${DATAROOT:-/pscratch/sd/s/shas1693/data/weather/era5}"
REGISTRY="$(readlink -f "${REGISTRY:-${SCRIPT_DIR}/registry}")"
OUTPUT="$(readlink -f "${OUTPUT:-${SCRIPT_DIR}/expts}")"
NODES="${SLURM_JOB_NUM_NODES:-1}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29511
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CPP_LOG_LEVEL=ERROR

if [[ -z "${EVAL_RUNS:-}" ]]; then
  echo "error: EVAL_RUNS not set (space-separated sweep _main.yaml paths)" >&2
  exit 1
fi

rc_all=0
for Y in ${EVAL_RUNS}; do
  TAG=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('${Y}'))['hydra_overrides']['run_tag'])")
  echo "==== $(date +%FT%T) eval ${TAG} (${Y})"
  srun --overlap -N "${NODES}" --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
    --image="${IMAGE}" --module=gpu,nccl-plugin shifter --image="${IMAGE}" --module=gpu,nccl-plugin \
    -V "${DATAROOT}:/data;${OUTPUT}:/expts;${REGISTRY}:/registry" \
    bash -c "
      export CUDA_VISIBLE_DEVICES=3,2,1,0
      cd '${SCRIPT_DIR}'
      python -u eval_isoflop.py --run-tag '${TAG}' --sweep-yaml '${Y}' ${EVAL_ARGS:-}
    "
  rc=$?
  echo "==== $(date +%FT%T) finished ${TAG} exit=${rc}"
  [[ ${rc} -ne 0 ]] && rc_all=${rc}
done
exit ${rc_all}
