#!/usr/bin/env bash
# Chained per-config training jobs for ERA5 Swin isoflop sweeps (Slurm dependencies).

set -euo pipefail

usage() {
  cat <<'EOF'
Submit N chained training jobs per *.yaml config. Each step runs run_multi_gpu.sh
with the same TRAIN_CONFIG; step k>1 waits on step k-1 via --dependency=afterok:JOBID.
Forces LOAD_HIGHEST_CHECKPOINT=1 so each step resumes from the latest epoch/iter checkpoint (overrides YAML load_checkpoint / branch_from).

Usage: bash submit_isoflop_sweep_chain.sh [options] [DIR]

  -d, --dir DIR      Config directory (required, e.g. scripts_6E17_flops)
  -s, --steps N      Chained jobs per config (default: 2, min: 1)
  -n, --dry-run      Print sbatch commands; do not submit
  -h, --help

Per-config Slurm allocation (optional top-level keys; defaults match run_multi_gpu.sh):

  slurm:
    nodes: 1
    gpus_per_node: 4

Requires python3 with PyYAML. Override interpreter with PYTHON=/path/to/python.

Examples:
  bash submit_isoflop_sweep_chain.sh -n -d scripts_6E17_flops --steps 3
  bash submit_isoflop_sweep_chain.sh -d scripts_6E17_flops
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT_NAME="run_multi_gpu.sh"
JOB_NAME_PREFIX="era5"
CONFIG_DIR=""
STEPS=2
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)
      [[ $# -ge 2 ]] || { echo "error: $1 requires an argument" >&2; exit 1; }
      CONFIG_DIR="$2"
      shift 2
      ;;
    -s|--steps)
      [[ $# -ge 2 ]] || { echo "error: $1 requires an argument" >&2; exit 1; }
      STEPS="$2"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${CONFIG_DIR}" ]]; then
        echo "error: only one config directory allowed (got extra: $1)" >&2
        exit 1
      fi
      CONFIG_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_DIR}" ]]; then
  echo "error: config directory required (use -d scripts_<budget>_flops)" >&2
  usage >&2
  exit 1
elif [[ "${CONFIG_DIR}" != /* ]]; then
  CONFIG_DIR="${ROOT}/${CONFIG_DIR}"
fi

JOB_SCRIPT="${ROOT}/${JOB_SCRIPT_NAME}"

if [[ ! "${STEPS}" =~ ^[0-9]+$ ]] || [[ "${STEPS}" -lt 1 ]]; then
  echo "error: --steps must be a positive integer (got: ${STEPS})" >&2
  exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "error: config directory not found: ${CONFIG_DIR}" >&2
  exit 1
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "error: batch script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

cd "${ROOT}"

CONFIG_DIR_ABS="$(cd "${CONFIG_DIR}" && pwd)"
ROOT_ABS="$(cd "${ROOT}" && pwd)"
if [[ "${CONFIG_DIR_ABS}" == "${ROOT_ABS}" ]]; then
  folder_tag="."
elif [[ "${CONFIG_DIR_ABS}" == "${ROOT_ABS}"/* ]]; then
  folder_tag="${CONFIG_DIR_ABS#"${ROOT_ABS}"/}"
else
  folder_tag="$(basename "${CONFIG_DIR_ABS}")"
fi
folder_tag="${folder_tag//\//__}"

shopt -s nullglob
configs=( "${CONFIG_DIR}"/*.yaml )

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "error: no *.yaml in ${CONFIG_DIR}" >&2
  exit 1
fi

LOG_DIR="${ROOT}/slurm_logs"
MANIFEST_DIR="${LOG_DIR}/chain_manifests"
mkdir -p "${MANIFEST_DIR}"

manifest="${MANIFEST_DIR}/${folder_tag}_$(date +%Y%m%d_%H%M%S)_$$.txt"
{
  echo "# config_path  step  job_id  depends_on"
} > "${manifest}"

slurm_resources_from_config() {
  local cfg_path="$1"
  "${PYTHON:-python3}" -c "
import sys
try:
    import yaml
except ImportError:
    print('error: PyYAML is required to read slurm.* from configs', file=sys.stderr)
    sys.exit(1)
with open(sys.argv[1], encoding='utf-8') as f:
    c = yaml.safe_load(f)
if c is None:
    c = {}
slurm = c.get('slurm') or {}
try:
    nodes = int(slurm.get('nodes', 1))
    gpus = int(slurm.get('gpus_per_node', 4))
except (TypeError, ValueError):
    print('error: slurm.nodes and slurm.gpus_per_node must be integers', file=sys.stderr)
    sys.exit(1)
if nodes < 1 or gpus < 1:
    print('error: slurm.nodes and slurm.gpus_per_node must be >= 1', file=sys.stderr)
    sys.exit(1)
print(nodes, gpus)
" "$cfg_path"
}

parse_sbatch_job_id() {
  local output="$1"
  local job_id
  job_id="$(awk '/^Submitted batch job / { print $NF; exit }' <<< "${output}")"
  if [[ -z "${job_id}" ]]; then
    echo "error: failed to parse job id from sbatch output: ${output}" >&2
    exit 1
  fi
  echo "${job_id}"
}

echo "configs=${#configs[@]} steps=${STEPS} manifest=${manifest}"

for config in "${configs[@]}"; do
  abs="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
  base="$(basename "${config}")"
  base="${base%.yaml}"
  log_base="${LOG_DIR}/${folder_tag}__${base}"

  IFS=' ' read -r SLURM_NODES SLURM_GPUS < <(slurm_resources_from_config "${abs}")

  prev_job_id=""
  for (( step=1; step<=STEPS; step++ )); do
    job_name="${JOB_NAME_PREFIX}_${base}_s${step}"
    step_log_base="${log_base}_s${step}"
    dep_flag=()
    dep_display="-"

    if [[ "${step}" -gt 1 ]]; then
      dep_flag=(--dependency="afterok:${prev_job_id}")
      dep_display="${prev_job_id}"
    fi

    echo "config=${abs} step=${step}/${STEPS} job_name=${job_name} nodes=${SLURM_NODES} gpus_per_node=${SLURM_GPUS} depends_on=${dep_display}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      if [[ "${step}" -gt 1 ]]; then
        echo "  sbatch --job-name=${job_name} --nodes=${SLURM_NODES} --gpus-per-node=${SLURM_GPUS} \\"
        echo "    --ntasks-per-node=${SLURM_GPUS} --output=${step_log_base}.out --error=${step_log_base}.err \\"
        echo "    --dependency=afterok:${prev_job_id} \\"
        echo "    --export=ALL,TRAIN_CONFIG=${abs},LOAD_HIGHEST_CHECKPOINT=1 ${JOB_SCRIPT}"
      else
        echo "  sbatch --job-name=${job_name} --nodes=${SLURM_NODES} --gpus-per-node=${SLURM_GPUS} \\"
        echo "    --ntasks-per-node=${SLURM_GPUS} --output=${step_log_base}.out --error=${step_log_base}.err \\"
        echo "    --export=ALL,TRAIN_CONFIG=${abs},LOAD_HIGHEST_CHECKPOINT=1 ${JOB_SCRIPT}"
      fi
      prev_job_id="<step${step}_jobid>"
      printf '%s  %s  %s  %s\n' "${abs}" "${step}" "<dry-run>" "${dep_display}" >> "${manifest}"
      continue
    fi

    sbatch_out="$(
      sbatch \
        --job-name="${job_name}" \
        --nodes="${SLURM_NODES}" \
        --gpus-per-node="${SLURM_GPUS}" \
        --ntasks-per-node="${SLURM_GPUS}" \
        --output="${step_log_base}.out" \
        --error="${step_log_base}.err" \
        "${dep_flag[@]}" \
        --export=ALL,TRAIN_CONFIG="${abs}",LOAD_HIGHEST_CHECKPOINT=1 \
        "${JOB_SCRIPT}" 2>&1
    )"
    job_id="$(parse_sbatch_job_id "${sbatch_out}")"
    echo "  submitted job_id=${job_id}"
    printf '%s  %s  %s  %s\n' "${abs}" "${step}" "${job_id}" "${dep_display}" >> "${manifest}"
    prev_job_id="${job_id}"
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "(dry-run: $(( ${#configs[@]} * STEPS )) job(s) not submitted)"
else
  echo "submitted $(( ${#configs[@]} * STEPS )) job(s); manifest=${manifest}"
fi
