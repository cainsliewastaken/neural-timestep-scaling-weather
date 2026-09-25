#!/usr/bin/env bash
# One Slurm job array for generated isoflop sweep configs (ERA5 Swin).

set -euo pipefail

usage() {
  cat <<'EOF'
Submit one sbatch array for *.yaml configs produced by scripts/generate_isoflop_configs.py.
Each array task reads one config path from a manifest and run_multi_gpu.sh launches train.py.

Usage: bash submit_isoflop_sweep.sh [options] [DIR]

  -d, --dir DIR      Config directory (default: <repo>/scripts_<budget>_flops)
  -n, --dry-run      Print sbatch command; do not submit
  -h, --help

Array submissions use run_multi_gpu.sh. All configs in the sweep must share the same
Slurm allocation. Resources are read from the top-level slurm: block in each YAML.

Requires python3 with PyYAML. Override interpreter with PYTHON=/path/to/python.

Examples:
  bash submit_isoflop_sweep.sh -n
  bash submit_isoflop_sweep.sh -d scripts_6E17_flops
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT_NAME="run_multi_gpu.sh"
CONFIG_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)
      [[ $# -ge 2 ]] || { echo "error: $1 requires an argument" >&2; exit 1; }
      CONFIG_DIR="$2"
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
mkdir -p "${LOG_DIR}"

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

first_abs="$(cd "$(dirname "${configs[0]}")" && pwd)/$(basename "${configs[0]}")"
IFS=' ' read -r SLURM_NODES SLURM_GPUS < <(slurm_resources_from_config "${first_abs}")

for config in "${configs[@]}"; do
  abs="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
  IFS=' ' read -r NODES GPUS < <(slurm_resources_from_config "${abs}")
  if [[ "${NODES}" != "${SLURM_NODES}" || "${GPUS}" != "${SLURM_GPUS}" ]]; then
    echo "error: array submission requires identical slurm resources for every config" >&2
    echo "  expected nodes=${SLURM_NODES} gpus_per_node=${SLURM_GPUS} (from ${first_abs})" >&2
    echo "  found    nodes=${NODES} gpus_per_node=${GPUS} in ${abs}" >&2
    exit 1
  fi
done

manifest_dir="${LOG_DIR}/array_manifests"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/${folder_tag}_$(date +%Y%m%d_%H%M%S)_$$.txt"

for config in "${configs[@]}"; do
  abs="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
  printf '%s\n' "${abs}" >> "${manifest}"
done

array_end=$(( ${#configs[@]} - 1 ))
log_base="${LOG_DIR}/${folder_tag}__array_%A_%a"

echo "array_tasks=${#configs[@]} range=0-${array_end} nodes=${SLURM_NODES} gpus_per_node=${SLURM_GPUS} logs=${log_base}.{out,err}"
echo "manifest=${manifest}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "  sbatch --job-name=era5_isoflop_${folder_tag} --array=0-${array_end} --nodes=${SLURM_NODES} --gpus-per-node=${SLURM_GPUS} \\"
  echo "    --ntasks-per-node=${SLURM_GPUS} --output=${log_base}.out --error=${log_base}.err \\"
  echo "    --export=ALL,ARRAY_CONFIG_MANIFEST=${manifest} ${JOB_SCRIPT}"
  echo "(dry-run: array job not submitted)"
  exit 0
fi

sbatch \
  --job-name="era5_isoflop_${folder_tag}" \
  --array="0-${array_end}" \
  --nodes="${SLURM_NODES}" \
  --gpus-per-node="${SLURM_GPUS}" \
  --ntasks-per-node="${SLURM_GPUS}" \
  --output="${log_base}.out" \
  --error="${log_base}.err" \
  --export=ALL,ARRAY_CONFIG_MANIFEST="${manifest}" \
  "${JOB_SCRIPT}"
