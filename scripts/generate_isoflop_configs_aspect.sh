#!/usr/bin/env bash
# Generate aspect-ratio isoflop configs inside the upstream weather-pytorch Shifter image.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage: bash scripts/generate_isoflop_configs_aspect.sh [args for generate_isoflop_configs_aspect.py]

Example (one line is fine; dt-scales defaults to 1 2 3 4 6 8 12):
  bash scripts/generate_isoflop_configs_aspect.sh --budget-label 12T --target-rollout-flops 12e12 --train-final-time-hours 24

Or multiline with an explicit dt sweep:
  bash scripts/generate_isoflop_configs_aspect.sh \
    --budget-label 3e9 \
    --target-rollout-flops 3e9 \
    --total-train-flops 1e18 \
    --train-final-time-hours 24 \
    --dt-scales 1 2 3 4 6 8 12 \
    --min-depth 2 --max-depth 30 \
    --min-aspect-ratio 40 --max-aspect-ratio 100

For a quick smoke test on a reduced grid:
  bash scripts/generate_isoflop_configs_aspect.sh \
    --budget-label TEST_ASPECT --target-rollout-flops 1e12 --total-train-flops 1e15 --train-final-time-hours 24 \
    --dt-scales 1 2 --grid-h 144 --grid-w 288 --min 32 --max 512 \
    --min-depth 2 --max-depth 30 \
    --output-dir scripts_TEST_ASPECT_flops
EOF
  exit 1
fi

# Drop empty / whitespace-only tokens (can appear when pasting "\  " continuations).
args=()
for arg in "$@"; do
  if [[ "${arg//[$' \t\r\n']/}" != "" ]]; then
    args+=("$arg")
  fi
done

if [[ ${#args[@]} -eq 0 ]]; then
  echo "error: no non-empty arguments after filtering" >&2
  exit 1
fi

# Pass argv without nested printf quoting (avoids stray '' tokens for --dt-scales).
shifter --image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.06 \
  bash -lc 'cd "$1" && shift && exec python scripts/generate_isoflop_configs_aspect.py "$@"' \
  _ "${ROOT}" "${args[@]}"
