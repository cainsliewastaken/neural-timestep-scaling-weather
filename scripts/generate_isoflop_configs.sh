#!/usr/bin/env bash
# Generate isoflop configs inside the upstream weather-pytorch Shifter image (needs transformer_engine).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage: bash scripts/generate_isoflop_configs.sh [args for generate_isoflop_configs.py]

Example (one line is fine):
  bash scripts/generate_isoflop_configs.sh --budget-label 3e9 --target-rollout-flops 3e9 --total-train-flops 1e18 --train-final-time-hours 24 --dt-scales 1 2 3 4 6 8 12 --depth 12

Or multiline:
  bash scripts/generate_isoflop_configs.sh \
    --budget-label 3e9 \
    --target-rollout-flops 3e9 \
    --total-train-flops 1e18 \
    --train-final-time-hours 24 \
    --dt-scales 1 2 3 4 6 8 12 \
    --depth 12

For a quick smoke test on a reduced grid:
  bash scripts/generate_isoflop_configs.sh \
    --budget-label TEST --target-rollout-flops 1e12 --total-train-flops 1e15 --train-final-time-hours 24 \
    --dt-scales 1 2 --depth 4 --grid-h 144 --grid-w 288 --min 32 --max 512 \
    --output-dir scripts_TEST_flops
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
  bash -lc 'cd "$1" && shift && exec python scripts/generate_isoflop_configs.py "$@"' \
  _ "${ROOT}" "${args[@]}"
