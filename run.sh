#!/usr/bin/env bash
set -euo pipefail

# LabOS Robot Runtime remote MCP launcher.
# Copy .env.example to .env and fill LABOS_URL, LABOS_API_KEY, LABOS_DEVICE_ID.
#
# Usage:
#   ./run.sh
#   ./run.sh --headless
#   ./run.sh --mock

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MOCK=0
ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--mock" ]]; then
    MOCK=1
  else
    ARGS+=("${arg}")
  fi
done

if [[ "${MOCK}" == "1" ]]; then
  exec python run_mock_mcp.py "${ARGS[@]}"
fi

exec python run_mcp.py "${ARGS[@]}"
