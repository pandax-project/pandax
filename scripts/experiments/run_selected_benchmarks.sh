#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# You listed 6 benchmarks; all are included here.
BENCHMARKS=(
  "comprehensive-data-exploration-with-python"
  "retail-supermarket-store-analysis"
  "adidas-retail-eda-data-visualization"
  "indian-startup-growth-analysis"
  "imdb-dataset-eda-project"
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/main_runs/${TIMESTAMP}}"
SUMMARY_FILE="${LOG_DIR}/summary.txt"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "${REPO_ROOT}"

echo "Logs directory: ${LOG_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Mode: DRY RUN (no commands will be executed)"
fi
echo

SUCCESS_COUNT=0
FAIL_COUNT=0

for benchmark in "${BENCHMARKS[@]}"; do
  log_file="${LOG_DIR}/${benchmark}.log"
  cmd="python -u main.py ${benchmark}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "=== Would run ${benchmark} ==="
    echo "Command: ${cmd}"
    echo "Log file: ${log_file}"
    echo
    continue
  fi

  mkdir -p "${LOG_DIR}"
  echo "=== Running ${benchmark} ==="
  echo "Command: ${cmd}" | tee "${log_file}"
  echo "Started: $(date)" | tee -a "${log_file}"

  if ${cmd} >> "${log_file}" 2>&1; then
    echo "Status: SUCCESS" | tee -a "${log_file}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo "${benchmark}: SUCCESS (${log_file})" >> "${SUMMARY_FILE}"
  else
    exit_code=$?
    echo "Status: FAILED (exit ${exit_code})" | tee -a "${log_file}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "${benchmark}: FAILED (exit ${exit_code}) (${log_file})" >> "${SUMMARY_FILE}"
  fi

  echo "Finished: $(date)" | tee -a "${log_file}"
  echo | tee -a "${log_file}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete."
else
  echo "Completed benchmark runs."
  echo "Success: ${SUCCESS_COUNT}"
  echo "Failed: ${FAIL_COUNT}"
  echo "Summary written to: ${SUMMARY_FILE}"
fi
