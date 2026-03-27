#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./verify_datasets_are_copied.sh [base_dir ...]
# If no base dir is provided, defaults to ./dias_notebooks.
if [ "$#" -eq 0 ]; then
  set -- "./dias_notebooks"
fi

for base_dir in "$@"; do
  if ! [ -d "$base_dir" ]; then
    echo "Base directory does not exist: $base_dir"
    exit 1
  fi

  while IFS= read -r -d '' dir; do
    if ! [ -d "$dir/input" ]; then
      echo "input/ was not found in $dir. This means that the dataset was not copied successfully."
      exit 1
    fi
  done < <(find "$base_dir" -mindepth 2 -maxdepth 2 -type d -print0)
done
