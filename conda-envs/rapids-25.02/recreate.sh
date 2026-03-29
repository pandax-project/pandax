#!/usr/bin/env bash
set -euo pipefail
# Usage: ./recreate.sh [NEW_ENV_NAME]
# If NEW_ENV_NAME is omitted, we'll use the name embedded in the YAML.

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_HISTORY="$THIS_DIR/environment_from_history.yml"
FULL="$THIS_DIR/environment.yml"
PIP_REQS="$THIS_DIR/pip-requirements.txt"

NEW_NAME="${1:-}"

create_from_yaml () {
  local yaml="$1"
  if [[ -n "$NEW_NAME" ]]; then
    # Create with a new name by overriding in-place via a temp file.
    tmp="$(mktemp)"
    # Replace 'name:' in YAML (best-effort). If not present, conda will use dir name.
    if grep -qE '^name:' "$yaml"; then
      sed -E "s/^name:.*/name: ${NEW_NAME}/" "$yaml" > "$tmp"
    else
      printf 'name: %s\n' "$NEW_NAME" > "$tmp"
      cat "$yaml" >> "$tmp"
    fi
    conda env create -f "$tmp"
    rm -f "$tmp"
  else
    conda env create -f "$yaml"
  fi
}

if [[ -s "$FROM_HISTORY" ]]; then
  echo "Creating conda env from minimal spec (from-history)…"
  create_from_yaml "$FROM_HISTORY" || {
    echo "from-history creation failed; trying full environment.yml…" >&2
    create_from_yaml "$FULL"
  }
else
  echo "No from-history export; creating from full environment.yml…"
  create_from_yaml "$FULL"
fi

# Determine the name we actually created (NEW_NAME or the YAML's name)
ENV_NAME="$NEW_NAME"
if [[ -z "$ENV_NAME" ]]; then
  # Extract from YAML's name: field
  ENV_NAME="$(grep -E '^name:' "$FROM_HISTORY" "$FULL" 2>/dev/null | head -n1 | awk '{print $2}')"
fi

if [[ -s "$PIP_REQS" ]]; then
  echo "Installing pip packages into '$ENV_NAME'…"
  conda run -n "$ENV_NAME" python -m pip install -r "$PIP_REQS"
else
  echo "No pip-requirements.txt found or it is empty; skipping pip installs."
fi

echo "✅ Done. To use it:  conda activate ${ENV_NAME}"
