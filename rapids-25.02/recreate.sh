#!/usr/bin/env bash
set -euo pipefail
# Usage: ./recreate.sh [NEW_ENV_NAME] [ENV_PREFIX]
# If NEW_ENV_NAME is omitted, we'll use the name embedded in the YAML.
# If ENV_PREFIX is provided, conda creates/uses the env at that prefix.

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_HISTORY="$THIS_DIR/environment_from_history.yml"
FULL="$THIS_DIR/environment.yml"
PIP_REQS="$THIS_DIR/pip-requirements.txt"

NEW_NAME="${1:-}"
ENV_PREFIX="${2:-}"

create_from_yaml () {
  local yaml="$1"
  if [[ -n "$ENV_PREFIX" ]]; then
    # Create at an explicit prefix path.
    conda env create --file="$yaml" --prefix="$ENV_PREFIX"
  elif [[ -n "$NEW_NAME" ]]; then
    # Create with a new name by overriding in-place via a temp file.
    tmp="$(mktemp)"
    # Replace 'name:' in YAML (best-effort). If not present, conda will use dir name.
    if grep -qE '^name:' "$yaml"; then
      sed -E "s/^name:.*/name: ${NEW_NAME}/" "$yaml" > "$tmp"
    else
      printf 'name: %s\n' "$NEW_NAME" > "$tmp"
      cat "$yaml" >> "$tmp"
    fi
    # Use the long option to avoid deprecated code paths:
    conda env create --file="$tmp"
    rm -f "$tmp"
  else
    # Use the long option to avoid deprecated code paths:
    conda env create --file="$yaml"
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

# Determine the target we actually created (prefix or name)
ENV_TARGET="$ENV_PREFIX"
ENV_TARGET_FLAG="--prefix"
if [[ -z "$ENV_TARGET" ]]; then
  ENV_TARGET="$NEW_NAME"
  ENV_TARGET_FLAG="--name"
  if [[ -z "$ENV_TARGET" ]]; then
    # Extract from YAML's name: field (first match across the two files)
    ENV_TARGET="$(grep -E '^name:' "$FROM_HISTORY" "$FULL" 2>/dev/null | head -n1 | awk '{print $2}')"
  fi
fi

# If FULL already contains a pip: section, conda handled pip installs; skip extra step.
if grep -Eq '^\s*-\s+pip:\s*$' "$FULL"; then
  echo "environment.yml contains a pip: section; skipping extra pip-requirements step."
elif [[ -s "$PIP_REQS" ]]; then
  echo "Installing pip packages into '$ENV_TARGET'…"
  conda run "$ENV_TARGET_FLAG" "$ENV_TARGET" python -m pip install -r "$PIP_REQS" --no-deps
else
  echo "No pip-requirements.txt found or it is empty; skipping pip installs."
fi

echo "✅ Done. To use it:  conda activate ${ENV_TARGET}"
