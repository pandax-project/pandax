#!/usr/bin/env bash
set -euo pipefail
# Usage: ./recreate.sh [NEW_ENV_NAME]
# If NEW_ENV_NAME is omitted, we'll use the name embedded in the YAML.
# Env location defaults to: /opt/conda/envs/<env-name>
# Override parent dir with ENV_HOME_DIR, e.g.:
#   ENV_HOME_DIR="/mnt/fast-ssd/conda-envs" ./recreate.sh rapids-25.02

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_HISTORY="$THIS_DIR/environment_from_history.yml"
FULL="$THIS_DIR/environment.yml"
PIP_REQS="$THIS_DIR/pip-requirements.txt"

NEW_NAME="${1:-}"
ENV_HOME_DIR="${ENV_HOME_DIR:-/opt/conda/envs}"

# Resolve conda in non-interactive shells where shell init files
# (and the conda function) may not be loaded.
if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="$(command -v conda)"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_CMD="${CONDA_EXE}"
else
  for candidate in \
    "/opt/conda/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA_CMD="$candidate"
      break
    fi
  done
fi

if [[ -z "${CONDA_CMD:-}" ]]; then
  if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
    CONDA_CMD="$(command -v conda || true)"
  fi
fi

if [[ -z "${CONDA_CMD:-}" ]]; then
  echo "ERROR: conda not found in PATH and no known install path worked." >&2
  echo "Try exporting CONDA_EXE (e.g. /opt/conda/bin/conda) and rerun." >&2
  exit 127
fi

create_from_yaml () {
  local yaml="$1"
  "$CONDA_CMD" env create -p "$ENV_PREFIX" -f "$yaml"
}

resolve_env_name () {
  if [[ -n "$NEW_NAME" ]]; then
    printf '%s\n' "$NEW_NAME"
    return
  fi
  local name
  name="$(awk '/^name:/{print $2; exit}' "$FROM_HISTORY" 2>/dev/null || true)"
  if [[ -z "$name" ]]; then
    name="$(awk '/^name:/{print $2; exit}' "$FULL" 2>/dev/null || true)"
  fi
  printf '%s\n' "$name"
}

remove_env_if_exists () {
  local prefix="$1"
  [[ -z "$prefix" ]] && return 0
  if "$CONDA_CMD" env list | awk 'NR>2 {print $NF}' | grep -Fxq "$prefix"; then
    echo "Existing env at '$prefix' found; removing it before recreate…"
    "$CONDA_CMD" env remove -p "$prefix" -y
  elif [[ -d "$prefix" ]]; then
    # Handle stale directories that are not registered as conda envs.
    echo "Removing stale env directory '$prefix'…"
    rm -rf "$prefix"
  fi
}

ENV_NAME="$(resolve_env_name)"
if [[ -z "$ENV_NAME" ]]; then
  echo "ERROR: Could not determine environment name from YAML or argument." >&2
  echo "Pass an explicit env name: ./recreate.sh <NEW_ENV_NAME>" >&2
  exit 2
fi

if [[ ! -d "$ENV_HOME_DIR" ]]; then
  if ! mkdir -p "$ENV_HOME_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create '$ENV_HOME_DIR' (permission denied)." >&2
    echo "Run with sudo for shared locations, or set ENV_HOME_DIR to a writable path." >&2
    exit 13
  fi
fi

if [[ ! -w "$ENV_HOME_DIR" ]]; then
  echo "ERROR: '$ENV_HOME_DIR' is not writable by user '$USER'." >&2
  echo "Use sudo for shared locations (e.g. /opt/conda/envs), or set ENV_HOME_DIR." >&2
  exit 13
fi

ENV_PREFIX="$ENV_HOME_DIR/$ENV_NAME"

echo "Target env prefix: $ENV_PREFIX"
remove_env_if_exists "$ENV_PREFIX"

if [[ -s "$FROM_HISTORY" ]]; then
  echo "Creating conda env from minimal spec (from-history)…"
  create_from_yaml "$FROM_HISTORY" || {
    echo "from-history creation failed; trying full environment.yml…" >&2
    # If a partial env was created before failing, clear it out first.
    remove_env_if_exists "$ENV_PREFIX"
    create_from_yaml "$FULL"
  }
else
  echo "No from-history export; creating from full environment.yml…"
  create_from_yaml "$FULL"
fi

if [[ -s "$PIP_REQS" ]]; then
  echo "Installing pip packages into '$ENV_PREFIX'…"
  "$CONDA_CMD" run -p "$ENV_PREFIX" python -m pip install -r "$PIP_REQS"
else
  echo "No pip-requirements.txt found or it is empty; skipping pip installs."
fi

echo "✅ Done. To use it:  conda activate ${ENV_PREFIX}"
