#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-conda-env-exports}"
SKIP_BASE="${SKIP_BASE:-1}"  # set to 0 to include "base"

mkdir -p "$OUTDIR"

# Ensure jq is installed (used to parse JSON). If you don't have jq, install it or
# replace the jq usage with a pure-grep/awk variant.
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Please install jq (e.g., sudo apt-get install jq) and re-run." >&2
  exit 1
fi

# Make sure conda is on PATH.
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Source your conda setup first (e.g., 'source ~/miniconda3/etc/profile.d/conda.sh')." >&2
  exit 1
fi

# List all env paths as JSON, then iterate.
mapfile -t ENV_PATHS < <(conda env list --json | jq -r '.envs[]')

for E in "${ENV_PATHS[@]}"; do
  NAME="$(basename "$E")"

  if [[ "$NAME" == "base" && "$SKIP_BASE" == "1" ]]; then
    echo "Skipping base (set SKIP_BASE=0 to include it)."
    continue
  fi

  echo "Exporting env: $NAME ($E)"
  DEST="$OUTDIR/$NAME"
  mkdir -p "$DEST"

  # 1) Full export (includes exact versions, channels, and pip section if present)
  #    --no-builds keeps it more portable while still pinning versions.
  if ! conda env export -n "$NAME" --no-builds > "$DEST/environment.yml"; then
    echo "  ⚠️  conda env export failed for $NAME; continuing..."
  fi

  # 2) From-history export (minimal set of packages you explicitly asked for via conda)
  #    This is closest to the “initial create” intent.
  if ! conda env export -n "$NAME" --from-history > "$DEST/environment_from_history.yml"; then
    echo "  ⚠️  from-history export failed for $NAME; continuing..."
  fi

  # 3) Exact pip packages (only pip-managed ones, independent of conda)
  #    Use python -m pip via conda run to ensure we capture the env’s pip.
  if ! conda run -n "$NAME" python -m pip freeze --exclude-editable > "$DEST/pip-requirements.txt"; then
    echo "  ⚠️  pip freeze failed for $NAME; continuing..."
  fi

  # 4) Recreate script: tries minimal repro first, falls back to full, then pip.
  cat > "$DEST/recreate.sh" <<'EOS'
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
EOS

  chmod +x "$DEST/recreate.sh"

  # 5) A tiny README for convenience
  cat > "$DEST/README.txt" <<EOF
Environment: $NAME
Path: $E

Files:
- environment.yml                 → Full export (with versions & pip section)
- environment_from_history.yml    → Minimal spec of conda packages you requested
- pip-requirements.txt            → Exact pip packages (from pip freeze)
- recreate.sh                     → Rebuild script (uses from-history first, then full; then pip)

Quick rebuild:
  cd "$NAME"
  ./recreate.sh                   # creates env with the name in YAML
  # or ./recreate.sh ${NAME}-clone

Notes:
- "from-history" is closest to the original 'conda create ...' + subsequent 'conda install ...' commands.
- Pip packages are installed after the conda env is created.
- If an env was created entirely from a YAML (and not via 'conda install'), from-history may be sparse; the script falls back to full export.
EOF

done

echo "All done. Exports are in: $OUTDIR"