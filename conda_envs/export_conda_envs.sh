#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./export_conda_envs.sh [OUTDIR]
  ./export_conda_envs.sh --outdir OUTDIR [--prefix ENV_PREFIX]

Options:
  --outdir OUTDIR      Output directory (default: conda-env-exports)
  --prefix ENV_PREFIX  Export only the environment at this full prefix path
  -h, --help           Show this help
EOF
}

OUTDIR="conda-env-exports"
TARGET_PREFIX="${TARGET_PREFIX:-}"
OUTDIR_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="${2:?missing value for --outdir}"
      OUTDIR_SET=1
      shift 2
      ;;
    --prefix)
      TARGET_PREFIX="${2:?missing value for --prefix}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ "$OUTDIR_SET" -eq 0 ]]; then
        OUTDIR="$1"
        OUTDIR_SET=1
        shift
      else
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

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

if [[ -n "$TARGET_PREFIX" ]]; then
  TARGET_PREFIX_NORM="$(readlink -f "$TARGET_PREFIX" 2>/dev/null || echo "$TARGET_PREFIX")"
  FILTERED_ENV_PATHS=()
  for E in "${ENV_PATHS[@]}"; do
    E_NORM="$(readlink -f "$E" 2>/dev/null || echo "$E")"
    if [[ "$E_NORM" == "$TARGET_PREFIX_NORM" ]]; then
      FILTERED_ENV_PATHS+=("$E")
      break
    fi
  done
  if [[ "${#FILTERED_ENV_PATHS[@]}" -eq 0 ]]; then
    echo "No conda environment found for prefix: $TARGET_PREFIX" >&2
    exit 1
  fi
  ENV_PATHS=("${FILTERED_ENV_PATHS[@]}")
fi

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
  if ! conda env export -p "$E" --no-builds > "$DEST/environment.yml"; then
    echo "  ⚠️  conda env export failed for $NAME; continuing..."
  fi

  # 2) From-history export (minimal set of packages you explicitly asked for via conda)
  #    This is closest to the “initial create” intent.
  if ! conda env export -p "$E" --from-history > "$DEST/environment_from_history.yml"; then
    echo "  ⚠️  from-history export failed for $NAME; continuing..."
  fi

  # 3) Pip requirements without local @file paths, excluding conda-managed installs.
  #    We intentionally avoid `pip freeze` because it can emit non-portable lines like
  #    `pkg @ file:///...`. Instead, we use `pip list --format=json` and emit
  #    pinned `name==version` entries for packages that are not conda-managed.
  CONDA_LIST_JSON="$DEST/.conda_list.json"
  PIP_LIST_JSON="$DEST/.pip_list.json"
  if conda list -p "$E" --json > "$CONDA_LIST_JSON" \
    && conda run -p "$E" python -m pip list --format=json > "$PIP_LIST_JSON"; then
    if ! python - "$CONDA_LIST_JSON" "$PIP_LIST_JSON" "$DEST/environment_from_history.yml" "$DEST/pip-requirements.txt" <<'PY'
import json
import re
import sys
from pathlib import Path

conda_json_path = Path(sys.argv[1])
pip_json_path = Path(sys.argv[2])
from_history_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])


def norm_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_from_history_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if not item or item == "pip:":
            continue
        pkg_name = re.split(r"[<>=!~ ]", item, maxsplit=1)[0]
        if pkg_name:
            names.add(norm_name(pkg_name))
    return names


conda_pkgs = json.loads(conda_json_path.read_text(encoding="utf-8"))
pip_pkgs = json.loads(pip_json_path.read_text(encoding="utf-8"))

# Any package with channel != pypi is considered conda-managed.
conda_managed = {
    norm_name(pkg.get("name", ""))
    for pkg in conda_pkgs
    if pkg.get("name") and pkg.get("channel") != "pypi"
}

# Also exclude packages explicitly requested in `conda env create/install` history.
conda_managed |= parse_from_history_names(from_history_path)

lines: list[str] = []
for pkg in sorted(pip_pkgs, key=lambda p: norm_name(p.get("name", ""))):
    name = pkg.get("name")
    version = pkg.get("version")
    if not name:
        continue
    if norm_name(name) in conda_managed:
        continue
    if version:
        lines.append(f"{name}=={version}")
    else:
        lines.append(name)

out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
PY
    then
      echo "  ⚠️  failed to build filtered pip requirements for $NAME; continuing..."
    fi
  else
    echo "  ⚠️  pip list export failed for $NAME; continuing..."
  fi
  rm -f "$CONDA_LIST_JSON" "$PIP_LIST_JSON"

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
- pip-requirements.txt            → Pip packages (no local @file entries; excludes conda-managed packages)
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

