#!/usr/bin/env bash
set -euo pipefail

# Create a RAPIDS 26.02 env at a prefix, then install non-CUDA extras
# from the rapids-25.02 snapshot directory.
#
# Usage:
#   bash rapids-25.02/create_rapids_26_02_prefix.sh
#   bash rapids-25.02/create_rapids_26_02_prefix.sh /scratch/jieq/conda-envs/rapids-26.02
#
# Optional env vars:
#   RAPIDS_VERSION=26.02
#   PYTHON_VERSION=3.10
#   CUDA_VERSION_RANGE=">=13.0,<=13.1"
#   STRICT_NONCUDA=1   # fail if any extracted conda package cannot be installed

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PREFIX="${1:-/scratch/jieq/conda-envs/rapids-26.02}"

RAPIDS_VERSION="${RAPIDS_VERSION:-26.02}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION_RANGE="${CUDA_VERSION_RANGE:->=13.0,<=13.1}"
STRICT_NONCUDA="${STRICT_NONCUDA:-0}"

ENV_YAML="$THIS_DIR/environment.yml"
PIP_REQS="$THIS_DIR/pip-requirements-rapids-26.02.txt"
if [[ ! -f "$PIP_REQS" ]]; then
  PIP_REQS="$THIS_DIR/pip-requirements.txt"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$ENV_YAML" ]]; then
  echo "Missing $ENV_YAML" >&2
  exit 1
fi

echo "==> Creating base RAPIDS env at: $TARGET_PREFIX"
conda create -y \
  --prefix "$TARGET_PREFIX" \
  -c rapidsai \
  -c conda-forge \
  "rapids=${RAPIDS_VERSION}" \
  "python=${PYTHON_VERSION}" \
  "cuda-version${CUDA_VERSION_RANGE}"

tmp_noncuda="$(mktemp)"
tmp_failed="$(mktemp)"
trap 'rm -f "$tmp_noncuda" "$tmp_failed"' EXIT

echo "==> Extracting non-CUDA conda specs from $ENV_YAML"
python - "$ENV_YAML" "$tmp_noncuda" <<'PY'
import re
import sys
from pathlib import Path

env_yaml = Path(sys.argv[1])
out_path = Path(sys.argv[2])

drop_patterns = [
    r"^python(?:=|$)",
    r"^python_abi(?:=|$)",
    r"^rapids(?:=|$)",
    r"^rapids-dask-dependency(?:=|$)",
    r"^cudf(?:=|$)",
    r"^cuml(?:=|$)",
    r"^cupy(?:=|$)",
    r"^cupy-core(?:=|$)",
    r"^cuda(?:-|=|$)",
    r"^cudatoolkit(?:=|$)",
    r"^cubinlinker(?:=|$)",
    r"^numba-cuda(?:=|$)",
    r"^nvidia-ml-py(?:=|$)",
    r"^nccl(?:=|$)",
    r"^nv(?:comp|tx)(?:=|$)",
    r"^pylibcudf(?:=|$)",
    r"^pylibraft(?:=|$)",
    r"^raft-dask(?:=|$)",
    r"^rmm(?:=|$)",
    r"^libcudf(?:=|$)",
    r"^libcuml(?:=|$)",
    r"^libcumlprims(?:=|$)",
    r"^libcuvs(?:=|$)",
    r"^librmm(?:=|$)",
    r"^libraft(?:=|$)",
    r"^libraft-headers(?:=|$)",
    r"^libraft-headers-only(?:=|$)",
    r"^libkvikio(?:=|$)",
    r"^libucxx(?:=|$)",
    r"^ucx(?:=|$)",
    r"^ucx-py(?:=|$)",
    r"^ucxx(?:=|$)",
    r"^distributed-ucxx(?:=|$)",
    r"^dask-cuda(?:=|$)",
    r"^dask-cudf(?:=|$)",
    r"^libcu.*",
]
drop_re = re.compile("|".join(f"(?:{p})" for p in drop_patterns))

specs = []
for raw in env_yaml.read_text().splitlines():
    line = raw.rstrip()
    if not line.startswith("  - "):
        continue
    spec = line[4:].strip()
    if not spec or spec.startswith("pip:"):
        continue
    # Skip YAML map entries (e.g., "- pip:")
    if ":" in spec and not any(op in spec for op in ["=", "<", ">"]):
        continue
    if drop_re.match(spec):
        continue
    specs.append(spec)

# Deduplicate while preserving order.
seen = set()
ordered = []
for s in specs:
    if s not in seen:
        seen.add(s)
        ordered.append(s)

out_path.write_text("\n".join(ordered) + ("\n" if ordered else ""))
print(f"kept_specs={len(ordered)}")
PY

echo "==> Installing extracted non-CUDA conda packages"
if [[ -s "$tmp_noncuda" ]]; then
  # Fast path: install all in one solve.
  if ! conda install -y --prefix "$TARGET_PREFIX" -c conda-forge --file "$tmp_noncuda"; then
    echo "Bulk install failed; retrying package-by-package..." >&2
    while IFS= read -r spec; do
      [[ -z "$spec" ]] && continue
      if ! conda install -y --prefix "$TARGET_PREFIX" -c conda-forge "$spec"; then
        echo "$spec" >> "$tmp_failed"
      fi
    done < "$tmp_noncuda"
  fi
fi

if [[ -s "$tmp_failed" ]]; then
  echo "==> Some non-CUDA conda specs could not be installed:"
  sed 's/^/  - /' "$tmp_failed"
  if [[ "$STRICT_NONCUDA" == "1" ]]; then
    echo "STRICT_NONCUDA=1 -> failing." >&2
    exit 2
  fi
fi

if [[ -s "$PIP_REQS" ]]; then
  echo "==> Installing pip requirements from $PIP_REQS"
  # --no-deps keeps conda solver in control for core packages.
  conda run --prefix "$TARGET_PREFIX" python -m pip install -r "$PIP_REQS" --no-deps
fi

echo "==> Done."
echo "Activate with:"
echo "  conda activate $TARGET_PREFIX"
