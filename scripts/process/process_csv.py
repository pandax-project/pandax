#!/usr/bin/env python3
"""
process_csv.py

Recursively find only bench.ipynb & small_bench.ipynb under a base dir,
extract every pd.read_csv / pd.read_parquet / pd.read_table call (preserving
all its args), then for each referenced file:

  • re-run the same pd.read_* call with all flags
  • write it back to the same path in the same format
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from utils.notebook_data_calls import gather_data_files


def reload_and_dump(path, loader, args, kw_json):
    """Re-run pd.read_* with the same args/kwargs, then overwrite file."""
    kwargs = json.loads(kw_json)
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return
    try:
        if loader == "csv":
            df = pd.read_csv(path, *args, **kwargs)
            df.to_csv(path, index=False)
        elif loader == "parquet":
            df = pd.read_parquet(path, *args, **kwargs)
            df.to_parquet(path, index=False)
        elif loader == "table":
            df = pd.read_table(path, *args, **kwargs)
            df.to_csv(path, sep=kwargs.get("sep", "\t"), index=False)
        print(f"  ✔ Reloaded & dumped ({loader}) → {path}")
    except Exception as e:
        print(f"  ✖ Error processing {path} with args={args} kw={kwargs}: {e}")


def main():
    p = argparse.ArgumentParser(
        description="Reload & re-dump data files referenced by pd.read_* in bench notebooks, preserving all read args"
    )
    p.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory to search (default: current dir)",
    )
    args = p.parse_args()

    print(f"Scanning for bench.ipynb & small_bench.ipynb under: {args.base_dir!r}")
    files = gather_data_files(args.base_dir, verbose=True)
    if not files:
        print("No pd.read_* calls found in bench notebooks.")
        return

    print(f"Found {len(files)} unique data files. Processing:\n")
    for path, loader, args, kw_json in sorted(files):
        if "Billionaires Statistics Dataset.csv" in path or "uk_pm.parquet" in path:
            continue
        reload_and_dump(path, loader, args, kw_json)


if __name__ == "__main__":
    main()