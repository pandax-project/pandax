#!/usr/bin/env python3
"""
verify_csv.py

Scan only bench.ipynb & small_bench.ipynb for pd.read_csv / read_parquet / read_table calls
(preserving all args), then for each referenced file:

  1. Load with pandas.
  2. Load with cudf.
  3. Compare and report column dtypes side by side.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cudf
import pandas as pd

LOADERS = {
    "csv": ("read_csv", pd.read_csv, cudf.read_csv),
    "parquet": ("read_parquet", pd.read_parquet, cudf.read_parquet),
    "table": ("read_table", pd.read_table, pd.read_table),
}

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from utils.notebook_data_calls import gather_data_files


def compare_dtypes(path, loader_key, args, kw_json):
    """
    Load with pandas & cudf, then compare dtypes.
    Returns a list of (column, pandas_dtype, cudf_dtype).
    """
    pd_loader = LOADERS[loader_key][1]
    cudf_loader = LOADERS[loader_key][2]
    kwargs = json.loads(kw_json)

    # Load
    df_pd = pd_loader(path, *args, **kwargs)
    df_cudf = cudf_loader(path, *args, **kwargs)

    # Align columns
    cols = list(df_pd.columns)
    report = []
    for col in cols:
        pd_dt = df_pd.dtypes[col]
        cudf_dt = df_cudf.dtypes[col]

        accepted_dtypes = [
            "int64",
            "float64",
            "object",
            "datetime64[ns]",
            # "bool",
            "int16",
            "string",
        ]
        if str(pd_dt) not in accepted_dtypes:
            print(f"❌ {col} has dtype {pd_dt} which is not in {accepted_dtypes}")
            continue
        report.append((col, str(pd_dt), str(cudf_dt)))
    return report


def main():
    p = argparse.ArgumentParser(
        description="Compare pandas vs cuDF column dtypes for files referenced in bench notebooks"
    )
    p.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Project root to scan (default: current directory)",
    )
    args = p.parse_args()

    files = gather_data_files(args.base_dir)
    if not files:
        print("No data-loading calls found in bench.ipynb or small_bench.ipynb.")
        return

    print(f"Found {len(files)} files. Comparing dtypes:\n")
    for path, fn_key, args, kw_json in sorted(files):
        print(f"File: {path}")
        print(
            f"  Loader: pd.{LOADERS[fn_key][0]}( ..., args={args}, kwargs={json.loads(kw_json)})"
        )
        if not os.path.exists(path):
            print("  ⚠️  File not found, skipping.\n")
            continue

        try:
            diffs = compare_dtypes(path, fn_key, args, kw_json)
        except Exception as e:
            print(f"  ✖ Error loading with pd or cudf: {e}\n")
            continue

        # Check for mismatches
        mismatches = [row for row in diffs if row[1] != row[2]]
        if not mismatches:
            print("  ✅ All column dtypes match.")
        else:
            print("  ❌ Mismatched dtypes:")
            for col, pd_dt, cudf_dt in mismatches:
                print(f"    – {col!r}: pandas={pd_dt}, cudf={cudf_dt}")
        print()


if __name__ == "__main__":
    main()
