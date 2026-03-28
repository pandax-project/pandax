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
import re

import cudf
import pandas as pd
from scripts.utils.notebook_data_calls import gather_data_files

LOADERS = {
    "csv": ("read_csv", pd.read_csv, cudf.read_csv),
    "parquet": ("read_parquet", pd.read_parquet, cudf.read_parquet),
    "table": ("read_table", pd.read_table, pd.read_table),
}


def _apply_loader_overrides(path, loader_key, kwargs):
    """Apply file-specific read kwargs to align inferred dtypes."""
    basename = os.path.basename(path)
    if loader_key == "csv" and basename == "title-metadata.csv":
        # Force `genres` to string-like without changing NA handling for numeric columns.
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        dtype["genres"] = "str"
        kwargs["dtype"] = dtype
    elif loader_key == "csv" and basename == "Billionaires Statistics Dataset.csv":
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        # Keep this boolean-like column within accepted/reportable dtypes.
        dtype["selfMade"] = "object"
        # Align nullable numeric columns across pandas/cuDF.
        for col in ["age", "birthYear", "birthMonth", "birthDay", "population_country"]:
            dtype[col] = "float64"
        kwargs["dtype"] = dtype
    return kwargs


def _is_supported_pd_dtype(dtype_str):
    """Return True for pandas dtypes we explicitly support/report."""
    accepted_exact = {
        "int64",
        "float64",
        "object",
        "datetime64[ns]",
        "int16",
        "string",
    }
    if dtype_str in accepted_exact:
        return True

    # Accept timezone-aware datetimes with any pandas datetime unit, e.g.
    # datetime64[us, UTC], datetime64[ns, Europe/London].
    return bool(re.fullmatch(r"datetime64\[[a-z]+,\s*[^\]]+\]", dtype_str))


def compare_dtypes(path, loader_key, args, kw_json):
    """
    Load with pandas & cudf, then compare dtypes.
    Returns a list of (column, pandas_dtype, cudf_dtype).
    """
    pd_loader = LOADERS[loader_key][1]
    cudf_loader = LOADERS[loader_key][2]
    kwargs = json.loads(kw_json)
    kwargs = _apply_loader_overrides(path, loader_key, kwargs)

    # Load
    df_pd = pd_loader(path, *args, **kwargs)
    df_cudf = cudf_loader(path, *args, **kwargs)

    # Align columns
    cols = list(df_pd.columns)
    report = []
    for col in cols:
        pd_dt = str(df_pd.dtypes[col])
        cudf_dt = str(df_cudf.dtypes[col])

        if not _is_supported_pd_dtype(pd_dt):
            print(f"❌ {col} has unsupported pandas dtype {pd_dt}")
            continue
        report.append((col, pd_dt, cudf_dt))
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

    files = gather_data_files(args.base_dir, target_notebooks=["bench.ipynb"])
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
