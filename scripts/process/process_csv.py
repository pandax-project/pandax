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
import csv
import json
import os
import pandas as pd

from scripts.utils.notebook_data_calls import gather_data_files

def apply_loader_overrides(path, loader, kwargs):
    """Apply file-specific read kwargs to align downstream dtype behavior."""
    if loader == "csv" and os.path.basename(path) == "title-metadata.csv":
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        dtype["genres"] = "str"
        kwargs["dtype"] = dtype
    elif loader == "csv" and os.path.basename(path) == "Billionaires Statistics Dataset.csv":
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        dtype["selfMade"] = "object"
        for col in ["age", "birthYear", "birthMonth", "birthDay", "population_country"]:
            dtype[col] = "float64"
        kwargs["dtype"] = dtype
    return kwargs


def normalize_uk_pm_parquet(df):
    """Normalize known parquet columns to align pandas/cuDF dtype behavior."""
    text_cols = [
        "text",
        "username",
        "hashtags",
        "language",
        "quotedtweet",
        "inReplyToUser",
        "mentionedUsers",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if "created_at" in df.columns:
        dt = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["created_at"] = dt.dt.tz_convert(None).astype("datetime64[ns]")

    return df


def normalize_to_pandas_ground_truth(df):
    """Normalize columns using pandas-inferred dtypes as ground truth.

    If pandas reads a column as numeric, keep it explicitly numeric so dumping
    emits canonical numeric tokens that cuDF can infer consistently.
    """
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(series, errors="coerce").astype("int64")
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif pd.api.types.is_bool_dtype(dtype):
            df[col] = series.astype("boolean")
    return df


def reload_and_dump(path, loader, args, kw_json):
    """Re-run pd.read_* with the same args/kwargs, then overwrite file."""
    kwargs = json.loads(kw_json)
    kwargs = apply_loader_overrides(path, loader, kwargs)
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return
    if loader == "csv":
        df = pd.read_csv(path, *args, **kwargs)
        df = normalize_to_pandas_ground_truth(df)
        # Keep text field escaping stable while preserving numeric tokens.
        df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    elif loader == "parquet":
        df = pd.read_parquet(path, *args, **kwargs)
        if os.path.basename(path) == "uk_pm.parquet":
            df = normalize_uk_pm_parquet(df)
        df.to_parquet(path, index=False)
    elif loader == "table":
        df = pd.read_table(path, *args, **kwargs)
        df = normalize_to_pandas_ground_truth(df)
        df.to_csv(
            path,
            sep=kwargs.get("sep", "\t"),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
    else:
        print(f"  ⚠️  Unsupported loader {loader!r}, skipping: {path}")
        return
    print(f"  ✔ Reloaded & dumped ({loader}) → {path}")

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
        reload_and_dump(path, loader, args, kw_json)


if __name__ == "__main__":
    main()