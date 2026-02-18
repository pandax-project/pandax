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
import ast
import json
import os

import cudf
import pandas as pd

TARGET_NOTEBOOKS = {"bench.ipynb", "small_bench.ipynb"}
LOADERS = {
    "read_csv": ("csv", pd.read_csv, cudf.read_csv),
    "read_parquet": ("parquet", pd.read_parquet, cudf.read_parquet),
    "read_table": ("table", pd.read_table, pd.read_table),
}


def find_data_calls_in_notebook(nb_path):
    """
    Parse code cells with ast and return a list of:
      (loader_key, rel_path, extra_args, kwargs)
    """
    calls = []
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == "pd":
                    fn = node.func.attr
                    if (
                        fn in LOADERS
                        and node.args
                        and isinstance(node.args[0], ast.Str)
                    ):
                        rel_path = node.args[0].s
                        # collect extra positional args
                        extra_args = [ast.literal_eval(arg) for arg in node.args[1:]]
                        # collect keyword args
                        kw = {
                            kw.arg: ast.literal_eval(kw.value)
                            for kw in node.keywords
                            if kw.arg
                        }
                        kw_json = json.dumps(kw, sort_keys=True)
                        calls.append((fn, rel_path, extra_args, kw_json))
    return calls


def gather_files(base_dir):
    """
    Walk base_dir, inspect only bench notebooks, and return a set of:
      (absolute_path, loader_key, args, frozenset(kwargs.items()))
    """
    files = set()
    for root, _, fnames in os.walk(base_dir):
        for fname in fnames:
            if fname in TARGET_NOTEBOOKS:
                nb_path = os.path.join(root, fname)
                for fn, rel_path, args, kw_json in find_data_calls_in_notebook(nb_path):
                    abs_path = (
                        rel_path
                        if os.path.isabs(rel_path)
                        else os.path.normpath(os.path.join(root, rel_path))
                    )
                    files.add((abs_path, fn, tuple(args), kw_json))
    return files


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

    files = gather_files(args.base_dir)
    if not files:
        print("No data-loading calls found in bench.ipynb or small_bench.ipynb.")
        return

    print(f"Found {len(files)} files. Comparing dtypes:\n")
    for path, fn_key, args, kw_json in sorted(files):
        print(f"File: {path}")
        print(f"  Loader: pd.{fn_key}( ..., args={args}, kwargs={json.loads(kw_json)})")
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
