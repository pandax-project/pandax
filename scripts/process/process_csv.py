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
import ast
import json
import os

import pandas as pd

TARGET_NOTEBOOKS = {"bench.ipynb", "small_bench.ipynb"}
LOADERS = {"read_csv": "csv", "read_parquet": "parquet", "read_table": "table"}


def find_data_calls_in_notebook(nb_path):
    """
    Parses each code cell with ast, returns a list of tuples:
      (loader, rel_path, args_list, kwargs_dict)
    where loader is 'csv'|'parquet'|'table'.
    """
    calls = []
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "pd":
                    fn = node.func.attr
                    if fn in LOADERS:
                        # must have at least one positional arg for filepath
                        if not node.args or not isinstance(node.args[0], ast.Str):
                            continue
                        rel_path = node.args[0].s
                        # capture extra positional args (after path)
                        extra_args = []
                        for arg in node.args[1:]:
                            extra_args.append(ast.literal_eval(arg))
                        # capture all keywords
                        kw = {
                            k.arg: ast.literal_eval(k.value)
                            for k in node.keywords
                            if k.arg
                        }
                        kw_json = json.dumps(kw, sort_keys=True)
                        calls.append((LOADERS[fn], rel_path, extra_args, kw_json))
    return calls


def gather_data_files(base_dir):
    """
    Walk base_dir looking only for bench.ipynb / small_bench.ipynb,
    extract all read_* calls (with their args), return a set of
    (abs_path, loader, args, kwargs).
    """
    files = set()
    for root, _, fnames in os.walk(base_dir):
        for fname in fnames:
            if fname not in TARGET_NOTEBOOKS:
                continue
            nb_path = os.path.join(root, fname)
            for loader, rel_path, args, kw_json in find_data_calls_in_notebook(nb_path):
                abs_path = (
                    rel_path
                    if os.path.isabs(rel_path)
                    else os.path.normpath(os.path.join(root, rel_path))
                )
                files.add((abs_path, loader, tuple(args), kw_json))

    return files


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
    files = gather_data_files(args.base_dir)
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
    import pandas as pd

    path = "/home/dias-benchmarks/notebooks/mpwolke/just-you-wait-rishi-sunak/input/latest-elected-uk-prime-minister-rishi-sunak/uk_pm.parquet"

    # 1) load normally
    df = pd.read_parquet(path)

    # 2) cast all string columns from pandas' new StringDtype to plain object
    string_cols = [
        "text",
        "username",
        "hashtags",
        "language",
        "quotedtweet",
        "inReplyToUser",
        "mentionedUsers",
    ]
    for c in string_cols:
        df[c] = df[c].astype("object")

    # 3) strip off the timezone so dtype becomes naive datetime64[ns]
    if df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)

    # 4) now df.dtypes should match what cudf.read_parquet produces:
    print(df.dtypes)

    # 5) (optional) overwrite the original parquet if you like
    df.to_parquet(path, index=False)

    import pandas as pd

    path = "/home/dias-benchmarks/notebooks/josecode1/billionaires-statistics-2023/input/Billionaires Statistics Dataset.csv"

    # load with 'rank' as the DataFrame index
    data = pd.read_csv(path, index_col="rank")

    # overwrite (or write to a new file) and preserve the index column
    data.to_csv(path, index_label="rank")
