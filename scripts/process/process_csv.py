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


def _path_file_parent_levels(node):
    """
    Return number of trailing `.parent` accesses for expressions like:
      Path(__file__).parent
      Path(__file__).parent.parent
    Returns None if node is not that shape.
    """
    levels = 0
    cur = node
    while isinstance(cur, ast.Attribute) and cur.attr == "parent":
        levels += 1
        cur = cur.value

    if (
        levels > 0
        and isinstance(cur, ast.Call)
        and isinstance(cur.func, ast.Name)
        and cur.func.id == "Path"
        and len(cur.args) == 1
        and isinstance(cur.args[0], ast.Name)
        and cur.args[0].id == "__file__"
    ):
        return levels
    return None


def _resolve_path_arg(arg_node, nb_path):
    """
    Resolve file path arg from ast node.
    Supports:
      - string literals: "data.csv"
      - f-strings like: f"{Path(__file__).parent}/data.csv"
    Returns path string or None if unsupported dynamic expression.
    """
    if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
        return arg_node.value

    if isinstance(arg_node, ast.JoinedStr):
        parts = []
        for v in arg_node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
                continue

            if isinstance(v, ast.FormattedValue):
                parent_levels = _path_file_parent_levels(v.value)
                if parent_levels is None:
                    return None

                base = os.path.dirname(nb_path)
                for _ in range(parent_levels - 1):
                    base = os.path.dirname(base)
                parts.append(base)
                continue

            return None
        return "".join(parts)

    return None


def find_data_calls_in_notebook(nb_path):
    """
    Parses each code cell with ast, returns a list of tuples:
      (loader, rel_path, args_list, kwargs_dict)
    where loader is 'csv'|'parquet'|'table'.
    """
    calls = []
    print(f"Processing notebook: {nb_path}")
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
                        if not node.args:
                            continue

                        rel_path = _resolve_path_arg(node.args[0], nb_path)
                        if not rel_path:
                            continue
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
            nb_path = os.path.abspath(os.path.join(root, fname))
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