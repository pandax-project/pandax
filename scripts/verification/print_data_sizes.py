#!/usr/bin/env python3
"""
print_data_sizes.py

For each benchmark:
1) Parse data-loading calls from `bench.ipynb`.
2) Load each referenced dataset with pandas.
3) Concatenate loaded datasets with `pd.concat`.
4) Scale by benchmark factor:
   - integer factor -> repeat with `pd.concat`
   - floating-point factor -> subsample with `DataFrame.sample(frac=...)`
5) Print base and scaled data sizes.

Usage:
    python scripts/verification/print_data_sizes.py
    python scripts/verification/print_data_sizes.py --benchmarks imdb nyc-taxi
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.utils.notebook_data_calls import find_data_calls_in_notebook
from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS, FACTOR_MAP
from utils.verification import extract_factors, load_code_lines

LOADERS = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
    "table": pd.read_table,
}


def _apply_loader_overrides(path: str, loader_key: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply file-specific read kwargs to align inferred dtypes."""
    basename = os.path.basename(path)
    if loader_key == "csv" and basename == "title-metadata.csv":
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        dtype["genres"] = "str"
        kwargs["dtype"] = dtype
    elif loader_key == "csv" and basename == "Billionaires Statistics Dataset.csv":
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, dict):
            dtype = {}
        dtype["selfMade"] = "object"
        for col in ["age", "birthYear", "birthMonth", "birthDay", "population_country"]:
            dtype[col] = "float64"
        kwargs["dtype"] = dtype
    return kwargs


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _parse_factor(benchmark_name: str, bench_nb_path: Path) -> float:
    lines = load_code_lines(str(bench_nb_path))
    factors = extract_factors(lines)
    if factors:
        first_factor = factors[0]
        try:
            parsed = ast.literal_eval(first_factor)
        except Exception:
            parsed = None
        if isinstance(parsed, (int, float)):
            return float(parsed)
    return float(FACTOR_MAP[benchmark_name])


def _load_concat_df(bench_nb_path: Path) -> pd.DataFrame:
    data_calls = find_data_calls_in_notebook(str(bench_nb_path))
    if not data_calls:
        raise ValueError("No pandas data-loading calls found in bench.ipynb")

    root = bench_nb_path.parent
    loaded_dfs: list[pd.DataFrame] = []
    for loader_key, rel_path, extra_args, kw_json in data_calls:
        loader = LOADERS.get(loader_key)
        if loader is None:
            continue

        abs_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(root, rel_path))
        kwargs = json.loads(kw_json)
        kwargs = _apply_loader_overrides(abs_path, loader_key, kwargs)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing data file: {abs_path}")
        df = loader(abs_path, *extra_args, **kwargs)
        loaded_dfs.append(df)

    if not loaded_dfs:
        raise ValueError("No loadable pandas dataframes extracted from bench.ipynb")

    return pd.concat(loaded_dfs, ignore_index=True, sort=False)


def _apply_factor(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    if float(factor).is_integer():
        repeats = int(factor)
        if repeats == 1:
            return df
        return pd.concat([df] * repeats, ignore_index=True, sort=False)

    frac = float(factor)
    return df.sample(frac=frac, random_state=0)


def _print_sizes(benchmark_name: str, factor: float, base_df: pd.DataFrame, scaled_df: pd.DataFrame) -> None:
    base_bytes = int(base_df.memory_usage(index=True, deep=True).sum())
    scaled_bytes = int(scaled_df.memory_usage(index=True, deep=True).sum())
    print(f"{benchmark_name}:")
    print(f"  factor: {factor}")
    print(f"  base_shape: {base_df.shape}")
    print(f"  base_size:  {base_bytes} bytes ({_format_bytes(base_bytes)})")
    print(f"  scaled_shape: {scaled_df.shape}")
    print(f"  scaled_size:  {scaled_bytes} bytes ({_format_bytes(scaled_bytes)})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-benchmark base/scaled dataset sizes.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=BENCHMARK_NAMES,
        help="Benchmark names to process (default: all benchmarks in utils.benchmarks).",
    )
    args = parser.parse_args()

    for benchmark_name in args.benchmarks:
        if benchmark_name in {
            "nlp-on-student-writing-eda",
            "getting-started-with-a-movie-recommendation-system",
            "kaggle-survey-2022-all-results",
        }:
            print(f"Skipping {benchmark_name} because it's not a normal load.")
            continue
        
        if benchmark_name not in BENCHMARKS_TO_PATHS:
            print(f"{benchmark_name}: ✖ unknown benchmark name")
            continue

        benchmark_src_dir = Path(BENCHMARKS_TO_PATHS[benchmark_name])
        bench_nb_path = benchmark_src_dir / "bench.ipynb"
        if not bench_nb_path.exists():
            print(f"{benchmark_name}: ✖ missing {bench_nb_path}")
            continue

        try:
            factor = _parse_factor(benchmark_name, bench_nb_path)
            base_df = _load_concat_df(bench_nb_path)
            scaled_df = _apply_factor(base_df, factor)
            _print_sizes(benchmark_name, factor, base_df, scaled_df)
        except Exception as exc:
            print(f"{benchmark_name}: ✖ {exc}")


if __name__ == "__main__":
    main()
