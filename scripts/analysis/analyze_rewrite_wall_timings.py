#!/usr/bin/env python3
"""Analyze rewrite wall timings grouped by cell index.

Usage:
    python scripts/analysis/analyze_rewrite_wall_timings.py <benchmark_name>
"""

import argparse
import pandas as pd

from utils.benchmarks import get_stats_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute min/max/avg elapsed_seconds from rewrite wall timings, "
            "aggregated by cell_index."
        )
    )
    parser.add_argument(
        "benchmark_name",
        type=str,
        help="Benchmark key from utils.benchmarks.BENCHMARKS_TO_PATHS",
    )
    return parser.parse_args()


def get_wall_time_timings(benchmark_name: str) -> pd.DataFrame:
    csv_path = get_stats_dir(benchmark_name) / "rewrite_wall_time_timings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Timing CSV not found: {csv_path}")
    return pd.read_csv(csv_path)

def analyze_rewrite_wall_timings(df: pd.DataFrame) -> None:
    grouped = (
        df[df["category"] == "total"]
        .groupby("cell_index", as_index=False)
        .agg(total_elapsed_seconds=("elapsed_seconds", "sum"))
        .sort_values("cell_index")
    )
    print(grouped["total_elapsed_seconds"].max())

def main() -> None:
    args = parse_args()
    analyze_rewrite_wall_timings(get_wall_time_timings(args.benchmark_name))


if __name__ == "__main__":
    main()
