#!/usr/bin/env python3
"""Analyze rewrite wall timings grouped by cell index.

Usage:
    python scripts/analysis/analyze_rewrite_wall_timings.py <benchmark_name> [<benchmark_name> ...] [--cell-window-size N]
"""

import argparse
import pandas as pd

from utils.benchmarks import get_stats_dir


def get_wall_time_timings(benchmark_name: str) -> pd.DataFrame:
    csv_path = get_stats_dir(benchmark_name) / "rewrite_wall_time_timings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Timing CSV not found: {csv_path}")
    return pd.read_csv(csv_path)

def analyze_rewrite_wall_timings(df: pd.DataFrame, cell_window_size: int) -> tuple[float, float]:
    if cell_window_size <= 0:
        raise ValueError("--cell-window-size must be a positive integer")

    total_rows = df[df["category"] == "total"].copy()

    # If run_id exists, only keep rows from the latest run_id per cell_index.
    # "Latest" is defined by row order in the CSV (last occurrence wins).
    if "run_id" in total_rows.columns:
        latest_run_id = total_rows.groupby("cell_index", sort=False)["run_id"].transform("last")
        total_rows = total_rows[
            total_rows["run_id"].astype(str) == latest_run_id.astype(str)
        ]

    grouped = (
        total_rows
        .groupby("cell_index", as_index=False)
        .agg(total_elapsed_seconds=("elapsed_seconds", "sum"))
    )

    if grouped.empty:
        return 0.0, 0.0

    grouped["cell_index"] = pd.to_numeric(grouped["cell_index"])
    grouped = grouped.sort_values("cell_index")

    global_max = float(grouped["total_elapsed_seconds"].max())

    grouped["window_id"] = (grouped["cell_index"] // cell_window_size).astype(int)
    window_max_sum = float(
        grouped.groupby("window_id", as_index=False)["total_elapsed_seconds"].max()[
            "total_elapsed_seconds"
        ].sum()
    )

    return global_max, window_max_sum

def analyze_category_totals(df: pd.DataFrame) -> pd.DataFrame:
    category_rows = df[df["category"] != "total"].copy()

    if category_rows.empty:
        return pd.DataFrame(columns=["category", "total_elapsed_seconds"])

    # If run_id exists, keep only rows from the latest run_id for each
    # (cell_index, category) pair (last occurrence wins by CSV row order).
    if "run_id" in category_rows.columns:
        latest_run_id = category_rows.groupby(
            ["cell_index", "category"], sort=False
        )["run_id"].transform("last")
        category_rows = category_rows[
            category_rows["run_id"].astype(str) == latest_run_id.astype(str)
        ]

    grouped = (
        category_rows
        .groupby("category", as_index=False)
        .agg(total_elapsed_seconds=("elapsed_seconds", "sum"))
        .sort_values("total_elapsed_seconds", ascending=False)
    )

    return grouped

def analyze_agent_call_ratio(df: pd.DataFrame) -> tuple[float, float, float]:
    category_totals = analyze_category_totals(df)
    if category_totals.empty:
        return 0.0, 0.0, 0.0

    non_total_categories_sum = float(category_totals["total_elapsed_seconds"].sum())
    agent_call_elapsed = float(
        category_totals.loc[
            category_totals["category"] == "agent_call", "total_elapsed_seconds"
        ].sum()
    )
    other_elapsed = non_total_categories_sum - agent_call_elapsed

    if other_elapsed <= 0:
        ratio = float("inf") if agent_call_elapsed > 0 else 0.0
    else:
        ratio = agent_call_elapsed / other_elapsed

    return agent_call_elapsed, other_elapsed, ratio

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze rewrite wall timings, including grouped total timing "
            "and optional per-category totals."
        )
    )
    parser.add_argument(
        "benchmark_names",
        nargs="+",
        type=str,
        help="One or more benchmark keys from utils.benchmarks.BENCHMARKS_TO_PATHS",
    )
    parser.add_argument(
        "--cell-window-size",
        type=int,
        default=5,
        help="Cell window size for grouped max-sum (default: 5).",
    )
    parser.add_argument(
        "--category-totals",
        action="store_true",
        help=(
            "Also print summed elapsed_seconds per category, excluding "
            "category 'total'."
        ),
    )
    parser.add_argument(
        "--agent-call-ratio",
        action="store_true",
        help=(
            "Also print ratio of category 'agent_call' elapsed_seconds "
            "to all other non-total category elapsed_seconds."
        ),
    )
    args = parser.parse_args()
    
    for benchmark_name in args.benchmark_names:
        df = get_wall_time_timings(benchmark_name)
        global_max, window_max_sum = analyze_rewrite_wall_timings(
            df, args.cell_window_size
        )
        # print(f"{benchmark_name}: {window_max_sum}")
        if args.category_totals:
            category_totals = analyze_category_totals(df)
            non_total_categories_sum = float(
                category_totals["total_elapsed_seconds"].sum()
            ) if not category_totals.empty else 0.0

            if category_totals.empty:
                print("  category_totals: (no non-total categories found)")
            else:
                print("  category_totals:")
                for row in category_totals.itertuples(index=False):
                    print(
                        f"    {row.category}: "
                        f"{float(row.total_elapsed_seconds):.6f}"
                    )
            print(f"  non_total_categories_sum: {non_total_categories_sum:.6f}")

        if args.agent_call_ratio:
            agent_call_elapsed, other_elapsed, ratio = analyze_agent_call_ratio(df)
            ratio_display = "inf" if ratio == float("inf") else f"{ratio:.6f}"
            print(f"  agent_call_elapsed_seconds: {agent_call_elapsed:.6f}")
            print(f"  other_elapsed_seconds: {other_elapsed:.6f}")
            print(f"  agent_call_to_other_ratio: {ratio_display}")


if __name__ == "__main__":
    main()
