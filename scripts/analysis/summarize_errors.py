#!/usr/bin/env python3
# Summarize the errors in all the transfer_costs.csv.
# To use, run:
# python summarize_errors.py <root_dir>

import os
import sys

import pandas as pd


def find_transfer_csvs(root):
    """
    Yield paths to transfer_costs.csv only in dirs that also have bench.ipynb.
    """
    for dirpath, _, files in os.walk(root):
        if "bench.ipynb" in files and "transfer_costs.csv" in files:
            yield os.path.join(dirpath, "transfer_costs.csv")


def analyze(root):
    per_file_stats = []

    for csv_path in find_transfer_csvs(root):
        df = pd.read_csv(csv_path)
        df = df[df.df_col.notna()]
        # compute per-file means
        cpu_to_gpu_error_percentages = (
            abs(df["cpu->gpu"] - df["cost-cpu->gpu"]) / df["cpu->gpu"]
        )
        gpu_to_cpu_error_percentages = (
            abs(df["gpu->cpu"] - df["cost-gpu->cpu"]) / df["gpu->cpu"]
        )
        cpu_mean = cpu_to_gpu_error_percentages.mean()
        gpu_mean = gpu_to_cpu_error_percentages.mean()
        abs_cpu_error = abs(df["cpu->gpu"] - df["cost-cpu->gpu"]).mean()
        abs_gpu_error = abs(df["gpu->cpu"] - df["cost-gpu->cpu"]).mean()
        per_file_stats.append(
            {
                "directory": os.path.dirname(csv_path),
                "cpu_mean": cpu_mean,
                "gpu_mean": gpu_mean,
                "abs_cpu_error": abs_cpu_error,
                "abs_gpu_error": abs_gpu_error,
            }
        )
        print(csv_path)
        print("cpu mean error percentage", cpu_mean)
        print("gpu mean error percentage", gpu_mean)
        print("cpu abs error", abs_cpu_error)
        print("gpu abs error", abs_gpu_error)

        print()

    # report
    stats_df = pd.DataFrame(per_file_stats)
    print("\nPer-directory means:")
    print(stats_df.to_string(index=False))

    # Mean of the per-file cpu means
    mean_of_cpu_means = stats_df["cpu_mean"].mean()
    # Mean of the per-file gpu means
    mean_of_gpu_means = stats_df["gpu_mean"].mean()

    print(f"Mean of per-file cpu_error_percentages means: {mean_of_cpu_means:.3f}")
    print(f"Mean of per-file gpu_error_percentages means: {mean_of_gpu_means:.3f}")

    median_cpu = stats_df["cpu_mean"].median()
    median_gpu = stats_df["gpu_mean"].median()
    print(f"Median of per-file cpu_error_percentages means: {median_cpu:.3f}")
    print(f"Median of per-file gpu_error_percentages means: {median_gpu:.3f}")


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze(root_dir)
