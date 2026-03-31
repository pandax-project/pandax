"""Predict per-cell timings for all benchmarks.

This script writes `prediction_times.csv` next to each selected rewritten notebook:
- GPU mode: `<benchmark>/rewritten/o4_mini_high.ipynb`
- CPU mode: `<benchmark>/rewritten_cpu/o4_mini_high.ipynb`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS
from utils.prediction import modify_factor_and_run
from utils.verification import extract_factors, load_code_lines

BENCHMARK_NAMES = [
    "feedback3-eda-hf-custom-trainer-sift",
    "beautiful-kaggle-2022-analysis",
    "creating-player-stats-using-tracking-data",
    "comprehensive-data-exploration-with-python",
    "retail-supermarket-store-analysis",
    "adidas-retail-eda-data-visualization",
    "indian-startup-growth-analysis",
]

def main(use_gpu: bool) -> None:
    failed_paths: list[str] = []
    rewritten_dir = "rewritten" if use_gpu else "rewritten_cpu"

    for name in BENCHMARK_NAMES:
        path = BENCHMARKS_TO_PATHS[name]
        nb_path = Path(path) / rewritten_dir / "o4_mini_high.ipynb"
        print(f"\n{name} ({'GPU' if use_gpu else 'CPU'})")

        bench_lines = load_code_lines(nb_path)
        factors = extract_factors(bench_lines)
        assert len(factors) == 1, f"Expected 1 factor, got {factors}"
        factor = int(factors[0])
        use_float = factor == 1

        (
            error_percentages,
            original_cell_times_lst,
            predicted_times_lst,
            cell_times_across_factors,
        ) = modify_factor_and_run(
            nb_path,
            multipliers=[0.3, 0.4, 0.5, 0.6],
            use_gpu=use_gpu,
            use_float=use_float,
        )

        absolute_error_lst = [
            abs(predicted_times_lst[i] - original_cell_times_lst[i])
            for i in range(len(predicted_times_lst))
        ]

        for factor_key in cell_times_across_factors.keys():
            cell_times_across_factors[factor_key] = [
                value
                for _, value in sorted(
                    cell_times_across_factors[factor_key].items(),
                    key=lambda x: x[0],
                )
            ]

        factor_timings = {
            f"factor_{factor_key}": cell_times_across_factors[factor_key]
            for factor_key in sorted(cell_times_across_factors.keys())
        }

        data = pd.DataFrame(
            {
                **factor_timings,
                "original_times": original_cell_times_lst,
                "error_percentages": error_percentages,
                "absolute_error": absolute_error_lst,
                "predicted_times": predicted_times_lst,
            }
        )
        output_path = nb_path.parent / "prediction_times.csv"
        print("Writing data to CSV file...", output_path)
        data.to_csv(output_path, index=False)

        print("Average error percentage:", np.mean(error_percentages))
        print("Average absolute error:", np.mean(absolute_error_lst))
        print("total original time:", sum(original_cell_times_lst))
        print("total predicted time:", sum(predicted_times_lst))
        print("total absolute error:", sum(absolute_error_lst))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use GPU rewritten notebooks (`rewritten/`) when true, "
            "CPU rewritten notebooks (`rewritten_cpu/`) when false."
        ),
    )
    args = parser.parse_args()
    main(use_gpu=args.use_gpu)
