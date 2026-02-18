# Get the cpu timings for all benchmarks. These includes the original timing, as well as the timings for the modified factors and the predicted timings.
# Note this is timings cell by cell, not end to end.
from pathlib import Path

from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS
from utils.notebook import load_notebook
from utils.prediction import modify_factor_and_run
from utils.verification import extract_factors, load_code_lines

failed_paths = []
for name in BENCHMARK_NAMES:
    path = BENCHMARKS_TO_PATHS[name]
    nb_path = Path(path) / "rewritten" / "o4_mini_high.ipynb"
    cpu_nb = load_notebook(nb_path)
    bench_lines = load_code_lines(nb_path)
    factors = extract_factors(bench_lines)
    assert len(factors) == 1, f"Expected 1 factor, got {factors}"
    factor = int(factors[0])

    if factor == 1:
        use_float = True
    else:
        use_float = False

    (
        error_percentages,
        original_cell_times_lst,
        predicted_times_lst,
        cell_times_across_factors,
    ) = modify_factor_and_run(
        nb_path, multipliers=[0.3, 0.4, 0.5, 0.6], use_gpu=True, use_float=use_float
    )

    absolute_error_lst = [
        abs(predicted_times_lst[i] - original_cell_times_lst[i])
        for i in range(len(predicted_times_lst))
    ]

    import pandas as pd

    for factor in cell_times_across_factors.keys():
        cell_times_across_factors[factor] = [
            v
            for k, v in sorted(
                cell_times_across_factors[factor].items(), key=lambda x: x[0]
            )
        ]

    factor_timings = {
        f"factor_{factor}": cell_times_across_factors[factor]
        for factor in sorted(cell_times_across_factors.keys())
    }

    # Dump the data to one single CSV file.
    data = pd.DataFrame(
        {
            **factor_timings,
            "original_times": original_cell_times_lst,
            "error_percentages": error_percentages,
            "absolute_error": absolute_error_lst,
            "predicted_times": predicted_times_lst,
        }
    )
    print("Writing data to CSV file...", nb_path.parent / "prediction_times.csv")
    data.to_csv(nb_path.parent / "prediction_times.csv", index=False)
    import numpy as np

    print("Average error percentage:", np.mean(error_percentages))
    print("Average absolute error:", np.mean(absolute_error_lst))

    print("total original cpu time:", sum(original_cell_times_lst))
    print("total predicted cpu time:", sum(predicted_times_lst))
    print("total absolute error:", sum(absolute_error_lst))
print("Failed paths:", failed_paths)
