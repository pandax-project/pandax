import copy
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.execution import execute_notebook, find_cell_times
from utils.notebook import (
    annotate_notebook,
    load_notebook,
    make_notebook,
    save_notebook,
    update_factor_in_cell,
)


def modify_factor_and_run(
    nb_path: Path,
    multipliers: list[float] = [0.3, 0.4, 0.5, 0.6],
    factor_var: str = "factor",
    use_gpu: bool = False,
    use_float: bool = False,
) -> tuple[list[float], list[float], list[float], dict[int | float, dict[int, float]]]:
    # Load notebook
    nb = load_notebook(nb_path)

    # Find the cell with the `factor = x` line
    factor_cell_index = None
    original_value = None
    factor_pattern = re.compile(rf"{factor_var}\s*=\s*([\d.]+)")

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        match = factor_pattern.search(cell.source)
        if match:
            try:
                if use_float:
                    original_value = float(match.group(1))
                else:
                    original_value = int(match.group(1))
                factor_cell_index = idx
                break
            except ValueError:
                continue

    if original_value is None:
        raise ValueError(f"Could not find `{factor_var} = x` assignment in notebook.")

    print(f"Found `{factor_var} = {original_value}` in cell {factor_cell_index}")

    # Directory to store outputs
    out_dir = Path("notebook_outputs")
    out_dir.mkdir(exist_ok=True)
    cell_times_across_factors: dict[int | float, dict[int, float]] = {}

    for multiplier in multipliers:
        if use_float:
            new_value = original_value * multiplier
        else:
            new_value = round(original_value * multiplier)
        modified_nb_cells = copy.deepcopy(nb.cells)
        modified_nb = make_notebook(modified_nb_cells)

        orig_source = modified_nb.cells[factor_cell_index].source
        modified_source = update_factor_in_cell(orig_source, factor_var, new_value)
        modified_nb.cells[factor_cell_index].source = modified_source

        modified_nb_path = out_dir / f"{nb_path.stem}_{factor_var}_{new_value}.ipynb"
        save_notebook(modified_nb, modified_nb_path)

        annotate_notebook(
            original_notebook_path=modified_nb_path,
            annotated_notebook_path=modified_nb_path,
            add_timing_code=True,
            add_record_events=False,
            add_checkpoints=False,
            track_column_info=False,
            add_cudf_profile=False,
            use_gpu=use_gpu,
        )
        modified_nb = load_notebook(modified_nb_path)

        # Execute with papermill
        print(f"Running with `{factor_var} = {new_value}`...")
        start_time = time.time()
        execute_notebook(modified_nb)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        # Now get the timings for each cell.
        cell_times = find_cell_times(modified_nb)
        # Dump the timings to a CSV file.
        cell_times_across_factors[new_value] = cell_times

    # Now plot the data.
    predicted_times_lst = predict_cell_times(cell_times_across_factors, original_value)

    # Now run the original notebook.
    annotated_nb_path = nb_path.parent / "original_with_timing.ipynb"
    annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=annotated_nb_path,
        add_timing_code=True,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        add_cudf_profile=False,
        use_gpu=use_gpu,
    )
    annotated_nb = load_notebook(annotated_nb_path)
    execute_notebook(annotated_nb)
    original_cell_times = find_cell_times(annotated_nb)

    error_percentages: list[float] = []
    original_cell_times_lst: list[float] = []
    for cell_idx, predicted_time in enumerate(predicted_times_lst):
        annotated_time = original_cell_times[cell_idx]
        print(f"Cell {cell_idx}: Predicted {predicted_time}, Actual {annotated_time}")
        print(f"Error: {abs(predicted_time - annotated_time) / annotated_time}")
        error_percentages.append(abs(predicted_time - annotated_time) / annotated_time)
        original_cell_times_lst.append(annotated_time)

    return (
        error_percentages,
        original_cell_times_lst,
        predicted_times_lst,
        cell_times_across_factors,
    )


def smooth(values: np.ndarray, window_size: int = 3) -> np.ndarray:
    if len(values) < window_size:
        return values
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = window_size // 2
    return np.concatenate(
        [
            values[:pad],
            smoothed,
            values[-pad:] if window_size % 2 == 1 else values[-pad + 1 :],
        ]
    )


def predict_cell_times(
    cell_times_across_factors: dict[int | float, dict[int, float]],
    original_factor: int | float,
) -> list[float]:
    # Plot each group and predict for factor = 1000
    print("Predictions at factor = 1000:")
    # All values of cell_times_across_factors are dicts, and they should have the same keys.
    cell_indices = sorted(list(cell_times_across_factors.values())[0].keys())
    predictions: dict[int, float] = {}
    for cell_idx in cell_indices:
        xs = []
        ys = []
        for factor, cell_times in sorted(
            cell_times_across_factors.items(), key=lambda x: x[0]
        ):
            xs.append(factor)
            ys.append(cell_times[cell_idx])
        xs = np.array(xs)
        ys = np.array(ys)
        smoothed_ys = smooth(ys, window_size=3)
        increasing_steps = sum(
            1
            for i in range(len(smoothed_ys) - 1)
            if smoothed_ys[i + 1] > smoothed_ys[i]
        )
        monotonicity = increasing_steps / (len(smoothed_ys) - 1)
        avg = np.mean(ys)

        if monotonicity >= 0.7:
            model = LinearRegression()
            model.fit(xs.reshape(-1, 1), ys)
            y_pred = model.predict(np.array([[original_factor]]))[0]
            predictions[cell_idx] = float(y_pred)
        else:
            predictions[cell_idx] = float(avg)

    predicted_times_lst: list[float] = []
    for cell_idx, prediction in sorted(predictions.items(), key=lambda x: x[0]):
        print(f"Cell {cell_idx}: {prediction}")
        predicted_times_lst.append(prediction)
    return predicted_times_lst


def predict_cell_times_for_nb(nb_path: Path, use_float: bool) -> pd.DataFrame:
    """Run the notebook at `nb_path` at multiple factors to get the predicted cell times.

    use_float indicates whether to use float or int for the factors.
    """
    (
        cpu_error_percentages,
        cpu_original_cell_times_lst,
        cpu_predicted_times_lst,
        cpu_cell_times_across_factors,
    ) = modify_factor_and_run(
        nb_path, multipliers=[0.3, 0.4, 0.5, 0.6], use_gpu=False, use_float=use_float
    )
    (
        gpu_error_percentages,
        gpu_original_cell_times_lst,
        gpu_predicted_times_lst,
        gpu_cell_times_across_factors,
    ) = modify_factor_and_run(
        nb_path, multipliers=[0.3, 0.4, 0.5, 0.6], use_gpu=True, use_float=use_float
    )

    # Dump the data to one single CSV file.
    data = pd.DataFrame(
        {
            "original_cpu_times": cpu_original_cell_times_lst,
            "original_gpu_times": gpu_original_cell_times_lst,
            "cpu_error_percentages": cpu_error_percentages,
            "gpu_error_percentages": gpu_error_percentages,
            "cpu_predicted_times": cpu_predicted_times_lst,
            "gpu_predicted_times": gpu_predicted_times_lst,
        }
    )
    print("Writing data to CSV file...", nb_path.parent / "prediction_times.csv")
    data.to_csv(nb_path.parent / "prediction_times.csv", index=False)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=str, help="Path to the notebook")
    args = parser.parse_args()
    nb_path = Path(args.notebook)
    data = predict_cell_times_for_nb(nb_path, use_float=False)
    print("Writing data to CSV file...", nb_path.parent / "prediction_times.csv")
    data.to_csv(nb_path.parent / "prediction_times.csv", index=False)
