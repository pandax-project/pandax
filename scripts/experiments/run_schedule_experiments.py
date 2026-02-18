import pickle
from pathlib import Path

import pandas as pd

from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS
from utils.schedule import get_actual_time_for_schedule, get_schedule_and_cost

for name in BENCHMARK_NAMES:
    path = Path(BENCHMARKS_TO_PATHS[name])
    with open(path / "transfer_times.pkl", "rb") as f:
        transfer_times = pickle.load(f)
    with open(path / "input_df_cols.pkl", "rb") as f:
        input_df_cols = pickle.load(f)
    with open(path / "output_df_cols.pkl", "rb") as f:
        output_df_cols = pickle.load(f)

    # Get the cpu and gpu times.
    gpu_path = path / "rewritten"
    orig_gpu_times = pd.read_csv(
        gpu_path / "prediction_times.csv"
    ).original_times.tolist()
    predicted_gpu_times = pd.read_csv(
        gpu_path / "prediction_times.csv"
    ).predicted_times.tolist()

    cpu_path = Path(BENCHMARKS_TO_PATHS[name]) / "rewritten_cpu"
    orig_cpu_times = pd.read_csv(
        cpu_path / "prediction_times.csv"
    ).original_times.tolist()
    predicted_cpu_times = pd.read_csv(
        cpu_path / "prediction_times.csv"
    ).predicted_times.tolist()

    # Ground truth schedule.
    ground_truth_schedule, ground_truth_total_cost = get_schedule_and_cost(
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
        use_cost_model=False,
    )
    actual_time_for_ground_truth_schedule = get_actual_time_for_schedule(
        ground_truth_schedule,
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
    )
    print(f"Ground truth schedule: {ground_truth_schedule}")
    print(f"Ground truth total cost: {ground_truth_total_cost}")
    assert round(ground_truth_total_cost, 2) == round(
        actual_time_for_ground_truth_schedule, 2
    )
    print("Ground truth schedule is correct")
    print()

    # Just use the cost model for transfer costs.
    cost_model_schedule, cost_model_total_cost = get_schedule_and_cost(
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
        use_cost_model=True,
    )
    actual_time_for_cost_model_schedule = get_actual_time_for_schedule(
        cost_model_schedule,
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
    )
    print(f"Cost model schedule: {cost_model_schedule}")
    print(f"Cost model total cost: {cost_model_total_cost}")
    print(f"Actual time for cost model schedule: {actual_time_for_cost_model_schedule}")
    print()

    # just use the predicted times.
    predicted_schedule, predicted_total_cost = get_schedule_and_cost(
        predicted_cpu_times,
        predicted_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
        use_cost_model=False,
    )
    actual_time_for_predicted_schedule = get_actual_time_for_schedule(
        predicted_schedule,
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
    )
    print(f"Predicted schedule: {predicted_schedule}")
    print(f"Predicted total cost: {predicted_total_cost}")
    print(f"Actual time for predicted schedule: {actual_time_for_predicted_schedule}")
    print()

    # use both the predicted times and the cost model for transfer costs.
    predicted_cost_model_schedule, predicted_cost_model_total_cost = (
        get_schedule_and_cost(
            predicted_cpu_times,
            predicted_gpu_times,
            input_df_cols,
            output_df_cols,
            transfer_times,
            use_cost_model=True,
        )
    )
    actual_time_for_predicted_cost_model_schedule = get_actual_time_for_schedule(
        predicted_cost_model_schedule,
        orig_cpu_times,
        orig_gpu_times,
        input_df_cols,
        output_df_cols,
        transfer_times,
    )
    print(f"Predicted + cost model schedule: {predicted_cost_model_schedule}")
    print(f"Predicted + cost model total cost: {predicted_cost_model_total_cost}")
    print(
        f"Actual time for predicted + cost model schedule: {actual_time_for_predicted_cost_model_schedule}"
    )
    print()

    # Dump these results to a csv file.
    results = pd.DataFrame(
        {
            "ground_truth_schedule": ground_truth_schedule,
            "ground_truth_total_cost": ground_truth_total_cost,
            "cost_model_schedule": cost_model_schedule,
            "cost_model_total_cost": cost_model_total_cost,
            "actual_time_for_cost_model_schedule": actual_time_for_cost_model_schedule,
            "predicted_schedule": predicted_schedule,
            "predicted_total_cost": predicted_total_cost,
            "actual_time_for_predicted_schedule": actual_time_for_predicted_schedule,
            "predicted_cost_model_schedule": predicted_cost_model_schedule,
            "predicted_cost_model_total_cost": predicted_cost_model_total_cost,
            "actual_time_for_predicted_cost_model_schedule": actual_time_for_predicted_cost_model_schedule,
        }
    )

    print("Schedule errors:")
    print(
        "Predicted error percentage",
        sum(results.ground_truth_schedule != results.predicted_schedule)
        / len(results.ground_truth_schedule),
    )
    print(
        "Cost model error percentage",
        sum(results.ground_truth_schedule != results.cost_model_schedule)
        / len(results.ground_truth_schedule),
    )
    print(
        "Predicted + cost model error percentage",
        sum(results.ground_truth_schedule != results.predicted_cost_model_schedule)
        / len(results.ground_truth_schedule),
    )
    print()

    print("Time errors:")
    print(
        "Cost model time error percentage",
        abs(actual_time_for_cost_model_schedule - ground_truth_total_cost)
        / ground_truth_total_cost,
    )
    print(
        "Predicted time error percentage",
        abs(actual_time_for_predicted_schedule - ground_truth_total_cost)
        / ground_truth_total_cost,
    )
    print(
        "Predicted + cost model time error percentage",
        abs(actual_time_for_predicted_cost_model_schedule - ground_truth_total_cost)
        / ground_truth_total_cost,
    )
    print()

    print(
        ground_truth_total_cost,
        actual_time_for_predicted_schedule,
        actual_time_for_cost_model_schedule,
        actual_time_for_predicted_cost_model_schedule,
    )
    # results.ground_truth_schedule != results.cost_model_schedule
    print(f"Dumped results to {path / 'schedule_results.csv'}")

    results.to_csv(path / "schedule_results.csv", index=False)
