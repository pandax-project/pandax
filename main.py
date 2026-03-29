import asyncio
import logging
import pickle
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

from utils.benchmarks import BENCHMARKS_TO_PATHS, FACTOR_MAP, TRANSFER_COST_FACTOR_MAP
from utils.execution import CostModelInput, merge_cpu_and_gpu_cost_model_inputs
from utils.notebook import (
    load_notebook,
    replace_factor,
    save_notebook,
)
from utils.prediction import predict_cell_times_for_nb
from utils.rewrite import checkpoint_and_get_cudf_profile_info, rewrite_notebook
from utils.logging_utils import log_precompute_timing
from utils.schedule import (
    get_cell_exec_info,
    get_cost_model_inputs,
    get_schedule_and_cost,
    get_transfer_times_and_input_output_df_cols,
)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=Path("codeagent.log"),
        filemode="w",
    )
    shell = InteractiveShell.instance()

    # Take in benchmark name and derive notebook paths from benchmark mapping.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the notebook")
    parser.add_argument(
        "--disable_scheduling",
        action="store_true",
        help="Disable scheduling (default: False)"
    )
    parser.add_argument(
        "--cell_exec_info_path",
        type=str,
        default=None,
        help="Optional path to precomputed cell execution info pickle.",
    )
    parser.add_argument(
        "--cudf_profile_info_path",
        type=str,
        default=None,
        help="Optional path to precomputed cuDF profile info pickle.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run ID for CSV logging. Auto-generated when not provided.",
    )
    parser.add_argument(
        "--start_cell_index",
        type=int,
        default=0,
        help="Annotated cell index to start rewriting from (default: 0).",
    )
    parser.add_argument(
        "--end_cell_index",
        type=int,
        default=None,
        help="Annotated cell index to end rewriting at (default: None).",
    )
    args = parser.parse_args()
    run_id = args.run_id if args.run_id is not None else f"{int(time.time())}-{uuid4().hex[:8]}"
    benchmark_base_path = BENCHMARKS_TO_PATHS.get(args.name)
    if benchmark_base_path is None:
        raise ValueError(f"Unknown benchmark name: {args.name}. ")
    benchmark_base_path = Path(benchmark_base_path)

    original_notebook_path = benchmark_base_path / "bench.ipynb"
    small_notebook_path = benchmark_base_path / "small_bench.ipynb"
    if not original_notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {original_notebook_path}")
    if not small_notebook_path.exists():
        raise FileNotFoundError(f"Small notebook not found: {small_notebook_path}")
    notebook_base_dir = original_notebook_path.parent
    disable_scheduling = args.disable_scheduling
    intermediate_dir = notebook_base_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Mapping from annotated cell to inputs to cost model.
    cost_model_inputs: dict[int, tuple[list[CostModelInput], list[CostModelInput]]] = {}

    # First annotate the notebook to add checkpoints and timing code.
    cell_exec_info_path = (
        Path(args.cell_exec_info_path)
        if args.cell_exec_info_path is not None
        else intermediate_dir / "cell_exec_info.pkl"
    )
    cell_exec_start_time = time.time()
    if cell_exec_info_path.exists():
        with open(cell_exec_info_path, "rb") as f:
            cell_exec_info = pickle.load(f)
        cell_exec_elapsed_seconds = time.time() - cell_exec_start_time
        log_precompute_timing(
            benchmark_name=args.name,
            run_id=run_id,
            stage="cell_exec_info",
            source="cache",
            elapsed_seconds=cell_exec_elapsed_seconds,
        )
        print(f"Loaded cell exec info from {cell_exec_info_path}")
    else:
        cell_exec_info = get_cell_exec_info(small_notebook_path, shell)
        cell_exec_elapsed_seconds = time.time() - cell_exec_start_time
        cell_exec_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cell_exec_info_path, "wb") as f:
            pickle.dump(cell_exec_info, f)
        log_precompute_timing(
            benchmark_name=args.name,
            run_id=run_id,
            stage="cell_exec_info",
            source="compute",
            elapsed_seconds=cell_exec_elapsed_seconds,
        )
        print(
            f"Ran small notebook to get cell exec info in {cell_exec_elapsed_seconds} seconds"
        )

    # Now we can rewrite the notebook.
    # Get cudf profile information.
    # Use the small notebook to get the cudf profile information.
    try:
        cudf_profile_info_path = (
            Path(args.cudf_profile_info_path)
            if args.cudf_profile_info_path is not None
            else intermediate_dir / "cudf_profile_info.pkl"
        )
        cudf_profile_start_time = time.time()
        if cudf_profile_info_path.exists():
            with open(cudf_profile_info_path, "rb") as f:
                cudf_profile_infos = pickle.load(f)
            cudf_profile_elapsed_seconds = time.time() - cudf_profile_start_time
            log_precompute_timing(
                benchmark_name=args.name,
                run_id=run_id,
                stage="cudf_profile_info",
                source="cache",
                elapsed_seconds=cudf_profile_elapsed_seconds,
            )
            print(f"Loaded cuDF profile info from {cudf_profile_info_path}")
        else:
            cudf_profile_infos = checkpoint_and_get_cudf_profile_info(small_notebook_path)
            cudf_profile_elapsed_seconds = time.time() - cudf_profile_start_time
            cudf_profile_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cudf_profile_info_path, "wb") as f:
                pickle.dump(cudf_profile_infos, f)
            log_precompute_timing(
                benchmark_name=args.name,
                run_id=run_id,
                stage="cudf_profile_info",
                source="compute",
                elapsed_seconds=cudf_profile_elapsed_seconds,
            )
            print(
                f"Ran small notebook to get cudf profile info in {cudf_profile_elapsed_seconds} seconds"
            )
    except Exception as e:
        print(f"Error getting cudf profile info: {e}")
        exit(1)

    rewrite_start_time = time.time()
    await rewrite_notebook(
        benchmark_name=args.name,
        run_id=run_id,
        small_nb_path=small_notebook_path,
        cudf_profile_infos=cudf_profile_infos,
        cell_exec_infos=cell_exec_info,
        shell=shell,
        start_cell_index=args.start_cell_index,
        end_cell_index=args.end_cell_index,
    )
    rewrite_end_time = time.time()
    print(f"Rewrote notebook in {rewrite_end_time - rewrite_start_time} seconds")
    if disable_scheduling:
        return

    # Now that we have rewritten the notebook, we can get the transfer times and input/output df cols.
    # We use scaled down factors to get sample df sizes, and then scale the factors to get the "actual" df sizes.
    original_notebook = load_notebook(original_notebook_path)
    new_notebook = replace_factor(
        original_notebook, TRANSFER_COST_FACTOR_MAP[args.name]
    )
    transfer_cost_notebook_path = (
        original_notebook_path.parent / "transfer_cost_bench.ipynb"
    )
    save_notebook(new_notebook, transfer_cost_notebook_path)
    print(f"Transfer cost notebook saved to {transfer_cost_notebook_path}")
    df_size_multiplier = FACTOR_MAP[args.name] // TRANSFER_COST_FACTOR_MAP[args.name]

    cpu_to_gpu_cost_model_inputs = get_cost_model_inputs(
        nb_path=transfer_cost_notebook_path,
        cell_exec_info=cell_exec_info,
        shell=shell,
        cpu_to_gpu=True,
        df_size_multiplier=df_size_multiplier,
    )
    gpu_to_cpu_cost_model_inputs = get_cost_model_inputs(
        nb_path=transfer_cost_notebook_path,
        cell_exec_info=cell_exec_info,
        shell=shell,
        cpu_to_gpu=False,
        df_size_multiplier=df_size_multiplier,
    )
    with open(
        original_notebook_path.parent / "cpu_to_gpu_cost_model_inputs.pkl", "wb"
    ) as f:
        pickle.dump(cpu_to_gpu_cost_model_inputs, f)
    with open(
        original_notebook_path.parent / "gpu_to_cpu_cost_model_inputs.pkl", "wb"
    ) as f:
        pickle.dump(gpu_to_cpu_cost_model_inputs, f)

    # Merge the CPU to GPU and GPU to CPU cost model inputs to one data structure and dump it.
    cost_model_inputs = merge_cpu_and_gpu_cost_model_inputs(
        cpu_to_gpu_cost_model_inputs, gpu_to_cpu_cost_model_inputs
    )
    # Dump the cost_model_inputs in case we need to change the cost model in the future.
    with open(original_notebook_path.parent / "cost_model_inputs.pkl", "wb") as f:
        pickle.dump(cost_model_inputs, f)

    # Get the ground truth transfer times and input/output df cols.
    transfer_times, input_df_cols, output_df_cols = (
        get_transfer_times_and_input_output_df_cols(
            original_notebook_path,
            cell_exec_info,
            cost_model_inputs,
        )
    )
    with open(original_notebook_path.parent / "transfer_times.pkl", "wb") as f:
        pickle.dump(transfer_times, f)
    with open(original_notebook_path.parent / "input_df_cols.pkl", "wb") as f:
        pickle.dump(input_df_cols, f)
    with open(original_notebook_path.parent / "output_df_cols.pkl", "wb") as f:
        pickle.dump(output_df_cols, f)

    # # We can choose to record the transfer times to a csv file.
    # transfer_cost_csv_path = original_notebook_path.parent / Path("transfer_costs.csv")
    # record_transfer_times(transfer_times, transfer_cost_csv_path)

    # Predict the cell times by running the notebook at multiple factors.
    # This function also runs the notebook at the original factor to get the ground truth cell times.
    predicted_times = predict_cell_times_for_nb(original_notebook_path, use_float=False)
    print(
        "Writing data to CSV file...",
        original_notebook_path.parent / "prediction_times.csv",
    )
    predicted_times.to_csv(
        original_notebook_path.parent / "prediction_times.csv", index=False
    )

    # Get the ground truth schedule and cost, using the ground truth cell times and transfer times.
    ground_truth_schedule, ground_truth_total_cost = get_schedule_and_cost(
        cpu_times=predicted_times.original_cpu_times.tolist(),
        gpu_times=predicted_times.original_gpu_times.tolist(),
        input_df_cols=input_df_cols,
        output_df_cols=output_df_cols,
        transfer_times=transfer_times,
        use_cost_model=False,
    )

    # Get the cost model schedule and cost, using the predicted cell times and predicted transfer times.
    cost_model_schedule, cost_model_total_cost = get_schedule_and_cost(
        cpu_times=predicted_times.cpu_predicted_times.tolist(),
        gpu_times=predicted_times.gpu_predicted_times.tolist(),
        input_df_cols=input_df_cols,
        output_df_cols=output_df_cols,
        transfer_times=transfer_times,
        use_cost_model=True,
    )
    print(f"Ground truth total cost: {ground_truth_total_cost}")
    print(f"Cost model total cost: {cost_model_total_cost}")
    print(f"Ground truth schedule: {ground_truth_schedule}")
    print(f"Cost model schedule: {cost_model_schedule}")

    # Check how many cells are scheduled differently.
    unmatched_count = 0
    matches: list[bool] = []
    for gs, cs in zip(ground_truth_schedule, cost_model_schedule):
        if gs != cs:
            unmatched_count += 1
        matches.append(gs == cs)
    print("Error rate", unmatched_count / len(ground_truth_schedule))

    # Dump the schedule info, as well as the matches, to a csv file.
    schedule_info = pd.DataFrame(
        {
            "ground_truth_schedule": ground_truth_schedule,
            "cost_model_schedule": cost_model_schedule,
            "matches": matches,
        }
    )
    schedule_info.to_csv(original_notebook_path / "schedule_info.csv")

    # TODO(jie): clear checkpoints.
    # Clear all checkpoints in the original and annotated notebooks.


if __name__ == "__main__":
    asyncio.run(main())
