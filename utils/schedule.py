## 5 parts to scheduling.
# 1. Run a small notebook to figure out cell dependencies.
# 2. Run the original CPU notebook for times.
# 3. Run the original GPU notebook for times.
# 4. Run the original / small CPU notebook to get the transfer code and df sizes.
# 5. Run the CPU to GPU and GPU to CPU notebooks to get both the real transfer times and the cost model transfer times.
# 6. Get the schedule based on the cost model transfer times.
# 7. Get the schedule based on the real transfer times.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from nbclient import NotebookClient
from nbformat import NotebookNode

from utils.execution import (
    CellExecInfo,
    CostModelInput,
    execute_cell,
    execute_code,
    execute_notebook,
    find_cell_times,
    get_cost_model_transfer_times,
    make_transfer_cells_and_get_next_index,
    parse_wall_time_to_ms_from_all_outputs,
    raise_errors_from_cell_outputs,
    reset_shell,
    run_cell_and_get_all_transfer_inputs,
)
from utils.notebook import (
    annotate_notebook,
    cell_annotation_pattern,
    load_notebook,
    make_dummy_transfer_cell,
    save_notebook,
)

Device = Literal["cpu", "gpu"]


### Step 1: Run a small notebook to figure out cell dependencies.
def get_cell_exec_info(
    small_nb_path: Path, shell: InteractiveShell
) -> dict[int, CellExecInfo]:
    """Returns a dictionary mapping cell indices to cell execution info."""
    cell_exec_info_nb_path = small_nb_path.parent / "cell_exec_info.ipynb"
    annotate_notebook(
        original_notebook_path=small_nb_path,
        annotated_notebook_path=cell_exec_info_nb_path,
        add_timing_code=False,
        add_record_events=True,
        add_checkpoints=False,
        track_column_info=True,
        use_gpu=False,
    )
    cell_exec_info_nb = load_notebook(cell_exec_info_nb_path)
    # We have to run this notebook through iPython shell because we need record cell exec info.
    # In the end, we want to save the cell info.
    for cell in cell_exec_info_nb.cells:
        execute_cell(cell, shell)
    # You can see the code for this magic command in the elastic-notebook repo.
    execute_code("%PrintCellInfo cell_exec_info", shell)
    cell_exec_info: dict[int, CellExecInfo] = shell.user_ns["cell_exec_info"]

    # # delete the cell_exec_info_nb notebook because we don't want to clutter our directory.
    # cell_exec_info_nb_path.unlink()
    # Reset the shell state for future use.
    reset_shell(shell)
    return cell_exec_info


### Step 2: Run the original CPU notebook for times.
def get_cpu_times(nb_path: Path) -> dict[int, float]:
    """Returns a dictionary mapping cell indices to cell times."""
    original_cpu_notebook_path = nb_path.parent / Path("original_annotated_cpu.ipynb")
    annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=original_cpu_notebook_path,
        add_timing_code=True,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=False,
    )
    cpu_notebook = load_notebook(original_cpu_notebook_path)
    print("Executing CPU notebook")
    execute_notebook(cpu_notebook)
    print("Finished executing CPU notebook")
    return find_cell_times(cpu_notebook)


### Step 3: Run the original GPU notebook for times.
def get_gpu_times(nb_path: Path) -> dict[int, float]:
    """Returns a dictionary mapping cell indices to cell times."""
    original_gpu_notebook_path = nb_path.parent / Path("original_annotated_gpu.ipynb")
    annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=original_gpu_notebook_path,
        add_timing_code=True,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=True,
    )
    gpu_notebook = load_notebook(original_gpu_notebook_path)
    print("Executing GPU notebook")
    execute_notebook(gpu_notebook)
    print("Finished executing GPU notebook")
    return find_cell_times(gpu_notebook)


### Step 4: Run the original / small CPU notebook to get the transfer code and df sizes.
def get_cost_model_inputs(
    *,
    nb_path: Path,
    cell_exec_info: dict[int, CellExecInfo],
    shell: InteractiveShell,
    cpu_to_gpu: bool,
    df_size_multiplier: int,
) -> dict[int, tuple[list[CostModelInput], list[CostModelInput]]]:
    """Returns a dictionary mapping cell indices to cost model inputs."""
    annotated_cell_idx_to_cost_model_inputs: dict[
        int, tuple[list[CostModelInput], list[CostModelInput]]
    ] = {}
    if cpu_to_gpu:
        cost_model_inputs_nb_path = nb_path.parent / Path(
            "cpu_to_gpu_cost_model_inputs.ipynb"
        )
    else:
        cost_model_inputs_nb_path = nb_path.parent / Path(
            "gpu_to_cpu_cost_model_inputs.ipynb"
        )
    if cpu_to_gpu:
        use_gpu = False
    else:
        use_gpu = True

    cost_model_inputs_annotated_cell_idx_to_cell_data = annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=cost_model_inputs_nb_path,
        add_timing_code=True,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=use_gpu,
    )
    cost_model_inputs_nb = load_notebook(cost_model_inputs_nb_path)

    for annotated_cell_idx, info in sorted(cell_exec_info.items()):
        cell_data = cost_model_inputs_annotated_cell_idx_to_cell_data[
            annotated_cell_idx
        ]
        curr_cell = cost_model_inputs_nb.cells[cell_data.cell_idx_in_annotated_notebook]
        if annotated_cell_idx == 0:
            for cell in cost_model_inputs_nb.cells[
                : cell_data.cell_idx_in_annotated_notebook
            ]:
                execute_cell(cell, shell)

        _, input_transfer_inputs, output_transfer_inputs = (
            run_cell_and_get_all_transfer_inputs(
                curr_cell=curr_cell,
                info=info,
                shell=shell,
                cpu_to_gpu=cpu_to_gpu,
                df_size_multiplier=df_size_multiplier,
            )
        )
        annotated_cell_idx_to_cost_model_inputs[annotated_cell_idx] = (
            input_transfer_inputs,
            output_transfer_inputs,
        )

    # Delete the cost_model_inputs_nb notebook because we don't want to clutter our directory.
    cost_model_inputs_nb_path.unlink()

    # Reset the shell state for future use.
    reset_shell(shell)

    return annotated_cell_idx_to_cost_model_inputs


### Step 5: Run the CPU to GPU and GPU to CPU notebooks to get both the real transfer times and the cost model transfer times.
# TODO(jie): fix the type.
def _get_transfer_times_nb(
    nb_path: Path,
    cpu_to_gpu: bool,
    cost_model_inputs: dict[int, tuple[list[CostModelInput], list[CostModelInput]]],
) -> NotebookNode:
    """Returns a dictionary mapping cell indices to transfer times."""
    if cpu_to_gpu:
        use_gpu = False
        transfer_times_nb_path = nb_path.parent / Path(
            "cpu_to_gpu_transfer_times.ipynb"
        )
    else:
        use_gpu = True
        transfer_times_nb_path = nb_path.parent / Path(
            "gpu_to_cpu_transfer_times.ipynb"
        )

    annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=transfer_times_nb_path,
        add_timing_code=True,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=use_gpu,
    )
    transfer_times_nb = load_notebook(transfer_times_nb_path)

    # If transfering from CPU to GPU, we need to warm up the GPU with each data type first.
    if cpu_to_gpu:
        dummy_transfer_cells = make_dummy_transfer_cell(cpu_to_gpu=True)
        # Insert dummy transfer cells after the first cell.
        transfer_times_nb.cells.insert(1, dummy_transfer_cells)

    index = 0
    while index < len(transfer_times_nb.cells):
        cell = transfer_times_nb.cells[index]
        if cell.cell_type != "code":
            index += 1
            continue  # Skip non-code cells

        # Check if this cell is an annotated cell.
        match = cell_annotation_pattern.search(cell.source)
        annotated_cell_idx: int | None = None
        input_transfer_inputs: list[CostModelInput] | None = None
        output_transfer_inputs: list[CostModelInput] | None = None

        if match:
            annotated_cell_idx = int(match.group(1))

        if annotated_cell_idx is not None:
            input_transfer_inputs, output_transfer_inputs = cost_model_inputs[
                annotated_cell_idx
            ]
            make_transfer_cells_and_get_next_index(
                annotated_cell_idx=annotated_cell_idx,
                transfers=input_transfer_inputs,
                cpu_to_gpu=cpu_to_gpu,
                next_index=transfer_times_nb.cells.index(cell),
                notebook=transfer_times_nb,
                pre_exec=True,
            )
            next_execute_index = make_transfer_cells_and_get_next_index(
                annotated_cell_idx=annotated_cell_idx,
                transfers=output_transfer_inputs,
                cpu_to_gpu=cpu_to_gpu,
                next_index=transfer_times_nb.cells.index(cell) + 1,
                notebook=transfer_times_nb,
                pre_exec=False,
            )
            index = next_execute_index
        else:
            cell_idx = transfer_times_nb.cells.index(cell)
            index = cell_idx + 1

    if cpu_to_gpu:
        transfer_times_nb_path = nb_path.parent / Path(
            "cpu_to_gpu_transfer_times.ipynb"
        )
    else:
        transfer_times_nb_path = nb_path.parent / Path(
            "gpu_to_cpu_transfer_times.ipynb"
        )

    # Here we are keeping the tansfer times notebook for debugging purposes. We might not need to save in the future.
    save_notebook(transfer_times_nb, transfer_times_nb_path)
    # This was a helper notebook so we can delete it now.
    transfer_times_nb_path.unlink()

    return transfer_times_nb


def _populate_transfer_times(
    cells: list[NotebookNode],
    series_input_transfers: dict[int, list[CostModelInput]],
    series_output_transfers: dict[int, list[CostModelInput]],
    df_input_transfers: dict[int, list[CostModelInput]],
    df_output_transfers: dict[int, list[CostModelInput]],
    cpu_to_gpu: bool,
) -> None:
    # populate the actual GPU to CPU transfer times in the input and output cost model inputs.
    for cell in cells:
        # This means it's a transfer cell.
        if "transfer" not in cell.metadata:
            continue

        # Now we want to find the transfer that corresponds to this transfer cell.
        is_series = cell.metadata["transfer"]["is_series"]
        df_name = cell.metadata["transfer"]["df_name"]
        annotated_cell_idx = cell.metadata["transfer"]["annotated_cell_idx"]
        pre_exec = cell.metadata["transfer"]["pre_exec"]
        curr_transfer: Optional[CostModelInput] = None

        related_transfers: list[CostModelInput] = []

        if is_series:
            if pre_exec:
                related_transfers = series_input_transfers[annotated_cell_idx]
            else:
                related_transfers = series_output_transfers[annotated_cell_idx]
        else:
            if pre_exec:
                related_transfers = df_input_transfers[annotated_cell_idx]
            else:
                related_transfers = df_output_transfers[annotated_cell_idx]

        for transfer in related_transfers:
            if transfer.df_name == df_name:
                curr_transfer = transfer
                break

        if curr_transfer is None:
            raise RuntimeError(
                "Could not find the transfer that corresponds to this cell."
            )

        exec_time = parse_wall_time_to_ms_from_all_outputs(cell.outputs)
        if exec_time is None:
            raise RuntimeError(
                f"Could not find the transfer time for cell {annotated_cell_idx}."
            )

        if is_series:
            if cpu_to_gpu:
                curr_transfer.cpu_to_gpu_transfer_time = exec_time
            else:
                curr_transfer.gpu_to_cpu_transfer_time = exec_time
        else:
            col_name = cell.metadata["transfer"]["col_name"]
            if cpu_to_gpu:
                if curr_transfer.cpu_to_gpu_col_transfer_times is None:
                    curr_transfer.cpu_to_gpu_col_transfer_times = {}
                curr_transfer.cpu_to_gpu_col_transfer_times[col_name] = exec_time
            else:
                if curr_transfer.gpu_to_cpu_col_transfer_times is None:
                    curr_transfer.gpu_to_cpu_col_transfer_times = {}
                curr_transfer.gpu_to_cpu_col_transfer_times[col_name] = exec_time


def get_transfer_times_and_input_output_df_cols(
    nb_path: Path,
    cell_exec_info: dict[int, CellExecInfo],
    cost_model_inputs: dict[int, tuple[list[CostModelInput], list[CostModelInput]]],
) -> tuple[
    list[dict[DfCol, dict[str, float]]],
    list[list[DfCol]],
    list[list[DfCol]],
]:
    """
    Runs the notebook at `nb_path` to get the actual transfer times.
    Uses the `cost_model_inputs` to get the predicted transfer times.

    Returns a tuple of:
        - list of dictionaries mapping cell indices to dictionaries mapping df columns to transfer times.
        - input df columns: list of lists of df columns.
        - output df columns: list of lists of df columns.
    """
    series_input_transfers: dict[int, list[CostModelInput]] = {}
    series_output_transfers: dict[int, list[CostModelInput]] = {}
    df_input_transfers: dict[int, list[CostModelInput]] = {}
    df_output_transfers: dict[int, list[CostModelInput]] = {}
    for annotated_cell_idx, (
        input_transfer_inputs,
        output_transfer_inputs,
    ) in cost_model_inputs.items():
        series_input_transfers[annotated_cell_idx] = []
        series_output_transfers[annotated_cell_idx] = []
        df_input_transfers[annotated_cell_idx] = []
        df_output_transfers[annotated_cell_idx] = []

        for input_transfer in input_transfer_inputs:
            if input_transfer.is_series:
                series_input_transfers[annotated_cell_idx].append(input_transfer)
            else:
                df_input_transfers[annotated_cell_idx].append(input_transfer)
        for output_transfer in output_transfer_inputs:
            if output_transfer.is_series:
                series_output_transfers[annotated_cell_idx].append(output_transfer)
            else:
                df_output_transfers[annotated_cell_idx].append(output_transfer)

    # Now we will run the CPU to GPU and GPU to CPU notebooks to get the real transfer times.
    for cpu_to_gpu in [False, True]:
        if cpu_to_gpu:
            print("Getting CPU to GPU transfer times")
        else:
            print("Getting GPU to CPU transfer times")

        # Get a notebook with all the transfer cells and execute it.
        transfer_nb = _get_transfer_times_nb(
            nb_path=nb_path,
            cpu_to_gpu=cpu_to_gpu,
            cost_model_inputs=cost_model_inputs,
        )
        NotebookClient(transfer_nb).execute()
        raise_errors_from_cell_outputs(transfer_nb.cells)

        # Now we have executed the transfer cells, we can populate the actual transfer times in the input and output cost model inputs.
        _populate_transfer_times(
            cells=transfer_nb.cells,
            series_input_transfers=series_input_transfers,
            series_output_transfers=series_output_transfers,
            df_input_transfers=df_input_transfers,
            df_output_transfers=df_output_transfers,
            cpu_to_gpu=cpu_to_gpu,
        )

        if cpu_to_gpu:
            print("Finished getting CPU to GPU transfer times")
        else:
            print("Finished getting GPU to CPU transfer times")

    # Now we will set the transfer times for each cost model input.
    # Make a list of input df cols.
    num_total_cells = list(sorted(cell_exec_info.keys()))[-1] + 1
    input_df_cols = [[] for _ in range(num_total_cells)]
    output_df_cols = [[] for _ in range(num_total_cells)]
    transfer_times: list[dict[DfCol, dict[str, float]]] = [
        {} for _ in range(num_total_cells)
    ]
    for annotated_cell_idx, (
        input_transfer_inputs,
        output_transfer_inputs,
    ) in cost_model_inputs.items():
        print("cell index", annotated_cell_idx)
        print("input transfer inputs", input_transfer_inputs)
        print("output transfer inputs", output_transfer_inputs)
        cpu_to_gpu_input_cost_model_transfer_times = get_cost_model_transfer_times(
            input_transfer_inputs, cpu_to_gpu=True
        )
        cpu_to_gpu_output_cost_model_transfer_times = get_cost_model_transfer_times(
            output_transfer_inputs, cpu_to_gpu=True
        )
        gpu_to_cpu_input_cost_model_transfer_times = get_cost_model_transfer_times(
            input_transfer_inputs, cpu_to_gpu=False
        )
        gpu_to_cpu_output_cost_model_transfer_times = get_cost_model_transfer_times(
            output_transfer_inputs, cpu_to_gpu=False
        )
        for idx, input_transfer in enumerate(input_transfer_inputs):
            input_transfer.validate()
            # get the cost model times
            if input_transfer.is_series:
                assert input_transfer.cpu_to_gpu_transfer_time is not None
                assert input_transfer.gpu_to_cpu_transfer_time is not None
                transfer_times[annotated_cell_idx][
                    DfCol(df_name=input_transfer.df_name, col_name=None)
                ] = {
                    "cpu->gpu": input_transfer.cpu_to_gpu_transfer_time,
                    "gpu->cpu": input_transfer.gpu_to_cpu_transfer_time,
                    # FIXME(sahil): add 0 cost for series.
                    "cost-cpu->gpu": 0,
                    "cost-gpu->cpu": 0,
                }
            else:
                for col_name in input_transfer.col_names or []:
                    assert input_transfer.cpu_to_gpu_col_transfer_times is not None
                    assert input_transfer.gpu_to_cpu_col_transfer_times is not None
                    input_df_cols[annotated_cell_idx].append(
                        DfCol(df_name=input_transfer.df_name, col_name=col_name)
                    )
                    transfer_times[annotated_cell_idx][
                        DfCol(df_name=input_transfer.df_name, col_name=col_name)
                    ] = {
                        "cpu->gpu": input_transfer.cpu_to_gpu_col_transfer_times[
                            col_name
                        ],
                        "gpu->cpu": input_transfer.gpu_to_cpu_col_transfer_times[
                            col_name
                        ],
                        "cost-cpu->gpu": cpu_to_gpu_input_cost_model_transfer_times[
                            idx
                        ][col_name],
                        "cost-gpu->cpu": gpu_to_cpu_input_cost_model_transfer_times[
                            idx
                        ][col_name],
                    }
        for idx, output_transfer in enumerate(output_transfer_inputs):
            output_transfer.validate()
            cpu_to_gpu_output_cost_model_transfer_times = get_cost_model_transfer_times(
                output_transfer_inputs, cpu_to_gpu=True
            )
            if output_transfer.is_series:
                assert output_transfer.cpu_to_gpu_transfer_time is not None
                assert output_transfer.gpu_to_cpu_transfer_time is not None
                transfer_times[annotated_cell_idx][
                    DfCol(df_name=output_transfer.df_name, col_name=None)
                ] = {
                    "cpu->gpu": output_transfer.cpu_to_gpu_transfer_time,
                    "gpu->cpu": output_transfer.gpu_to_cpu_transfer_time,
                    "cost-cpu->gpu": 0,
                    "cost-gpu->cpu": 0,
                }
            else:
                for col_name in output_transfer.col_names or []:
                    assert output_transfer.cpu_to_gpu_col_transfer_times is not None
                    assert output_transfer.gpu_to_cpu_col_transfer_times is not None
                    output_df_cols[annotated_cell_idx].append(
                        DfCol(df_name=output_transfer.df_name, col_name=col_name)
                    )
                    transfer_times[annotated_cell_idx][
                        DfCol(df_name=output_transfer.df_name, col_name=col_name)
                    ] = {
                        "cpu->gpu": output_transfer.cpu_to_gpu_col_transfer_times[
                            col_name
                        ],
                        "gpu->cpu": output_transfer.gpu_to_cpu_col_transfer_times[
                            col_name
                        ],
                        "cost-cpu->gpu": cpu_to_gpu_output_cost_model_transfer_times[
                            idx
                        ][col_name],
                        "cost-gpu->cpu": gpu_to_cpu_output_cost_model_transfer_times[
                            idx
                        ][col_name],
                    }
    return transfer_times, input_df_cols, output_df_cols


@dataclass(frozen=True)
class DfCol:
    df_name: str
    col_name: str | None = None


def _build_last_use(input_df_cols) -> dict[DfCol, int]:
    """Map each column to the index of its last *read*."""
    last = defaultdict(int)
    for i, reads in enumerate(input_df_cols):
        for df_col in reads:
            last[df_col] = i
    return last


def get_schedule_and_cost(
    cpu_times: list[float],
    gpu_times: list[float],
    input_df_cols: list[list[DfCol]],
    output_df_cols: list[list[DfCol]],
    transfer_times: list[dict[DfCol, dict[str, float]]],
    use_cost_model: bool,
) -> tuple[list[Device], float]:
    n = len(cpu_times)
    last_read: dict[DfCol, int] = _build_last_use(input_df_cols)
    parent = {}  # (i, sigma) → chosen device

    @lru_cache(maxsize=None)
    def dp(i: int, sigma: frozenset[tuple[DfCol, Device]]) -> float:
        if i == n:
            return 0.0

        best_cost, best_dev = float("inf"), None
        sigma_dict = dict(sigma)  # make mutable copy once

        for dev in ("cpu", "gpu"):
            exec_time = cpu_times[i] if dev == "cpu" else gpu_times[i]
            xfer_cost = 0.0
            new_sigma = sigma_dict.copy()

            # bring inputs
            for col in input_df_cols[i]:
                loc = new_sigma.get(col, dev)  # unseen ⇒ assume dev
                if loc != dev:
                    if use_cost_model:
                        cost_key = f"cost-{loc}->{dev}"
                        real_key = f"{loc}->{dev}"
                        cost_model_time = transfer_times[i][col][cost_key]
                        xfer_cost += cost_model_time
                        # print(f'Cell {i}, col {col}: {loc}->{dev}, cost_model={cost_model_time}, real={real_time}, total_xfer_cost={xfer_cost}')
                    else:
                        real_key = f"{loc}->{dev}"
                        real_time = transfer_times[i][col][real_key]
                        xfer_cost += real_time
                        # print(f'Cell {i}, col {col}: {loc}->{dev}, real={real_time}, total_xfer_cost={xfer_cost}')
                    new_sigma[col] = dev

            # place outputs
            for col in output_df_cols[i]:
                new_sigma[col] = dev

            # GC columns whose last read is <= i
            for col in list(new_sigma):
                if last_read.get(col, -1) <= i:
                    new_sigma.pop(col)
            try:
                cost = exec_time + xfer_cost + dp(i + 1, frozenset(new_sigma.items()))
            except Exception as e:
                print(f"Error occurred: {e}")
            if cost < best_cost:
                best_cost, best_dev = cost, dev

        parent[(i, sigma)] = best_dev
        return best_cost

    # run DP from empty layout
    root = frozenset()
    total_cost = dp(0, root)

    # reconstruct schedule
    schedule, i, sigma = [], 0, root
    while i < n:
        dev = parent[(i, sigma)]
        schedule.append(dev)

        # advance one step (mirror of transition logic)
        sigma_dict = dict(sigma)
        for col in input_df_cols[i]:
            loc = sigma_dict.get(col, dev)
            if loc != dev:
                sigma_dict[col] = dev
        for col in output_df_cols[i]:
            sigma_dict[col] = dev
        for col in list(sigma_dict):
            if last_read.get(col, -1) <= i:
                sigma_dict.pop(col)

        sigma = frozenset(sigma_dict.items())
        i += 1

    print("Schedule:", schedule)
    print("Total cost:", total_cost)
    print("sigma size:", len(sigma))
    print("parent size:", len(parent))
    return schedule, total_cost


def record_transfer_times(
    transfer_times: list[dict[DfCol, dict[str, float]]], csv_path: Path
) -> None:
    """Writes the transfer times to a csv file."""
    cpu_gpu_times: list[dict[str, float | int | str]] = []
    for cell_idx, times in enumerate(transfer_times):
        for df_col, time_info in times.items():
            cpu_gpu_times.append(
                {
                    "cell_idx": cell_idx,
                    "df_name": df_col.df_name,
                    "df_col": df_col.col_name or "None",
                    "cpu->gpu": time_info["cpu->gpu"],
                    "cost-cpu->gpu": time_info["cost-cpu->gpu"],
                    "gpu->cpu": time_info["gpu->cpu"],
                    "cost-gpu->cpu": time_info["cost-gpu->cpu"],
                }
            )
    cpu_gpu_times_df = pd.DataFrame(cpu_gpu_times)
    cpu_gpu_times_df.to_csv(csv_path, index=False)


def get_actual_time_for_schedule(
    schedule: list[Device],
    cpu_times: list[float],
    gpu_times: list[float],
    input_df_cols: list[list[DfCol]],
    output_df_cols: list[list[DfCol]],
    transfer_times: list[dict[DfCol, dict[str, float]]],
) -> float:
    # We start with nothing. Will add columns as we go.
    sigma_dict = {}

    # We don't use cost model here.
    total_time = 0.0
    for i, dev in enumerate(schedule):
        total_time += cpu_times[i] if dev == "cpu" else gpu_times[i]
        for col in input_df_cols[i]:
            loc = sigma_dict.get(col, dev)  # unseen ⇒ assume dev
            if loc != dev:
                total_time += transfer_times[i][col][f"{loc}->{dev}"]
                sigma_dict[col] = dev
        for col in output_df_cols[i]:
            sigma_dict[col] = dev

    return total_time
