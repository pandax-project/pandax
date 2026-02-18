import copy
import io
import logging
import re
import sys
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from elastic.core.common.cell_exec_info import CellExecInfo
from elastic.core.common.pandas import (
    DFType,
    get_df_size,
    get_df_type,
    is_type_dataframe,
    is_type_df,
)
from IPython.core.interactiveshell import InteractiveShell
from nbclient import NotebookClient
from nbformat import NotebookNode

from utils.notebook import (
    cell_annotation_pattern,
    get_transfer_cpu_to_gpu_cells,
    get_transfer_gpu_to_cpu_cells,
    make_code_cell,
    maybe_annotate_code_with_time,
)


@dataclass
class CudfProfileInfo:
    gpu_calls: int
    cpu_calls: int
    gpu_time: float
    cpu_time: float
    df_table: pd.DataFrame


@dataclass
class CpuProfileInfo:
    profiling_info: str
    df_table: pd.DataFrame


@dataclass
class CostModelInput:
    df_name: str
    rows: int
    cols: int
    is_series: bool
    col_data: dict[str, tuple[DFType, int | None, int | None]]
    # The following fields are only used for dataframes.
    col_names: list[str] | None
    cpu_to_gpu_col_transfer_times: dict[str, float] | None = None
    gpu_to_cpu_col_transfer_times: dict[str, float] | None = None
    # The following fields are only used for series.
    cpu_to_gpu_transfer_time: float | None = None
    gpu_to_cpu_transfer_time: float | None = None

    @property
    def key(self) -> tuple:
        return (
            self.df_name,
            # self.rows,
            # self.cols,
            # self.is_series,
            # set(self.col_names or []),
        )

    def validate(self) -> None:
        if self.is_series:
            assert self.cpu_to_gpu_transfer_time is not None
            assert self.gpu_to_cpu_transfer_time is not None
        else:
            # print("Expected col_names:", sorted(self.col_names))
            # print("cpu_to_gpu_col_transfer_times:", sorted(self.cpu_to_gpu_col_transfer_times.keys()))
            # print("gpu_to_cpu_col_transfer_times:", sorted(self.gpu_to_cpu_col_transfer_times.keys()))
            # check only if the col_names are not empty
            if self.col_names:
                assert self.cpu_to_gpu_col_transfer_times is not None
                assert self.gpu_to_cpu_col_transfer_times is not None
                assert sorted(self.cpu_to_gpu_col_transfer_times.keys()) == sorted(
                    self.col_names
                )
                assert sorted(self.gpu_to_cpu_col_transfer_times.keys()) == sorted(
                    self.col_names
                )
        for col_data_value in self.col_data.values():
            assert col_data_value[0] is not None
            assert col_data_value[1] is not None
            assert col_data_value[2] is not None


def sort_cost_model_inputs(
    cost_model_inputs: list[CostModelInput],
) -> list[CostModelInput]:
    # Sort the inputs by the df_name and df_type
    return sorted(cost_model_inputs, key=lambda x: x.key)


def merge_cpu_and_gpu_cost_model_inputs(
    cpu_to_gpu_cost_model_inputs: dict[
        int, tuple[list[CostModelInput], list[CostModelInput]]
    ],
    gpu_to_cpu_cost_model_inputs: dict[
        int, tuple[list[CostModelInput], list[CostModelInput]]
    ],
) -> dict[int, tuple[list[CostModelInput], list[CostModelInput]]]:
    final_cost_model_inputs: dict[
        int, tuple[list[CostModelInput], list[CostModelInput]]
    ] = {}
    if set(cpu_to_gpu_cost_model_inputs.keys()) != set(
        gpu_to_cpu_cost_model_inputs.keys()
    ):
        raise ValueError(
            "The CPU to GPU and GPU to CPU inputs must have the same keys."
        )
    for key in cpu_to_gpu_cost_model_inputs.keys():
        tuple_merged_cost_model_inputs: list[list[CostModelInput]] = []
        for tuple_idx in [0, 1]:
            curr_cpu_to_gpu_cost_model_inputs = sort_cost_model_inputs(
                cpu_to_gpu_cost_model_inputs[key][tuple_idx]
            )
            curr_gpu_to_cpu_cost_model_inputs = sort_cost_model_inputs(
                gpu_to_cpu_cost_model_inputs[key][tuple_idx]
            )
            if len(curr_cpu_to_gpu_cost_model_inputs) != len(
                curr_gpu_to_cpu_cost_model_inputs
            ):
                raise ValueError(
                    "The number of CPU to GPU and GPU to CPU inputs must be the same."
                )
            merged_cost_model_inputs = []
            for cpu_to_gpu_input, gpu_to_cpu_input in zip(
                curr_cpu_to_gpu_cost_model_inputs, curr_gpu_to_cpu_cost_model_inputs
            ):
                if cpu_to_gpu_input.key != gpu_to_cpu_input.key:
                    raise ValueError(
                        "The CPU to GPU and GPU to CPU inputs must have the same key."
                    )
                final_input = copy.deepcopy(cpu_to_gpu_input)
                all_col_names = set(final_input.col_data.keys()).intersection(
                    set(gpu_to_cpu_input.col_data.keys())
                )
                for col_name in all_col_names:
                    cpu_to_gpu_time = cpu_to_gpu_input.col_data.get(
                        col_name, (None, None, None)
                    )[1]
                    gpu_to_cpu_time = gpu_to_cpu_input.col_data.get(
                        col_name, (None, None, None)
                    )[2]
                    cpu_to_gpu_time = 0 if cpu_to_gpu_time is None else cpu_to_gpu_time
                    gpu_to_cpu_time = 0 if gpu_to_cpu_time is None else gpu_to_cpu_time
                    final_input.col_data[col_name] = cast(
                        tuple[DFType, int, int],
                        (
                            final_input.col_data[col_name][0],
                            cpu_to_gpu_time,
                            gpu_to_cpu_time,
                        ),
                    )
                merged_cost_model_inputs.append(final_input)
            tuple_merged_cost_model_inputs.append(merged_cost_model_inputs)
        final_cost_model_inputs[key] = (
            tuple_merged_cost_model_inputs[0],
            tuple_merged_cost_model_inputs[1],
        )
    return final_cost_model_inputs


def execute_code(code: str, shell: InteractiveShell) -> str:
    print("Executing code:", code)
    custom_output = io.StringIO()
    # Redirect stdout to the custom buffer
    old_stdout = sys.stdout  # Save the current stdout
    sys.stdout = custom_output  # Replace stdout with the custom StringIO buffer
    captured_output = None
    shell.run_cell(code)
    captured_output = custom_output.getvalue()
    # Restore the original stdout
    sys.stdout = old_stdout
    logging.info("Captured output:")
    logging.info(captured_output)
    # This is a hack, because result.raise_error doesn't seem to work as the documentation says.
    if "Error" in captured_output:
        raise Exception(f"Error in code execution: {captured_output}")
    return captured_output


def execute_cell(cell: NotebookNode, shell: InteractiveShell) -> str | None:
    if cell.cell_type == "code":
        return execute_code(cell.source, shell)
    return None


def execute_cells(
    cells: list[NotebookNode], shell: InteractiveShell
) -> list[str | None]:
    return [execute_cell(cell, shell) for cell in cells]


def parse_wall_time_to_ms_from_all_outputs(
    outputs: list[dict[str, Any]],
) -> float | None:
    for output in outputs:
        # print(f"Output: {output}")
        wall_time = parse_wall_time_to_ms(output.get("text", ""))
        if wall_time is not None:
            return wall_time
    return None


def parse_total_time_to_ms_from_all_outputs(
    outputs: list[dict[str, Any]],
) -> float | None:
    for output in outputs:
        total_time = parse_total_time_to_ms(output.get("text", ""))
        if total_time is not None:
            return total_time
    return None


def parse_cudf_profile_info_from_all_outputs(
    outputs: list[dict[str, Any]],
) -> CudfProfileInfo | None:
    for output in outputs:
        try:
            captured_output = output["data"]["text/plain"]
            gpu_calls, cpu_calls = extract_total_calls(captured_output)
            gpu_time, cpu_time = extract_total_time(captured_output)

            df_table = parse_cudf_profile_table(captured_output)
            return CudfProfileInfo(
                gpu_calls=gpu_calls,
                cpu_calls=cpu_calls,
                gpu_time=gpu_time,
                cpu_time=cpu_time,
                df_table=df_table,
            )
        except Exception:
            continue
    # raise RuntimeError("Could not find the cudf profile information.")


def parse_cpu_profile_info_from_all_outputs(
    outputs: list[dict[str, Any]],
) -> CpuProfileInfo | None:
    def extract_profile_table(text, marker="[PandasProfile] Summary"):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if marker in line:
                return "\n".join(lines[i + 1 :])  # includes header + data
        return None

    for output in outputs:
        try:
            captured_output = extract_profile_table(output["text"])
            if captured_output is None:
                continue
            df_table = pd.read_csv(io.StringIO(captured_output), sep="\t")
            if df_table is None:
                continue
            return CpuProfileInfo(
                profiling_info=output["text"],
                df_table=df_table,
            )
        except Exception:
            continue


def parse_wall_time_to_ms(time_output: str) -> float | None:
    """
    Convert 'Wall time: …' strings to milliseconds.

    Handles strings like:
      - 'Wall time: 85.2 ms'
      - 'Wall time: 4min 39s'
      - 'Wall time: 1h 3min 7.5s'
      - 'Wall time: 250µs'
      - 'Wall time: 1e+03 us'

    Returns
    -------
    float | None
        Wall time in milliseconds, or None if nothing could be parsed.
    """

    time_output = time_output.lower()

    # match sequences like "1e+03 ms", "4min", "3.2 s", etc.
    pattern = r"wall time:\s*((?:[\deE\.\+\-]+\s*[a-zµμ]+(?:\s+)?)*)"
    match = re.search(pattern, time_output)
    if not match:
        return None

    duration_str = match.group(1)

    # Extract all number + unit pairs from the matched string
    pair_pattern = r"([\deE\.\+\-]+)\s*([a-zµμ]+)"
    pairs = re.findall(pair_pattern, duration_str)
    if not pairs:
        return None

    unit_to_ms = {
        "ns": 1 / 1_000_000,
        "µs": 1 / 1000,
        "μs": 1 / 1000,
        "us": 1 / 1000,
        "ms": 1,
        "s": 1000,
        "sec": 1000,
        "m": 60_000,  # allow 'm' as shorthand for minutes
        "min": 60_000,
        "h": 3_600_000,
        "hr": 3_600_000,
    }

    total_ms = 0.0
    for value_str, unit in pairs:
        unit = unit.lower()
        if unit not in unit_to_ms:
            print(f"Unknown unit '{unit}' in time output: {time_output}")
            return None
        try:
            total_ms += float(value_str) * unit_to_ms[unit]
        except ValueError:
            return None

    return total_ms


def parse_total_time_to_ms(time_output: str) -> float | None:
    """
    Convert 'total time: …' strings to milliseconds.

    Handles strings like:
      - 'total time: 85.2 ms'
      - 'total time: 4min 39s'
      - 'total time: 1h 3min 7.5s'
      - 'total time: 250µs'
      - 'total time: 1e+03 us'

    Returns
    -------
    float | None
        total time in milliseconds, or None if nothing could be parsed.
    """
    # Regex: match float or scientific notation (e.g., 1.2e+03)
    pattern = r"total time:\s*((?:\d+(?:\.\d*)?|\.\d+|(?:\d+(?:\.\d+)?[eE][+-]?\d+))\s*[a-zμ]+)"
    matches = re.findall(pattern, time_output.lower())

    if not matches:
        return None

    unit_to_ms = {
        "ns": 1 / 1_000_000,
        "μs": 1 / 1000,
        "us": 1 / 1000,
        "ms": 1,
        "s": 1000,
        "min": 60_000,
        "h": 3_600_000,
    }

    total_ms = 0.0
    for match in matches:
        # Extract number and unit
        pair_match = re.match(
            r"((?:\d+(?:\.\d*)?|\.\d+|(?:\d+(?:\.\d+)?[eE][+-]?\d+)))\s*([a-zμ]+)",
            match,
        )
        if not pair_match:
            return None
        value, unit = pair_match.groups()
        if unit not in unit_to_ms:
            print(f"Unknown unit '{unit}' in time output: {time_output}")
            return None
        total_ms += float(value) * unit_to_ms[unit]

    return total_ms


def parse_memory_and_time(profile_output: str | None):
    """
    output is of the following format:
    Cell runtime: 8.149862289428711
    Memory usage before cell: 2.93 GB (2999.07 MB)
    Memory usage after cell: 2.93 GB (2999.07 MB)
    Peak usage of cell: 2.93 GB (2999.07 MB)
    """
    if not profile_output:
        return None

    time_match = re.search(r"Cell runtime:\s*([\d.]+)", profile_output)
    memory_before_match = re.search(
        r"Memory usage before cell:\s*([\d.]+)\s*GB", profile_output
    )
    memory_after_match = re.search(
        r"Memory usage after cell:\s*([\d.]+)\s*GB", profile_output
    )
    peak_memory_match = re.search(
        r"Peak usage of cell:\s*([\d.]+)\s*GB", profile_output
    )

    time = float(time_match.group(1)) if time_match else None
    memory_before = float(memory_before_match.group(1)) if memory_before_match else None
    memory_after = float(memory_after_match.group(1)) if memory_after_match else None
    peak_memory = float(peak_memory_match.group(1)) if peak_memory_match else None

    if (
        time is None
        or memory_before is None
        or memory_after is None
        or peak_memory is None
    ):
        return None
    return time, memory_before, memory_after, peak_memory


def extract_total_calls(output: str | None) -> tuple[int, int]:
    gpu_match = re.search(r"(\d+)\s+GPU function calls", output)
    cpu_match = re.search(r"(\d+)\s+CPU function calls", output)

    gpu_calls = int(gpu_match.group(1)) if gpu_match else 0
    cpu_calls = int(cpu_match.group(1)) if cpu_match else 0
    return gpu_calls, cpu_calls


def extract_total_time(output: str | None) -> tuple[float, float]:
    gpu_cpu_times = re.findall(
        r"(?:GPU|CPU) function calls in (\d+\.\d+) seconds", output
    )
    if len(gpu_cpu_times) != 2:
        raise ValueError(f"Expected 2 times, but got {len(gpu_cpu_times)}")
    return int(float(gpu_cpu_times[0]) * 1000), int(float(gpu_cpu_times[1]) * 1000)


def get_cudf_profile_stats(output: str | None) -> str | None:
    if not output:
        return None
    match = re.search(r"Stats\s+([\s\S]*?)(?=\nTo request GPU support|$)", output)
    stats_section = match.group(1) if match else None
    return stats_section


def parse_cudf_profile_table(stats_section: str) -> pd.DataFrame:
    """
    Parses the provided stats_section table and returns a DataFrame with the first two columns:
    'Function' and 'GPU ncalls'.
    """
    rows = []
    lines = stats_section.splitlines()

    # Identify the start of data rows: after the header divider line
    # We assume data rows start with a line that begins with "│" after the header divider ("┡")
    data_started = False
    for line in lines:
        if line.startswith("┡"):
            data_started = True
            continue
        # Process only lines that start with a data cell marker and once data has started.
        if data_started and line.startswith("│"):
            # Remove leading and trailing box characters and then split using the vertical bar as delimiter.
            parts = [field.strip() for field in line.strip("│").split("│")]
            # Ensure the row has at least two columns (Function, GPU calls)
            if len(parts) >= 5:
                # Append a tuple with the function name and GPU calls (converted to int)
                try:
                    gpu_calls = int(parts[1])
                    cpu_calls = int(parts[4])
                except ValueError:
                    gpu_calls = None  # if conversion fails
                rows.append(
                    {
                        "Function": parts[0],
                        "GPU calls": gpu_calls,
                        "CPU calls": cpu_calls,
                    }
                )

    df = pd.DataFrame(rows)
    return df


def is_rewritten_code_better(
    orig_cudf_profile_info: CudfProfileInfo,
    opt_cudf_profile_info: CudfProfileInfo,
    orig_time: float,
    opt_time: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> bool:
    # Calculate GPU call percentages
    if orig_cudf_profile_info.gpu_calls == 0 and orig_cudf_profile_info.cpu_calls == 0:
        gpu_calls_orig_perc = 0
    else:
        gpu_calls_orig_perc = orig_cudf_profile_info.gpu_calls / (
            orig_cudf_profile_info.gpu_calls + orig_cudf_profile_info.cpu_calls
        )
    if opt_cudf_profile_info.gpu_calls == 0 and opt_cudf_profile_info.cpu_calls == 0:
        gpu_calls_opt_perc = 0
    else:
        gpu_calls_opt_perc = opt_cudf_profile_info.gpu_calls / (
            opt_cudf_profile_info.gpu_calls + opt_cudf_profile_info.cpu_calls
        )
    print(
        f"gpu_calls_orig_perc: {gpu_calls_orig_perc}, gpu_calls_opt_perc: {gpu_calls_opt_perc}"
    )
    print(f"orig_time: {orig_time}, opt_time: {opt_time}")
    # If both the call ratio and the time are better, then the rewritten code is better.
    if gpu_calls_opt_perc >= gpu_calls_orig_perc and opt_time <= orig_time:
        print("Both the call ratio and the time are better.")
        return True

    # If the call ratio is worse and the time is worse, then the rewritten code is better.
    if gpu_calls_opt_perc < gpu_calls_orig_perc and opt_time > orig_time:
        print("Both the call ratio and the time are worse.")
        return False

    speedup = orig_time / opt_time
    slowdown = opt_time / orig_time
    if gpu_calls_opt_perc < gpu_calls_orig_perc:
        print("Ratio drops.")
        # Ratio drops.
        # If the speedup makes up for the drop in call ratio, then the rewritten code is better.
        if speedup >= 1 + alpha * (gpu_calls_orig_perc - gpu_calls_opt_perc):
            print("Speedup makes up for the drop in call ratio.")
            return True
        else:
            print("Speedup does not make up for the drop in call ratio.")
            return False
    else:
        print("Ratio improves.")
        # Ratio improves, but the time is worse.
        # If the improvement in percentage makes up for the slowdown, then the rewritten code is better.
        if slowdown <= 1 + beta * (gpu_calls_opt_perc - gpu_calls_orig_perc):
            print("Improvement in percentage makes up for the slowdown.")
            return True
        else:
            print("Improvement in percentage does not make up for the slowdown.")
            return False


def execute_notebook(notebook: NotebookNode) -> None:
    nb_client = NotebookClient(notebook, allow_errors=False)
    nb_client.execute()
    raise_errors_from_cell_outputs(notebook.cells)


def cost_transfer_cpu_to_gpu(
    df_type: DFType,
    df_size: int,
    num_rows: int,
) -> float:
    """integer	(243.7483 * Memory_GB) + 0.6051
    float	(359.8324 * Memory_GB) + 1.2119
    string	(1774.8175 * Memory_GB) + -0.3635
    object	(1038.4338 * Memory_GB) + 1.1120
    datetime	(328.7391 * Memory_GB) + 1.1654"
    """
    df_size_gb = df_size / (10**9)
    if df_type == DFType.INT64:
        # cost = 243.7483 * df_size_gb + 0.6051
        # cost = (288.6167 * df_size_gb) + 0.7718
        cost = (229.0760 * df_size_gb) + 0.8473
    elif df_type == DFType.INT16:
        # cost = 425.3453 * df_size_gb + 0.3707
        cost = (425.3453 * df_size_gb) + 0.3707
    elif df_type == DFType.FLOAT64:
        # cost = 359.8324 * df_size_gb + 1.2119
        cost = (1185.0288 * df_size_gb) + 0.8123
    elif df_type == DFType.STRING:
        # cost = 1776.9979 * df_size_gb
        cost = (1232.3932 * df_size_gb) + 0.0000706963 * num_rows + 0.1386
    elif df_type == DFType.OBJECT:
        # cost = 1038.4338 * df_size_gb + 1.1120
        cost = (1167.8079 * df_size_gb) + 0.0000262355 * num_rows
    elif df_type == DFType.DATETIME:
        # cost = 328.7391 * df_size_gb + 1.1654
        cost = (360.8108 * df_size_gb) + 0.4300
    elif df_type == DFType.CATEGORICAL:
        cost = 111409.9677 * df_size_gb + 0.6465
    elif df_type == DFType.BOOLEAN:
        cost = (1660.8111 * df_size_gb) + 0.3887
    else:
        raise ValueError(f"Unknown DataFrame type: {df_type}")
    return cost


def cost_transfer_gpu_to_cpu(df_type: DFType, df_size: int, num_rows: int) -> float:
    """
    computes the cost of transferring data column from GPU to CPU and returns the cost in ms.
    """

    df_size_gb = df_size / (10**9)
    if df_type == DFType.INT64:
        # cost = 3834.9758 * df_size_gb + 2.4559
        # cost = 2238.9442 * df_size_gb + 1.0860
        # cost = (3935.8492 * df_size_gb) + 1.9986
        cost = (2026.6461 * df_size_gb) + 1.1384
    elif df_type == DFType.INT16:
        cost = 1775.5872 * df_size_gb + 1.4806
    elif df_type == DFType.FLOAT64:
        cost = 2715.6053 * df_size_gb + 0.0565
        # cost = 3661.7365 * df_size_gb + 3.4888
    elif df_type == DFType.STRING:
        cost = 20708.8545 * df_size_gb + 6.7869
    elif df_type == DFType.OBJECT:
        # cost = 20681.7519 * df_size + 1.8726
        # cost = (21684.9700 * df_size_gb) + 6.7963
        # cost = 2331.6524 * df_size_gb + 0.0003 * num_rows + 3.8415
        cost = 590.0244 * df_size_gb + 0.0000392422 * num_rows + 1.7382
    elif df_type == DFType.DATETIME:
        # cost = 4020.7354 * df_size_gb + 5.0196
        # cost = 3298.6397 * df_size_gb + 3.2079
        cost = 2027.8784 * df_size_gb + 0.2281
    elif df_type == DFType.CATEGORICAL:
        cost = 41575.6405 * df_size_gb + 4.4107
    elif df_type == DFType.BOOLEAN:
        cost = (1259.4013 * df_size_gb) + 1.7150
    else:
        raise ValueError(f"Unknown DataFrame type: {df_type}")
    return cost


def schedule_dp_data_on_cpu(
    cpu_times: list[float],
    gpu_times: list[float],
    transfer_times: list[float],
) -> list[str]:
    # Total number of tasks.
    n = len(cpu_times)
    # minimum total time to reach task i ending on CPU.
    dp_cpu = [0.0] * n
    # minimum total time to reach task i ending on GPU.
    dp_gpu = [0.0] * n
    # where task i came from if ending on CPU — "c" or "g".
    prev_cpu = [""] * n
    # where task i came from if ending on GPU — "c" or "g".
    prev_gpu = [""] * n

    # base case
    dp_cpu[0] = cpu_times[0]
    dp_gpu[0] = gpu_times[0]

    # Assume the data exists on CPU.
    # TODO(jie)

    for i in range(1, n):
        # Stay on CPU.
        stay_on_cpu = dp_cpu[i - 1] + cpu_times[i]
        # Last was GPU, now CPU.
        gpu_to_cpu = dp_gpu[i - 1] + cpu_times[i]
        # Stay on GPU. We still need the transfer times.
        stay_on_gpu = dp_gpu[i - 1] + gpu_times[i] + transfer_times[i]
        # Move to GPU from CPU.
        move_to_gpu_from_cpu = dp_cpu[i - 1] + gpu_times[i] + transfer_times[i]

        if stay_on_cpu <= gpu_to_cpu:
            dp_cpu[i] = stay_on_cpu
            prev_cpu[i] = "c"
        else:
            dp_cpu[i] = gpu_to_cpu
            prev_cpu[i] = "g"

        if stay_on_gpu <= move_to_gpu_from_cpu:
            dp_gpu[i] = stay_on_gpu
            prev_gpu[i] = "g"
        else:
            dp_gpu[i] = move_to_gpu_from_cpu
            prev_gpu[i] = "c"

    dev = "c" if dp_cpu[-1] < dp_gpu[-1] else "g"
    schedule = []
    for i in range(n - 1, -1, -1):
        schedule.append("c" if dev == "c" else "g")
        dev = prev_cpu[i] if dev == "c" else prev_gpu[i]
    schedule.reverse()
    return schedule


def schedule_dp_data_on_gpu(
    cpu_times: list[float],
    gpu_times: list[float],
    transfer_times: list[float],
) -> list[str]:
    # Total number of tasks.
    n = len(cpu_times)
    # minimum total time to reach task i ending on CPU.
    dp_cpu = [0.0] * n
    # minimum total time to reach task i ending on GPU.
    dp_gpu = [0.0] * n
    # where task i came from if ending on CPU — "c" or "g".
    prev_cpu = [""] * n
    # where task i came from if ending on GPU — "c" or "g".
    prev_gpu = [""] * n

    # base case
    dp_cpu[0] = cpu_times[0]
    dp_gpu[0] = gpu_times[0]

    # Assume the data exists on CPU.
    # TODO(jie)

    for i in range(1, n):
        # Stay on CPU.
        stay_on_cpu = dp_cpu[i - 1] + cpu_times[i] + transfer_times[i]
        # Last was GPU, now CPU.
        gpu_to_cpu = dp_gpu[i - 1] + cpu_times[i] + transfer_times[i]
        # Stay on GPU. We still need the transfer times.
        stay_on_gpu = dp_gpu[i - 1] + gpu_times[i]
        # Move to GPU from CPU.
        move_to_gpu_from_cpu = dp_cpu[i - 1] + gpu_times[i]

        if stay_on_cpu <= gpu_to_cpu:
            dp_cpu[i] = stay_on_cpu
            prev_cpu[i] = "c"
        else:
            dp_cpu[i] = gpu_to_cpu
            prev_cpu[i] = "g"

        if stay_on_gpu <= move_to_gpu_from_cpu:
            dp_gpu[i] = stay_on_gpu
            prev_gpu[i] = "g"
        else:
            dp_gpu[i] = move_to_gpu_from_cpu
            prev_gpu[i] = "c"

    dev = "c" if dp_cpu[-1] < dp_gpu[-1] else "g"
    schedule = []
    for i in range(n - 1, -1, -1):
        schedule.append("c" if dev == "c" else "g")
        dev = prev_cpu[i] if dev == "c" else prev_gpu[i]
    schedule.reverse()
    return schedule


def get_cost_model_transfer_times(
    transfers: list[CostModelInput],
    cpu_to_gpu: bool,
) -> list[dict[str, float]]:
    """
    Returns a list of dictionaries, where each dictionary maps column names to transfer times.
    dict[i] is the transfer times for the ith cell.
    """
    all_transfer_times = []
    for transfer in transfers:
        col_transfer_times: dict[str, float] = {}
        num_rows = transfer.rows

        # get the transfer times for each column.
        for col_name, (
            col_type,
            col_size_cpu_to_gpu,
            col_size_gpu_to_cpu,
        ) in transfer.col_data.items():
            assert col_size_cpu_to_gpu is not None
            assert col_size_gpu_to_cpu is not None
            # Prepare the input for the cost model.
            if cpu_to_gpu:
                transfer_time = cost_transfer_cpu_to_gpu(
                    col_type, col_size_cpu_to_gpu, num_rows
                )
            else:
                transfer_time = cost_transfer_gpu_to_cpu(
                    col_type, col_size_gpu_to_cpu, num_rows
                )
            col_transfer_times[col_name] = transfer_time
        all_transfer_times.append(col_transfer_times)
    return all_transfer_times


def make_transfer_cells_and_get_next_index(
    annotated_cell_idx: int,
    transfers: list[CostModelInput],
    cpu_to_gpu: bool,
    next_index: int,
    notebook: NotebookNode,
    pre_exec: bool,
) -> int:
    """Returns the cell index to be executed next."""
    curr_insert_index = next_index - 1
    for transfer in transfers:
        # First, make a new cudf dataframe.
        is_series, df_name, col_names = (
            transfer.is_series,
            transfer.df_name,
            transfer.col_names,
        )
        if is_series:
            if cpu_to_gpu:
                if pre_exec:
                    transfer_code_annotations = f"## Transfer_pre {annotated_cell_idx} ##\ncudf_{df_name} = cudf.from_pandas({df_name})"
                else:
                    transfer_code_annotations = f"## Transfer_post {annotated_cell_idx} ##\ncudf_{df_name} = cudf.from_pandas({df_name})"
                transfer_code = maybe_annotate_code_with_time(transfer_code_annotations)
                transfer_cell = make_code_cell(transfer_code)
                curr_insert_index += 1
                notebook.cells.insert(curr_insert_index, transfer_cell)
            else:
                # TODO(sahil): find out if series are ever placed on the GPU.
                # add a dummy transfer cell for series.
                if pre_exec:
                    transfer_code_annotations = f"## Transfer_pre {annotated_cell_idx} ##\npd_{df_name} = {df_name}"
                else:
                    transfer_code_annotations = f"## Transfer_post {annotated_cell_idx} ##\npd_{df_name} = {df_name}"
                transfer_code = maybe_annotate_code_with_time(transfer_code_annotations)
                transfer_cell = make_code_cell(transfer_code)
                curr_insert_index += 1
                notebook.cells.insert(curr_insert_index, transfer_cell)

            transfer_cell.metadata = {
                "transfer": {
                    "df_name": df_name,
                    "is_series": is_series,
                    "pre_exec": pre_exec,
                    "annotated_cell_idx": annotated_cell_idx,
                }
            }
        else:
            assert col_names is not None
            if cpu_to_gpu:
                # Initialize the new dataframe.
                initialization_code = [
                    "import cudf",
                    f"""
if 'cudf_{df_name}' in globals():
    del cudf_{df_name}
cudf_{df_name} = cudf.DataFrame(index={df_name}.index)
                    """,
                ]
                initialization_cell = make_code_cell(initialization_code)
                curr_insert_index += 1
                notebook.cells.insert(curr_insert_index, initialization_cell)
                transfer_cells = get_transfer_cpu_to_gpu_cells(
                    df_name, col_names, annotated_cell_idx, pre_exec
                )
            else:
                # Initialize the new dataframe.
                initialization_code = f"""
import cudf
if 'pd_{df_name}' in globals():
    del pd_{df_name}
try:
    pd_{df_name} = cudf.DataFrame(index={df_name}.index).to_pandas()
except Exception:
    pd_{df_name} = cudf.DataFrame().to_pandas()
                """
                initialization_cell = make_code_cell(initialization_code)
                curr_insert_index += 1
                notebook.cells.insert(curr_insert_index, initialization_cell)
                transfer_cells = get_transfer_gpu_to_cpu_cells(
                    df_name, col_names, annotated_cell_idx, pre_exec
                )

            for col_name, transfer_cell in zip(col_names, transfer_cells):
                curr_insert_index += 1
                notebook.cells.insert(curr_insert_index, transfer_cell)
                transfer_cell.metadata = {
                    "transfer": {
                        "df_name": df_name,
                        "col_name": col_name,
                        "is_series": is_series,
                        "pre_exec": pre_exec,
                        "annotated_cell_idx": annotated_cell_idx,
                    }
                }

    return curr_insert_index + 1


def run_cell_and_get_all_transfer_inputs(
    curr_cell: NotebookNode,
    info: CellExecInfo,
    shell: InteractiveShell,
    cpu_to_gpu: bool,
    df_size_multiplier: int,
) -> tuple[float, list[CostModelInput], list[CostModelInput]]:
    all_input_transfer_inputs: list[CostModelInput] = []
    # In the case when df is input but not in df_exec_info. In this case, we will just transfer the entire df.
    df_names_in_exec_info = set(
        df_exec_info.df_name for df_exec_info in info.df_exec_infos
    )
    for input_var_info in info.input_vars:
        if (
            is_type_df(input_var_info.type)
            and input_var_info.name not in df_names_in_exec_info
        ):
            df = shell.user_ns[input_var_info.name]
            if is_type_dataframe(input_var_info.type):
                input_rows, input_cols = df.shape
                input_col_names = list(df.columns)
                if cpu_to_gpu:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            get_df_size(df[col]) * df_size_multiplier,
                            None,
                        )
                        for col in input_col_names
                    }
                else:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            None,
                            get_df_size(df[col]) * df_size_multiplier,
                        )
                        for col in input_col_names
                    }
                is_series = False
            else:
                input_rows, input_cols = df.shape[0], 1
                input_col_names = None
                is_series = True
                # FIXME(sahil): check if the name should be None or empty string.
                if cpu_to_gpu:
                    col_data = {
                        "": (
                            get_df_type(df),
                            get_df_size(df) * df_size_multiplier,
                            None,
                        )
                    }
                else:
                    col_data = {
                        "": (
                            get_df_type(df),
                            None,
                            get_df_size(df) * df_size_multiplier,
                        )
                    }
            curr_input = CostModelInput(
                df_name=input_var_info.name,
                rows=input_rows,
                cols=input_cols,
                is_series=is_series,
                col_names=input_col_names,
                col_data=col_data,  # type: ignore
            )
            all_input_transfer_inputs.append(curr_input)

    # Find other dataframes whose columns are used in this cell.
    for df_exec_info in info.df_exec_infos:
        df = shell.user_ns[df_exec_info.df_name]
        input_transfer_cols = df_exec_info.input_transfer_cols
        try:
            input_transfer_cols = {
                col for col in input_transfer_cols if col in df.columns
            }
        except Exception:
            print(f"input_transfer_cols: {input_transfer_cols}")
            print(f"df.columns: {df}")
            raise ValueError(
                f"input_transfer_cols: {input_transfer_cols} is not a subset of df.columns: {df.columns}"
            )
        if len(input_transfer_cols) == 0:
            input_transfer_cols = set(df.columns)
        if input_transfer_cols:
            if is_type_dataframe(type(df)):
                input_rows, input_cols = df.shape[0], len(input_transfer_cols)
                input_transfer_cols = list(input_transfer_cols)
                is_series = False
                if cpu_to_gpu:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            get_df_size(df[col]) * df_size_multiplier,
                            None,
                        )
                        for col in input_transfer_cols
                    }
                else:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            None,
                            get_df_size(df[col]) * df_size_multiplier,
                        )
                        for col in input_transfer_cols
                    }
            else:
                input_rows, input_cols = df.shape[0], 1
                input_transfer_cols = None
                is_series = True
                if cpu_to_gpu:
                    col_data = {
                        "": (
                            get_df_type(df),
                            get_df_size(df) * df_size_multiplier,
                            None,
                        )
                    }
                else:
                    col_data = {
                        "": (
                            get_df_type(df),
                            None,
                            get_df_size(df) * df_size_multiplier,
                        )
                    }
            curr_input = CostModelInput(
                df_name=df_exec_info.df_name,
                rows=input_rows,
                cols=input_cols,
                is_series=is_series,
                col_names=input_transfer_cols,
                col_data=col_data,  # type: ignore
            )
            all_input_transfer_inputs.append(curr_input)

    output = execute_cell(curr_cell, shell)
    exec_time = parse_wall_time_to_ms(output) if output is not None else None
    if exec_time is None:
        raise ValueError(f"Execution time is None for cell {curr_cell.source}")
    all_output_transfer_inputs: list[CostModelInput] = []

    # These are output variables that are used downstream.
    for output_var_info in info.active_vars:
        if (
            is_type_df(output_var_info.type)
            and output_var_info.name not in df_names_in_exec_info
        ):
            try:
                df = shell.user_ns[output_var_info.name]
            except Exception:
                print(f"output_var_info.name: {output_var_info.name}")
                print(f"shell.user_ns: {shell.user_ns}")
                raise ValueError(
                    f"output_var_info.name: {output_var_info.name} is not in shell.user_ns: {shell.user_ns}"
                )
            if is_type_dataframe(output_var_info.type):
                input_rows, input_cols = df.shape
                input_col_names = list(df.columns)
                is_series = False
                # populate col_data
                if cpu_to_gpu:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            get_df_size(df[col]) * df_size_multiplier,
                            None,
                        )
                        for col in input_col_names
                    }
                else:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            None,
                            get_df_size(df[col]) * df_size_multiplier,
                        )
                        for col in input_col_names
                    }
            else:
                input_rows, input_cols = df.shape[0], 1
                input_col_names = None
                is_series = True
                if cpu_to_gpu:
                    col_data = {
                        "": (
                            get_df_type(df),
                            get_df_size(df) * df_size_multiplier,
                            None,
                        )
                    }
                else:
                    col_data = {
                        "": (
                            get_df_type(df),
                            None,
                            get_df_size(df) * df_size_multiplier,
                        )
                    }
            curr_input = CostModelInput(
                df_name=output_var_info.name,
                rows=input_rows,
                cols=input_cols,
                is_series=is_series,
                col_names=input_col_names,
                col_data=col_data,  # type: ignore
            )
            all_output_transfer_inputs.append(curr_input)

    for df_exec_info in info.df_exec_infos:
        df = shell.user_ns[df_exec_info.df_name]
        output_transfer_cols = df_exec_info.output_transfer_cols
        if len(output_transfer_cols) == 0 and len(df_exec_info.deleted_cols) > 0:
            output_transfer_cols = set(df.columns)
        output_transfer_cols = {
            col
            for col in output_transfer_cols
            if col in df.columns and col not in df_exec_info.deleted_cols
        }
        if output_transfer_cols:
            if is_type_dataframe(type(df)):
                output_rows, output_cols = df.shape
                output_transfer_cols = list(output_transfer_cols)
                is_series = False
                # populate col_data
                if cpu_to_gpu:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            get_df_size(df[col]) * df_size_multiplier,
                            None,
                        )
                        for col in output_transfer_cols
                    }
                else:
                    col_data = {
                        col: (
                            get_df_type(df[col]),
                            None,
                            get_df_size(df[col]) * df_size_multiplier,
                        )
                        for col in output_transfer_cols
                    }
            else:
                if cpu_to_gpu:
                    col_data = {
                        "": (
                            get_df_type(df),
                            get_df_size(df) * df_size_multiplier,
                            None,
                        )
                    }
                else:
                    col_data = {
                        "": (
                            get_df_type(df),
                            None,
                            get_df_size(df) * df_size_multiplier,
                        )
                    }
                output_rows, output_cols = df.shape[0], 1
                output_transfer_cols = None
                is_series = True
            curr_input = CostModelInput(
                df_name=df_exec_info.df_name,
                rows=output_rows,
                cols=output_cols,
                is_series=is_series,
                col_names=output_transfer_cols,
                col_data=col_data,  # type: ignore
            )
            all_output_transfer_inputs.append(curr_input)

    return exec_time, all_input_transfer_inputs, all_output_transfer_inputs


def reset_shell(shell: InteractiveShell) -> None:
    # Reset the state, and run the notebook fresh.
    execute_code("%reset -f", shell)
    # We need to reload elastic notebook because we need to reset the dependency graph.
    execute_code("%reload_ext ElasticNotebook", shell)


def raise_errors_from_cell_outputs(cells: list[NotebookNode]) -> None:
    for cell in cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(
                    f"Execution failed in cell {cell.source}: {output.get('evalue')}"
                )


def find_cell_times(notebook: NotebookNode) -> dict[int, float]:
    """Returns a dictionary mapping cell indices to cell times."""
    cell_times: dict[int, float] = {}
    for cell in notebook.cells:
        match = cell_annotation_pattern.search(cell.source)
        if match:
            cell_idx = int(match.group(1))
            exec_time = parse_wall_time_to_ms_from_all_outputs(cell.outputs)
            assert exec_time is not None
            cell_times[cell_idx] = exec_time
    return cell_times
