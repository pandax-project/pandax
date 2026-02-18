import copy
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nbformat
from nbformat import NotebookNode

cell_annotation_pattern = re.compile(r"### cell (\d+) ###")
transfer_annotation_pattern = re.compile(r"## Transfer_(pre|post) (\d+) ##")


def is_code_cell(cell: NotebookNode) -> bool:
    """Whether a cell is a code cell."""
    return cell.cell_type == "code"


def is_pandas_cell(cell: NotebookNode) -> bool:
    """Whether a cell has pd calls."""
    if not is_code_cell(cell):
        return False
    pandas_pattern = re.compile(r"\b(pd\.\w+)")
    return code_has_pattern(cell.source, pandas_pattern)


def is_executable_cell(cell: NotebookNode) -> bool:
    """Whether a cell has executable code."""
    if not is_code_cell(cell):
        return False
    return is_executable_code(cell.source)


def is_executable_code(code: str | list[str]) -> bool:
    if isinstance(code, list):
        code = "\n".join(code)
    non_comment_pattern = re.compile(r"^(?!\s*#).+", re.MULTILINE)
    return code_has_pattern(code, non_comment_pattern)


def make_code_cell(code: str | list[str]) -> nbformat.NotebookNode:
    cell_id = str(uuid.uuid4())
    if isinstance(code, list):
        code = "\n".join(code)
    return nbformat.v4.new_code_cell(code, id=cell_id)


def code_has_pattern(
    code: str,
    pattern: re.Pattern[str],
) -> bool:
    """Whether a code string contains a pattern."""
    return pattern.search(code) is not None


def maybe_annotate_code_with_cell_index(
    code: str | list[str], annotated_cell_idx: int
) -> str:
    if isinstance(code, list):
        code = "\n".join(code)
    annotation = f"### cell {annotated_cell_idx} ###\n"
    if not code_has_pattern(code, cell_annotation_pattern):
        return "\n".join([annotation, code])
    return code


def maybe_annotate_code_with_time(
    code: str | list[str],
) -> str:
    if isinstance(code, list):
        code = "\n".join(code)
    if code.strip().startswith("%"):
        return code
    time_magic_command = "%%time"
    return "\n".join([time_magic_command, code])


def maybe_annotate_code_with_record_event(
    code: str | list[str], track_column_info: bool = False
) -> str:
    if isinstance(code, list):
        code = "\n".join(code)
    if not is_executable_code(code):
        return code
    # If we are loading the extension, we cannot annotate with record events.
    if code_has_pattern(code, re.compile(re.escape("%load_ext ElasticNotebook"))):
        return code
    if track_column_info:
        record_event_magic_command = "%%RecordEventWithColumnInfo"
    else:
        record_event_magic_command = "%%RecordEvent"
    return "\n".join([record_event_magic_command, code])


def maybe_annotate_code_with_cudf_profile(code: str | list[str]) -> str:
    if isinstance(code, list):
        code = "\n".join(code)
    if not is_executable_code(code):
        return code
    if not code_has_pattern(code, cell_annotation_pattern):
        return code
    return "\n".join(["%%cudf.pandas.profile", code])


def maybe_annotate_code_with_cpu_profile(code: str | list[str]) -> str:
    if isinstance(code, list):
        code = "\n".join(code)
    if not is_executable_code(code):
        return code
    if not code_has_pattern(code, cell_annotation_pattern):
        return code
    return "\n".join(["%%PandasProfile", code])


def ensure_cell_ids(notebook: NotebookNode) -> None:
    """Ensure every cell in the notebook has a unique ID."""
    for cell in notebook.cells:
        if "id" not in cell:
            cell["id"] = str(uuid.uuid4())


def clear_all_checkpoints(notebook_path: Path) -> None:
    checkpoint_dir = notebook_path.parent / notebook_path.stem / "checkpoints"
    for checkpoint_file in checkpoint_dir.glob("*.pickle"):
        checkpoint_file.unlink()


def get_pre_checkpoint_path(notebook_path: Path, annotated_cell_idx: int) -> Path:
    checkpoint_dir = notebook_path.parent / notebook_path.stem / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir.resolve() / f"pre_cell_{annotated_cell_idx}.pickle"


def get_post_checkpoint_path(
    notebook_path: Path,
    annotated_cell_idx: int,
    try_num: Optional[int] = None,
) -> Path:
    checkpoint_dir = notebook_path.parent / notebook_path.stem / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if try_num is None:
        return checkpoint_dir.resolve() / f"post_cell_{annotated_cell_idx}.pickle"
    else:
        return (
            checkpoint_dir.resolve()
            / f"post_cell_{annotated_cell_idx}_try_{try_num}.pickle"
        )


def load_notebook(notebook_path: Path) -> NotebookNode:
    """Load a Jupyter notebook from path."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    ensure_cell_ids(notebook)
    return notebook


def save_notebook(notebook: NotebookNode, output_path: Path) -> None:
    """Save a notebook to the specified path."""
    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nbformat.validator.normalize(notebook)  # Ensure cell IDs are properly assigned
    nbformat.validate(notebook)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def make_notebook(cells: list[NotebookNode]) -> NotebookNode:
    new_notebook = nbformat.v4.new_notebook()
    new_notebook.cells = cells
    return new_notebook


def get_load_elastic_notebook_cell() -> NotebookNode:
    code = [
        "import sys, os",
        "%load_ext ElasticNotebook",
        "from elastic.core.common.pandas import compare_df, convert_col",
        "import pickle",
    ]
    if os.getenv("USE_GPU") == "True":
        code.append("import cudf")
    return make_code_cell(code)


def make_dummy_transfer_cell(cpu_to_gpu: bool) -> NotebookNode:
    if cpu_to_gpu:
        return make_code_cell(
            [
                "%%time",
                "import numpy as np",
                "import pandas as pd",
                "import cudf",
                'dummy_i64 = pd.Series([0], dtype="int64")   # 1-row int64 Series',
                "cudf.Series(dummy_i64)                      # triggers int64 kernels & pool alloc",
                'dummy_float64 = pd.Series([0], dtype="float64")   # 1-row int64 Series',
                "cudf.Series(dummy_float64)                      # triggers int64 kernels & pool alloc",
                "dummy_str = pd.Series(['warm-up'], dtype='object')   # <— 1-row object column",
                "cudf.Series(dummy_str)                               # triggers string-kernels",
            ]
        )
    else:
        return make_code_cell(
            [
                # — cuDF Series → pandas Series (GPU → CPU)
                "import cudf",
                'gpu_i64   = cudf.Series([0],   dtype="int64")    # 1-row int64 on GPU',
                "host_i64  = gpu_i64.to_pandas()                  # triggers GPU→CPU copy",
                'gpu_f64   = cudf.Series([0.0], dtype="float64")  # 1-row float64 on GPU',
                "host_f64  = gpu_f64.to_pandas()                  # triggers GPU→CPU copy",
                "gpu_str   = cudf.Series(['warm-up'], dtype=\"object\")  # 1-row string on GPU",
                "host_str  = gpu_str.to_pandas()                  # triggers GPU→CPU copy",
            ]
        )


def get_load_cudf_ext_cell() -> NotebookNode:
    return make_code_cell("%load_ext cudf.pandas")


def get_load_checkpoint_cell(checkpoint_path: Path) -> NotebookNode:
    return make_code_cell(f"%LoadCheckpoint {checkpoint_path}")


def get_save_checkpoint_cell(checkpoint_path: Path) -> NotebookNode:
    return make_code_cell(f"%Checkpoint {checkpoint_path}")


def get_transfer_cpu_to_gpu_cells(
    df_name: str, col_names: list[str], curr_insert_index: int, pre_exec: bool
) -> list[NotebookNode]:
    transfer_cells: list[NotebookNode] = []
    for col_name in col_names:
        if pre_exec:
            transfer_code_annotations = f"## Transfer_pre {curr_insert_index} ##\n"
        else:
            transfer_code_annotations = f"## Transfer_post {curr_insert_index} ##\n"
        # Escape the single quotes in the column name.
        col_name = col_name.replace('"', '\\"')
        lines = [
            # "%%time",
            # transfer_code_annotations,
            # f"cudf_{df_name}['{col_name}'] = {df_name}['{col_name}']"
            "%%time",
            "from elastic.core.common.pandas import convert_col",
            transfer_code_annotations,
            "import cudf",
            f'cudf_{df_name}["""{col_name}"""] = convert_col({df_name}["""{col_name}"""]).values',
            f'del cudf_{df_name}["""{col_name}"""]',
            # "from pandas.api.types import is_numeric_dtype, is_object_dtype, infer_dtype",
            # "",
            # f"if is_object_dtype({df_name}['{col_name}']) and infer_dtype({df_name}['{col_name}']) == 'floating':",
            # f"  cudf_{df_name}['{col_name}'] = pd.to_numeric({df_name}['{col_name}'], errors='coerce').astype('float64')",
            # "else:",
            # f"  cudf_{df_name}['{col_name}'] = {df_name}['{col_name}']",
        ]
        transfer_code = "\n".join(lines)

        transfer_cells.append(make_code_cell(transfer_code))
    return transfer_cells


def get_transfer_gpu_to_cpu_cells(
    df_name: str, col_names: list[str], curr_insert_index: int, pre_exec: bool
) -> list[NotebookNode]:
    transfer_cells: list[NotebookNode] = []
    for col_name in col_names:
        if pre_exec:
            transfer_code_annotations = f"## Transfer_pre {curr_insert_index} ##\n"
        else:
            transfer_code_annotations = f"## Transfer_post {curr_insert_index} ##\n"
        col_name = col_name.replace('"', '\\"')
        code = [
            f'{transfer_code_annotations}pd_{df_name}["""{col_name}"""] = {df_name}["""{col_name}"""].to_numpy()',
            f'del pd_{df_name}["""{col_name}"""]',
        ]
        transfer_code = maybe_annotate_code_with_time(code)
        transfer_cells.append(make_code_cell(transfer_code))
    return transfer_cells


@dataclass
class CellData:
    cell_idx_in_annotated_notebook: int
    cell_idx_in_original_notebook: int


def annotate_notebook(
    original_notebook_path: Path,
    annotated_notebook_path: Path | None = None,
    add_timing_code: bool = True,
    add_record_events: bool = True,
    add_checkpoints: bool = True,
    track_column_info: bool = False,
    add_cudf_profile: bool = False,
    add_cpu_profile: bool = False,
    use_gpu: bool = False,
) -> dict[int, CellData]:
    """
    Reads a Jupyter Notebook, for cells after the first pandas statement that have executable code, annotate with cell number and add code to record timings.

    Parameters:
        notebook_path (str): Path to the input .ipynb notebook file.
        output_path (str): Path to save the annotated notebook.
    """
    notebook = load_notebook(original_notebook_path)

    annotated_cell_idx = 0
    first_pandas_cell_idx = None
    notebook_cells: list[NotebookNode] = notebook.cells

    # First we want to find the index of the first pandas cell.
    for cell_idx, cell in enumerate(notebook_cells):
        if is_pandas_cell(cell) and is_executable_cell(cell):
            first_pandas_cell_idx = cell_idx
            break
    if first_pandas_cell_idx is None:
        raise RuntimeError("No pandas cell found in the notebook.")

    # Next we want to find the index of the last pandas cell.
    for cell_idx, cell in enumerate(notebook.cells):
        if is_executable_cell(cell) and cell_idx >= first_pandas_cell_idx:
            last_executable_cell_idx = cell_idx
    if last_executable_cell_idx is None:
        raise RuntimeError("No pandas cell found in the notebook.")

    new_notebook_cells: list[NotebookNode] = []
    # In the new notebook, map the annotated cell indices to a CellData object.
    annotated_cell_idx_to_cell_data: dict[int, CellData] = {}
    if use_gpu:
        new_notebook_cells.append(get_load_cudf_ext_cell())
    # Add a new cell that loads the ElasticNotebook extension.
    if add_record_events or add_checkpoints or track_column_info:
        new_notebook_cells.append(get_load_elastic_notebook_cell())

    for cell_idx, cell in enumerate(notebook_cells):
        # Find the first pandas cell.
        # Also annotate the cells with `### cell n ###` if they are not already annotated.
        if not is_executable_cell(cell):
            new_notebook_cells.append(cell)
        elif cell_idx < first_pandas_cell_idx:
            code = cell.source
            if add_cudf_profile:
                code = maybe_annotate_code_with_cudf_profile(code)
            if add_record_events:
                code = maybe_annotate_code_with_record_event(code, track_column_info)
            if add_cpu_profile:
                code = maybe_annotate_code_with_cpu_profile(code)

            # TODO(jie): remove this
            if add_record_events:
                code = maybe_annotate_code_with_record_event(code, track_column_info)
            cell = make_code_cell(code)
            new_notebook_cells.append(cell)
        elif cell_idx >= first_pandas_cell_idx:
            if add_checkpoints:
                pre_checkpoint_path = get_pre_checkpoint_path(
                    original_notebook_path, annotated_cell_idx
                )
                checkpoint_cell = get_save_checkpoint_cell(pre_checkpoint_path)
                new_notebook_cells.append(checkpoint_cell)
            # For non-last cells, add a checkpoint before the cell.
            code = cell.source
            code = maybe_annotate_code_with_cell_index(code, annotated_cell_idx)
            if add_cudf_profile:
                code = maybe_annotate_code_with_cudf_profile(code)
            if add_cpu_profile:
                code = maybe_annotate_code_with_cpu_profile(code)
            if add_timing_code:
                code = maybe_annotate_code_with_time(code)
            if add_record_events:
                code = maybe_annotate_code_with_record_event(code, track_column_info)

            annotated_cell_idx_to_cell_data[annotated_cell_idx] = CellData(
                cell_idx_in_annotated_notebook=len(new_notebook_cells),
                cell_idx_in_original_notebook=cell_idx,
            )
            new_notebook_cells.append(make_code_cell(code))
            annotated_cell_idx += 1
        else:
            # We are not supposed to get into this state at all.
            raise RuntimeError("Unexpected state.")

    # Save the new notebook.
    if annotated_notebook_path:
        new_notebook = make_notebook(new_notebook_cells)
        save_notebook(new_notebook, annotated_notebook_path)
        print(f"Annotated notebook saved to {annotated_notebook_path}")

    return annotated_cell_idx_to_cell_data


def remove_magic_commands(code: str) -> str:
    """Remove magic commands from code string."""
    magic_command_pattern = re.compile(r"^(\s*%.*\n)", re.MULTILINE)
    return re.sub(magic_command_pattern, "", code)


def replace_factor(notebook: NotebookNode, factor: int) -> NotebookNode:
    cells: list[NotebookNode] = []
    for cell in notebook.cells:
        if is_code_cell(cell):
            code = cell.source
            code = update_factor_in_cell(code, "factor", factor)
            cell.source = code
        cells.append(copy.deepcopy(cell))
    return make_notebook(cells)


def update_factor_in_cell(
    cell_source: str, factor_var: str, new_value: int | float
) -> str:
    lines = cell_source.splitlines()
    pattern = re.compile(rf"^\s*{factor_var}\s*=\s*[\d.]+(\s*#.*)?$")
    new_lines = []
    replaced = False

    for line in lines:
        if pattern.match(line) and not replaced:
            # Preserve any inline comment
            comment = line.split("#", 1)[-1] if "#" in line else ""
            newline = f"{factor_var} = {new_value}"
            if comment:
                newline += f"  # {comment.strip()}"
            new_lines.append(newline)
            replaced = True
            print(f"Replaced factor in cell with {new_value}")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def _clean_up_cell(cell: NotebookNode):
    """Remove unwanted lines from a code cell."""
    if not is_code_cell(cell):
        return

    # Ensure source is a list of lines
    if isinstance(cell.source, str):
        lines = cell.source.splitlines()
    else:
        lines = cell.source

    # Filter out unwanted lines
    cleaned_lines = [line for line in lines if not line.strip().startswith("%%")]

    # Join back to string
    cell.source = "\n".join(cleaned_lines)


def clean_up_rewritten_notebook(nb_path: str, out_path: str = None):
    """
    Clean up a rewritten notebook by:
    - removing unwanted annotations in each cell,
    - and removing the elastic import cell at the top of the notebook.

    Args:
        nb_path: path to the notebook to clean
        out_path: path to save cleaned notebook; if None, overwrite original
    """
    nb = nbformat.read(nb_path, as_version=4)

    # Remove the first cell (elastic import)
    nb.cells = nb.cells[1:]

    # Clean up all remaining cells
    for cell in nb.cells:
        _clean_up_cell(cell)

    if out_path is None:
        out_path = nb_path

    nbformat.write(nb, out_path)
    print(f"Notebook cleaned and saved to {out_path}")
