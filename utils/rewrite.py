# Step 1. Get cudf profile information.

import copy
import time
from pathlib import Path

from elastic.core.common.cell_exec_info import CellExecInfo
from IPython.core.interactiveshell import InteractiveShell
from nbclient import NotebookClient
from nbformat import NotebookNode

from utils.agent_flow import NUM_TRIES_PER_CELL, CodeInfo, call_rewrite_agent
from utils.execution import (
    CudfProfileInfo,
    execute_cell,
    execute_code,
    execute_notebook,
    is_rewritten_code_better,
    parse_cudf_profile_info_from_all_outputs,
    parse_wall_time_to_ms_from_all_outputs,
    raise_errors_from_cell_outputs,
    reset_shell,
)
from utils.notebook import (
    CellData,
    annotate_notebook,
    clear_all_checkpoints,
    get_load_checkpoint_cell,
    get_load_cudf_ext_cell,
    get_load_elastic_notebook_cell,
    get_post_checkpoint_path,
    get_pre_checkpoint_path,
    get_save_checkpoint_cell,
    load_notebook,
    make_code_cell,
    make_notebook,
    maybe_annotate_code_with_cell_index,
    maybe_annotate_code_with_cudf_profile,
    maybe_annotate_code_with_record_event,
    maybe_annotate_code_with_time,
    remove_magic_commands,
    save_notebook,
)
from utils.logging_utils import log_rewrite_timing
from utils.testing import get_test_code_from_cell_exec_info


def checkpoint_and_get_cudf_profile_info(nb_path: Path) -> dict[int, CudfProfileInfo]:
    """
    Returns a dictionary mapping cell indices to CudfProfileInfo objects.
    """
    # Parse the notebook to get the cell indices.
    # For each cell, read the output file and extract the relevant information.
    print("Getting cudf profile information...")
    cudf_nb_path = nb_path.parent / Path("cudf_profile.ipynb")
    cudf_annotated_cell_idx_to_cell_data = annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=cudf_nb_path,
        add_timing_code=False,
        add_record_events=True,
        add_checkpoints=True,
        track_column_info=False,
        add_cudf_profile=True,
        use_gpu=True,
    )
    cudf_nb = load_notebook(cudf_nb_path)
    execute_notebook(cudf_nb)

    cudf_profile_infos: dict[int, CudfProfileInfo] = {}
    for annotated_cell_index, cell_data in cudf_annotated_cell_idx_to_cell_data.items():
        # FIXME(jie): this is a bug. cell_data does not have output. we want to get the outputs from cudf_notebook.
        cudf_profile_info = parse_cudf_profile_info_from_all_outputs(
            cudf_nb.cells[cell_data.cell_idx_in_annotated_notebook].outputs
        )
        if cudf_profile_info is None:
            raise ValueError(
                f"Cudf profile info is None for cell {annotated_cell_index}."
            )
        cudf_profile_infos[annotated_cell_index] = cudf_profile_info
    print("Done getting cudf profile information.")
    return cudf_profile_infos


async def rewrite_notebook(
    benchmark_name: str,
    run_id: str,
    nb_path: Path,
    small_nb_path: Path,
    cudf_profile_infos: dict[int, CudfProfileInfo],
    cell_exec_infos: dict[int, CellExecInfo],
    shell: InteractiveShell,
    # The index of the first annotated cell to rewrite. This is for debugging purposes only.
    start_cell_index: int = 0,
) -> tuple[NotebookNode, dict[int, float]]:
    # Now, we will rewrite the notebook. Note that `nb_path` should not be the original notebook.'

    # For debugging purposes, we will keep track of the cells that have been executed.
    executed_cells: list[NotebookNode] = []
    small_executed_cells: list[NotebookNode] = []
    rewritten_times: dict[int, float] = {}

    # Get the annotated notebook for rewriting.
    annotated_nb_path = nb_path.parent / Path("annotated.ipynb")
    annotated_cell_idx_to_cell_data: dict[int, CellData] = annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=annotated_nb_path,
        add_timing_code=True,
        add_record_events=True,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=True,
    )
    # Get the small annotated notebook.
    small_annotated_nb_path = nb_path.parent / Path("small_annotated.ipynb")
    annotate_notebook(
        original_notebook_path=small_nb_path,
        annotated_notebook_path=small_annotated_nb_path,
        add_timing_code=True,
        add_record_events=True,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=True,
    )
    annotated_nb = load_notebook(annotated_nb_path)
    small_annotated_nb = load_notebook(small_annotated_nb_path)
    rewritten_nb_path = nb_path.parent / Path("rewritten") / Path("o4_mini_high.ipynb")
    small_rewritten_nb_path = (
        nb_path.parent / Path("rewritten") / Path("o4_mini_high_small.ipynb")
    )
    rewritten_nb_cells: list[NotebookNode] = copy.deepcopy(annotated_nb.cells)
    small_rewritten_nb_cells: list[NotebookNode] = copy.deepcopy(
        small_annotated_nb.cells
    )

    # Make sure we are starting with a fresh shell.
    reset_shell(shell)

    # Set the pre-checkpoint paths.
    pre_checkpoint_path = None
    rewritten_pre_checkpoint_path = None
    small_rewritten_pre_checkpoint_path = None

    # Set up the post-checkpoint paths.
    post_checkpoint_path = None
    rewritten_post_checkpoint_path = None
    small_rewritten_post_checkpoint_path = None

    # Now we will iterate over the annotated cells and try to rewrite them.
    for annotated_cell_idx, cell_data in sorted(
        annotated_cell_idx_to_cell_data.items(), key=lambda x: x[0]
    ):
        if annotated_cell_idx < start_cell_index:
            continue

        if annotated_cell_idx == start_cell_index:
            for cell in annotated_nb.cells[: cell_data.cell_idx_in_annotated_notebook]:
                executed_cells.append(cell)
                small_executed_cells.append(cell)

            # Make the pre_checkpoint_path for this cell.
            print(
                f"Executing {len(executed_cells)} cells up to cell {annotated_cell_idx}."
            )
            pre_checkpoint_path = _checkpoint_before_cell(
                annotated_nb_path=annotated_nb_path,
                annotated_cell_idx=annotated_cell_idx,
                executed_cells=executed_cells,
            )
            rewritten_pre_checkpoint_path = pre_checkpoint_path

            small_rewritten_pre_checkpoint_path = _checkpoint_before_cell(
                annotated_nb_path=small_annotated_nb_path,
                annotated_cell_idx=annotated_cell_idx,
                executed_cells=small_executed_cells,
            )

            print("========================================================")
            print("Cell ", annotated_cell_idx)
            print(f"Rewritten pre-checkpoint path: {rewritten_pre_checkpoint_path}")
            print(f"Pre-checkpoint path: {pre_checkpoint_path}")
            print(
                f"Small rewritten pre-checkpoint path: {small_rewritten_pre_checkpoint_path}"
            )
            print("========================================================")

        else:
            # not the first cell to rewrite.
            pre_checkpoint_path = post_checkpoint_path
            rewritten_pre_checkpoint_path = rewritten_post_checkpoint_path
            small_rewritten_pre_checkpoint_path = small_rewritten_post_checkpoint_path

        cell = annotated_nb.cells[cell_data.cell_idx_in_annotated_notebook]

        # Now we will try to rewrite the cell.
        rewrite_start_time = time.time()
        # Get the pre-checkpoint path for the _next_ cell.
        if pre_checkpoint_path is None:
            raise ValueError(
                f"Pre-checkpoint path is None for cell {annotated_cell_idx}"
            )
        if rewritten_pre_checkpoint_path is None:
            raise ValueError(
                f"Rewritten pre-checkpoint path is None for cell {annotated_cell_idx}"
            )
        if small_rewritten_pre_checkpoint_path is None:
            raise ValueError(
                f"Small rewritten pre-checkpoint path is None for cell {annotated_cell_idx}"
            )
        (
            rewritten_time,
            rewritten_code,
            rewritten_post_checkpoint_path,
            small_rewritten_post_checkpoint_path,
            post_checkpoint_path,
        ) = await _rewrite_cell(
            benchmark_name=benchmark_name,
            run_id=run_id,
            cell=cell,
            cell_exec_info=cell_exec_infos[annotated_cell_idx],
            cudf_profile_info=cudf_profile_infos[annotated_cell_idx],
            annotated_cell_idx=annotated_cell_idx,
            pre_checkpoint_path=pre_checkpoint_path,
            rewritten_pre_checkpoint_path=rewritten_pre_checkpoint_path,
            small_rewritten_pre_checkpoint_path=small_rewritten_pre_checkpoint_path,
            nb_path=annotated_nb_path,
            small_nb_path=small_nb_path,
            rewritten_nb_path=rewritten_nb_path,
            small_rewritten_nb_path=small_rewritten_nb_path,
        )
        print("========================================================")
        print("After rewriting cell ", annotated_cell_idx)
        print(f"Rewritten post-checkpoint path: {rewritten_post_checkpoint_path}")
        print(
            f"Small rewritten post-checkpoint path: {small_rewritten_post_checkpoint_path}"
        )
        print("========================================================")
        rewritten_times[annotated_cell_idx] = rewritten_time
        rewrite_end_time = time.time()
        print(
            f"Cell {annotated_cell_idx} rewriting took {rewrite_end_time - rewrite_start_time} seconds."
        )

        cell_index = cell_data.cell_idx_in_annotated_notebook
        if rewritten_code is not None:
            # This means we couldn't rewrite anything.
            rewritten_nb_cells[cell_index] = make_code_cell(rewritten_code)
            small_rewritten_nb_cells[cell_index] = make_code_cell(rewritten_code)
        else:
            rewritten_nb_cells[cell_index] = make_code_cell(
                remove_magic_commands(cell.source)
            )
            small_rewritten_nb_cells[cell_index] = make_code_cell(
                remove_magic_commands(cell.source)
            )

    # Clear all the checkpoints we made.
    clear_all_checkpoints(annotated_nb_path)
    clear_all_checkpoints(small_nb_path)
    rewritten_nb = make_notebook(rewritten_nb_cells)
    small_rewritten_nb = make_notebook(small_rewritten_nb_cells)
    save_notebook(rewritten_nb, rewritten_nb_path)
    save_notebook(small_rewritten_nb, small_rewritten_nb_path)
    print(f"Rewritten notebook saved to {rewritten_nb_path}")
    print(f"Small rewritten notebook saved to {small_rewritten_nb_path}")
    return rewritten_nb, rewritten_times


def _checkpoint_before_cell(
    *,
    annotated_nb_path: Path,
    annotated_cell_idx: int,
    executed_cells: list[NotebookNode],
) -> Path:
    print("========================================================")
    print(f"Checkpointing before runing cell {annotated_cell_idx}.")
    time.time()
    # Write the checkpoint before the annotated cell is run.
    # Check if we have a post last cell checkpoint. If we do, then we will load that checkpoint.
    pre_checkpoint_path = None
    if annotated_cell_idx > 0:
        last_post_checkpoint_path = get_post_checkpoint_path(
            annotated_nb_path, annotated_cell_idx - 1
        )
        if last_post_checkpoint_path.exists():
            pre_checkpoint_path = last_post_checkpoint_path
            print(
                f"No need to checkpoint. Found existing checkpoint at {pre_checkpoint_path}."
            )
    if pre_checkpoint_path is None:
        pre_checkpoint_path = get_pre_checkpoint_path(
            annotated_nb_path, annotated_cell_idx
        )
        pre_checkpoint_cell = get_save_checkpoint_cell(pre_checkpoint_path)
        time.time()
        executed_cells.append(pre_checkpoint_cell)
        execute_notebook(make_notebook(executed_cells))
    return pre_checkpoint_path


def _run_cell(
    *,
    annotated_nb: NotebookNode,
    annotated_cell_idx: int,
    cell_data: CellData,
    executed_cells: list[NotebookNode],
    shell: InteractiveShell,
) -> None:
    # Run this annotated cell in the original notebook.
    print("========================================================")
    print(f"Running cell {annotated_cell_idx} in the original notebook.")

    cell = annotated_nb.cells[cell_data.cell_idx_in_annotated_notebook]
    executed_cells.append(cell)
    execute_code("Out.clear()", shell)
    execute_code(cell.source, shell)

    # We need to save the output of the cell for testing purposes.
    save_output_code = "orig_output = Out.get(1)"
    save_output_code = maybe_annotate_code_with_record_event(save_output_code)
    executed_cells.append(make_code_cell(save_output_code))
    execute_code(save_output_code, shell)


def _checkpoint_after_cell(
    annotated_nb_path: Path,
    annotated_cell_idx: int,
    executed_cells: list[NotebookNode],
    shell: InteractiveShell,
) -> Path:
    # Checkpoint the post execution state.
    print("========================================================")
    print(f"Checkpointing after running cell {annotated_cell_idx}.")
    checkpoint_start_time = time.time()
    post_checkpoint_path = get_post_checkpoint_path(
        annotated_nb_path, annotated_cell_idx
    )
    post_checkpoint_cell = get_save_checkpoint_cell(post_checkpoint_path)
    executed_cells.append(post_checkpoint_cell)
    execute_cell(post_checkpoint_cell, shell)
    checkpoint_end_time = time.time()
    print(f"Checkpointing took {checkpoint_end_time - checkpoint_start_time} seconds.")
    return post_checkpoint_path


# TODO(jie): figure out the type.
async def _rewrite_cell(
    benchmark_name: str,
    run_id: str,
    cell: NotebookNode,
    cell_exec_info: CellExecInfo,
    cudf_profile_info: CudfProfileInfo,
    annotated_cell_idx: int,
    pre_checkpoint_path: Path,
    rewritten_pre_checkpoint_path: Path,
    small_rewritten_pre_checkpoint_path: Path,
    nb_path: Path,
    small_nb_path: Path,
    rewritten_nb_path: Path,
    small_rewritten_nb_path: Path,
) -> tuple[float, str | None, Path, Path, Path]:
    # Now we will try to rewrite the cell.
    rewritten_cell: NotebookNode | None = None
    rewritten_time: float = 0.0
    best_rewritten_code: str | None = None
    best_rewritten_post_checkpoint_path: Path | None = None

    num_tries = 0
    # last_attempt_exec_states_equal = True
    is_correct = False
    rewritten_cell_notebook_save_path = None
    all_rewritten_code_info: list[CodeInfo] = []
    rewritten_code_info = None
    rewritten_cell_cudf_profile_info = None

    # Get the original time.
    print("Getting original timing for cell", annotated_cell_idx)
    original_start_time = time.time()
    # cell 0: load the ElasticNotebook extension.
    load_elastic_notebook_cell = get_load_elastic_notebook_cell()
    # cell 1: load the cudf extension.
    load_cudf_extension_cell = get_load_cudf_ext_cell()
    # cell 2: load the checkpoint before the annotated cell.
    load_checkpoint_cell = get_load_checkpoint_cell(pre_checkpoint_path)
    # cell 3: clear the output.
    clear_output_cell = make_code_cell("Out.clear()")
    # cell 3: the original cell.
    original_cell = cell
    # cell 4: save the output of the original cell.
    save_output_code = maybe_annotate_code_with_record_event("orig_output = Out.get(5)")
    save_output_cell = make_code_cell(save_output_code)
    # cell 5: checkpoint the post-execution state.
    post_checkpoint_path = get_post_checkpoint_path(nb_path, annotated_cell_idx)
    post_checkpoint_cell = get_save_checkpoint_cell(post_checkpoint_path)
    # cell 6: save the output of the original cell.
    original_cell_notebook = make_notebook(
        [
            load_elastic_notebook_cell,
            load_cudf_extension_cell,
            load_checkpoint_cell,
            clear_output_cell,
            original_cell,
            save_output_cell,
            post_checkpoint_cell,
        ]
    )
    # Save the original cell notebook.
    original_cell_notebook_save_path = (
        nb_path.parent / f"original_cell_{annotated_cell_idx}.ipynb"
    )
    save_notebook(original_cell_notebook, original_cell_notebook_save_path)
    # Execute the original cell notebook.
    execute_notebook(original_cell_notebook)
    original_time = parse_wall_time_to_ms_from_all_outputs(
        original_cell_notebook.cells[4]["outputs"]
    )
    if original_time is None:
        raise ValueError(f"Original time is None for cell {annotated_cell_idx}")
    original_end_time = time.time()
    print(f"Original time for cell {annotated_cell_idx} took {original_end_time - original_start_time} seconds.")
    log_rewrite_timing(
        benchmark_name=benchmark_name,
        run_id=run_id,
        cell_idx=annotated_cell_idx,
        try_num=None,
        category="original_execution",
        elapsed_seconds=original_end_time - original_start_time,
    )
    
    # We want to keep track of some info for the best solution. By default, they are the original solution
    # at the beginning.
    best_rewritten_time = original_time
    best_rewritten_cudf_profile_info = cudf_profile_info

    best_rewritten_post_checkpoint_path = post_checkpoint_path
    best_small_rewritten_post_checkpoint_path = get_pre_checkpoint_path(
        small_nb_path, annotated_cell_idx + 1
    )
    print(
        "Setting initially best_rewritten_post_checkpoint_path: ",
        best_rewritten_post_checkpoint_path,
    )
    print(
        "Setting initially best_small_rewritten_post_checkpoint_path: ",
        best_small_rewritten_post_checkpoint_path,
    )

    # TODO(jie): move this to before running the original cell.
    while num_tries < NUM_TRIES_PER_CELL:
        try_start_time = time.time()
        print("========================================================")
        print(f"Rewriting cell {annotated_cell_idx}... Try {num_tries}... ")
        active_vars = [var.name for var in cell_exec_info.active_vars]
        future_vars = [var.name for var in cell_exec_info.future_vars]
        original_cudf_df = cudf_profile_info.df_table

        cleaned_cell_code = remove_magic_commands(original_cell.source)
        original_code_info = CodeInfo(
            code=cleaned_cell_code,
            profiling_info=(
                original_cudf_df.to_string() if original_cudf_df is not None else None
            ),
            active_vars=active_vars,
            future_vars=future_vars,
            execution_time=original_time,
        )
        # FIXME(sahil): once the bug is fixed, remove the hardcoded code.
        rewrite_agent_start_time = time.time()
        rewritten_code = await call_rewrite_agent(
            num_tries=num_tries,
            original_code_info=original_code_info,
            rewritten_code_info=rewritten_code_info,
            benchmark_name=benchmark_name,
            run_id=run_id,
            cell_index=annotated_cell_idx,
        )
        rewrite_agent_end_time = time.time()
        print(
            f"Try {num_tries}: calling rewrite agent for cell {annotated_cell_idx} took {rewrite_agent_end_time - rewrite_agent_start_time} seconds."
        )
        log_rewrite_timing(
            benchmark_name=benchmark_name,
            run_id=run_id,
            cell_idx=annotated_cell_idx,
            try_num=num_tries,
            category="agent_call",
            elapsed_seconds=rewrite_agent_end_time - rewrite_agent_start_time,
        )
        if num_tries == 0 and rewritten_code is None:
            print("There is no need to rewrite the cell.")
            break

        if rewritten_code is None:
            break

        rewritten_code = maybe_annotate_code_with_cell_index(
            rewritten_code, annotated_cell_idx
        )

        print("Executing rewritten code...", rewritten_code)
        # Execute the rewritten cell.
        # cell 0: load the ElasticNotebook extension.
        load_elastic_notebook_cell = get_load_elastic_notebook_cell()
        # cell 1: load the cudf extension.
        load_cudf_extension_cell = get_load_cudf_ext_cell()
        # cell 2: load the checkpoint before the annotated cell.
        load_checkpoint_cell = get_load_checkpoint_cell(rewritten_pre_checkpoint_path)
        # cell 3: the rewritten cell.
        rewritten_cell = make_code_cell(
            maybe_annotate_code_with_record_event(
                maybe_annotate_code_with_time(rewritten_code)
            )
        )
        # cell 4: checkpoint the post-execution state.
        rewritten_post_checkpoint_path = get_post_checkpoint_path(
            rewritten_nb_path, annotated_cell_idx, try_num=num_tries
        )
        rewritten_post_checkpoint_cell = get_save_checkpoint_cell(
            rewritten_post_checkpoint_path
        )
        # cell 5: save the cell exec info.
        save_cell_exec_info_cell = make_code_cell("%PrintCellInfo opt_cell_exec_info")
        # cell 6: dump the cell exec info to json.
        opt_cell_exec_info_pkl_path = (
            nb_path.parent
            / f"opt_cell_exec_info_{annotated_cell_idx}_try_{num_tries}.pkl"
        )
        dump_cell_exec_info_cell = make_code_cell(
            f"""
with open("{opt_cell_exec_info_pkl_path}", "wb") as f:
    pickle.dump(opt_cell_exec_info[{annotated_cell_idx}], f)
"""
        )
        # cell 7: save the output of the rewritten cell.
        save_output_cell = make_code_cell("opt_output = Out.get(4)")

        # Save the rewritten cell notebook.
        rewritten_cell_notebook_save_path = (
            nb_path.parent
            / f"new_rewrite_cell_{annotated_cell_idx}_try_{num_tries}.ipynb"
        )
        rewritten_cell_notebook = make_notebook(
            [
                load_elastic_notebook_cell,
                load_cudf_extension_cell,
                load_checkpoint_cell,
                rewritten_cell,
                rewritten_post_checkpoint_cell,
                save_cell_exec_info_cell,
                dump_cell_exec_info_cell,
                save_output_cell,
            ]
        )
        # TODO(jie): this is for debugging. We can remove it later.
        save_notebook(rewritten_cell_notebook, rewritten_cell_notebook_save_path)

        # Run the new notebook.
        new_nb_client = NotebookClient(rewritten_cell_notebook, allow_errors=False)
        rewritten_time = float("inf")

        # We have to go through the cell outputs to see if there are any errors. This is
        # because once a cell gets annotated with %%time, the error thrown in that cell doesn't stop
        # the notebook from executing.
        print("Running rewritten cell notebook")
        rewrite_execution_start_time = time.time()
        try:
            with new_nb_client.setup_kernel():
                index = 0
                while index < len(rewritten_cell_notebook.cells):
                    cell = rewritten_cell_notebook.cells[index]
                    if cell.cell_type != "code":
                        index += 1
                        continue  # Skip non-code cells

                    # Execute the current cell
                    new_nb_client.execute_cell(cell, index)

                    # We just dumped the json cell exec info.
                    if index == 6:
                        test_code = get_test_code_from_cell_exec_info(
                            cell_exec_info,
                            opt_cell_exec_info_pkl_path,
                            post_checkpoint_path,
                        )
                        test_cell = make_code_cell(test_code)
                        rewritten_cell_notebook.cells.append(test_cell)
                        save_notebook(
                            rewritten_cell_notebook, rewritten_cell_notebook_save_path
                        )
                    index += 1

                    # Check if the outputs have any errors.
                    raise_errors_from_cell_outputs([cell])

            # If we reach this point, then we know that the rewritten cell has already passed the test.
            # We find the execution time.
            curr_rewritten_time = parse_wall_time_to_ms_from_all_outputs(
                rewritten_cell_notebook.cells[3]["outputs"]
            )
            if curr_rewritten_time is None:
                raise ValueError(
                    f"Rewritten time is None for cell {annotated_cell_idx}"
                )
            rewritten_time = curr_rewritten_time
            execution_output = (
                f"Rewritten time: {rewritten_time} ms\n"
                f"Original time: {original_time} ms"
            )
            is_correct = True
            print("Finished executing the rewritten code.")
            rewrite_execution_end_time = time.time()
            print(
                f"Try {num_tries}: executing the rewritten code for cell {annotated_cell_idx} took {rewrite_execution_end_time - rewrite_execution_start_time} seconds."
            )
            log_rewrite_timing(
                benchmark_name=benchmark_name,
                run_id=run_id,
                cell_idx=annotated_cell_idx,
                try_num=num_tries,
                category="rewrite_execution",
                elapsed_seconds=rewrite_execution_end_time - rewrite_execution_start_time,
            )
        except Exception as e:
            if "AssertionError" in str(e):
                execution_output = (
                    f"The rewritten code was not equivalent to the original code. {e}"
                )
                print(f"Test failed. {e}")
            else:
                execution_output = (
                    f"The rewritten code resulted in a runtime error. {e}"
                )
                print(f"An unexpected error occurred {e}")
            is_correct = False
        finally:
            # dump the rewritten cell notebook.
            rewritten_cell_notebook_save_path = (
                nb_path.parent
                / f"rewrite_cell_{annotated_cell_idx}_try_{num_tries}.ipynb"
            )
            save_notebook(rewritten_cell_notebook, rewritten_cell_notebook_save_path)

        print(f"Rewritten time: {rewritten_time} ms")

        print("Checkpointing the rewritten cell for small notebook...")
        small_rewritten_post_checkpoint_path = get_post_checkpoint_path(
            small_rewritten_nb_path, annotated_cell_idx, try_num=num_tries
        )
        load_small_checkpoint_cell = get_load_checkpoint_cell(
            small_rewritten_pre_checkpoint_path
        )
        small_rewritten_post_checkpoint_cell = get_save_checkpoint_cell(
            small_rewritten_post_checkpoint_path
        )
        small_rewritten_cell_notebook = make_notebook(
            [
                load_elastic_notebook_cell,
                load_cudf_extension_cell,
                load_small_checkpoint_cell,
                rewritten_cell,
                small_rewritten_post_checkpoint_cell,
            ]
        )

        if is_correct:
            # Save the small rewritten cell notebook.
            small_rewritten_cell_notebook_save_path = (
                small_nb_path.parent
                / f"small_rewrite_cell_{annotated_cell_idx}_try_{num_tries}.ipynb"
            )
            save_notebook(
                small_rewritten_cell_notebook, small_rewritten_cell_notebook_save_path
            )
            execute_notebook(small_rewritten_cell_notebook)

            print("The rewritten code is correct. Profiling cudf...")
            # cell 0: load the ElasticNotebook extension.
            load_elastic_notebook_cell = get_load_elastic_notebook_cell()
            # cell 1: load the cudf extension.
            load_cudf_extension_cell = get_load_cudf_ext_cell()
            # cell 2: load the checkpoint before the annotated cell. Note this is the checkpoint for the small notebook.
            load_checkpoint_cell = get_load_checkpoint_cell(
                small_rewritten_pre_checkpoint_path
            )
            # cell 3: cudf profile the rewritten code.
            cudf_cell = make_code_cell(
                maybe_annotate_code_with_cudf_profile(rewritten_code)
            )
            # Save the new cell notebook for profiling
            rewritten_cell_profile_notebook_save_path = (
                nb_path.parent
                / f"rewrite_cell_{annotated_cell_idx}_try_{num_tries}_profile.ipynb"
            )
            rewritten_cell_profile_notebook = make_notebook(
                [
                    load_elastic_notebook_cell,
                    load_cudf_extension_cell,
                    load_checkpoint_cell,
                    cudf_cell,
                ]
            )
            save_notebook(
                rewritten_cell_profile_notebook,
                rewritten_cell_profile_notebook_save_path,
            )
            cudf_profile_start_time = time.time()
            execute_notebook(rewritten_cell_profile_notebook)
            cudf_profile_end_time = time.time()
            print(
                f"Try {num_tries}: cudf profiling the rewritten code for cell {annotated_cell_idx} took {cudf_profile_end_time - cudf_profile_start_time} seconds."
            )
            log_rewrite_timing(
                benchmark_name=benchmark_name,
                run_id=run_id,
                cell_idx=annotated_cell_idx,
                try_num=num_tries,
                category="cudf_profiling",
                elapsed_seconds=cudf_profile_end_time - cudf_profile_start_time,
            )

            rewritten_cell_cudf_profile_info = parse_cudf_profile_info_from_all_outputs(
                rewritten_cell_profile_notebook.cells[-1].outputs
            )
            if (
                best_rewritten_cudf_profile_info is None
                or rewritten_cell_cudf_profile_info is None
            ):
                raise ValueError(
                    f"CudfProfileInfo is None for cell {annotated_cell_idx}"
                )
            accepted_rewritten_code = is_rewritten_code_better(
                orig_cudf_profile_info=best_rewritten_cudf_profile_info,
                opt_cudf_profile_info=rewritten_cell_cudf_profile_info,
                orig_time=best_rewritten_time,
                opt_time=rewritten_time,
            )
            print(
                f"Rewritten code took {rewritten_time}ms. The previous fastest solution took {original_time}ms."
            )
            print(
                "Previous best solution num GPU and CPU calls: ",
                best_rewritten_cudf_profile_info.gpu_calls,
                best_rewritten_cudf_profile_info.cpu_calls,
            )
            print(
                "New solution num GPU and CPU calls: ",
                rewritten_cell_cudf_profile_info.gpu_calls,
                rewritten_cell_cudf_profile_info.cpu_calls,
            )
            if accepted_rewritten_code:
                print("Accepted the rewritten code.")
                best_rewritten_code = rewritten_code
                print(
                    "Setting best_rewritten_post_checkpoint_path to ",
                    rewritten_post_checkpoint_path,
                )
                best_rewritten_post_checkpoint_path = rewritten_post_checkpoint_path
                print(
                    "Setting best_small_rewritten_post_checkpoint_path to ",
                    small_rewritten_post_checkpoint_path,
                )
                best_small_rewritten_post_checkpoint_path = (
                    small_rewritten_post_checkpoint_path
                )
                best_rewritten_time = rewritten_time
                best_rewritten_cudf_profile_info = rewritten_cell_cudf_profile_info

        cudf_df = (
            rewritten_cell_cudf_profile_info.df_table
            if rewritten_cell_cudf_profile_info is not None
            else None
        )
        rewritten_code_info = CodeInfo(
            code=rewritten_code,
            profiling_info=cudf_df.to_string() if cudf_df is not None else None,
            active_vars=active_vars,
            future_vars=future_vars,
            execution_time=rewritten_time,
            execution_output=execution_output,
        )
        all_rewritten_code_info.append(rewritten_code_info)
        try_end_time = time.time()
        log_rewrite_timing(
            benchmark_name=benchmark_name,
            run_id=run_id,
            cell_idx=annotated_cell_idx,
            try_num=num_tries,
            category="total",
            elapsed_seconds=try_end_time - try_start_time,
        )
        num_tries += 1

    if best_rewritten_code is None:
        print("Could not rewrite the cell. Continuing to next cell...")

    return (
        best_rewritten_time,
        best_rewritten_code,
        best_rewritten_post_checkpoint_path,
        best_small_rewritten_post_checkpoint_path,
        post_checkpoint_path,
    )
