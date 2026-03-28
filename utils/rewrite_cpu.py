import copy
import time
from pathlib import Path

from elastic.core.common.cell_exec_info import CellExecInfo
from IPython.core.interactiveshell import InteractiveShell
from nbclient import NotebookClient
from nbformat import NotebookNode

from utils.agent_flow import CodeInfo
from utils.agent_flow_cpu import call_rewrite_agent_cpu
from utils.execution import (
    CpuProfileInfo,
    execute_cell,
    execute_notebook,
    parse_cpu_profile_info_from_all_outputs,
    parse_wall_time_to_ms_from_all_outputs,
    raise_errors_from_cell_outputs,
    reset_shell,
)
from utils.notebook import (
    CellData,
    annotate_notebook,
    get_load_checkpoint_cell,
    get_load_elastic_notebook_cell,
    get_post_checkpoint_path,
    get_pre_checkpoint_path,
    get_save_checkpoint_cell,
    load_notebook,
    make_code_cell,
    make_notebook,
    maybe_annotate_code_with_cell_index,
    maybe_annotate_code_with_cpu_profile,
    maybe_annotate_code_with_record_event,
    maybe_annotate_code_with_time,
    remove_magic_commands,
    save_notebook,
)
from utils.testing import get_test_code_from_cell_exec_info


def checkpoint_and_get_cpu_profile_info(nb_path: Path) -> dict[int, CpuProfileInfo]:
    """
    Returns a dictionary mapping cell indices to CpuProfileInfo objects.
    """
    # Parse the notebook to get the cell indices.
    # For each cell, read the output file and extract the relevant information.
    print("Getting cpu profile information...")
    cpu_nb_path = nb_path.parent / Path("cpu_profile.ipynb")
    cpu_annotated_cell_idx_to_cell_data = annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=cpu_nb_path,
        add_timing_code=False,
        add_record_events=True,
        add_checkpoints=True,
        track_column_info=False,
        use_gpu=False,
        add_cpu_profile=True,
    )
    cpu_nb = load_notebook(cpu_nb_path)
    execute_notebook(cpu_nb)

    cpu_profile_infos: dict[int, CpuProfileInfo] = {}
    for annotated_cell_index, cell_data in cpu_annotated_cell_idx_to_cell_data.items():
        cpu_profile_info = parse_cpu_profile_info_from_all_outputs(
            cpu_nb.cells[cell_data.cell_idx_in_annotated_notebook].outputs
        )
        if cpu_profile_info is None:
            raise ValueError(
                f"Cpu profile info is None for cell {annotated_cell_index}."
            )
        cpu_profile_infos[annotated_cell_index] = cpu_profile_info
    print("Done getting cpu profile information.")
    return cpu_profile_infos


# def is_magic_only_cell(cell):
#     """
#     Returns True if the code cell contains only one or more `%%` cell magic lines
#     followed by nothing or just whitespace.
#     """
#     if cell.cell_type != "code":
#         return False

#     lines = cell.source.strip().splitlines()
#     if not lines:
#         return True  # completely empty

#     # Count leading %% lines
#     i = 0
#     while i < len(lines) and lines[i].strip().startswith("%%"):
#         i += 1

#     # If everything after magic lines is empty/whitespace → invalid
#     body_lines = lines[i:]
#     return all(not line.strip() for line in body_lines)


async def rewrite_notebook_cpu(
    nb_path: Path,
    small_nb_path: Path,
    cpu_profile_infos: dict[int, CpuProfileInfo],
    cell_exec_infos: dict[int, CellExecInfo],
    shell: InteractiveShell,
    # The index of the first annotated cell to rewrite. This is for debugging purposes only.
    start_cell_index: int = 0,
    num_tries_per_cell: int = 5,
) -> tuple[NotebookNode, dict[int, float]]:
    # For debugging purposes, we will keep track of the cells that have been executed.
    executed_cells: list[NotebookNode] = []
    original_execution_times: dict[int, float] = {}
    rewritten_execution_times: dict[int, float] = {}

    # Get the annotated notebook for rewriting.
    annotated_nb_path = nb_path.parent / Path("annotated_cpu.ipynb")
    annotated_cell_idx_to_cell_data: dict[int, CellData] = annotate_notebook(
        original_notebook_path=nb_path,
        annotated_notebook_path=annotated_nb_path,
        add_timing_code=True,
        add_record_events=True,
        add_checkpoints=False,
        track_column_info=False,
        use_gpu=False,
    )
    annotated_nb = load_notebook(annotated_nb_path)
    rewritten_nb_path = (
        nb_path.parent / Path("rewritten_cpu") / Path("o4_mini_high.ipynb")
    )
    small_rewritten_nb_path = (
        nb_path.parent / Path("rewritten_cpu") / Path("o4_mini_high.ipynb")
    )
    rewritten_nb_cells: list[NotebookNode] = copy.deepcopy(annotated_nb.cells)

    # Make sure we are starting with a fresh shell.
    reset_shell(shell)

    # Now we will iterate over the annotated cells and try to rewrite them.
    for annotated_cell_idx, cell_data in sorted(
        annotated_cell_idx_to_cell_data.items(), key=lambda x: x[0]
    ):
        if annotated_cell_idx < start_cell_index:
            continue

        if annotated_cell_idx == start_cell_index:
            for cell in annotated_nb.cells[: cell_data.cell_idx_in_annotated_notebook]:
                execute_cell(cell, shell)
                executed_cells.append(cell)

            # Make the pre_checkpoint_path for this cell.
            checkpoint_start_time = time.time()
            pre_checkpoint_path = get_pre_checkpoint_path(
                annotated_nb_path, annotated_cell_idx
            )
            pre_checkpoint_cell = get_save_checkpoint_cell(pre_checkpoint_path)
            checkpoint_end_time = time.time()
            executed_cells.append(pre_checkpoint_cell)
            execute_cell(pre_checkpoint_cell, shell)
            print(
                f"Checkpointing took {checkpoint_end_time - checkpoint_start_time} seconds."
            )
            rewritten_pre_checkpoint_path = pre_checkpoint_path
            small_rewritten_pre_checkpoint_path = get_pre_checkpoint_path(
                small_nb_path, annotated_cell_idx
            )
            print("========================================================")
            print("Cell ", annotated_cell_idx)
            print(f"Rewritten pre-checkpoint path: {rewritten_pre_checkpoint_path}")
            print(f"Pre-checkpoint path: {pre_checkpoint_path}")
            print(
                f"Small rewritten pre-checkpoint path: {small_rewritten_pre_checkpoint_path}"
            )
            print("========================================================")

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
        (
            original_execution_time,
            rewritten_execution_time,
            rewritten_code,
            rewritten_pre_checkpoint_path,
            small_rewritten_pre_checkpoint_path,
            pre_checkpoint_path,
        ) = await _rewrite_cell_cpu(
            cell=cell,
            cell_exec_info=cell_exec_infos[annotated_cell_idx],
            cpu_profile_info=cpu_profile_infos[annotated_cell_idx],
            annotated_cell_idx=annotated_cell_idx,
            pre_checkpoint_path=pre_checkpoint_path,
            rewritten_pre_checkpoint_path=rewritten_pre_checkpoint_path,
            small_rewritten_pre_checkpoint_path=small_rewritten_pre_checkpoint_path,
            nb_path=annotated_nb_path,
            small_nb_path=small_nb_path,
            rewritten_nb_path=rewritten_nb_path,
            small_rewritten_nb_path=small_rewritten_nb_path,
            num_tries_per_cell=num_tries_per_cell,
        )
        print("========================================================")
        print("After rewriting cell ", annotated_cell_idx)
        print(f"Rewritten pre-checkpoint path: {rewritten_pre_checkpoint_path}")
        print(f"Pre-checkpoint path: {pre_checkpoint_path}")
        print("========================================================")
        original_execution_times[annotated_cell_idx] = round(original_execution_time, 2)
        rewritten_execution_times[annotated_cell_idx] = round(
            rewritten_execution_time, 2
        )
        rewrite_end_time = time.time()
        print(
            f"Cell {annotated_cell_idx} rewriting took {rewrite_end_time - rewrite_start_time} seconds."
        )

        cell_index = cell_data.cell_idx_in_annotated_notebook
        if rewritten_code is not None:
            # This means we couldn't rewrite anything.
            rewritten_nb_cells[cell_index] = make_code_cell(rewritten_code)
        else:
            rewritten_nb_cells[cell_index] = make_code_cell(
                remove_magic_commands(cell.source)
            )

    # Clear all the checkpoints we made.
    # TODO(jie): remove this.
    # clear_all_checkpoints(annotated_nb_path)
    rewritten_nb = make_notebook(rewritten_nb_cells)
    save_notebook(rewritten_nb, rewritten_nb_path)

    return rewritten_nb, rewritten_execution_times, original_execution_times


async def _get_original_cell_execution_info(
    cell: NotebookNode,
    cell_exec_info: CellExecInfo,
    cpu_profile_info: CpuProfileInfo,
    annotated_cell_idx: int,
    pre_checkpoint_path: Path,
    nb_path: Path,
    is_rewriting_cell: bool = False,
):
    """
    A function to get original cell execution time.
    If the cell is being rewritten, set is_rewriting_cell to True and save the original cell notebook.
    Returns the original cell execution time in ms.
    """
    print("Getting original execution timing for cell", annotated_cell_idx)
    # cell 0: load the ElasticNotebook extension.
    load_elastic_notebook_cell = get_load_elastic_notebook_cell()
    # cell 2: load the checkpoint before the annotated cell.
    print("HAHA original checkpoint path: ", pre_checkpoint_path)
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
            load_checkpoint_cell,
            clear_output_cell,
            original_cell,
            save_output_cell,
            post_checkpoint_cell,
        ]
    )
    # Save the original cell notebook, only if this cell is going to be rewritten.
    if is_rewriting_cell:
        original_cell_notebook_save_path = (
            nb_path.parent / f"original_cell_{annotated_cell_idx}_cpu.ipynb"
        )
        save_notebook(original_cell_notebook, original_cell_notebook_save_path)

    # Execute the original cell notebook.
    execute_notebook(original_cell_notebook)
    original_execution_time = parse_wall_time_to_ms_from_all_outputs(
        original_cell_notebook.cells[3]["outputs"]
    )
    if original_execution_time is None:
        # raise ValueError(f"Original time is None for cell {annotated_cell_idx}")
        original_execution_time = 0
    return original_execution_time, post_checkpoint_path


# cpu version of _rewrite_cell, stripped of cpu
async def _rewrite_cell_cpu(
    cell: NotebookNode,
    cell_exec_info: CellExecInfo,
    cpu_profile_info: CpuProfileInfo,
    annotated_cell_idx: int,
    pre_checkpoint_path: Path,
    rewritten_pre_checkpoint_path: Path,
    small_rewritten_pre_checkpoint_path: Path,
    nb_path: Path,
    small_nb_path: Path,
    rewritten_nb_path: Path,
    small_rewritten_nb_path: Path,
    num_tries_per_cell: int,
) -> tuple[float, str | None, Path, Path, Path]:
    # Now we will try to rewrite the cell.
    rewritten_cell: NotebookNode | None = None
    rewritten_execution_time: float = 0.0
    best_rewritten_code: str | None = None
    best_rewritten_post_checkpoint_path: Path | None = None

    num_tries = 0
    # last_attempt_exec_states_equal = True
    is_correct = False
    rewritten_cell_notebook_save_path = None
    all_rewritten_code_info: list[CodeInfo] = []
    rewritten_code_info = None
    rewritten_cell_cpu_profile_info = None

    # Get original cell execution time.
    (
        original_execution_time,
        post_checkpoint_path,
    ) = await _get_original_cell_execution_info(
        cell,
        cell_exec_info,
        cpu_profile_info,
        annotated_cell_idx,
        pre_checkpoint_path,
        nb_path,
        is_rewriting_cell=True,
    )

    # We want to keep track of some info for the best solution. By default, they are the original solution
    # at the beginning.
    best_rewritten_execution_time = original_execution_time
    best_rewritten_cpu_profile_info = cpu_profile_info

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
    while num_tries < num_tries_per_cell:
        print("========================================================")
        print(f"Rewriting cell {annotated_cell_idx}... Try {num_tries}... ")
        active_vars = [var.name for var in cell_exec_info.active_vars]
        future_vars = [var.name for var in cell_exec_info.future_vars]
        original_cpu_df = cpu_profile_info.df_table

        cleaned_cell_code = remove_magic_commands(cell.source)
        original_code_info = CodeInfo(
            code=cleaned_cell_code,
            profiling_info=(
                original_cpu_df.to_string() if original_cpu_df is not None else None
            ),
            active_vars=active_vars,
            future_vars=future_vars,
            execution_time=original_execution_time,
        )
        # FIXME(sahil): once the bug is fixed, remove the hardcoded code.
        rewrite_agent_start_time = time.time()
        rewritten_code = await call_rewrite_agent_cpu(
            num_tries, original_code_info, rewritten_code_info, num_tries_per_cell
        )
        rewrite_agent_end_time = time.time()
        print(
            f"Try {num_tries}: calling rewrite agent for cell {annotated_cell_idx} took {rewrite_agent_end_time - rewrite_agent_start_time} seconds."
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
        rewritten_execution_time = float("inf")

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
                rewritten_cell_notebook.cells[2]["outputs"]
            )
            if curr_rewritten_time is None:
                raise ValueError(
                    f"Rewritten time is None for cell {annotated_cell_idx}"
                )
            rewritten_execution_time = curr_rewritten_time
            execution_output = (
                f"Rewritten execution time: {rewritten_execution_time} ms\n"
                f"Original execution time: {original_execution_time} ms"
            )
            is_correct = True
            print("Finished executing the rewritten code.")
            rewrite_execution_end_time = time.time()
            print(
                f"Try {num_tries}: executing the rewritten code for cell {annotated_cell_idx} took {rewrite_execution_end_time - rewrite_execution_start_time} seconds."
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

        print(f"Rewritten execution time: {rewritten_execution_time} ms")

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

            print("The rewritten code is correct. Profiling cpu...")
            # cell 0: load the ElasticNotebook extension.
            load_elastic_notebook_cell = get_load_elastic_notebook_cell()
            # cell 2: load the checkpoint before the annotated cell. Note this is the checkpoint for the small notebook.
            load_checkpoint_cell = get_load_checkpoint_cell(
                small_rewritten_pre_checkpoint_path
            )
            # cell 3: cpu profile the rewritten code.
            cpu_cell = make_code_cell(
                maybe_annotate_code_with_cpu_profile(rewritten_code)
            )
            # Save the new cell notebook for profiling
            rewritten_cell_profile_notebook_save_path = (
                nb_path.parent
                / f"rewrite_cell_{annotated_cell_idx}_try_{num_tries}_profile_cpu.ipynb"
            )
            rewritten_cell_profile_notebook = make_notebook(
                [
                    load_elastic_notebook_cell,
                    load_checkpoint_cell,
                    cpu_cell,
                ]
            )
            save_notebook(
                rewritten_cell_profile_notebook,
                rewritten_cell_profile_notebook_save_path,
            )
            cpu_profile_start_time = time.time()
            execute_notebook(rewritten_cell_profile_notebook)
            cpu_profile_end_time = time.time()
            print(
                f"Try {num_tries}: cpu profiling the rewritten code for cell {annotated_cell_idx} took {cpu_profile_end_time - cpu_profile_start_time} seconds."
            )

            rewritten_cell_cpu_profile_info = parse_cpu_profile_info_from_all_outputs(
                rewritten_cell_profile_notebook.cells[-1].outputs
            )
            if (
                best_rewritten_cpu_profile_info is None
                or rewritten_cell_cpu_profile_info is None
            ):
                raise ValueError(
                    f"CpuProfileInfo is None for cell {annotated_cell_idx}"
                )
            # TODO(colin): maybe need more sophisticated way
            accepted_rewritten_code = rewritten_execution_time < original_execution_time
            # accepted_rewritten_code = is_rewritten_code_better(
            #     orig_cpu_profile_info=best_rewritten_cpu_profile_info,
            #     opt_cpu_profile_info=rewritten_cell_cpu_profile_info,
            #     orig_time=best_rewritten_execution_time,
            #     opt_time=rewritten_execution_time,
            # )
            print(
                f"Rewritten code took {rewritten_execution_time}ms. The previous fastest solution took {original_execution_time}ms."
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
                best_rewritten_execution_time = rewritten_execution_time
                best_rewritten_cpu_profile_info = rewritten_cell_cpu_profile_info

        cpu_df = (
            rewritten_cell_cpu_profile_info.df_table
            if rewritten_cell_cpu_profile_info is not None
            else None
        )
        rewritten_code_info = CodeInfo(
            code=rewritten_code,
            profiling_info=cpu_df.to_string() if cpu_df is not None else None,
            active_vars=active_vars,
            future_vars=future_vars,
            execution_time=rewritten_execution_time,
            execution_output=execution_output,
        )
        all_rewritten_code_info.append(rewritten_code_info)
        num_tries += 1

    if best_rewritten_code is None:
        print("Could not rewrite the cell. Continuing to next cell...")

    return (
        original_execution_time,
        best_rewritten_execution_time,
        best_rewritten_code,
        best_rewritten_post_checkpoint_path,
        best_small_rewritten_post_checkpoint_path,
        post_checkpoint_path,
    )
