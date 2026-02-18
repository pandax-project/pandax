# Get the end to end timings for all benchmarks. Note, this is done by adding a start and end time cell to the notebook and executing it.
# It's different from timing the notebook cell by cell.
import copy
import re

# Get the cpu timings for all benchmarks. These includes the original timing, as well as the timings for the modified factors and the predicted timings.
from pathlib import Path

from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS
from utils.execution import execute_notebook
from utils.notebook import load_notebook, make_code_cell, make_notebook
from utils.verification import extract_factors, load_code_lines

failed_paths = []
for name in BENCHMARK_NAMES:
    path = BENCHMARKS_TO_PATHS[name]

    # check if the path is of format q01, q02, etc.
    # use regex.
    match = re.match(r"q(\d+)", name)
    if match:
        i_str = match.group(1)
        nb_path = Path(path) / f"q{i_str}_rewrite.ipynb"
    else:
        nb_path = Path(path) / "bench.ipynb"

    cpu_nb = load_notebook(nb_path)
    bench_lines = load_code_lines(nb_path)
    factors = extract_factors(bench_lines)
    assert len(factors) == 1, f"Expected 1 factor, got {factors}"
    factor = int(factors[0])

    cell_0_idx = None
    for i, cell in enumerate(cpu_nb.cells):
        if cell.cell_type == "code" and "### cell 0 ###" in cell.source:
            cell_0_idx = i
            break
    new_cells = copy.deepcopy(cpu_nb.cells)
    assert cell_0_idx is not None, f"No code cell found in {nb_path}"
    # execute the notebook from the first code cell to the end.
    start_time_code = """
import time
start_time = time.time()
    """
    end_time_code = """
end_time = time.time()
print(f'Execution time: {end_time - start_time} seconds')
"""

    new_cells.insert(cell_0_idx, make_code_cell(start_time_code))
    # new_cells.insert(0, make_code_cell("%load_ext cudf.pandas"))
    new_cells.insert(0, make_code_cell("import dias.rewriter"))
    new_cells.append(make_code_cell(end_time_code))
    new_nb = make_notebook(new_cells)

    execute_notebook(new_nb)
    for output in new_cells[-1].outputs:
        # try to parse the execution time.
        # try to match the regex "Execution time: \d+\.\d+ seconds"
        match = re.search(r"Execution time: (\d+\.\d+) seconds", output.get("text", ""))
        if match:
            execution_time = float(match.group(1))
            break
        if execution_time is None:
            raise Exception(f"Failed to parse execution time from {nb_path}")
    print(f"{nb_path} {execution_time}")
