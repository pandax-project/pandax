# Run all cells in a notebook at factor extracted from the environment variable $FACTOR.
# Print the factor, cell number, and wall time for each cell to screen.
import os
import sys
from copy import deepcopy

from nbclient import NotebookClient

from utils.execution import parse_wall_time_to_ms
from utils.notebook import (
    cell_annotation_pattern,
    load_notebook,
    make_code_cell,
    make_notebook,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: FACTOR={factor} python script.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    factor = os.getenv("FACTOR")

    if factor is None:
        raise EnvironmentError("FACTOR environment variable is not set.")

    cell_timings: dict[int, float] = {}

    notebook = load_notebook(notebook_path)
    wall_times = []

    # Inject the factor code cell
    notebook.cells.insert(0, make_code_cell(f"factor = {factor}"))

    # Prepare notebook for execution
    new_cells = deepcopy(notebook.cells)
    new_notebook = make_notebook(new_cells)

    # Execute
    client = NotebookClient(new_notebook, shutdown_kernel="immediate")
    client.execute()

    # Check for errors
    for cell in new_notebook.cells:
        outputs = cell.get("outputs", [])
        for output in outputs:
            if output.get("output_type") == "error":
                raise RuntimeError(
                    f"Execution failed in cell {cell.source}: {output.get('evalue')}"
                )
        if not outputs:
            continue

        wall_time = parse_wall_time_to_ms(outputs[0]["text"])
        match = cell_annotation_pattern.search(cell.source)
        if match and wall_time is None:
            raise RuntimeError(f"Cell {cell.source} is missing a wall time.")
        elif not match:
            continue
        cell_number = int(match.group(1))

        cell_timings[cell_number] = wall_time
        print(factor, cell_number, wall_time)
