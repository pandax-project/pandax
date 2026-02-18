# Run all cells in a notebook at factor extracted from the environment variable $FACTOR.
# Print the factor, wall time, memory before, memory after, and peak memory for each cell to screen.
import os
import sys
from copy import deepcopy

from nbclient import NotebookClient

from utils.execution import parse_memory_and_time
from utils.notebook import load_notebook, make_code_cell, make_notebook

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: FACTOR={factor} python script.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    factor = os.getenv("FACTOR")

    if factor is None:
        raise EnvironmentError("FACTOR environment variable is not set.")

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
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(
                    f"Execution failed in cell {cell.source}: {output.get('evalue')}"
                )

        # Parse wall time
        wall_time, memory_before, memory_after, peak = parse_memory_and_time(
            cell["outputs"][-1]["text"]
        )
        print(factor, wall_time, memory_before, memory_after, peak)
