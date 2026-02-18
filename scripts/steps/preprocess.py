import pickle
import time
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

from utils.schedule import get_cell_exec_info

if __name__ == "__main__":
    shell = InteractiveShell.instance()
    small_notebook_path = Path(
        "/Users/jieq/Desktop/pandax-meng/notebooks/spscientist/student-performance-in-exams/src/small_bench_meng.ipynb"
    )

    # First annotate the notebook to add checkpoints and timing code.
    cell_exec_start_time = time.time()
    cell_exec_info = get_cell_exec_info(small_notebook_path, shell)
    cell_exec_end_time = time.time()
    print(
        f"Ran small notebook to get cell exec info in {cell_exec_end_time - cell_exec_start_time} seconds"
    )
    # Dump the cell exec info to a pickle file.
    pkl_file_path = small_notebook_path.parent / "cell_exec_info_meng.pkl"
    with open(pkl_file_path, "wb") as f:
        pickle.dump(cell_exec_info, f)
