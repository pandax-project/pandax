import os
from pathlib import Path

from utils.diff import compute_diff
from utils.main_cpu import main
from utils.notebook import clean_up_rewritten_notebook


async def rewrite_cpu_html(
    small_notebook_path: str, start_cell_idx=0, num_tries_per_cell=5
):
    """
    Simple rewrite: just display a new notebook with one cell with comment "# rewritten with cpu".
    """
    full_notebook_path = "./notebooks/spscientist/student-performance-in-exams/src/small_bench_meng.ipynb"

    try:
        rewritten_execution_times, original_execution_times = await main(
            small_notebook_path=small_notebook_path,
            full_notebook_path=full_notebook_path,
            start_cell_index=start_cell_idx,
            num_tries_per_cell=num_tries_per_cell,
        )
        nb_path = Path(full_notebook_path)
        rewritted_path = (
            nb_path.parent / Path("rewritten_cpu") / Path("o4_mini_high.ipynb")
        )
        cleaned_path = rewritted_path.parent / Path("o4_mini_high_cleaned.ipynb")
        clean_up_rewritten_notebook(rewritted_path, cleaned_path)
        relative_path = os.path.relpath(cleaned_path, start=os.getcwd())

        diff, diff_path = compute_diff(small_notebook_path, cleaned_path)
        return (
            str(relative_path),
            diff,
            str(diff_path),
            rewritten_execution_times,
            original_execution_times,
        )

    except Exception as e:
        print(f"Error: failed to fetch rewritten notebook, str{e}")
