import argparse
import asyncio
import logging
import pickle
import time
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

from utils.rewrite_cpu import checkpoint_and_get_cpu_profile_info, rewrite_notebook_cpu
from utils.schedule import (
    get_cell_exec_info,
)


async def main(
    small_notebook_path="",
    full_notebook_path="",
    start_cell_index=0,
    num_tries_per_cell=5,
):
    assert small_notebook_path and full_notebook_path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=Path("cpu_code_agent.log"),
        filemode="w",
    )
    shell = InteractiveShell.instance()

    # First annotate the notebook to add checkpoints and timing code.
    original_notebook_path = Path(full_notebook_path)
    small_notebook_path = Path(small_notebook_path)

    try:
        cell_exec_start_time = time.time()
        cell_exec_info = get_cell_exec_info(small_notebook_path, shell)
        cell_exec_end_time = time.time()
        with open("cell_exec_info.pkl", "wb") as f:
            pickle.dump(cell_exec_info, f)
        print(
            f"Ran small notebook to get cell exec info in {cell_exec_end_time - cell_exec_start_time} seconds"
        )
    except Exception as e:
        print(f"Error getting cell exec info: {e}")

    # ############# cpu rewriter
    # Get cpu profile information.
    # Use the small notebook to get the cpu profile information.
    try:
        cpu_profile_start_time = time.time()
        cpu_profile_infos = checkpoint_and_get_cpu_profile_info(small_notebook_path)
        cpu_profile_end_time = time.time()
        with open("cpu_profile_info.pkl", "wb") as f:
            pickle.dump(cpu_profile_infos, f)
        print(
            f"Ran small notebook to get cpu profile info in {cpu_profile_end_time - cpu_profile_start_time} seconds"
        )
    except Exception as e:
        print(f"Error getting cpu profile info: {e}")

    print(cpu_profile_infos)

    rewrite_start_time = time.time()
    (
        _,
        rewritten_execution_times_cpu,
        original_execution_times_cpu,
    ) = await rewrite_notebook_cpu(
        nb_path=original_notebook_path,
        small_nb_path=small_notebook_path,
        cpu_profile_infos=cpu_profile_infos,
        cell_exec_infos=cell_exec_info,
        shell=shell,
        start_cell_index=start_cell_index,
        num_tries_per_cell=num_tries_per_cell,
    )
    rewrite_end_time = time.time()
    print(f"Rewrote CPU notebook in {rewrite_end_time - rewrite_start_time} seconds")
    print("rewritten execution times:", rewritten_execution_times_cpu)
    print("original execution times:", original_execution_times_cpu)
    return rewritten_execution_times_cpu, original_execution_times_cpu
    ####################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_notebook_path", type=str, required=True)
    parser.add_argument("--full_notebook_path", type=str, required=True)

    args = parser.parse_args()
    asyncio.run(
        main(
            small_notebook_path=args.small_notebook_path,
            full_notebook_path=args.full_notebook_path,
        )
    )
