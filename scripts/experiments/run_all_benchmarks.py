#!/usr/bin/env python3
import subprocess
from pathlib import Path

from utils.benchmarks import BENCHMARK_NAMES, BENCHMARKS_TO_PATHS


def process_notebooks(notebooks):
    """
    Iterate over notebook pairs, calling main.py on each.
    If a call fails, log the error and continue.
    """
    for name, notebook_path, small_notebook_path in notebooks:
        try:
            print(
                f"Running: python main.py {name} {notebook_path} {small_notebook_path}"
            )
            subprocess.run(
                [
                    "python",
                    "-u",
                    "utils/main.py",
                    name,
                    notebook_path,
                    small_notebook_path,
                ],
                check=True,
            )
            print(f"✅ Success: {notebook_path}")
        except subprocess.CalledProcessError as e:
            # CalledProcessError captures returncode, cmd, output, stderr
            print(f"❌ Failed ({e.returncode}) on {notebook_path}:\n")
            print(e.output)
        except Exception as e:
            print(f"⚠️ Unexpected error on {notebook_path}: {e}")


if __name__ == "__main__":
    notebooks = []
    for name in BENCHMARK_NAMES:
        notebooks.append(
            [
                name,
                Path(BENCHMARKS_TO_PATHS[name]) / "bench.ipynb",
                Path(BENCHMARKS_TO_PATHS[name]) / "small_bench.ipynb",
            ]
        )
    process_notebooks(notebooks)
