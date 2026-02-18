#!/usr/bin/env python3
"""
verify_bench.py

1. For every directory containing bench.ipynb:
   - Verify small_bench.ipynb, rewritten_cpu/o4_mini_high.ipynb, rewritten/o4_mini_high.ipynb exist.
   - Verify the only code‐cell differences between them are lines of the form `factor = <...>`.
     Print out the factor values in each notebook.
   - If there are other diffs, print a unified diff.

2. Ensure neither bench.ipynb nor small_bench.ipynb:
   - Contains any IPython magic (% or %%).
   - Defines `start_time = time.time()` in any code cell.
   - Contains loading cudf.pandas extension.
   - Contains commented out loading cudf.pandas extension.

Usage:
    chmod +x verify_bench.py
    ./verify_bench.py /path/to/base/dir
"""

import argparse
import difflib
import os
from pathlib import Path

from utils.benchmarks import FACTOR_MAP
from utils.notebook import load_notebook
from utils.verification import (
    check_forbidden,
    extract_factors,
    get_largest_cell,
    load_code_lines,
    only_factor_diff,
)


def main():
    p = argparse.ArgumentParser(
        description="Validate bench.ipynb / small_bench.ipynb / rewritten_cpu/o4_mini_high.ipynb / rewritten/o4_mini_high.ipynb consistency and style"
    )
    p.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory to search (default: current directory)",
    )
    args = p.parse_args()

    for root, dirs, files in os.walk(args.base_dir):
        if "google" in root:
            continue
        if "bench.ipynb" not in files:
            continue

        bench_nb = os.path.join(root, "bench.ipynb")
        small_nb = os.path.join(root, "small_bench.ipynb")
        cpu_nb = os.path.join(root, "rewritten_cpu", "o4_mini_high.ipynb")
        gpu_nb = os.path.join(root, "rewritten", "o4_mini_high.ipynb")
        loaded_bench_nb = load_notebook(Path(bench_nb))
        loaded_small_nb = load_notebook(Path(small_nb))
        loaded_cpu_nb = load_notebook(Path(cpu_nb))
        loaded_gpu_nb = load_notebook(Path(gpu_nb))
        largest_cell_idx_bench = get_largest_cell(loaded_bench_nb)
        largest_cell_idx_small = get_largest_cell(loaded_small_nb)
        largest_cell_idx_cpu = get_largest_cell(loaded_cpu_nb)
        largest_cell_idx_gpu = get_largest_cell(loaded_gpu_nb)

        if (
            largest_cell_idx_bench != largest_cell_idx_small
            or largest_cell_idx_bench != largest_cell_idx_cpu
            or largest_cell_idx_bench != largest_cell_idx_gpu
        ):
            print("  ✖ Largest cell index mismatch between bench, cpu, and gpu")
            print(f"    bench largest cell index: {largest_cell_idx_bench}")
            print(f"    small largest cell index: {largest_cell_idx_small}")
            print(f"    cpu largest cell index: {largest_cell_idx_cpu}")
            print(f"    gpu largest cell index: {largest_cell_idx_gpu}")
            continue

        if Path(root).name == "src":
            name = Path(root).parent.name
        else:
            name = Path(root).name

        ideal_factor = [f"{FACTOR_MAP[name]}"]

        print(f"\nDirectory: {root}")

        # 1) Check existence of small_bench.ipynb
        if not os.path.exists(small_nb):
            print("  ✖ Missing small_bench.ipynb")
            continue
        if not os.path.exists(cpu_nb):
            print("  ✖ Missing rewritten_cpu/o4_mini_high.ipynb")
            continue
        if not os.path.exists(gpu_nb):
            print("  ✖ Missing rewritten/o4_mini_high.ipynb")
            continue

        # 2) Load code lines
        bench_lines = load_code_lines(bench_nb)
        small_lines = load_code_lines(small_nb)
        cpu_lines = load_code_lines(cpu_nb)
        gpu_lines = load_code_lines(gpu_nb)

        # 3) Check diffs
        if only_factor_diff(bench_lines, small_lines):
            factors_bench = extract_factors(bench_lines)
            factors_small = extract_factors(small_lines)
            print("  ✅ Only factor differences")
            print(f"    bench factors: {factors_bench}")
            print(f"    small factors: {factors_small}")
        else:
            print("  ✖ Other differences detected:")
            diff = difflib.unified_diff(
                bench_lines,
                small_lines,
                fromfile="bench.ipynb",
                tofile="small_bench.ipynb",
                lineterm="",
            )
            for line in diff:
                print("    " + line)

        # Check cpu and gpu and bench must have the same factor
        factors_bench = extract_factors(bench_lines)
        factors_cpu = extract_factors(cpu_lines)
        factors_gpu = extract_factors(gpu_lines)
        if (
            factors_bench != factors_cpu
            or factors_bench != factors_gpu
            or factors_bench != ideal_factor
        ):
            print("  ✖ Factors mismatch between bench, cpu, and gpu")
            print(f"    ideal factor: {ideal_factor}")
            print(f"    bench factors: {factors_bench}")
            print(f"    cpu factors: {factors_cpu}")
            print(f"    gpu factors: {factors_gpu}")
        else:
            print("  ✅ Factors match between bench, cpu, and gpu")
            print(f"    ideal factor: {ideal_factor}")
            print(f"    bench factors: {factors_bench}")
            print(f"    cpu factors: {factors_cpu}")
            print(f"    gpu factors: {factors_gpu}")

        # 4) Check forbidden patterns in both notebooks
        for code_lines, name in [
            (bench_lines, "bench.ipynb"),
            (small_lines, "small_bench.ipynb"),
            (cpu_lines, "rewritten_cpu/o4_mini_high.ipynb"),
            (gpu_lines, "rewritten/o4_mini_high.ipynb"),
        ]:
            offenders = check_forbidden(code_lines)
            if offenders:
                print(f"  ✖ Forbidden usage in {name}:")
                for lineno, content in offenders:
                    print(f"    line {lineno}: {content.strip()}")

        # 5) Check if they all have the same number of cells
        if (
            len(loaded_bench_nb.cells) != len(loaded_small_nb.cells)
            or len(loaded_bench_nb.cells) != len(loaded_cpu_nb.cells)
            or len(loaded_bench_nb.cells) != len(loaded_gpu_nb.cells)
        ):
            print("  ✖ Number of cells mismatch between bench, cpu, and gpu")
            print(f"    bench number of cells: {len(loaded_bench_nb.cells)}")
            print(f"    small number of cells: {len(loaded_small_nb.cells)}")
            print(f"    cpu number of cells: {len(loaded_cpu_nb.cells)}")
            print(f"    gpu number of cells: {len(loaded_gpu_nb.cells)}")


if __name__ == "__main__":
    main()
