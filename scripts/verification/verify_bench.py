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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    p.add_argument(
        "--mode",
        choices=["input", "all"],
        default="all",
        help="Comparison scope: 'input' checks only bench/small_bench, 'all' checks all 4 notebooks",
    )
    args = p.parse_args()

    for root, _, files in os.walk(args.base_dir):
        if "bench.ipynb" not in files:
            continue

        print(f"\nDirectory: {root}")

        bench_nb = os.path.join(root, "bench.ipynb")
        small_nb = os.path.join(root, "small_bench.ipynb")
        notebook_paths = {
            "bench.ipynb": bench_nb,
            "small_bench.ipynb": small_nb,
        }
        if args.mode == "all":
            notebook_paths["rewritten_cpu/o4_mini_high.ipynb"] = os.path.join(
                root, "rewritten_cpu", "o4_mini_high.ipynb"
            )
            notebook_paths["rewritten/o4_mini_high.ipynb"] = os.path.join(
                root, "rewritten", "o4_mini_high.ipynb"
            )

        missing = [name for name, path in notebook_paths.items() if not os.path.exists(path)]
        if missing:
            for name in missing:
                print(f"  ✖ Missing {name}")
            continue

        loaded = {name: load_notebook(Path(path)) for name, path in notebook_paths.items()}
        largest_cell_idx = {
            name: get_largest_cell(nb_obj) for name, nb_obj in loaded.items()
        }
        unique_largest = set(largest_cell_idx.values())
        if len(unique_largest) != 1:
            print("  ✖ Largest cell index mismatch")
            for name, idx in largest_cell_idx.items():
                print(f"    {name} largest cell index: {idx}")
            continue

        if Path(root).name == "src":
            name = Path(root).parent.name
        else:
            name = Path(root).name

        ideal_factor = [f"{FACTOR_MAP[name]}"]
        code_lines = {name: load_code_lines(path) for name, path in notebook_paths.items()}
        bench_lines = code_lines["bench.ipynb"]
        small_lines = code_lines["small_bench.ipynb"]

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

        # Check factors
        factors_bench = extract_factors(bench_lines)
        if args.mode == "all":
            factors_cpu = extract_factors(code_lines["rewritten_cpu/o4_mini_high.ipynb"])
            factors_gpu = extract_factors(code_lines["rewritten/o4_mini_high.ipynb"])
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
        else:
            if factors_bench != ideal_factor:
                print("  ✖ Bench factor does not match ideal factor")
                print(f"    ideal factor: {ideal_factor}")
                print(f"    bench factors: {factors_bench}")
            else:
                print("  ✅ Bench factor matches ideal factor")
                print(f"    ideal factor: {ideal_factor}")
                print(f"    bench factors: {factors_bench}")

        # 4) Check forbidden patterns
        for nb_name, nb_lines in code_lines.items():
            offenders = check_forbidden(nb_lines)
            if offenders:
                print(f"  ✖ Forbidden usage in {nb_name}:")
                for lineno, content in offenders:
                    print(f"    line {lineno}: {content.strip()}")

        # 5) Check cell counts match across selected notebooks
        cell_counts = {nb_name: len(nb_obj.cells) for nb_name, nb_obj in loaded.items()}
        if len(set(cell_counts.values())) != 1:
            print("  ✖ Number of cells mismatch")
            for nb_name, count in cell_counts.items():
                print(f"    {nb_name} number of cells: {count}")


if __name__ == "__main__":
    main()
