from pathlib import Path

from utils.notebook import annotate_notebook, load_notebook
from utils.verification import (
    check_forbidden,
    extract_factors,
    get_largest_cell,
    load_code_lines,
)

count = 0

for i in range(1, 23):
    # turn i into a string with leading zeros
    i_str = str(i).zfill(2)
    # run the verify_bench.py script
    nb_dir = Path(f"/home/dias-benchmarks/tpch/notebooks/q{i_str}")

    cpu_nb_path = nb_dir / "rewritten_cpu" / "o4_mini_high_scheduled.ipynb"
    gpu_nb_path = nb_dir / "rewritten" / "o4_mini_high_scheduled.ipynb"

    annotate_notebook(
        cpu_nb_path,
        cpu_nb_path,
        add_timing_code=False,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        add_cudf_profile=False,
        add_cpu_profile=False,
        use_gpu=False,
    )
    annotate_notebook(
        gpu_nb_path,
        gpu_nb_path,
        add_timing_code=False,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        add_cudf_profile=False,
        add_cpu_profile=False,
        use_gpu=False,
    )

    cpu_lines = load_code_lines(cpu_nb_path)
    gpu_lines = load_code_lines(gpu_nb_path)

    # Check neither notebook has forbidden patterns
    cpu_forbidden = check_forbidden(cpu_lines, "")
    gpu_forbidden = check_forbidden(gpu_lines, "")

    cpu_factors = extract_factors(cpu_lines)
    gpu_factors = extract_factors(gpu_lines)

    if cpu_factors != gpu_factors or cpu_factors != ["10"]:
        print(f"Factors mismatch between cpu and gpu for q{i_str}")
        print(f"CPU factors: {cpu_factors}")
        print(f"GPU factors: {gpu_factors}")
        print()

    if cpu_forbidden or gpu_forbidden:
        print(f"Forbidden patterns found in {cpu_nb_path} or {gpu_nb_path}")
        print(f"CPU forbidden: {cpu_forbidden}")
        print(f"GPU forbidden: {gpu_forbidden}")
        print()

    # check if number of cells is the same.
    annotate_notebook(
        cpu_nb_path,
        cpu_nb_path,
        add_timing_code=False,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        add_cudf_profile=False,
        add_cpu_profile=False,
        use_gpu=False,
    )
    annotate_notebook(
        gpu_nb_path,
        gpu_nb_path,
        add_timing_code=False,
        add_record_events=False,
        add_checkpoints=False,
        track_column_info=False,
        add_cudf_profile=False,
        add_cpu_profile=False,
        use_gpu=False,
    )
    cpu_nb = load_notebook(cpu_nb_path)
    gpu_nb = load_notebook(gpu_nb_path)
    cpu_largest_cell = get_largest_cell(cpu_nb)
    gpu_largest_cell = get_largest_cell(gpu_nb)

    if cpu_largest_cell != gpu_largest_cell:
        print(f"Largest cell mismatch between cpu and gpu for q{i_str}")
        print(f"CPU largest cell: {cpu_largest_cell}")
        print(f"GPU largest cell: {gpu_largest_cell}")
        print()

    if gpu_largest_cell > 0:
        count += 1
        print(f"GPU largest cell is {gpu_largest_cell} for q{i_str}")


print(count)
