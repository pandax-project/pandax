"""
Return the diff between the original and rewritten notebook in JSON format,
using nbdime library.
"""

import json
from pathlib import Path

# Load notebooks
import nbformat
from nbdime import diff_notebooks


def compute_diff(source_file, rewritten_file):
    with open(source_file) as f:
        orig_nb = nbformat.read(f, as_version=4)
    with open(rewritten_file) as f:
        rewritten_nb = nbformat.read(f, as_version=4)

    # Compute diff
    diff = diff_notebooks(orig_nb, rewritten_nb)

    diff_path = Path(source_file).parent / Path("diff.json")
    # Save diff as JSON
    with open(diff_path, "w") as f:
        json.dump(diff, f, indent=2)

    return diff, str(diff_path)
