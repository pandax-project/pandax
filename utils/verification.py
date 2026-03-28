# This file contains helper functions for verifying the validity of our benchmarks.
import difflib
import json
import re

from utils.notebook import cell_annotation_pattern


def get_largest_cell(nb):
    largest_cell_idx = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        match = cell_annotation_pattern.search(cell.source)
        if match:
            cell_number = int(match.group(1))
            if cell_number > largest_cell_idx:
                largest_cell_idx = cell_number
    return largest_cell_idx


# Regexes
FACTOR_RE = re.compile(r"^\s*factor\s*=\s*(.+)$")
# Match only real IPython magics (%name / %%name), not Python `% (...)` formatting.
MAGIC_RE = re.compile(r"^\s*%%?[A-Za-z_]\w*")
START_TIME_RE = re.compile(r"^\s*start_time\s*=\s*time\.time\(\)")
# Filter out %load_ext cudf.pandas
LOAD_CUDF_EXTENSION = re.compile(r"^\s*%load_ext\s+cudf\.pandas$")
# Filter out commented out %load_ext cudf.pandas
LOAD_CUDF_EXTENSION_COMMENTED = re.compile(r"^\s*#\s*%load_ext\s+cudf\.pandas$")
# Filter out %%time
TIME_RE = re.compile(r"^\s*%%time$")


def load_code_lines(nb_path):
    """Return a list of code-cell lines from the notebook."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = src if isinstance(src, str) else "".join(src)
        lines.extend(text.splitlines())
    return lines


def only_factor_diff(a_lines, b_lines):
    """
    Return True if the only differences between a_lines and b_lines
    are replacements of lines matching FACTOR_RE.
    """
    sm = difflib.SequenceMatcher(None, a_lines, b_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                if not (
                    FACTOR_RE.match(a_lines[i1 + k])
                    and FACTOR_RE.match(b_lines[j1 + k])
                ):
                    return False
        else:
            # no inserts, deletes, or multi-line mismatched replaces
            return False
    return True


def extract_factors(lines):
    """Extract all RHS values from lines matching FACTOR_RE."""
    factors = []
    for ln in lines:
        match = FACTOR_RE.match(ln)
        if match:
            factors.append(match.group(1).strip())
    return factors


def check_forbidden(lines):
    """Check for magic commands or start_time usage; return list of offending lines."""
    offenders = []
    for idx, ln in enumerate(lines, start=1):
        if MAGIC_RE.match(ln):
            offenders.append((idx, ln))
        elif START_TIME_RE.match(ln):
            offenders.append((idx, ln))
        elif LOAD_CUDF_EXTENSION.match(ln):
            offenders.append((idx, ln))
        elif LOAD_CUDF_EXTENSION_COMMENTED.match(ln):
            offenders.append((idx, ln))
        elif TIME_RE.match(ln):
            offenders.append((idx, ln))
    return offenders
