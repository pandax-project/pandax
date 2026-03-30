import ast
import json
import os
from pathlib import Path

from utils.benchmarks import BENCHMARKS_TO_PATHS

LOADERS = {"read_csv": "csv", "read_parquet": "parquet", "read_table": "table"}
TARGET_NOTEBOOKS = {"bench.ipynb", "small_bench.ipynb"}


def _path_file_parent_levels(node):
    """
    Return number of trailing `.parent` accesses for expressions like:
      Path(__file__).parent
      Path(__file__).parent.parent
    Returns None if node is not that shape.
    """
    levels = 0
    cur = node
    while isinstance(cur, ast.Attribute) and cur.attr == "parent":
        levels += 1
        cur = cur.value

    if (
        levels > 0
        and isinstance(cur, ast.Call)
        and isinstance(cur.func, ast.Name)
        and cur.func.id == "Path"
        and len(cur.args) == 1
        and isinstance(cur.args[0], ast.Name)
        and cur.args[0].id == "__file__"
    ):
        return levels
    return None


def _resolve_path_arg(arg_node, nb_path):
    """
    Resolve file path arg from ast node.
    Supports:
      - string literals: "data.csv"
      - f-strings like: f"{Path(__file__).parent}/data.csv"
    Returns path string or None if unsupported dynamic expression.
    """
    if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
        return arg_node.value

    if isinstance(arg_node, ast.JoinedStr):
        parts = []
        for v in arg_node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
                continue

            if isinstance(v, ast.FormattedValue):
                parent_levels = _path_file_parent_levels(v.value)
                if parent_levels is None:
                    return None

                base = os.path.dirname(nb_path)
                for _ in range(parent_levels - 1):
                    base = os.path.dirname(base)
                parts.append(base)
                continue

            return None
        return "".join(parts)

    return None


def _extract_local_string_bindings(tree: ast.AST) -> dict[str, str]:
    """Collect simple in-cell `name = "value"` bindings."""
    bindings: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            bindings[target.id] = node.value.value
    return bindings


def _merge_string_bindings(
    existing: dict[str, str], tree: ast.AST
) -> dict[str, str]:
    """Update notebook-level bindings with simple in-cell string assignments."""
    merged = dict(existing)
    merged.update(_extract_local_string_bindings(tree))
    return merged


def _resolve_benchmark_key_expr(
    node: ast.AST, local_bindings: dict[str, str]
) -> str | None:
    # Support either direct literal keys or simple variables defined in the cell.
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return local_bindings.get(node.id)
    return None


def _resolve_path_expr(
    node: ast.AST, nb_path: str, local_bindings: dict[str, str]
) -> str | None:
    """Resolve common path expressions to concrete strings.

    This intentionally handles only predictable expression shapes so we avoid
    guessing dynamic runtime values.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # Path(__file__)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        arg0 = node.args[0]
        if isinstance(arg0, ast.Name) and arg0.id == "__file__":
            return nb_path

        # Path(BENCHMARKS_TO_PATHS[benchmark_name])
        if (
            isinstance(arg0, ast.Subscript)
            and isinstance(arg0.value, ast.Name)
            and arg0.value.id == "BENCHMARKS_TO_PATHS"
        ):
            benchmark_key = _resolve_benchmark_key_expr(arg0.slice, local_bindings)
            if benchmark_key and benchmark_key in BENCHMARKS_TO_PATHS:
                return BENCHMARKS_TO_PATHS[benchmark_key]
        return None

    # `<path>.parent`
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _resolve_path_expr(node.value, nb_path, local_bindings)
        if base is None:
            return None
        return str(Path(base).parent)

    # Resolve `Path(...) / "input" / "file.csv"` style joins.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_path_expr(node.left, nb_path, local_bindings)
        right = _resolve_path_expr(node.right, nb_path, local_bindings)
        if left is None or right is None:
            return None
        return str(Path(left) / right)

    # f"{...}" where ... is a resolvable path expression
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
                continue
            if isinstance(v, ast.FormattedValue):
                resolved = _resolve_path_expr(v.value, nb_path, local_bindings)
                if resolved is None:
                    return None
                parts.append(resolved)
                continue
            return None
        return "".join(parts)

    return None


def find_data_calls_in_notebook(nb_path):
    """
    Parses each code cell with ast, returns a list of tuples:
      (loader, rel_path, args_list, kwargs_json)
    where loader is 'csv'|'parquet'|'table'.
    """
    calls = []
    # Track simple string assignments across cells in notebook order.
    notebook_bindings: dict[str, str] = {}
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        # Build per-cell bindings view from values known up to this point plus
        # any simple assignments in the current cell.
        local_bindings = _merge_string_bindings(notebook_bindings, tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "pd":
                    fn = node.func.attr
                    if fn in LOADERS:
                        # Must have at least one positional arg for filepath.
                        if not node.args:
                            continue

                        # First try the richer resolver (Path joins, BENCHMARKS mapping).
                        rel_path = _resolve_path_expr(
                            node.args[0], nb_path, local_bindings
                        )
                        if not rel_path:
                            # Fallback keeps legacy support for old f-string patterns.
                            rel_path = _resolve_path_arg(node.args[0], nb_path)
                        if not rel_path:
                            continue

                        extra_args = [ast.literal_eval(arg) for arg in node.args[1:]]
                        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                        kwargs_json = json.dumps(kwargs, sort_keys=True)
                        calls.append((LOADERS[fn], rel_path, extra_args, kwargs_json))
        # Persist newly seen bindings for later cells.
        notebook_bindings = local_bindings
    return calls


def gather_data_files(base_dir, target_notebooks=None, verbose=False):
    """
    Walk base_dir looking for target notebooks and return a set of:
      (abs_path, loader, args, kwargs_json)
    """
    notebook_names = target_notebooks or TARGET_NOTEBOOKS
    files = set()
    for root, _, fnames in os.walk(base_dir):
        for fname in fnames:
            if fname not in notebook_names:
                continue
            nb_path = os.path.abspath(os.path.join(root, fname))
            if verbose:
                print(f"Processing notebook: {nb_path}")
            for loader, rel_path, args, kw_json in find_data_calls_in_notebook(nb_path):
                abs_path = (
                    rel_path
                    if os.path.isabs(rel_path)
                    else os.path.normpath(os.path.join(root, rel_path))
                )
                files.add((abs_path, loader, tuple(args), kw_json))
    return files
