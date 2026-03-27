import ast
import json
import os

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


def find_data_calls_in_notebook(nb_path):
    """
    Parses each code cell with ast, returns a list of tuples:
      (loader, rel_path, args_list, kwargs_json)
    where loader is 'csv'|'parquet'|'table'.
    """
    calls = []
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "pd":
                    fn = node.func.attr
                    if fn in LOADERS:
                        # Must have at least one positional arg for filepath.
                        if not node.args:
                            continue

                        rel_path = _resolve_path_arg(node.args[0], nb_path)
                        if not rel_path:
                            continue

                        extra_args = [ast.literal_eval(arg) for arg in node.args[1:]]
                        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                        kwargs_json = json.dumps(kwargs, sort_keys=True)
                        calls.append((LOADERS[fn], rel_path, extra_args, kwargs_json))
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
