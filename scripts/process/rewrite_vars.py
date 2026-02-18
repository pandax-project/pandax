#!/usr/bin/env python3
# This script renames variables in a Jupyter notebook. This is because elastic notebook currently does not support
# re-assigning variables. In those cases, we need to rename the variables to avoid conflicts.
# To use:
# python rewrite_vars.py input.ipynb output.ipynb --suffix _something
# Note that this file is GPT generated and has bugs. So you probably want to manually check the output.
import argparse
import ast
import logging

import nbformat

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ScopedRename(ast.NodeTransformer):
    """
    Renames all Name nodes according to `mapping`, but does NOT descend
    into function or lambda bodies.
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_FunctionDef(self, node):
        return node  # skip renaming inside functions

    def visit_Lambda(self, node):
        return node  # skip renaming inside lambdas


def extract_targets(node):
    """Recursively pull out simple variable names from an AST target."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names = []
        for elt in node.elts:
            names += extract_targets(elt)
        return names
    return []


def collect_assigned_names(stmt):
    """Return a list of all names assigned at top level by this stmt."""
    if isinstance(stmt, ast.Assign):
        names = []
        for tgt in stmt.targets:
            names += extract_targets(tgt)
        return names
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def rename_cell(cell, defined, global_map, suffix, idx):
    """
    - `defined`: set of var-names known from *earlier* cells.
    - `global_map`: var → new_name for *all* renames that have already happened.
    Returns updated source, and mutates `defined` & `global_map`.
    """
    tree = ast.parse(cell.source)
    new_body = []

    for stmt in tree.body:
        # check if this is a re-assignment of a defined var
        assigned = collect_assigned_names(stmt)
        redefs = [v for v in assigned if v in defined]

        if redefs:
            # for each var being re-defined, pick its new name
            for v in redefs:
                global_map[v] = f"{v}{suffix}{idx}"

            # 1) rewrite RHS with *old* mapping (so we don't rename this var in its own value)
            old_map = {k: global_map[k] for k in global_map if k not in redefs}
            rhs = getattr(stmt, "value", None)
            if rhs is not None:
                rhs = ScopedRename(old_map).visit(rhs)

            # 2) rebuild the assignment node, renaming LHS targets only
            if isinstance(stmt, ast.Assign):
                new_targets = []
                for tgt in stmt.targets:
                    # rename the target name if in redefs
                    def _rename_tgt(n):
                        if isinstance(n, ast.Name) and n.id in global_map:
                            n.id = global_map[n.id]
                        return n

                    new_targets.append(
                        ast.fix_missing_locations(
                            ast.NodeTransformer().generic_visit(tgt)
                        )
                    )
                    new_targets[-1] = ScopedRename(global_map).visit(new_targets[-1])
                new = ast.Assign(new_targets, rhs)
            elif isinstance(stmt, ast.AnnAssign):
                new = ast.AnnAssign(
                    target=ScopedRename(global_map).visit(stmt.target),
                    annotation=stmt.annotation,
                    value=rhs,
                    simple=stmt.simple,
                )
            elif isinstance(stmt, ast.AugAssign):
                new = ast.AugAssign(
                    target=ScopedRename(global_map).visit(stmt.target),
                    op=stmt.op,
                    value=rhs,
                )
            else:
                new = stmt  # fallback

            ast.copy_location(new, stmt)
            ast.fix_missing_locations(new)
            new_body.append(new)

        else:
            # normal stmt: apply all existing renames
            new = ScopedRename(global_map).visit(stmt)
            ast.fix_missing_locations(new)
            new_body.append(new)

        # after handling the stmt, mark *all* assigned names as defined
        for v in assigned:
            defined.add(v)

    # emit back to source
    new_tree = ast.Module(body=new_body, type_ignores=[])
    cell.source = ast.unparse(new_tree)


def process_notebook(nb, suffix="_cell"):
    defined = set()
    global_map = {}

    for idx, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue
        logging.info(f"→ Processing cell {idx}")
        rename_cell(cell, defined, global_map, suffix, idx)

    return nb


def main():
    parser = argparse.ArgumentParser(
        description="Rename re-assigned variables in Jupyter cells (v5)."
    )
    parser.add_argument("input_ipynb", help="Path to input notebook")
    parser.add_argument("output_ipynb", help="Path to write transformed notebook")
    parser.add_argument(
        "--suffix",
        default="_cell",
        help="Suffix to append before the cell number (default: _cell)",
    )
    args = parser.parse_args()

    nb = nbformat.read(args.input_ipynb, as_version=4)
    nb = process_notebook(nb, suffix=args.suffix)
    nbformat.write(nb, args.output_ipynb)
    print(f"✔️  Written rewritten notebook to {args.output_ipynb}")


if __name__ == "__main__":
    main()
