#!/usr/bin/env python3
"""
Script to remove leading underscores from variable names in all code cells of a Jupyter notebook.

Usage:
    python rename_underscore_vars.py input.ipynb output.ipynb

Dependencies:
    pip install nbformat
"""

import ast
import sys

import nbformat


class RenameUnderscoreVars(ast.NodeTransformer):
    """
    AST transformer that renames variable names, function arguments, and keyword args
    by stripping leading underscores.
    """

    def visit_Name(self, node):
        # Only rename variables in load, store, or delete contexts
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)) and node.id.startswith(
            "_"
        ):
            new_id = node.id.lstrip("_")
            return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
        return node

    def visit_arg(self, node):
        # Rename function argument names
        if node.arg.startswith("_"):
            node.arg = node.arg.lstrip("_")
        return node

    def visit_keyword(self, node):
        # Rename keyword argument names
        if node.arg and node.arg.startswith("_"):
            node.arg = node.arg.lstrip("_")
        self.generic_visit(node)
        return node


def process_code(source):
    """
    Parse the source code into an AST, apply renaming, and unparse back to code.
    If parsing or unparsing fails, return the original source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    transformer = RenameUnderscoreVars()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        # Use built-in unparse (Python 3.9+)
        return ast.unparse(new_tree)
    except AttributeError:
        # Fallback to astor if unparse is unavailable
        import astor

        return astor.to_source(new_tree)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python rename_underscore_vars_notebook.py input.ipynb output.ipynb"
        )
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    # Read notebook without converting cell types
    nb = nbformat.read(input_path, as_version=nbformat.NO_CONVERT)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = process_code(cell.source)

    nbformat.write(nb, output_path)
    print(f"Processed notebook saved to {output_path}")


if __name__ == "__main__":
    main()
