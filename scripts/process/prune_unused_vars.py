#!/usr/bin/env python3
"""
Script to remove unused variables and functions from a Jupyter notebook (.ipynb).
Usage:
    python remove_unused_vars.py input.ipynb output.ipynb

This script parses all code cells, identifies defined names (function and variable definitions),
computes which are never used, and then filters out the corresponding definitions from each cell.
Note: This is a heuristic approach; more complex patterns (e.g., class methods, partial imports, dynamic uses)
may require extending the logic.
"""

import argparse
import ast

import nbformat


def collect_names(node):
    """Recursively collect variable names from targets."""
    names = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.extend(collect_names(elt))
    return names


def collect_defs_and_uses(cells):
    """Collect all defined names and used names across code cells."""
    defined = set()
    used = set()
    for cell in cells:
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Definitions
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    defined.update(collect_names(tgt))
            elif isinstance(node, ast.AnnAssign):
                defined.update(collect_names(node.target))
            # Usages
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return defined, used


def filter_cell(cell, unused):
    """Remove unused definitions from a single code cell."""
    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        return cell
    new_body = []
    for stmt in tree.body:
        # Remove unused functions
        if isinstance(stmt, ast.FunctionDef) and stmt.name in unused:
            continue
        # Remove unused assignments
        if isinstance(stmt, ast.Assign):
            tgt_names = []
            for tgt in stmt.targets:
                tgt_names.extend(collect_names(tgt))
            if tgt_names and all(name in unused for name in tgt_names):
                continue
        if isinstance(stmt, ast.AnnAssign):
            tgt_names = collect_names(stmt.target)
            if tgt_names and all(name in unused for name in tgt_names):
                continue
        new_body.append(stmt)
    # Only rewrite if we removed something
    if len(new_body) != len(tree.body):
        new_tree = ast.Module(body=new_body, type_ignores=[])
        try:
            cell.source = ast.unparse(new_tree)
        except AttributeError:
            # For Python <3.9, you can install astor and use astor.to_source(new_tree)
            import astor

            cell.source = astor.to_source(new_tree)
    return cell


def main():
    parser = argparse.ArgumentParser(
        description="Remove unused vars/functions from a Jupyter notebook"
    )
    parser.add_argument("input", help="Input notebook file (.ipynb)")
    parser.add_argument("output", help="Output cleaned notebook file (.ipynb)")
    args = parser.parse_args()

    nb = nbformat.read(args.input, as_version=4)
    cells = nb.cells
    defined, used = collect_defs_and_uses(cells)
    unused = defined - used
    if unused:
        print(f"Removing unused definitions: {', '.join(sorted(unused))}")
    else:
        print("No unused definitions found.")

    new_cells = []
    for cell in cells:
        if cell.cell_type == "code":
            cell = filter_cell(cell, unused)
        new_cells.append(cell)
    nb.cells = new_cells

    nbformat.write(nb, args.output)
    print(f"Cleaned notebook written to {args.output}")


if __name__ == "__main__":
    main()
