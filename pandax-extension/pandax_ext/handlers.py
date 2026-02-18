"""Jupyter server example handlers."""

import json
import os
import sys

import nbformat
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.extension.handler import ExtensionHandlerMixin
from nbconvert import HTMLExporter
from tornado.web import HTTPError

# Add the parent directory that contains both 'prototype' and 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.rewrite_cpu_html import rewrite_cpu_html


class PandaXHandler(ExtensionHandlerMixin, JupyterHandler):
    """Default API handler."""

    auth_resource = "simple_ext1:default"

    def authorized(self, action="read"):
        # Always allow
        return True

    # TODO (yingan): figure out if need @authorized decorator here.
    def get(self):
        """GET endpoint that displays a Jupyter notebook as HTML."""
        # Get file path from query parameters: ?path=/full/path/to/notebook.ipynb
        file_path = self.get_argument("path", None)
        if not file_path:
            raise HTTPError(400, reason="Query parameter 'path' is required.")

        try:
            # Load the notebook
            with open(file_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Convert notebook to HTML
            html_exporter = HTMLExporter()
            body, _ = html_exporter.from_notebook_node(nb)

            # Return HTML
            self.set_header("Content-Type", "text/html")
            self.finish(body)

        except FileNotFoundError:
            raise HTTPError(404, reason=f"Notebook not found: {file_path}")
        except Exception as e:
            raise HTTPError(500, reason=f"Error reading notebook: {str(e)}")

    async def post(self):
        """
        POST handler that:
        1. rewrite input Jupyter notebook and save it to a new notebook,
        2. displays the new Jupyter notebook as HTML.

        Simple rewrite: just display a new notebook with one cell with comment "# rewritten with cpu".
        """
        data = json.loads(self.request.body.decode("utf-8"))
        file_path = data.get("path")
        start_cell_idx = data.get("startCellIdx")
        num_tries_per_cell = data.get("numRewriteTries")
        if not file_path:
            raise HTTPError(400, reason="Query parameter 'path' is required.")

        try:
            (
                rewritted_path,
                diff,
                diff_path,
                rewritten_execution_times,
                original_execution_times,
            ) = await rewrite_cpu_html(file_path, start_cell_idx, num_tries_per_cell)
            response = {
                "rewritten_notebook_path": rewritted_path,
                "diffJson": diff,
                "diff_path": diff_path,
                "rewritten_execution_times": rewritten_execution_times,
                "original_execution_times": original_execution_times,
            }

            self.set_header("Content-Type", "application/json")
            self.finish(json.dumps(response))

        except Exception as e:
            raise HTTPError(500, reason=f"Failed to rewrite notebook: {str(e)}")
