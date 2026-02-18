import time

import IPython
from IPython.display import Markdown, display


@IPython.core.magic.register_line_magic
def include(file):
    f = open(file, "r")
    contents = f.read()
    f.close()
    IPython.get_ipython().run_cell(contents)


@IPython.core.magic.register_cell_magic
def time_cell(line, cell):
    ip = IPython.get_ipython()
    start = time.perf_counter()
    ip.run_cell(cell)
    end = time.perf_counter()
    ip.user_ns["_TIMED_CELL"] = end - start


def print_md(s):
    display(Markdown(s))
