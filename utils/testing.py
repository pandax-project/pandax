import pickle
from pathlib import Path

from elastic.core.common.cell_exec_info import CellExecInfo
from elastic.core.common.pandas import is_type_dataframe, is_type_df, is_type_series

from utils.notebook import get_load_checkpoint_cell


def get_test_code_from_cell_exec_info(
    original_cell_exec_info: CellExecInfo,
    opt_cell_exec_info_pkl_path: Path,
    checkpoint_path: Path,
) -> str:
    with open(opt_cell_exec_info_pkl_path, "rb") as f:
        opt_cell_exec_info = pickle.load(f)
    original_active_vars = original_cell_exec_info.active_vars
    original_future_vars = original_cell_exec_info.future_vars
    # This is because we run the rewritten cells one by one, so there will never be "active" vars since nothing
    # is used downstream.
    opt_active_vars = (
        opt_cell_exec_info.active_vars + opt_cell_exec_info.intermediate_vars
    )
    opt_active_var_names_to_types = {
        var_info.name: var_info.type for var_info in opt_active_vars
    }

    # All the original active vars must be in the rewritten active vars.
    for var_info in original_active_vars:
        if var_info.name in opt_active_var_names_to_types:
            if is_type_dataframe(var_info.type) and is_type_dataframe(
                opt_active_var_names_to_types[var_info.name]
            ):
                continue
            elif is_type_series(var_info.type) and is_type_series(
                opt_active_var_names_to_types[var_info.name]
            ):
                continue
        # if var_info not in opt_active_vars:
        #     return f"assert False, '{var_info.name} should be rewritten in the optimized code.'"

    # All the rewritten active vars must not be in the original future vars.
    for var_info in opt_active_vars:
        if var_info in original_future_vars:
            return f"assert False, '{var_info.name} is incorrectly modified in the optimized code.'"

    # First, we need to rename the optimized variables.
    rename_opt_vars_code = [
        f"{var_info.name}_opt = {var_info.name}" for var_info in original_active_vars
    ]

    # Then we need to load the original checkpoint.
    load_checkpoint_code = get_load_checkpoint_cell(checkpoint_path).source

    # We only need to compare the active vars.
    compare_vars_code: list[str] = []
    for var_info in original_active_vars:
        if is_type_df(var_info.type):
            # FIXME(sahil): this is a hack.
            # compare_vars_code.append(f"assert compare_df({var_info.name}_opt, {var_info.name})")
            continue
        else:
            compare_vars_code.append(f"assert {var_info.name}_opt == {var_info.name}")
        # FIXME(jie): handle styler objects here as well.

    # Compare the outputs.
    # TODO(jie): this can be potentially combined with the above code.
    compare_output_code = """
import numpy as np
if os.getenv("USE_GPU") == "True":
    import cudf
from elastic.core.common.pandas import is_type_styler
is_orig_output_pd = isinstance(orig_output, (pd.Series, pd.DataFrame, pd.Index))
is_opt_output_pd = isinstance(opt_output, (pd.Series, pd.DataFrame, pd.Index))
if os.getenv("USE_GPU") == "True":
    is_orig_output_array = isinstance(orig_output, (cudf.pandas._wrappers.numpy.ndarray, np.ndarray))
    is_opt_output_array = isinstance(opt_output, (cudf.pandas._wrappers.numpy.ndarray, np.ndarray))
else:
    is_orig_output_array = isinstance(orig_output, np.ndarray)
    is_opt_output_array = isinstance(opt_output, np.ndarray)

is_orig_output_styler = is_type_styler(type(orig_output))
is_opt_output_styler = is_type_styler(type(opt_output))
if is_orig_output_styler and is_opt_output_styler:
    assert orig_output.to_html() == opt_output.to_html()
elif is_orig_output_styler:
    assert orig_output.to_html() == opt_output.to_html()
elif is_opt_output_styler:
    assert opt_output.to_html() == orig_output

if is_orig_output_pd and is_opt_output_pd:
    assert orig_output.equals(opt_output)
# TODO(jie): this is a hack.
elif ((is_orig_output_pd or is_opt_output_pd) and (is_orig_output_array or is_opt_output_array)) or (is_orig_output_array and is_opt_output_array):
    assert list(orig_output) == list(opt_output)
else:
    assert orig_output == opt_output
"""

    return "\n".join(
        [
            *rename_opt_vars_code,
            load_checkpoint_code,
            *compare_vars_code,
            compare_output_code,
        ]
    )
