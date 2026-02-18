import argparse
import pickle
from pathlib import Path

from utils.schedule import (
    get_transfer_times_and_input_output_df_cols,
    record_transfer_times,
)

# Make the following a command line argument.
# example usage: python scripts/generate_transfer_costs.py notebooks/ampiiere/animal-crossing-villager-popularity-analysis/src/fast_bench.ipynb
parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=str, help="Path to the notebook")
args = parser.parse_args()
original_notebook_path = Path(args.notebook)

with open(original_notebook_path.parent / "cost_model_inputs.pkl", "rb") as f:
    cost_model_inputs = pickle.load(f)

with open("cell_exec_info.pkl", "rb") as f:
    cell_exec_info = pickle.load(f)

transfer_times, input_df_cols, output_df_cols = (
    get_transfer_times_and_input_output_df_cols(
        original_notebook_path,
        cell_exec_info,
        cost_model_inputs,
    )
)
with open(original_notebook_path.parent / "transfer_times.pkl", "wb") as f:
    pickle.dump(transfer_times, f)
with open(original_notebook_path.parent / "input_df_cols.pkl", "wb") as f:
    pickle.dump(input_df_cols, f)
with open(original_notebook_path.parent / "output_df_cols.pkl", "wb") as f:
    pickle.dump(output_df_cols, f)

transfer_cost_csv_path = original_notebook_path.parent / Path("transfer_costs.csv")
record_transfer_times(transfer_times, transfer_cost_csv_path)
