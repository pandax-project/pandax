import concurrent.futures
import os
import random
import string  # Needed for generating random characters

import numpy as np
import pandas as pd
from tqdm import tqdm


# --- Original Functions (Keep them) ---
def generate_column_data(n_rows, col_type):
    """
    Generate a column of random data with n_rows elements.
    For 'int', random integers between 0 and 99 are generated.
    For 'float', random floats between 0 and 1 are generated.
    """
    if col_type == "int":
        return np.random.randint(0, 100, size=n_rows)
    elif col_type == "float":  # Assumes float
        return np.random.rand(n_rows)
    else:
        return np.random.choice([True, False], size=n_rows)


def generate_dataframe():
    """
    Generate a DataFrame with:
      - A random number of rows (between 100,000 and 500,000)
      - A random number of columns (between 1 and 10)
      - Each column randomly chosen as 'int', 'float', or 'string'
    """
    n_rows = random.randint(1000, 1000000)
    n_cols = random.randint(1, 1)
    # for mixed
    # col_types = [random.choice(['int', 'float', 'string']) for _ in range(n_cols)]
    col_types = ["bool" for _ in range(n_cols)]
    results = []

    for i, col_type in enumerate(col_types):
        if col_type == "int":
            col_data = generate_column_data(n_rows, "int")
        elif col_type == "float":
            col_data = generate_column_data(n_rows, "float")
        elif col_type == "string":
            col_data = generate_string_column_data(n_rows)
        elif col_type == "bool":
            col_data = generate_column_data(n_rows, "bool")
        else:
            raise ValueError(f"Unsupported column type: {col_type}")

        col_name = f"col_{i + 1}_{col_type}"
        results.append((col_name, col_data))

    df = pd.DataFrame({name: data for name, data in results})

    # Cast string columns to pandas 'string' dtype explicitly
    for col in df.columns:
        if col.endswith("_string"):
            df[col] = df[col].astype("string")  # Can use 'object' if desired

    return df


def generate_and_dump_dataframe(index, output_dir):
    """
    Generate a mixed-type DataFrame, dump it to a Parquet file, return filename.
    """
    df = generate_dataframe()
    filename = os.path.join(output_dir, f"dataframe_{index + 1}.parquet")
    df.to_parquet(filename, compression="snappy")
    return filename


# --- NEW Functions for String DataFrames ---


def generate_string_column_data(n_rows, string_pool_size=500, min_len=5, max_len=20):
    """
    Generates a column (numpy array) of random strings sampled from a pool.
    Using a pool is generally more memory-efficient and faster for large n_rows
    than generating unique strings for every cell.
    """
    # 1. Create a pool of unique random strings
    # Using letters and digits for variety
    characters = string.ascii_letters + string.digits
    string_pool = set()  # Use a set to ensure uniqueness initially
    while len(string_pool) < string_pool_size:
        length = random.randint(min_len, max_len)
        random_str = "".join(random.choice(characters) for _ in range(length))
        string_pool.add(random_str)
    string_pool_list = list(string_pool)  # Convert back to list for numpy.random.choice

    # 2. Sample from the pool with replacement to fill the column
    # This is much faster than generating unique strings per row
    return np.random.choice(string_pool_list, size=n_rows, replace=True)


def generate_categorical_dataframe(
    min_rows=1000, max_rows=500000, min_cols=1, max_cols=1
):
    num_categories = np.random.randint(2, 10)
    if categories is None:  # noqa: F821
        categories = [f"cat_{i + 1}" for i in range(num_categories)]
    # random choice with replacement:
    values = np.random.choice(categories, size=n_rows)  # noqa: F821
    return pd.Categorical(values, categories=categories)


def generate_string_dataframe(min_rows=1000, max_rows=1000000, min_cols=1, max_cols=1):
    """
    Generate a DataFrame with:
      - A random number of rows (between min_rows and max_rows)
      - A random number of columns (between min_cols and max_cols)
      - Each column is filled with random string data sampled from a pool.
    """
    n_rows = random.randint(min_rows, max_rows)
    n_cols = random.randint(min_cols, max_cols)

    # Generate columns using the string data generator function
    results = [
        generate_string_column_data(n_rows) for _ in range(n_cols)
    ]  # Defaults for pool/len used

    # Create column names indicating string type
    col_names = [f"col_{i + 1}_string" for i in range(n_cols)]

    # Create DataFrame
    df = pd.DataFrame({name: data for name, data in zip(col_names, results)})
    # Important: Explicitly set dtype to 'string' (Pandas extension type) or 'object'
    # 'string' is generally preferred for actual string data.
    for col in df.columns:
        df[col] = df[col].astype("string")  # Or df[col].astype(object)
    return df


def generate_object_dataframe(min_rows=1000, max_rows=1000000, min_cols=1, max_cols=1):
    """
    Generate a DataFrame where each column is an object type:
    - Either a categorical column (Pandas Categorical)
    - Or a random string column (object dtype)

    Returns a DataFrame with mixed object-type columns.
    """
    n_rows = random.randint(min_rows, max_rows)
    n_cols = random.randint(min_cols, max_cols)

    results = {}
    for i in range(n_cols):
        col_type = "string"  # random.choice(['categorical', 'string'])  # randomly choose column type

        if col_type == "categorical":
            num_categories = random.randint(2, 10)
            categories = [f"cat_{j + 1}" for j in range(num_categories)]
            values = np.random.choice(categories, size=n_rows)  # noqa: F821,F841
            #             col_data = pd.Categorical(values, categories=categories)
            col_data = np.random.choice(categories, size=n_rows).astype(object)
            # print(col_data)
        else:  # random string
            col_data = generate_string_column_data(n_rows)
            col_data = col_data.astype(object)  # Ensure 'object' dtype

        col_name = f"col_{i + 1}_{col_type}"
        results[col_name] = col_data
    # print("generated data frame")
    return pd.DataFrame(results)


def generate_and_dump_object_dataframe(index, output_dir):
    """
    Generate a object DataFrame, dump it to a Parquet file, return filename.
    """
    try:
        df = generate_object_dataframe()  # Call the specific string generator
    except Exception as e:
        print(f"generating data frame failed {e}")
    # Optional: Give string dataframes a distinct naming pattern
    filename = os.path.join(output_dir, f"object_dataframe_{index + 1}.parquet")
    # Parquet handles pandas 'string' dtype well
    # print("here1")
    df.to_parquet(filename, compression="snappy")
    return filename


def generate_and_dump_string_dataframe(index, output_dir):
    """
    Generate a STRING DataFrame, dump it to a Parquet file, return filename.
    """
    df = generate_string_dataframe()  # Call the specific string generator
    # Optional: Give string dataframes a distinct naming pattern
    filename = os.path.join(output_dir, f"string_dataframe_{index + 1}.parquet")
    # Parquet handles pandas 'string' dtype well

    df.to_parquet(filename, compression="snappy")
    return filename


# --- New helper for datetime data ---
def generate_datetime_column_data(
    n_rows, start_date: str = "1990-01-01", end_date: str = "2025-12-31"
):
    """
    Generate an array of random Timestamps between start_date and end_date.
    Uses nanosecond resolution via pandas.Timestamp.value.
    """
    # Convert to nanosecond‐since‐epoch integers
    start_ns = pd.Timestamp(start_date).value
    end_ns = pd.Timestamp(end_date).value
    # Sample uniformly in that range
    rand_ns = np.random.randint(start_ns, end_ns + 1, size=n_rows)
    # Convert back to datetime64[ns]
    return pd.to_datetime(rand_ns)


# --- New DataFrame generator for datetime columns ---
def generate_datetime_dataframe(
    min_rows=1000,
    max_rows=1000000,
    min_cols=1,
    max_cols=1,
    start_date="2000-01-01",
    end_date="2025-12-31",
):
    """
    Generate a DataFrame with:
      - A random number of rows (between min_rows and max_rows)
      - A random number of columns (between min_cols and max_cols)
      - Each column filled with random datetimes in [start_date, end_date]
    """
    n_rows = random.randint(min_rows, max_rows)
    n_cols = random.randint(min_cols, max_cols)

    data = {
        f"col_{i + 1}_datetime": generate_datetime_column_data(
            n_rows, start_date, end_date
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


# --- New dump function ---
def generate_and_dump_datetime_dataframe(index, output_dir):
    """
    Generate a datetime‐only DataFrame, dump it to a Parquet file, return filename.
    """
    df = generate_datetime_dataframe()
    filename = os.path.join(output_dir, f"datetime_dataframe_{index + 1}.parquet")
    df.to_parquet(filename, compression="snappy")
    return filename


# --- Main Execution Logic ---


def main(generate_type="mixed"):  # Add argument to control type
    num_dataframes = 10000  # Or read from args
    output_dir = (
        "/home/jupyter/generated_dataframes_object_columns_new3"  # Use appropriate path
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_dataframes} DataFrames of type: {generate_type}")

    # Choose the target function based on the desired type
    if generate_type == "string":
        target_func = generate_and_dump_string_dataframe
        desc = "Generating String DataFrames"
    elif generate_type == "mixed":
        target_func = generate_and_dump_dataframe
        desc = "Generating Mixed DataFrames"
    elif generate_type == "boolean":
        target_func = generate_and_dump_dataframe
        desc = "Generating Boolean DataFrames"
    elif generate_type == "object":
        target_func = generate_and_dump_object_dataframe
        desc = "Generating object DataFrames"
    elif generate_type == "datetime":
        target_func = generate_and_dump_datetime_dataframe
        desc = "Generating datetime DataFrames"
    else:
        print(
            f"Error: Unknown generate_type '{generate_type}'. Choose 'mixed' or 'string'."
        )
        return

    # Parallelize DataFrame generation and saving.
    # max_workers can be adjusted based on your CPU cores
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count() // 2
    ) as executor:
        futures = [
            executor.submit(target_func, i, output_dir) for i in range(num_dataframes)
        ]
        # Wrap as_completed with tqdm to show progress.
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=num_dataframes, desc=desc
        ):
            try:
                future.result()  # Check for exceptions during generation/saving
            except Exception as e:
                print(f"\nError generating/saving a DataFrame: {e}")

    print(
        f"\nAll {generate_type} DataFrames have been generated and saved to {output_dir}."
    )


if __name__ == "__main__":
    # --- Choose which type of DataFrame to generate ---
    # main(generate_type='mixed')
    main(generate_type="object")
    # Or use command-line arguments to select
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--type', type=str, default='mixed', choices=['mixed', 'string'])
    # args = parser.parse_args()
    # main(generate_type=args.type)
