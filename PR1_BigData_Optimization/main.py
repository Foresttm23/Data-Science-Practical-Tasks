import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# DATASET USED https://huggingface.co/datasets/laion/relaion2B-en-research-safe/blob/main/part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet

SAMPLE_FILE_NAME = "part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet"  # Assuming the installed file is from the link above
SAMPLE_SELECTED_COLS_FILE_NAME = "metadata/selected_cols.json"  # Where the selected columns are saved


def get_selected_cols(metadata_file_name: str, data_file_name: str, cols: list) -> list:
    output_path = Path(metadata_file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if Path.exists(output_path):
        selected = json.loads(output_path.read_text())
    else:
        df = pd.read_parquet(data_file_name)
        selected = select_cols(df, cols)
        with open(output_path, "w") as f:
            json.dump(selected, f)

    print(f"\n\nSelected columns: {selected}")
    return selected


def select_cols(df: pd.DataFrame, cols: list) -> list:
    selected = []

    for col in cols:
        unique_count = df[col].nunique()
        total_count = len(df)
        print(f"Column '{col}':\n {unique_count} unique values out of {total_count}\n\n")

        unique_percentage = unique_count / total_count * 100
        if unique_percentage < 20:
            selected.append(col)

    print(selected)
    return selected


def categorize_df(df: pd.DataFrame, cols: list) -> None:
    mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"\n\nMemory before categorizing: {mem_before:.2f} MB")

    for col in cols:
        df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory after categorizing: {mem_after:.2f} MB")

    reduction = (1 - mem_after / mem_before) * 100
    print(f"% Memory saved: {reduction:.2f}%\n\n")
    print(f"Columns are now of type: \n{df.dtypes}")


def calculate_area_apply(row):
    return row['width'] * row['height']


def calculate_area_vectorize(df):
    return df['width'] * df['height']


def compare_apply_vs_vectorize(df):
    start_apply = time.perf_counter()
    df.apply(calculate_area_apply, axis=1)
    end_apply = time.perf_counter()

    res_apply_time = end_apply - start_apply
    print(f"\n\nApply time: {res_apply_time:.2f} seconds")

    start_vectorize = time.perf_counter()
    calculate_area_vectorize(df)
    end_vectorize = time.perf_counter()

    res_vectorize_time = end_vectorize - start_vectorize
    print(f"Vectorize time: {res_vectorize_time:.2f} seconds")


if __name__ == "__main__":
    schema = pq.read_schema(SAMPLE_FILE_NAME)

    all_columns = schema.names
    print(f"All sample colons names:\n{all_columns}")

    selected_cols = get_selected_cols(SAMPLE_SELECTED_COLS_FILE_NAME, SAMPLE_FILE_NAME, all_columns)
    # selected_cols = ['status']

    data_df = pd.read_parquet(SAMPLE_FILE_NAME, columns=selected_cols)

    columns_to_copy = ['width', 'height']
    copy_df = data_df[columns_to_copy].copy()

    categorize_df(data_df, selected_cols)
    compare_apply_vs_vectorize(copy_df)
