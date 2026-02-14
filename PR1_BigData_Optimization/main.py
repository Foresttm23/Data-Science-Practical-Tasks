import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# DATASET USED https://huggingface.co/datasets/laion/relaion2B-en-research-safe/blob/main/part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet

SAMPLE_FILE_NAME = "part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet"  # Assuming the installed file is from the link above
SELECTED_COLS_FILE_NAME = "metadata/selected_cols.json"  # Where the selected columns are saved
PREVIEW_FILE_NAME = "metadata/preview_sample.json"


def get_selected_cols(out_path: Path, sample_file_name: str, cols: list) -> list:
    if Path.exists(out_path):
        selected = json.loads(out_path.read_text())
    else:
        df = pd.read_parquet(sample_file_name)
        selected = select_cols(df, cols)
        with open(out_path, "w") as f:
            json.dump(selected, f)

    print(f"\n\nSelected columns: {selected}")
    return selected


def select_cols(df: pd.DataFrame, cols: list) -> list:
    selected = []

    for col in cols:
        unique_count: int = df[col].nunique()
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


def save_preview_data(out_path: Path, sample_file_name: str) -> None:
    if not Path.exists(out_path):
        preview_df = pd.read_parquet(sample_file_name, engine='pyarrow').head(10)
        preview_df.to_json(out_path, orient='records', indent=4)


if __name__ == "__main__":
    schema = pq.read_schema(SAMPLE_FILE_NAME)

    output_path = Path(PREVIEW_FILE_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_data(output_path, SAMPLE_FILE_NAME)

    all_columns = schema.names
    print(f"All sample colons names:\n{all_columns}")

    output_path = Path(SELECTED_COLS_FILE_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_cols = get_selected_cols(output_path, SAMPLE_FILE_NAME, all_columns)
    # selected_cols = ['status']

    sample_df = pd.read_parquet(SAMPLE_FILE_NAME, columns=selected_cols)
    columns_to_copy = ['width', 'height']
    copy_df = sample_df[columns_to_copy].copy()

    categorize_df(sample_df, selected_cols)
    compare_apply_vs_vectorize(copy_df)
