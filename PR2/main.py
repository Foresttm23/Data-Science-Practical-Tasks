import multiprocessing as mp
import time
from time import perf_counter

import numpy as np
import pandas as pd


def analyze_efficiency(data_chunk):
    id_name, group = data_chunk

    time.sleep(0.01)  # Since calculations are simple, add sleep time to showcase the speed differences.

    mean_val = group['energy_mean']
    max_val = group['energy_max']
    std_val = group['energy_std']

    load_factors = np.where(max_val > 0, mean_val / max_val, 0)
    avg_load_factor = np.mean(load_factors)

    var_calc = np.where(mean_val > 0, std_val / mean_val, 0)
    variability = np.mean(var_calc)

    total_energy = group['energy_sum'].sum()

    return {
        'LCLid': id_name,
        'Efficiency_Score': round(float(avg_load_factor), 4),
        'Stability_Score': round(float(1 / (1 + variability)), 4),
        'Total_KWh': round(float(total_energy), 2)
    }


def prepare_data(df):
    initial_count = len(df)
    # Обов'язково очищаємо критичні дані
    df = df.dropna(subset=['energy_mean', 'energy_max', 'energy_std'])
    print(f"Deleted {initial_count - len(df)} with missing values")

    grouped_data = list(df.groupby('LCLid'))
    print(f"Unique values: {len(grouped_data)}")
    return grouped_data


def run_sequential_processing(grouped_data):
    results = [analyze_efficiency(item) for item in grouped_data]
    return pd.DataFrame(results)


def run_parallel_processing(grouped_data):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(analyze_efficiency, grouped_data, chunksize=100)
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("daily_dataset.csv")
    prepared_data = prepare_data(df)

    print("\nParallel processing...")
    par_start = perf_counter()
    par_final_results = run_parallel_processing(prepared_data)
    par_end = perf_counter()

    print("Sequential processing...")
    seq_start = perf_counter()
    seq_final_results = run_sequential_processing(prepared_data)
    seq_end = perf_counter()

    print("\n--- Best 5 ---")
    print(par_final_results.nlargest(5, 'Efficiency_Score'))

    print(f"\nParallel time: {par_end - par_start:.4f} сек")
    print(f"Sequential time: {seq_end - seq_start:.4f} сек")

    speedup = (seq_end - seq_start) / (par_end - par_start)
    print(f"Speedup: {speedup:.2f}x")
