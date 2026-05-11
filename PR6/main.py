import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

EARTH_GRAVITY = 9.8


def analyze_acceleration(file_path, surface_name):
    df = pd.read_csv(file_path)

    col_name = 'Absolute acceleration (m/s^2)'

    if col_name not in df.columns:
        print(f"Error: Column '{col_name}' not in file: {file_path}")
        return None

    data = df[col_name]
    time = df['Time (s)']

    peaks, _ = find_peaks(data, height=EARTH_GRAVITY, distance=20)

    avg_impact = data[peaks].mean()

    print(f"Surface: {surface_name:10} | Avarage impact: {avg_impact:5.2f} m/s²")

    return time, data, peaks, avg_impact


surfaces = ['tile', 'laminate', 'bed']
results = {}

for s in surfaces:
    res = analyze_acceleration(f'{s}.csv', s)
    if res:
        results[s] = {'time': res[0], 'data': res[1], 'peaks': res[2], 'mean': res[3]}

if results:
    # Acceleration graph
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for s in results:
        t = results[s]['time']
        d = results[s]['data']
        plt.plot(t, d, label=s)

    plt.title('Acceleration graph')
    plt.legend()

    # averages for impact
    plt.subplot(1, 2, 2)
    labels = list(results.keys())
    means = [results[s]['mean'] for s in labels]
    plt.bar(labels, means, color=['blue', 'orange', 'green'])
    plt.title('Average acceleration/impact')
    plt.ylabel('m/s²')

    plt.tight_layout()
    plt.show()
