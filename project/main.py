import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataCleaningAgent:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        self.scaler = StandardScaler()

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method="linear") -> pd.DataFrame:
        """
        Interpolation
        """
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method=method, limit_direction='both')

        if df_cleaned.isnull().values.any():
            df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))

        return df_cleaned

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Isolation Forest.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        scaled_data = self.scaler.fit_transform(numeric_df)

        preds = self.iso_forest.fit_predict(scaled_data)

        df_result = df.copy()
        df_result['is_anomaly'] = preds
        return df_result

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Starting pipeline...")

        df_filled = self.handle_missing_values(df)

        df_with_anomalies = self.detect_anomalies(df_filled)

        clean_df = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1].drop(columns=['is_anomaly'])

        print(f"Pipeline finished. Removed {len(df) - len(clean_df)} anomalies/noise points.")
        return clean_df


def main():
    data = {
        'feature_1': [10, 12, np.nan, 14, 15, 100, 16, 18],
        'feature_2': [20, np.nan, 22, 23, 25, 24, 26, 1],
    }
    df_test = pd.DataFrame(data)

    agent = DataCleaningAgent(contamination=0.2)
    final_data = agent.run_pipeline(df_test)

    print("\nOriginal Data:")
    print(df_test)
    print("\nCleaned Data:")
    print(final_data)


if __name__ == "__main__":
    main()
