import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# ------------------------------------------------------
# Load model + scaler
# ------------------------------------------------------
model = pickle.load(open("model_output/model.pkl", "rb"))
scaler = pickle.load(open("model_output/scaler.pkl", "rb"))

# ------------------------------------------------------
# Feature engineering for full CSV
# ------------------------------------------------------
def build_features_for_csv(df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["UserID", "Time"])

    all_rows = []

    for user, group in df.groupby("UserID"):
        group = group.sort_values("Time").reset_index(drop=True)

        tx_times = group["Time"]
        amounts = group["Amount"]

        tx_count_5m = []
        tx_count_1h = []
        avg_amount_24h = []
        amount_z = []

        for i in range(len(group)):
            t = tx_times[i]

            mask5 = (tx_times >= t - timedelta(minutes=5)) & (tx_times < t)
            tx_count_5m.append(mask5.sum())

            mask1 = (tx_times >= t - timedelta(hours=1)) & (tx_times < t)
            tx_count_1h.append(mask1.sum())

            mask24 = (tx_times >= t - timedelta(hours=24)) & (tx_times <= t)
            avg_amount_24h.append(amounts[mask24].mean())

            mean_amt = amounts[: i + 1].mean()
            std_amt = amounts[: i + 1].std() if amounts[: i + 1].std() > 0 else 1
            amount_z.append((amounts[i] - mean_amt) / std_amt)

        group["tx_count_5m"] = tx_count_5m
        group["tx_count_1h"] = tx_count_1h
        group["avg_amount_24h"] = avg_amount_24h
        group["amount_zscore"] = amount_z

        sec = group["Time"].dt.hour * 3600 + group["Time"].dt.minute * 60 + group["Time"].dt.second
        group["tod_sin"] = np.sin(2 * np.pi * sec / 86400)
        group["tod_cos"] = np.cos(2 * np.pi * sec / 86400)

        group["missing_location"] = group["Location"].isna().astype(int)

        all_rows.append(group)

    final = pd.concat(all_rows)

    X = final[[
        "Amount",
        "amount_zscore",
        "tx_count_5m",
        "tx_count_1h",
        "avg_amount_24h",
        "tod_sin",
        "tod_cos",
        "missing_location",
    ]]

    return X, final


# ------------------------------------------------------
# Main process
# ------------------------------------------------------
def predict_batch(input_csv="transactions_unsupervised.csv", output_csv="fraud_results.csv"):
    print(" Loading input file...")
    df = pd.read_csv(input_csv)

    print(" Building features for all rows...")
    X, df_full = build_features_for_csv(df)

    print(" Scaling...")
    X_scaled = scaler.transform(X)

    print(" Predicting...")
    predictions = model.predict(X_scaled)          # -1 = anomaly
    scores = model.decision_function(X_scaled)     # lower = more suspicious

    df_full["fraud_score"] = scores
    df_full["prediction"] = ["FRAUD" if p == -1 else "NORMAL" for p in predictions]

    print(" Saving output file...")
    df_full.to_csv(output_csv, index=False)

    print("\n Done! Results saved in:", output_csv)
    print(" Fraud count:", (df_full["prediction"] == "FRAUD").sum())
    print(" Normal count:", (df_full["prediction"] == "NORMAL").sum())


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    predict_batch()
