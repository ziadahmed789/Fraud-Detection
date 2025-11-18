# train_fraud_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import os

# -----------------------------------------------------------
# 1) Feature Engineering Functions
# -----------------------------------------------------------

def time_of_day_features(ts):
    sec = ts.hour * 3600 + ts.minute * 60 + ts.second
    sin_t = np.sin(2 * np.pi * sec / 86400)
    cos_t = np.cos(2 * np.pi * sec / 86400)
    return sin_t, cos_t

def build_features(df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["UserID", "Time"])

    all_rows = []

    for user, group in df.groupby("UserID"):
        group = group.sort_values("Time")
        group = group.reset_index(drop=True)

        # Rolling features manually
        tx_times = group["Time"]
        amounts = group["Amount"]

        tx_count_5m = []
        tx_count_1h = []
        avg_amount_24h = []
        amount_z = []

        for i in range(len(group)):

            t = tx_times[i]

            # 5 min window
            mask5 = (tx_times >= t - timedelta(minutes=5)) & (tx_times < t)
            tx_count_5m.append(mask5.sum())

            # 1 hour window
            mask1 = (tx_times >= t - timedelta(hours=1)) & (tx_times < t)
            tx_count_1h.append(mask1.sum())

            # 24 hours average
            mask24 = (tx_times >= t - timedelta(hours=24)) & (tx_times <= t)
            avg_amount_24h.append(amounts[mask24].mean())

            # z-score
            mean_amt = amounts[: i + 1].mean()
            std_amt = amounts[: i + 1].std() if amounts[: i + 1].std() > 0 else 1
            amount_z.append((amounts[i] - mean_amt) / std_amt)

        group["tx_count_5m"] = tx_count_5m
        group["tx_count_1h"] = tx_count_1h
        group["avg_amount_24h"] = avg_amount_24h
        group["amount_zscore"] = amount_z

        sin_t, cos_t = zip(*group["Time"].apply(time_of_day_features))
        group["tod_sin"] = sin_t
        group["tod_cos"] = cos_t

        group["missing_location"] = group["Location"].isna().astype(int)

        all_rows.append(group)

    final_df = pd.concat(all_rows)

    features = final_df[[
        "Amount",
        "amount_zscore",
        "tx_count_5m",
        "tx_count_1h",
        "avg_amount_24h",
        "tod_sin",
        "tod_cos",
        "missing_location",
    ]]

    return features, final_df


# -----------------------------------------------------------
# 2) Train Model
# -----------------------------------------------------------

def train_model(input_csv="transactions_unsupervised.csv"):

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found — تأكدي من وجود الملف!")

    print("\n Loading CSV...")
    df = pd.read_csv(input_csv)

    print(" Building features (this may take 5–20 seconds)...")
    X, full_df = build_features(df)

    print(" Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(" Training IsolationForest model...")
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    print(" Saving model...")
    os.makedirs("model_output", exist_ok=True)

    pickle.dump(model, open("model_output/model.pkl", "wb"))
    pickle.dump(scaler, open("model_output/scaler.pkl", "wb"))

    print("\n Model training completed!")
    print("Saved files:")
    print(" - model_output/model.pkl")
    print(" - model_output/scaler.pkl")

    # Optional evaluation:
    print("\n Testing anomaly scores on first 10 rows:")
    scores = model.decision_function(X_scaled[:10])
    preds = model.predict(X_scaled[:10])
    print("Scores:", scores)
    print("Predictions (-1 = anomaly):", preds)


if __name__ == "__main__":
    train_model()
