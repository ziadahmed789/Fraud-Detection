import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------
# Load model + scaler
# ----------------------------------------------
model = pickle.load(open("model_output/model.pkl", "rb"))
scaler = pickle.load(open("model_output/scaler.pkl", "rb"))

# ----------------------------------------------
# Example transaction to test the model
# ----------------------------------------------

tx = {
    "UserID": "123",
    "Amount": 45000,  
    "Time": "2025-01-01 12:30:00",
    "Location": "Cairo, Egypt",
    "Device": "iPhone 12"
}

# ----------------------------------------------
# Fake user history (previous user transactions)
# ----------------------------------------------
user_history = [
    (datetime(2025, 1, 1, 12, 0), 200, "Cairo"),
    (datetime(2025, 1, 1, 12, 5), 250, "Cairo"),
    (datetime(2025, 1, 1, 12, 7), 100, "Cairo"),
]

# ----------------------------------------------
# Build real-time features
# ----------------------------------------------
now = pd.to_datetime(tx["Time"])
amount = tx["Amount"]

history = user_history + [(now, amount, tx["Location"])]

# 5 min count
tx_count_5m = sum(1 for t, _, _ in history if t >= now - timedelta(minutes=5)) - 1

# 1 hour count
tx_count_1h = sum(1 for t, _, _ in history if t >= now - timedelta(hours=1)) - 1

# 24h stats
amounts = [amt for _, amt, _ in history]
avg_amount_24h = np.mean(amounts)
std_amount = np.std(amounts) if np.std(amounts) > 0 else 1

# z-score
amount_z = (amount - avg_amount_24h) / std_amount

# cyclic time features
sec = now.hour*3600 + now.minute*60 + now.second
tod_sin = np.sin(2*np.pi*sec/86400)
tod_cos = np.cos(2*np.pi*sec/86400)

missing_location = 0

# feature vector
feature_vector = np.array([
    amount,
    amount_z,
    tx_count_5m,
    tx_count_1h,
    avg_amount_24h,
    tod_sin,
    tod_cos,
    missing_location
]).reshape(1, -1)

# ----------------------------------------------
# Scale + Predict
# ----------------------------------------------
scaled = scaler.transform(feature_vector)
prediction = model.predict(scaled)[0]
score = model.decision_function(scaled)[0]

print("\n===== RESULT =====")
print("Anomaly Score:", score)
print("Prediction:", " FRAUD" if prediction == -1 else " Normal")
print("==================\n")