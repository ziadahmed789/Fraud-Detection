# data_generating.py
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

def generate_realistic_transactions(n=10000, output_file="transactions_unsupervised.csv"):
    users = [f"User{i}" for i in range(1, 101)]
    locations = ["Cairo", "Giza", "Alex", "Luxor", "Aswan", "Sharm", "Hurghada"]
    devices = ["Android", "iOS", "Web", "Windows"]
    merchants = ["Amazon", "Jumia", "Noon", "Souq", "AliExpress", "eBay", "LocalStore"]

    user_profiles = {}
    for user in users:
        user_profiles[user] = {
            'usual_locations': random.sample(locations, 2),
            'preferred_device': random.choice(devices),
            'spending_tier': random.choice(['low', 'medium', 'high']),
            'avg_transaction': max(5, np.random.lognormal(4, 1))
        }

    start_time = datetime.now() - timedelta(days=30)
    data = []
    transaction_counter = 1000000

    # Normal transactions (95%)
    for i in range(int(n * 0.95)):
        user = random.choice(users)
        profile = user_profiles[user]
        transaction = generate_normal_transaction(user, profile, locations, devices, merchants, start_time, i, transaction_counter)
        data.append(transaction)
        transaction_counter += 1

    # Suspicious transactions (5%)
    for i in range(int(n * 0.05)):
        user = random.choice(users)
        profile = user_profiles[user]
        anomaly_type = random.choice(['rapid_fire', 'geo_change', 'high_amount', 'device_switch', 'merchant_risk'])
        transaction = generate_suspicious_transaction(user, profile, locations, devices, merchants, start_time, i, anomaly_type, transaction_counter)
        data.append(transaction)
        transaction_counter += 1

    random.shuffle(data)
    data = add_missing_values_and_noise(data, missing_rate=0.01)
    data = add_hidden_duplicates(data, duplicate_rate=0.005)

    df = pd.DataFrame(data)
    df['Amount'] = df['Amount'].astype(float)
    df['Time'] = pd.to_datetime(df['Time'])

    df.to_csv(output_file, index=False)
    print(f"✅ {len(df):,} transactions saved to '{output_file}'")
    return df

def generate_normal_transaction(user, profile, locations, devices, merchants, start_time, seq_num, txn_id):
    base_amount = profile['avg_transaction']
    amount = max(5, min(50000, np.random.lognormal(np.log(max(base_amount, 1)), 0.5)))
    hour = random.choices(range(24), weights=[0.3]*6 + [1.0]*6 + [1.5]*6 + [0.8]*6, k=1)[0]
    transaction_time = start_time + timedelta(days=random.randint(0, 30), hours=hour, minutes=random.randint(0,59), seconds=random.randint(0,59))
    return {
        "TransactionID": f"TXN{txn_id}",
        "UserID": user,
        "Amount": round(amount, 2),
        "Location": random.choice(profile['usual_locations']),
        "Time": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Device": profile['preferred_device'],
        "Merchant": random.choice(merchants)
    }

def generate_suspicious_transaction(user, profile, locations, devices, merchants, start_time, seq_num, anomaly_type, txn_id):
    base_transaction = generate_normal_transaction(user, profile, locations, devices, merchants, start_time, seq_num, txn_id)
    if anomaly_type == 'rapid_fire':
        base_transaction['Amount'] = round(random.uniform(5, 100), 2)
    elif anomaly_type == 'geo_change':
        unusual_locations = [loc for loc in locations if loc not in profile['usual_locations']]
        if unusual_locations:
            base_transaction['Location'] = random.choice(unusual_locations)
    elif anomaly_type == 'high_amount':
        base_transaction['Amount'] = round(random.uniform(20000, 50000), 2)
    elif anomaly_type == 'device_switch':
        unusual_devices = [dev for dev in devices if dev != profile['preferred_device']]
        if unusual_devices:
            base_transaction['Device'] = random.choice(unusual_devices)
    elif anomaly_type == 'merchant_risk':
        base_transaction['Merchant'] = random.choice(['AliExpress', 'eBay'])
        base_transaction['Amount'] = round(random.uniform(1000, 20000), 2)
    return base_transaction

def add_missing_values_and_noise(data, missing_rate=0.01):
    for transaction in data:
        if random.random() < missing_rate:
            field = random.choice(['Location', 'Device', 'Merchant'])
            transaction[field] = None
        if random.random() < 0.02:
            noise_factor = random.uniform(0.99, 1.01)
            transaction['Amount'] = round(transaction['Amount'] * noise_factor, 2)
        if random.random() < 0.01:
            time_obj = datetime.strptime(transaction['Time'], "%Y-%m-%d %H:%M:%S")
            time_obj += timedelta(minutes=random.randint(-2, 2))
            transaction['Time'] = time_obj.strftime("%Y-%m-%d %H:%M:%S")
    return data

def add_hidden_duplicates(data, duplicate_rate=0.005):
    num_duplicates = int(len(data) * duplicate_rate)
    duplicates_added = 0
    while duplicates_added < num_duplicates:
        original = random.choice(data)
        duplicate = original.copy()
        data.append(duplicate)
        duplicates_added += 1
    print(f"🔍 Added {duplicates_added} hidden duplicates")
    return data

if __name__ == "__main__":
    df = generate_realistic_transactions(n=10000, output_file="transactions_unsupervised.csv")
    print("Done.")
