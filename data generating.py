import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

def generate_realistic_transactions(n=10000, output_file="transactions_unsupervised.csv"):
    # Base data
    users = [f"User{i}" for i in range(1, 101)]
    locations = ["Cairo", "Giza", "Alex", "Luxor", "Aswan", "Sharm", "Hurghada"]
    devices = ["Android", "iOS", "Web", "Windows"]
    merchants = ["Amazon", "Jumia", "Noon", "Souq", "AliExpress", "eBay", "LocalStore"]
    
    # User profiles for realistic behavior
    user_profiles = {}
    for user in users:
        user_profiles[user] = {
            'usual_locations': random.sample(locations, 2),
            'preferred_device': random.choice(devices),
            'spending_tier': random.choice(['low', 'medium', 'high']),
            'avg_transaction': np.random.lognormal(5, 1)
        }
    
    start_time = datetime.now() - timedelta(days=30)
    data = []
    transaction_counter = 1000000
    
    # Generate normal transactions (95%)
    for i in range(int(n * 0.95)):
        user = random.choice(users)
        profile = user_profiles[user]
        transaction = generate_normal_transaction(user, profile, locations, devices, merchants, start_time, i, transaction_counter)
        data.append(transaction)
        transaction_counter += 1
    
    # Generate suspicious transactions (5%)
    for i in range(int(n * 0.05)):
        user = random.choice(users)
        profile = user_profiles[user]
        
        anomaly_type = random.choice(['rapid_fire', 'geo_change', 'high_amount', 'device_switch', 'merchant_risk'])
        transaction = generate_suspicious_transaction(user, profile, locations, devices, merchants, start_time, i, anomaly_type, transaction_counter)
        data.append(transaction)
        transaction_counter += 1
    
    # Shuffle the data
    random.shuffle(data)
    
    # Add missing values and noise
    data = add_missing_values_and_noise(data, missing_rate=0.01)
    
    # Add duplicates - بدون أي إشارة
    data = add_hidden_duplicates(data, duplicate_rate=0.005)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure proper data types
    df['Amount'] = df['Amount'].astype(float)
    df['Time'] = pd.to_datetime(df['Time'])
    
    try:
        df.to_csv(output_file, index=False)
        print(f"✅ {len(df):,} transactions saved to '{output_file}'")
        print(f"📊 Dataset prepared for UNSUPERVISED learning")
        print(f"📈 Amount stats: Mean=${df['Amount'].mean():.2f}, Max=${df['Amount'].max():.2f}")
        print(f"🎯 Challenge: Can you find the {int(n * 0.005)} hidden duplicates?")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
    
    return df

def generate_normal_transaction(user, profile, locations, devices, merchants, start_time, seq_num, txn_id):
    """Generate a normal transaction based on user profile"""
    base_amount = profile['avg_transaction']
    amount = max(10, min(50000, np.random.lognormal(np.log(base_amount), 0.5)))
    
    hour = random.choices(
        range(24), 
        weights=[0.3]*6 + [1.0]*6 + [1.5]*6 + [0.8]*6,
        k=1
    )[0]
    
    transaction_time = start_time + timedelta(
        days=random.randint(0, 30),
        hours=hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
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
    """Generate transactions with suspicious patterns"""
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
    """Add realistic missing values and noise"""
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
    """Add duplicates بدون أي إشارة - نفس البيانات بالظبط"""
    num_duplicates = int(len(data) * duplicate_rate)
    
    duplicates_added = 0
    while duplicates_added < num_duplicates:
        original = random.choice(data)
        
        # نعمل نسخة طبق الأصل من المعاملة
        duplicate = original.copy()
        
        # نضيفها للبيانات - نفس كل حرف في كل حقل
        data.append(duplicate)
        duplicates_added += 1
    
    print(f"🔍 Added {duplicates_added} hidden duplicates for you to find!")
    return data

def analyze_dataset(df):
    """Basic analysis of the generated dataset"""
    print("\n📊 Dataset Analysis (Unsupervised):")
    print(f"Total transactions: {len(df):,}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    print("\n💰 Amount Distribution:")
    print(f"Mean: ${df['Amount'].mean():.2f}")
    print(f"Median: ${df['Amount'].median():.2f}")
    print(f"Std: ${df['Amount'].std():.2f}")

if __name__ == "__main__":
    # Generate dataset for unsupervised learning
    df = generate_realistic_transactions(n=10000, output_file="transactions_unsupervised.csv")
    
    # Analyze the results
    analyze_dataset(df)
    
    print("\n🎯 Data Processing Challenge:")
    print("   - Find and remove hidden duplicates")
    print("   - Detect anomalies without labels") 
    print("   - Handle missing values")
    print("   - Prepare for ML models")