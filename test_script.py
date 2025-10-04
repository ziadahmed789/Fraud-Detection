import random
from datetime import datetime, timedelta

def generate_transactions(n=10000, output_file="transactions_data.py"):
    users = [f"User{i}" for i in range(1, 101)]
    locations = ["Cairo", "Giza", "Alex", "Luxor", "Aswan"]
    devices = ["Android", "iOS", "Web", "Windows"]
    merchants = ["Amazon", "Jumia", "Noon", "Souq"]
    
    start_time = datetime.now()
    data = []

    for i in range(n):
        transaction = {
            "UserID": random.choice(users),
            "Amount": round(random.uniform(10, 50000), 2),
            "Location": random.choice(locations),
            "Time": (start_time + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "Device": random.choice(devices),
            "Merchant": random.choice(merchants)
        }
        data.append(transaction)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("transactions = ")
            f.write(repr(data))
        print(f"✅ {n:,} transactions have been saved to '{output_file}' as a Python variable.")
    except Exception as e:
        print(f"❌ Error while saving the file: {e}")

if __name__ == "__main__":
    generate_transactions()
