
import json
import time
import os
import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CSV_FILE = "Data/transactions_unsupervised.csv"
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")

print(f"🔗 Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
print(f"📨 Topic: {TOPIC}")

try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=3,
        acks='all',
        batch_size=16384,
        linger_ms=10,
        request_timeout_ms=30000,
        api_version=(2, 0, 0)
    )
    print("✅ Producer created successfully!")
except Exception as e:
    print(f"❌ Error creating producer: {e}")
    print("💡 Tips:")
    print("   - Make sure Kafka is running: docker ps")
    print("   - Wait 30 seconds after starting Kafka")
    exit(1)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
print(f"📊 Loaded {len(df)} transactions from {CSV_FILE}")

sent_count = 0
error_count = 0

for _, row in df.iterrows():
    msg = row.where(pd.notnull(row), None).to_dict()
    try:
        future = producer.send(TOPIC, value=msg)
        future.get(timeout=10)
        sent_count += 1
        if sent_count % 100 == 0:
            print(f"✅ Sent {sent_count} messages...")
        time.sleep(0.02)
    except Exception as e:
        error_count += 1
        print(f"❌ Failed to send message {sent_count + error_count}: {e}")
        if error_count > 10:
            print("🛑 Too many errors, stopping...")
            break

producer.flush()
producer.close()

print(f"\n📊 Final Results:")
print(f"   ✅ Successfully sent: {sent_count}")
print(f"   ❌ Errors: {error_count}")
print(f"   📨 Total attempts: {sent_count + error_count}")