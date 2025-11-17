# consumer_print.py
import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Listening to transactions topic...")
for msg in consumer:
    tx = msg.value
    print("Received:", tx)
    
