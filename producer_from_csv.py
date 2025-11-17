# producer_from_csv.py
import json
import time
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=5
)

df = pd.read_csv("transactions_unsupervised.csv")  
topic = "transactions"

for _, row in df.iterrows():
    msg = row.dropna().to_dict()   
    producer.send(topic, msg)
    print("sent:", msg.get("TransactionID", msg.get("UserID")))
    time.sleep(0.05)  

producer.flush()
producer.close()
