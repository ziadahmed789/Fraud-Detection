from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StringType, DoubleType
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SENDER_EMAIL = os.getenv("SENDER_EMAIL", "ziadahmed554433@gmail.com")      
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "caxm mkbs glpf uetk")         
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "m.salah.khalil3@gmail.com")    
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
MYSQL_HOST = os.getenv("MYSQL_URL", "mysql:3306")


def send_fraud_alert_email(fraud_df):
    print("📧 Preparing to send Fraud Alert Email...")
    try:
        
        fraud_table_html = fraud_df[[
            "TransactionID", "UserID", "Amount", "Location", "Time"
        ]].to_html(index=False, border=1, justify="center")
        
        
        fraud_table_html = fraud_table_html.replace(
            '<table border="1" class="dataframe">', 
            '<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; font-family: Arial;">'
        ).replace(
            '<th>', '<th style="background-color: #d9534f; color: white; padding: 10px;">'
        ).replace('<td>', '<td style="padding: 10px; border: 1px solid #ddd; text-align: center;">')

        num_frauds = len(fraud_df)
        subject = f"🚨 URGENT: {num_frauds} Fraudulent Transactions Detected!"
        
        body = f"""
        <html><body>
            <h2 style="color: #d9534f;">⚠️ Immediate Fraud Detected</h2>
            <p>The system has flagged suspicious activities in real-time:</p>
            {fraud_table_html}
            <br><p style="color: gray;">System: Spark Real-time Engine</p>
        </body></html>
        """

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("✅ Alert Email Sent Successfully!")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def build_features_for_batch(df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["UserID", "Time"])
    all_rows = []
    for user, group in df.groupby("UserID"):
        group = group.sort_values("Time").reset_index(drop=True)
        tx_times, amounts = group["Time"], group["Amount"]
        tx_count_5m, tx_count_1h, avg_amount_24h, amount_z = [], [], [], []
        
        for i in range(len(group)):
            t = tx_times[i]
            mask5 = (tx_times >= t - timedelta(minutes=5)) & (tx_times < t)
            mask1 = (tx_times >= t - timedelta(hours=1)) & (tx_times < t)
            mask24 = (tx_times >= t - timedelta(hours=24)) & (tx_times <= t)
            
            tx_count_5m.append(mask5.sum())
            tx_count_1h.append(mask1.sum())
            avg_amount_24h.append(amounts[mask24].mean())
            
            mean = amounts[:i+1].mean()
            std = amounts[:i+1].std() if amounts[:i+1].std() > 0 else 1
            amount_z.append((amounts.iloc[i] - mean) / std)
            
        group["tx_count_5m"] = tx_count_5m
        group["tx_count_1h"] = tx_count_1h
        group["avg_amount_24h"] = avg_amount_24h
        group["amount_zscore"] = amount_z
        
        sec = group["Time"].dt.hour * 3600 + group["Time"].dt.minute * 60 + group["Time"].dt.second
        group["tod_sin"] = np.sin(2 * np.pi * sec / 86400)
        group["tod_cos"] = np.cos(2 * np.pi * sec / 86400)
        group["missing_location"] = group["Location"].isna().astype(int)
        all_rows.append(group)
    return pd.concat(all_rows) if all_rows else df


print("⏳ Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("AI_Fraud_Detection") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.jars.packages", "mysql:mysql-connector-j:8.0.33,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

print("🔍 Configuration Check:")
print(f"   Kafka: {KAFKA_BOOTSTRAP}")
print(f"   MySQL: {MYSQL_HOST}")

print("⏳ Loading AI Models...")
try:
    model = pickle.load(open("model_output/model.pkl", "rb"))
    scaler = pickle.load(open("model_output/scaler.pkl", "rb"))
    print("✅ Models Loaded!")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    model = None

schema = StructType() \
    .add("TransactionID", StringType()) \
    .add("UserID", StringType()) \
    .add("Amount", DoubleType()) \
    .add("Location", StringType()) \
    .add("Time", StringType()) \
    .add("Device", StringType()) \
    .add("Merchant", StringType())

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "earliest") \
    .load()

json_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

def process_batch(batch_df, batch_id):
    if batch_df.rdd.isEmpty(): 
        return
        
    print(f"🔄 Processing Batch {batch_id} with {batch_df.count()} records...")
    
    try:
        
        batch_df = batch_df.filter(col("Amount").isNotNull())
        pdf = batch_df.toPandas()
        
        if len(pdf) > 0:
            
            print(f"🧠 Building features for {len(pdf)} transactions...")
            proc_pdf = build_features_for_batch(pdf)
            
           
            if model:
                
                feats = proc_pdf[["Amount", "amount_zscore", "tx_count_5m", "tx_count_1h", "avg_amount_24h", "tod_sin", "tod_cos", "missing_location"]]
                preds = model.predict(scaler.transform(feats))
                proc_pdf["Is_Fraud"] = ["YES" if (p == -1 or amt > 2000) else "NO" for p, amt in zip(preds, proc_pdf["Amount"])]
            else:
                
                print("⚠️ Using Rule-Based Logic (No Model Found)")
                proc_pdf["Is_Fraud"] = ["YES" if amt > 2000 else "NO" for amt in proc_pdf["Amount"]]
            
            
            print("💾 Saving to MySQL...")
            spark_df = spark.createDataFrame(proc_pdf)
            
            spark_df.select("TransactionID", "UserID", "Amount", "Location", "Time", "Device", "Merchant", "Is_Fraud") \
                .write.format("jdbc") \
                .option("url", f"jdbc:mysql://{MYSQL_HOST}/transactions") \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("dbtable", "clean_data") \
                .option("user", "root") \
                .option("password", "root") \
                .mode("append") \
                .save()
            
            print(f"✅ Saved {len(proc_pdf)} records to DB.")
            
            
            frauds = proc_pdf[proc_pdf["Is_Fraud"] == "YES"]
            if not frauds.empty:
                print(f"🚨 ALERT: {len(frauds)} Frauds detected! Sending email...")
                send_fraud_alert_email(frauds)
            else:
                print("👍 No frauds in this batch.")
        else:
            print("⚠️ Empty batch after filtering.")
                
    except Exception as e:
        print(f"❌ Error in batch {batch_id}: {e}")


checkpoint_dir = "chk_point_dir" 

query = json_df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", checkpoint_dir) \
    .trigger(processingTime='30 seconds') \
    .start()

print("🚀 Streaming started! Waiting for data...")
print("📊 Check Spark UI: http://localhost:8080")
query.awaitTermination()