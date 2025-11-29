🛡️ Real-Time Financial Fraud Detection System

🚀 Overview

An End-to-End Data Engineering project designed to detect fraudulent financial transactions in Real-Time. The system ingests high-velocity transaction data, processes it using Apache Spark Structured Streaming, flags suspicious activities based on predefined logic, persists data in MySQL, triggers immediate Email Alerts, and visualizes live insights via a Streamlit Dashboard.

🏗️ Architecture

The pipeline follows a modern Lambda-like streaming architecture:

Ingestion: A Python Producer simulates real-time transaction data and streams it to Apache Kafka.

Processing: Apache Spark (PySpark) consumes the stream, performs data cleaning, feature engineering, and applies fraud detection rules.

Storage: Processed and flagged transactions are stored in a Dockerized MySQL database for auditing.

Alerting: The system triggers an SMTP Email Alert immediately upon detecting high-value fraud (> $2000).

Visualization: A real-time Streamlit dashboard monitors transaction traffic, fraud rates, and high-risk merchants.

<img width="609" height="286" alt="Screenshot 2025-11-29 at 1 22 51 PM" src="https://github.com/user-attachments/assets/bb718da0-0573-4f65-9bfd-51dd37a074eb" />
<img width="1470" height="657" alt="Screenshot 2025-11-29 at 1 34 03 AM" src="https://github.com/user-attachments/assets/09e4cea5-ea30-408c-9c64-d8c6cf1c2dab" />
<img width="1467" height="725" alt="Screenshot 2025-11-29 at 2 42 02 AM" src="https://github.com/user-attachments/assets/b81b79ec-a91e-4af6-b6d2-f851efd6e8c1" />
<img width="1470" height="622" alt="Screenshot 2025-11-29 at 2 42 18 AM" src="https://github.com/user-attachments/assets/d9348c66-2328-4af9-95c9-c52d8eb565d0" />
<img width="1466" height="581" alt="Screenshot 2025-11-29 at 2 42 27 AM" src="https://github.com/user-attachments/assets/7bce8256-a8d3-4a60-9137-783bdaaf2632" />
<img width="1468" height="588" alt="Screenshot 2025-11-29 at 2 42 36 AM" src="https://github.com/user-attachments/assets/a8696c46-9d70-42ca-b0f9-8e4e2cf59c27" />


🛠️ Tech Stack

Streaming: Apache Kafka, Zookeeper (Dockerized)

Processing: Apache Spark (Structured Streaming), PySpark

Database: MySQL (Dockerized)

Visualization: Streamlit, Plotly, SQLAlchemy

Infrastructure: Docker, Docker Compose

Language: Python 3.9+

📂 Project Structure

Fraud-Detection-System/
├── .streamlit/              # Streamlit theme configuration
│   └── config.toml

├── data/                    # Source data
│   └── transactions_unsupervised.csv

├── model_output/            # Pre-trained models (if applicable)

├── dashboard.py             # Real-time Visualization App

├── data_cleaning.py         # Spark ETL & Fraud Detection Logic

├── producer_from_csv.py     # Kafka Data Producer

├── train_fraud_model.py     # Model Training Script

├── docker-compose.yml       # Infrastructure (Kafka, Spark, MySQL)

├── requirements.txt         # Python Dependencies

└── README.md                # Project Documentation


⚙️ How to Run

1. Prerequisites

Docker & Docker Compose installed.

Python 3.9+ installed.

2. Start Infrastructure

Spin up the required services (Kafka, Zookeeper, Spark, MySQL) using Docker Compose:

docker-compose up -d


3. Install Dependencies

Create a virtual environment (recommended) and install the requirements:

pip install -r requirements.txt


4. Setup Database

Connect to the running MySQL container (or use a GUI like DBeaver) and create the database:

-- Connect to MySQL (User: root, Password: root)
CREATE DATABASE transactions;


5. Run the Pipeline Components

Open 3 separate terminals to simulate the real-time environment:

Terminal 1: Start the Dashboard

streamlit run dashboard.py


Terminal 2: Start the Spark Processor

python data_cleaning.py


(Wait until you see "Waiting for data..." or "Streaming started")

Terminal 3: Start the Data Producer

python producer_from_csv.py


📊 Features

Real-time Ingestion: Handles high-throughput data streams via Kafka.

Rule-Based/Hybrid Detection: Automatically flags transactions over $2,000 as potential fraud.

Instant Notifications: Sends HTML-formatted email alerts with transaction details.

Live Monitoring: Dashboard updates every 10 seconds with KPIs (Total Txns, Fraud Rate, Geo-Distribution).

Author: Ziad Ahmed
Connect on LinkedIn: [www.linkedin.com/in/-ziad-ali]
