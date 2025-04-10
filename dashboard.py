import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import winsound
from tkinter import messagebox, Tk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load your trained model
model = load_model("heart_risk_model.keras")

# Load training data for scaler fitting
train_data = pd.read_csv("dataset.csv")  # Replace with your actual training dataset
X_train = train_data.drop('class', axis=1)
scaler = StandardScaler()
scaler.fit(X_train)

# Initialize processed row count
processed_count = 0

print("📊 Real-Time Troponin Monitoring + Risk Prediction Started")

# Create or clear log file
log_path = "prediction_log.csv"
if os.path.exists(log_path):
    os.remove(log_path)  # Clear previous log

# Begin monitoring loop
while True:
    try:
        # Load current troponin data
        if not os.path.exists("simulated_data.csv"):
            print("Waiting for simulated_data.csv to appear...")
            time.sleep(300)
            continue

        df = pd.read_csv("simulated_data.csv")
        if df.empty:
            print("simulated_data.csv is empty.")
            time.sleep(300)
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Skip if no new data
        if len(df) <= processed_count:
            print("⏳ Waiting for new data...")
            time.sleep(300)  # 5 minutes
            continue

        new_data = df.iloc[processed_count:]
        processed_count = len(df)

        # Plot historical troponin values
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['troponin'], label='Troponin Level', color='blue', marker='o')
        plt.axhline(y=0.04, color='orange', linestyle='--', label='Clinical Threshold (0.04 ng/L)')

        risk_flag = False  # <-- Track if we need to alert

        # Loop through new data rows
        for index, row in new_data.iterrows():
            timestamp = row['timestamp']
            troponin = row['troponin']

            # Copy a row from training data to simulate full patient input
            input_row = X_train.iloc[0].copy()
            input_row['troponin'] = troponin
            scaled = scaler.transform([input_row])
            reshaped = scaled.reshape((1, 1, scaled.shape[1]))

            # Predict risk
            prediction = model.predict(reshaped)[0][0]
            risk = "R" if prediction > 0.5 else "S"
            color = 'red' if prediction > 0.5 else 'green'

            print(f"[{timestamp}] Troponin: {troponin:.4f} → Risk: {risk} ({prediction:.2f})")

            if prediction > 0.5:
                risk_flag = True  # Set flag if any row is at risk

            # Log prediction
            log_entry = pd.DataFrame([{
                'timestamp': timestamp,
                'troponin': troponin,
                'prediction_score': round(prediction, 4),
                'risk': risk
            }])

            # Write to log safely
            header_needed = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
            log_entry.to_csv(log_path, mode='a', index=False, header=header_needed)

            # Add prediction label to plot
            plt.text(timestamp, troponin + 0.005, f"{risk}", fontsize=9, color=color)

        # ✅ After processing batch, trigger alert if needed
        if risk_flag:
            winsound.Beep(1000, 500)
            root = Tk()
            root.withdraw()
            messagebox.showwarning("⚠️ Heart Risk Alert", "At least one reading in the past 5 mins shows high risk.")
            root.destroy()

        # Show and hold graph
        plt.title("AI-Based Troponin Risk Monitoring")
        plt.xlabel("Time")
        plt.ylabel("Troponin Level (ng/L)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    except KeyboardInterrupt:
        print("🛑 Monitoring stopped by user.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(300)
