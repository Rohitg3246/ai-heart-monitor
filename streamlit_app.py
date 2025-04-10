import asyncio
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("heart_risk_model.keras")
    train_data = pd.read_csv("dataset.csv")
    scaler = StandardScaler()
    scaler.fit(train_data.drop('class', axis=1))
    return model, scaler

model, scaler = load_model_and_scaler()

# App title
st.title("🫀 Real-Time Troponin Risk Monitor")
st.markdown("**Powered by AI for heart attack prediction**")

# Load data
if os.path.exists("simulated_data.csv"):
    df = pd.read_csv("simulated_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
else:
    st.warning("⚠️ simulated_data.csv not found.")
    st.stop()

# Select latest reading
latest = df.iloc[-1]
troponin = latest['troponin']
timestamp = latest['timestamp']

# Prepare input for model
sample = pd.read_csv("dataset.csv").drop('class', axis=1).iloc[0].copy()
sample['troponin'] = troponin
scaled = scaler.transform([sample])
reshaped = scaled.reshape((1, 1, scaled.shape[1]))
score = model.predict(reshaped)[0][0]
risk = "🟢 Safe" if score <= 0.5 else "🔴 At Risk"

# Show graph
st.subheader("Troponin Level Over Time")
fig, ax = plt.subplots()
ax.plot(df['timestamp'], df['troponin'], color='blue', marker='o', label="Troponin")
ax.axhline(y=0.04, color='orange', linestyle='--', label='Risk Threshold (0.04)')
ax.set_xlabel("Time")
ax.set_ylabel("Troponin Level")
ax.legend()
st.pyplot(fig)

# Show prediction result
st.subheader("Latest Risk Assessment")
st.metric(label="Current Troponin", value=f"{troponin:.4f} ng/L")
st.metric(label="Prediction Score", value=f"{score:.2f}")
st.markdown(f"### Current Risk Status: {risk}")

# Show last 5 predictions if log file exists
if os.path.exists("prediction_log.csv"):
    st.subheader("📋 Last 5 Predictions")
    log_df = pd.read_csv("prediction_log.csv").tail(5)
    st.dataframe(log_df, use_container_width=True)

# Optional: refresh button
if st.button("🔁 Simulate 5-Min Analysis"):
    st.rerun()
