import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your trained model
model = load_model("heart_risk_model.keras")

# Example new patient data (make sure the order matches training features!)
new_data = pd.DataFrame([{
    'age': 58,
    'gender': 1,
    'impluse': 70,
    'pressurehight': 130,
    'pressurelow': 85,
    'glucose': 200,
    'kcm': 3.5,
    'troponin': 0.02
},
{
        'age': 72,
        'gender': 1,
        'impluse': 45,
        'pressurehight': 170,
        'pressurelow': 110,
        'glucose': 280,
        'kcm': 2.8,
        'troponin': 0.07
    }])

# Fit scaler on original training data (IMPORTANT for consistency)
train_data = pd.read_csv("dataset.csv")  # Replace with your actual file
X_train = train_data.drop('class', axis=1)

scaler = StandardScaler()
scaler.fit(X_train)

# Scale new patient input
new_data_scaled = scaler.transform(new_data)

# Reshape for LSTM (samples, timesteps, features)
new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

# Make prediction
prediction = model.predict(new_data_reshaped)
# Scale and reshape
new_data_scaled = scaler.transform(new_data)
new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

# Predict
predictions = model.predict(new_data_reshaped)

# Output results
for i, score in enumerate(predictions):
    risk = "Positive (At Risk)" if score[0] > 0.5 else "Negative (Not at Risk)"
    print(f"\nPatient {i+1}:")
    print(f"  Predicted Risk Score: {score[0]:.4f}")
    print(f"  Risk Classification: {risk}")
