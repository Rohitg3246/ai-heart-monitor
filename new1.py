# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 📥 Load the dataset
df = pd.read_csv("dataset.csv")  # Change the filename as needed

# 🧹 Clean and preprocess
df.dropna(inplace=True)  # Remove missing values if any

# 🎯 Separate features and target
X = df[['pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']]
y = df['class']

# 🔢 Normalize feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🏷️ Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # One-hot encoding for classification

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)
