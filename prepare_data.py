import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("dataset.csv")  # Replace with your actual filename

# Preview the dataset
print("Dataset Preview:")
print(df.head())

# Convert class labels to binary: positive -> 1, negative -> 0
df['class'] = df['class'].map({'positive': 1, 'negative': 0})

# Separate features and labels
X = df.drop('class', axis=1)
y = df['class']

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.2, random_state=42)

# Reshape for LSTM input (samples, timesteps=1, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Save prepared data
np.savez("prepared_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Training and testing sets prepared.")
print("Unique classes in y_train:", np.unique(y_train))
