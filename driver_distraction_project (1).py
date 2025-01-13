# Full Project Code: Driver Distraction Detection System

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Step 1: Data Preparation
# Simulating improved synthetic data for sensors
np.random.seed(42)
synthetic_data = {
    'steering_grip': np.random.normal(0.5, 0.1, 1000),
    'seat_pressure': np.random.normal(50, 5, 1000),
    'ambient_sound': np.random.normal(60, 10, 1000),
    'imu_acceleration': np.random.normal(0.02, 0.01, 1000),
    # Introducing correlation: Higher sound -> higher chance of distraction
    'distraction_label': [1 if (sg < 0.45 or sp < 48 or asound > 65 or imu > 0.025) else 0 
                          for sg, sp, asound, imu in zip(
                              np.random.normal(0.5, 0.1, 1000),
                              np.random.normal(50, 5, 1000),
                              np.random.normal(60, 10, 1000),
                              np.random.normal(0.02, 0.01, 1000))]
}

# Creating a DataFrame
sensor_data = pd.DataFrame(synthetic_data)

# Step 2: Data Preprocessing
# Split features and labels
X = sensor_data[['steering_grip', 'seat_pressure', 'ambient_sound', 'imu_acceleration']]
y = sensor_data['distraction_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Random Forest Classifier
# Train the model with more trees
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)

# Step 4: LSTM Model for Sequential Data
# Reshaping data for LSTM
X_train_seq = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_seq = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model with additional layers
lstm_model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.3),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the LSTM model
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(X_train_seq, y_train, epochs=15, batch_size=32, validation_data=(X_test_seq, y_test))

# Evaluate the LSTM model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_seq, y_test)

# Step 5: Save Models
# Save the Random Forest model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the LSTM model
lstm_model.save('lstm_model.h5')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Step 6: Results and Deployment Instructions
print("Improved Random Forest Accuracy:", rf_accuracy)
print("Improved Random Forest Classification Report:\n", rf_classification_report)
print(f"Improved LSTM Accuracy: {lstm_accuracy * 100:.2f}%")

# Instructions for Running in Google Colab
# 1. Upload 'rf_model.pkl', 'lstm_model.h5', and 'scaler.pkl' to your Google Drive.
# 2. Mount Google Drive in Colab using:
#    from google.colab import drive
#    drive.mount('/content/drive')
# 3. Load the models and scaler in Colab using:
#    import joblib
#    from tensorflow.keras.models import load_model
#    rf_model = joblib.load('/content/drive/MyDrive/rf_model.pkl')
#    lstm_model = load_model('/content/drive/MyDrive/lstm_model.h5')
#    scaler = joblib.load('/content/drive/MyDrive/scaler.pkl')
# 4. Use real-time data input for predictions:
#    new_input = [[0.55, 52, 65, 0.03]]
#    new_input_scaled = scaler.transform(new_input)
#    rf_prediction = rf_model.predict(new_input_scaled)
#    lstm_prediction = lstm_model.predict(new_input_scaled.reshape(1, 1, -1))
