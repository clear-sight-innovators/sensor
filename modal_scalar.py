import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load saved models and scaler
rf_model = joblib.load('rf_model.pkl')
lstm_model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Example input data
new_input = [[0.55, 52, 65, 0.03]]  # Replace with real sensor data

# Preprocess the input
new_input_scaled = scaler.transform(new_input)

# Random Forest Prediction
rf_prediction = rf_model.predict(new_input_scaled)
print("Random Forest Prediction:", "Distracted" if rf_prediction[0] == 1 else "Attentive")

# LSTM Prediction
lstm_prediction = lstm_model.predict(new_input_scaled.reshape(1, 1, -1))
print("LSTM Prediction:", "Distracted" if lstm_prediction[0][0] > 0.5 else "Attentive")
