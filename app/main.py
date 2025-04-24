# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os

# Initialize FastAPI app (THIS MUST BE PRESENT)
app = FastAPI()

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Input schema
class SequenceData(BaseModel):
    sequence: list  # expects list of 12 values

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Demand Forecasting LSTM API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: SequenceData):
    if len(data.sequence) != 12:
        raise HTTPException(status_code=400, detail="Sequence must be exactly 12 time steps long.")
    
    seq_array = np.array(data.sequence).reshape(-1, 1)
    scaled_seq = scaler.transform(seq_array).reshape((1, 12, 1))

    pred_scaled = model.predict(scaled_seq)
    pred = scaler.inverse_transform(pred_scaled)
    
    return {"predicted_demand": float(pred[0][0])}
