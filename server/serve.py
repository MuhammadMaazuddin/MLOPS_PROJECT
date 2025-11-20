from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import os

MODEL_PATH = "/app/model/global_model.pkl"

app = FastAPI()

# Load model at startup
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠ No model found!")
        return None
    with open(MODEL_PATH, "rb") as f:
        params = pickle.load(f)
    return params

model_params = load_model()

@app.get("/")
def home():
    return {"status": "Model Serving API Running"}

@app.post("/predict")
def predict(features: dict):
    if model_params is None:
        return {"error": "No global model found"}

    df = pd.DataFrame([features])

    # ❗ Example prediction (replace with your real model)
    risk = int(df.sum(axis=1).values[0] % 3)

    labels = {0: "low", 1: "medium", 2: "high"}

    return {
        "prediction": labels[risk],
        "raw_value": risk
    }
