from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import tensorflow as tf

# Path to your saved Keras model
MODEL_PATH = "model/global_model.h5"

# Load the model at startup
if not tf.io.gfile.exists(MODEL_PATH):
    print("⚠ No model found!")
    model = None
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

# Initialize FastAPI
app = FastAPI()

# Instrument FastAPI with Prometheus
Instrumentator().instrument(app).expose(app)

# Define input schema
class ModelInput(BaseModel):
    features: list  # list of floats, same order as training data

@app.get("/")
def home():
    return {"status": "Model Serving API Running"}

@app.post("/predict")
def predict(data: ModelInput):
    if model is None:
        return {"error": "No global model found"}

    # Convert input to 2D numpy array
    X = np.array(data.features).reshape(1, -1)

    # Get prediction probabilities
    probs = model.predict(X)
    pred_class = int(np.argmax(probs, axis=1)[0])

    # Map to risk labels
    labels = {0: "low", 1: "medium", 2: "high"}

    return {
        "prediction": labels[pred_class],
        "raw_class": pred_class,
        "probabilities": probs.tolist()
    }
