import flwr as fl
import pandas as pd
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load local dataset
# -------------------------------
df = pd.read_csv("data/city_A.csv")

# Map string labels to integers
df["label"] = df["risk_level"].map({"low": 0, "medium": 1, "high": 2})

# Features and labels
X = df.drop(columns=["timestamp", "risk_level", "label"], errors="ignore")
y = df["label"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Define TensorFlow/Keras model
# -------------------------------
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(X_train.shape[1])

# -------------------------------
# Define Flower NumPyClient
# -------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self,config):
        # Return model weights as list of numpy arrays
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)
        # Train locally for 1 epoch
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        # Return updated model weights
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)
        # Evaluate locally
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return float(loss), len(self.X_test), {"accuracy": float(acc)}

# -------------------------------
# Start Flower client
# -------------------------------
if __name__ == "__main__":
    time.sleep(5)  # optional: wait for server to be ready
    client = FLClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
