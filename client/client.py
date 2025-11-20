import flwr as fl
import pandas as pd
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# -------------------------------
# Load local dataset for this node
# -------------------------------
NODE_DATA = os.environ.get("NODE_DATA")
if NODE_DATA is None:
    raise ValueError("NODE_DATA environment variable is not set!")

# Load dataset
df = pd.read_csv(NODE_DATA)

# Map string labels to integers
df["label"] = df["risk_level"].map({"low": 0, "medium": 1, "high": 2})

# Features and labels
X = df.drop(columns=["timestamp", "risk_level", "label"], errors="ignore")
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Define ML model
# -------------------------------
model = RandomForestClassifier(n_estimators=10, random_state=42)

# -------------------------------
# Define Flower client
# -------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self,config):
        # Return model parameters
        return self.model.get_params()

    def fit(self, parameters, config):
        # Update model parameters
        self.model.set_params(**parameters)
        # Train model on local data
        self.model.fit(self.X_train, self.y_train)
        # Return updated parameters
        return self.model.get_params(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_params(**parameters)
        # Ensure model is fitted
        if not hasattr(self.model, "n_estimators_"):
            self.model.fit(self.X_train, self.y_train)
        # Predictions
        preds = self.model.predict_proba(self.X_test)
        # Compute log loss
        loss = log_loss(self.y_test, preds)
        acc = (self.model.predict(self.X_test) == self.y_test).mean()
        return float(loss), len(self.X_test), {"accuracy": float(acc)}

# -------------------------------
# Start Flower client
# -------------------------------
if __name__ == "__main__":
    time.sleep(5)
    print(f"Starting FL client with dataset: {NODE_DATA}")
    client = FLClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="server:8080", client=client)
