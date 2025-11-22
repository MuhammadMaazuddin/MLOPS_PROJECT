import flwr as fl
from flwr.server.server import ServerConfig
import mlflow
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("federated_learning_experiment")

MODEL_PATH = "model/global_model.h5"
os.makedirs("model", exist_ok=True)

# Custom strategy to save TensorFlow model after each round
class SaveTFModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"[SERVER] Saving global TensorFlow model after round {rnd}...")
            # Convert Flower parameters (list of ndarrays) to TensorFlow model
            model = build_model()  # same model architecture as clients
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model.set_weights(params)
            model.save(MODEL_PATH)

            # Log metrics and model to MLflow
            with mlflow.start_run(run_name=f"round-{rnd}"):
                # Log metrics if available (assuming metrics contains 'accuracy' and 'loss')
                # Note: 'metrics' here is a tuple (loss, other_metrics_dict) if using standard FedAvg
                # But standard FedAvg aggregate_fit returns (parameters, metrics) where metrics is a dict of aggregated metrics
                # We'll assume metrics is a dictionary for now.
                if metrics:
                    mlflow.log_metrics(metrics, step=rnd)
                
                # Log the model
                mlflow.tensorflow.log_model(model, artifact_path=f"model_round_{rnd}")
                print(f"[SERVER] Logged model and metrics for round {rnd} to MLflow")

        return aggregated_parameters, metrics

# Example function to rebuild the same model architecture as clients
def build_model():
    input_shape = 11  # number of features
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")  # 3 classes: low, medium, high
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    strategy = SaveTFModelStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    server_config = ServerConfig(num_rounds=3)
    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config,
    )

if __name__ == "__main__":
    main()
