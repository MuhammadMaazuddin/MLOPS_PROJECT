import flwr as fl
from flwr.server.server import ServerConfig  # new import
import pickle
import os

MODEL_PATH = "model/global_model.pkl"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Custom strategy to save model after each aggregation
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"[SERVER] Saving global model after round {rnd}...")
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Example: using simple list of numpy arrays as "model"
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(params, f)

        return aggregated_parameters, metrics


def main():
    strategy = SaveModelStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )


    server_config = ServerConfig(num_rounds=5)
    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config,
    )


if __name__ == "__main__":
    main()
