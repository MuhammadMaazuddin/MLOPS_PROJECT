# BioShield AI: Federated Pandemic Response System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![Flower](https://img.shields.io/badge/flower-1.x-green.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)

**BioShield AI** is an advanced MLOps project designed to simulate and manage pandemic responses using **Federated Learning**. It enables privacy-preserving model training across distributed clients (hospitals, wearables) while providing real-time insights to health authorities and citizens through a premium dashboard.

## 🚀 Key Features

-   **Federated Learning**: Utilizes [Flower (flwr)](https://flower.dev/) to train global risk prediction models without sharing raw patient data.
-   **Privacy-First**: Data stays on the client devices; only model updates are sent to the server.
-   **Real-time Dashboard**:
    -   **Health Authorities**: View infection heatmaps, hospital capacity, and critical alerts.
    -   **Citizens**: Access personal risk scores and safety recommendations.
-   **MLOps Pipeline**:
    -   **Experiment Tracking**: Integrated **MLflow** for tracking training rounds, metrics, and models.
    -   **Monitoring**: **Prometheus** and **Grafana** for real-time system and model performance monitoring.
    -   **CI/CD**: Jenkins pipeline for automated building and deployment to Kubernetes.
-   **Synthetic Data Generation**: Scripts to generate realistic city, hospital, and wearable data for simulation.

## 🏗️ Architecture

The system consists of three main components:

1.  **Server**:
    -   **Federated Server (`fl_server.py`)**: Orchestrates the training rounds and aggregates model updates.
    -   **Model API (`serve.py`)**: Serves the global model for inference via FastAPI.
    -   **Dashboard**: Flask-based web interface for visualization.
    -   **Infrastructure**: Dockerized services including Prometheus, Grafana, and MLflow.

2.  **Client**:
    -   **Federated Client (`client.py`)**: Trains the model locally on private data and sends updates to the server.

3.  **Data Simulation**:
    -   Python scripts to generate synthetic datasets for cities, hospitals, and environmental factors.

## 🛠️ Tech Stack

-   **Core**: Python 3.10, TensorFlow/Keras
-   **Federated Learning**: Flower (flwr)
-   **Web & API**: Flask, FastAPI, Tailwind CSS
-   **Ops & Monitoring**: Docker, Kubernetes, Prometheus, Grafana, MLflow, Jenkins

## 🏁 Getting Started

### Prerequisites

-   Docker & Docker Compose
-   Python 3.10+ (for local development)
-   Kubernetes Cluster (optional, for production deployment)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MuhammadMaazuddin/MLOPS_PROJECT.git
    cd MLOPS_PROJECT
    ```

2.  **Generate Synthetic Data** (Optional):
    ```bash
    # Run simulation scripts to populate data/ directory
    python city.py
    python hospital.py
    ```

### 🏃‍♂️ Running with Docker Compose

The easiest way to run the entire stack (Server, Dashboard, Monitoring) is via Docker Compose.

1.  **Start the Server-side services**:
    ```bash
    cd server
    docker-compose up --build
    ```
    This will start:
    -   `fl_server` (Flower Server): Port `8080`
    -   `model_api` (Inference API): Port `8000`
    -   `dashboard` (Web UI): Port `5000`
    -   `prometheus`: Port `9090`
    -   `grafana`: Port `3000`

2.  **Start Clients**:
    Open a new terminal and run the clients to start training.
    ```bash
    cd client
    docker-compose up --build
    ```
    *Note: Ensure the server is running first.*

### 📊 Accessing Interfaces

-   **Dashboard**: [http://localhost:5000](http://localhost:5000)
    -   *Authority View*: `/authority`
    -   *Citizen View*: `/citizen`
-   **Grafana**: [http://localhost:3000](http://localhost:3000) (Default login: `admin`/`admin`)
-   **Prometheus**: [http://localhost:9090](http://localhost:9090)
-   **Model API**: [http://localhost:8000/docs](http://localhost:8000/docs)

## ☸️ Kubernetes Deployment

Manifests are located in `server/k8s/`.

1.  **Apply Manifests**:
    ```bash
    kubectl apply -f server/k8s/
    ```

2.  **Check Status**:
    ```bash
    kubectl get pods
    kubectl get services
    ```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.
