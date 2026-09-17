# Iris ML API – Monitored Machine Learning REST API

## Project Overview

Iris ML API is a machine learning REST API built using **FastAPI** and **Scikit-learn**.

The API accepts Iris flower measurements and predicts the flower species using a trained **Random Forest Classifier**.

The application is containerized using **Docker** and **Docker Compose**. It also includes API key security, request logging, request IDs, batch prediction, Prometheus monitoring, and API versioning.

### Technologies Used

- Python
- FastAPI
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Pydantic
- Docker
- Docker Compose
- Prometheus
- Pytest
- GitHub Actions

---

## Machine Learning Problem

This project solves a **multi-class classification problem** using the Iris dataset.

The model predicts one of three Iris flower species:

- Setosa
- Versicolor
- Virginica

### Input Features

The API accepts four numerical features:

- Sepal length
- Sepal width
- Petal length
- Petal width

### Machine Learning Model

The trained model is a:

**Random Forest Classifier**

The trained model is stored using Joblib and loaded by the FastAPI application when the application starts.

---

## Project Architecture

```text
                    Client
                      |
                      v
              Docker Compose
                      |
                      v
              +---------------+
              |   FastAPI     |
              |      API      |
              +---------------+
                /     |      \
               /      |       \
              v       v        v
          Security  Logging  Prometheus
              |
              v
       Input Validation
              |
              v
       Random Forest Model
              |
              v
          Prediction
              |
              v
         API Response
```

### Request Flow

```text
Client Request
      |
      v
API Key Verification
      |
      v
Input Validation
      |
      v
ML Model Prediction
      |
      v
Response Generation
      |
      +----> Request Logging
      |
      +----> Prometheus Metrics
```

---

## Project Structure

```text
iris-ml-api/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_config.py
│   ├── security.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── routers/
│       ├── v1.py
│       └── v2.py
│
├── ml/
│   ├── train.py
│   ├── predict.py
│   └── saved_model/
│       ├── model.joblib
│       └── model_metadata.json
│
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_model_info.py
│   ├── test_predict.py
│   ├── test_batch.py
│   └── test_security.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── load_test.py
├── TESTING.md
├── requirements.txt
└── README.md
```

---

# API Endpoints

All `/api/v1/*` and `/api/v2/*` endpoints require an API key.

The API key must be sent using the `X-API-Key` header.

Example:

```text
X-API-Key: your-api-key
```

---

## 1. Health Check

### Endpoint

```text
GET /api/v1/health
```

### Purpose

Checks whether the API is running and whether the ML model has been loaded.

### cURL

```bash
curl -X GET "http://localhost:8000/api/v1/health" \
  -H "X-API-Key: your-api-key"
```

### Example Response

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 2. Model Information

### Endpoint

```text
GET /api/v1/model-info
```

### Purpose

Returns metadata about the trained ML model.

### cURL

```bash
curl -X GET "http://localhost:8000/api/v1/model-info" \
  -H "X-API-Key: your-api-key"
```

### Example Response

```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0",
  "accuracy": 0.9
}
```

---

## 3. Single Prediction – V1

### Endpoint

```text
POST /api/v1/predict
```

### Purpose

Predicts the Iris flower species for a single set of measurements.

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Example Response

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "model_version": "1.0",
  "request_id": "generated-request-id"
}
```

---

## 4. Batch Prediction – V1

### Endpoint

```text
POST /api/v1/predict-batch
```

### Purpose

Predicts multiple Iris flower samples in a single API request.

The maximum batch size is controlled by the application configuration.

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "inputs": [
      {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      },
      {
        "sepal_length": 6.0,
        "sepal_width": 2.9,
        "petal_length": 4.5,
        "petal_width": 1.5
      }
    ]
  }'
```

### Example Response

```json
{
  "predictions": [
    {
      "prediction": "setosa",
      "confidence": 1.0,
      "model_version": "1.0",
      "request_id": "generated-request-id"
    },
    {
      "prediction": "versicolor",
      "confidence": 0.9,
      "model_version": "1.0",
      "request_id": "generated-request-id"
    }
  ]
}
```

---

## 5. Single Prediction – V2

### Endpoint

```text
POST /api/v2/predict
```

### Purpose

V2 provides the predicted class along with probability values for all supported Iris classes.

### cURL

```bash
curl -X POST "http://localhost:8000/api/v2/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Example Response

```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0",
  "request_id": "generated-request-id"
}
```

---

## 6. Prometheus Metrics

### Endpoint

```text
GET /metrics
```

### Purpose

Exposes application and ML prediction metrics in Prometheus format.

### cURL

```bash
curl -X GET "http://localhost:8000/metrics"
```

Example custom metric:

```text
ml_predictions_total{predicted_class="setosa"} 1.0
```

---

# API Documentation

FastAPI automatically provides interactive Swagger documentation.

After starting the application, open:

```text
http://localhost:8000/docs
```

Alternative OpenAPI documentation:

```text
http://localhost:8000/redoc
```

---

# Security

The API uses an API key for protected endpoints.

The API key is supplied using:

```text
X-API-Key
```

Requests without a valid API key are rejected.

Example:

```text
401 Unauthorized
```

The API key is configured through environment variables rather than hardcoded in the application.

---

# Logging and Request Tracking

The application includes request logging and request IDs.

Each request receives a unique request ID.

The request ID is:

- Added to application logs
- Returned in prediction responses
- Added to the `X-Request-ID` response header

This helps with debugging and tracing individual API requests.

---

# Monitoring

The application uses **Prometheus FastAPI Instrumentator** to expose API metrics.

The `/metrics` endpoint provides monitoring information.

The application also records ML prediction counts using the custom metric:

```text
ml_predictions_total
```

The metric is labelled by the predicted class.

Example:

```text
ml_predictions_total{predicted_class="setosa"} 1.0
```

---

# Docker Setup

## Prerequisites

Install and start:

- Docker Desktop
- Git

Make sure Docker Desktop is running before starting the application.

---

## Run with Docker Compose

From the project root directory:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Metrics:

```text
http://localhost:8000/metrics
```

---

## Run Without Rebuilding

If the Docker image has already been built:

```bash
docker compose up
```

---

## Stop the Application

Press:

```text
Ctrl + C
```

To stop and remove the containers and network:

```bash
docker compose down
```

---

# Environment Variables

The application uses environment variables for configuration.

A sample configuration is provided in:

```text
.env.example
```

The actual `.env` file should not be committed to GitHub.

Important configuration includes:

```text
API_KEY
MODEL_PATH
MODEL_METADATA_PATH
MODEL_VERSION
MAX_BATCH_SIZE
LOG_LEVEL
API_TITLE
```

---

# Model Storage

The trained model is stored in:

```text
ml/saved_model/model.joblib
```

Model metadata is stored in:

```text
ml/saved_model/model_metadata.json
```

Docker Compose mounts the saved model directory into the container so the model can be updated without rebuilding the complete Docker image.

---

# Testing

The project uses **Pytest** for automated testing.

Test coverage includes:

- Health endpoint
- Model information endpoint
- Single prediction
- Batch prediction
- API key security
- V1 and V2 API responses

Run the complete test suite with:

```bash
pytest
```

A successful test run should report all tests passing.

---

# Integration and Load Testing

The project also includes testing support for integration and load testing.

Load testing script:

```text
load_test.py
```

Detailed testing information is available in:

```text
TESTING.md
```

---

# Independent Extension – GitHub Actions

As the independent extension for the project, a **GitHub Actions automated testing workflow** is included.

The workflow runs the Pytest test suite automatically when changes are pushed to GitHub.

This provides continuous testing and helps identify issues before changes are considered complete.

Workflow location:

```text
.github/workflows/
```

---

# Running the ML Training Script

The model can be trained using:

```bash
python ml/train.py
```

The training process saves the trained model and model metadata in:

```text
ml/saved_model/
```

The prediction utility is available through:

```text
ml/predict.py
```

---

# What I Learned

During this project, I learned how to:

- Build REST APIs using FastAPI.
- Integrate a machine learning model into an API.
- Validate API input using Pydantic.
- Load and use a trained Scikit-learn model with Joblib.
- Implement single and batch prediction APIs.
- Design API versioning using V1 and V2 routers.
- Protect API endpoints using API key authentication.
- Add request IDs and structured logging.
- Monitor a FastAPI application using Prometheus metrics.
- Containerize a Python application using Docker.
- Run the application using Docker Compose.
- Write automated tests using Pytest.
- Perform integration and load testing.
- Manage the project using Git and GitHub.
- Automate testing with GitHub Actions.
- Organize a machine learning API project for deployment.

---

# Project Completion Checklist

- [x] FastAPI REST API
- [x] Iris ML model integration
- [x] Random Forest classification
- [x] V1 prediction API
- [x] V2 prediction API with probabilities
- [x] Batch prediction
- [x] API key security
- [x] Request logging
- [x] Request ID tracking
- [x] Prometheus monitoring
- [x] Dockerfile
- [x] Docker Compose
- [x] Automated Pytest tests
- [x] Integration/load testing
- [x] Clean requirements file
- [x] Complete project documentation
- [x] GitHub Actions testing workflow

---

# Project Status

The project is structured as a containerized and monitored machine learning REST API.

The application can be built and run locally using Docker Compose, tested using Pytest, monitored through Prometheus metrics, and maintained through GitHub version control and automated testing.

---