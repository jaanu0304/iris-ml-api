# Task 19 – Integration Testing, Load Testing, and Bug Fixing

## Objective

The objective of Task 19 was to perform end-to-end integration testing of the containerized Iris ML API, conduct a basic load test, identify and fix issues, and document the testing results.

## 1. Integration Testing

The API was tested while running inside the Docker container using the following endpoints.

### Health Check

**Endpoint:**
`GET /api/v1/health`

**Result:** PASS

The endpoint returned HTTP 200 and confirmed that the API was running and the ML model was loaded successfully.

### Single Prediction

**Endpoint:**
`POST /api/v1/predict`

**Result:** PASS

A valid Iris input was submitted with the required `X-API-Key`.

The API returned:
- Prediction: `setosa`
- Confidence: `1.0`
- Model version: `1.0`
- Request ID: Generated successfully

### Batch Prediction

**Endpoint:**
`POST /api/v1/predict-batch`

**Result:** PASS

Multiple Iris inputs were submitted successfully using the `inputs` field.

The API returned predictions and confidence values for all valid inputs.

### Prometheus Metrics

**Endpoint:**
`GET /metrics`

**Result:** PASS

The endpoint returned Prometheus metrics successfully with HTTP 200.

## 2. Load Testing

A basic asynchronous load test was performed using Python, `asyncio`, and `httpx`.

### Test Configuration

- Target endpoint: `POST /api/v1/predict`
- Total requests: 50
- Concurrent requests: 50
- API authentication: `X-API-Key`
- Docker container: Running successfully

### Load Test Results

| Metric | Result |
|---|---:|
| Total Requests | 50 |
| Successful Requests | 50 |
| Failed Requests | 0 |
| Total Time | 9.81 seconds |
| Requests/Second | 5.10 |

**Result:** PASS

All 50 requests were processed successfully with zero failed requests.

## 3. Bugs Identified and Fixed

### Bug 1 – Missing Prometheus Dependency

During the initial Docker integration test, the API container failed to start with:

`ModuleNotFoundError: No module named 'prometheus_fastapi_instrumentator'`

**Fix:**

Added the required Prometheus FastAPI Instrumentator package to `requirements.txt`.

### Bug 2 – Dependency Version Conflict

The initial package version caused a dependency conflict with the project's Starlette version.

The Prometheus FastAPI Instrumentator version was updated to:

`prometheus-fastapi-instrumentator==8.1.0`

The Docker image was then rebuilt successfully and the API container started correctly.

### Bug 3 – Load Test Timeout Handling

The initial load test stopped with an `httpx.ReadTimeout` exception when processing concurrent requests.

**Fix:**

- Added a 30-second HTTP client timeout.
- Added exception handling in the load test request function.
- Re-ran the load test successfully.

Final result:

- 50 successful requests
- 0 failed requests

## 4. Validation Summary

| Test | Result |
|---|---|
| Docker Container Startup | PASS |
| API Health Check | PASS |
| Single Prediction | PASS |
| Batch Prediction | PASS |
| Prometheus Metrics | PASS |
| 50 Concurrent Requests | PASS |
| Bug Fix Verification | PASS |

## 5. Conclusion

Task 19 testing confirmed that the containerized Iris ML API is functioning correctly under the tested integration and basic load-testing conditions. The identified dependency and load-testing issues were fixed and verified successfully.