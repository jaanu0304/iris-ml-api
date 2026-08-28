from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib
import numpy as np
import uuid
import time
from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logging

model = None
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")
    logger.info("ML model loaded successfully")

    yield


app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"duration={duration:.4f}s"
    )

    response.headers["X-Request-ID"] = request_id

    return response

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal value error occurred"
        }
    )


@app.get("/")
def root():
    return {"message": "ML API is alive"}



@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }





@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    try:
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
    except Exception as e:
        logger.error(
            f"request_id={request.state.request_id} "
            f"prediction_failed=true "
            f"error={e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = request.state.request_id

    logger.info(
        f"request_id={request_id} "
        f"prediction_success=true "
        f"prediction={prediction}"
    )

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "model_version": "1.0",
        "request_id": request_id
    }