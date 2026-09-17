from fastapi import APIRouter, HTTPException, Request, Depends
import numpy as np
import time

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)
from app.config import settings


from app.security import verify_api_key
from prometheus_client import Counter

prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of successful ML predictions",
    ["predicted_class"]
)

router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]
)


@router.get("/health")
def health(request: Request):
    from app.main import model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.get("/model-info")
def model_info(request: Request):
    import json

    metadata_path = settings.MODEL_METADATA_PATH

    try:
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        logger = request.app.state.logger

        logger.info(
            f"request_id={request.state.request_id} "
            f"model_info_success=true"
        )

        return metadata

    except Exception as e:
        logger = request.app.state.logger

        logger.error(
            f"request_id={request.state.request_id} "
            f"model_info_failed=true "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load model metadata"
        )

@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):
    from app.main import model

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
        logger = request.app.state.logger

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
    logger = request.app.state.logger

    logger.info(
        f"request_id={request_id} "
        f"prediction_success=true "
        f"prediction={prediction}"
    )

    prediction_counter.labels(predicted_class=str(prediction)).inc()

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "model_version": settings.MODEL_VERSION,
        "request_id": request_id
    }


@router.post("/predict-batch", response_model=PredictionBatchOutput)
def predict_batch(data: PredictionBatchInput, request: Request):
    from app.main import model

    logger = request.app.state.logger
    request_id = request.state.request_id

    start_time = time.perf_counter()

    batch_size = len(data.inputs)

    if batch_size > settings.MAX_BATCH_SIZE:
        logger.warning(
            f"request_id={request_id} "
            f"batch_prediction_rejected=true "
            f"batch_size={batch_size} "
            f"max_batch_size={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Batch size cannot exceed {settings.MAX_BATCH_SIZE}"
        )

    logger.info(
        f"request_id={request_id} "
        f"batch_prediction_started=true "
        f"batch_size={batch_size}"
    )

    features = np.array([
        [
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width
        ]
        for item in data.inputs
    ])

    try:
        # Predict the entire batch in one model call
        predictions = model.predict(features)

        # Get confidence scores for the entire batch
        probabilities = model.predict_proba(features)
        confidences = probabilities.max(axis=1)

    except Exception as e:
        duration = time.perf_counter() - start_time

        logger.error(
            f"request_id={request_id} "
            f"batch_prediction_failed=true "
            f"batch_size={batch_size} "
            f"duration={duration:.4f} "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )

    results = []

    for prediction, confidence in zip(predictions, confidences):
        results.append(
            PredictionOutput(
                prediction=str(prediction),
                confidence=float(confidence),
                model_version=settings.MODEL_VERSION,
                request_id=request_id
            )
        )

    duration = time.perf_counter() - start_time

    logger.info(
        f"request_id={request_id} "
        f"batch_prediction_success=true "
        f"batch_size={batch_size} "
        f"duration={duration:.4f}"
    )

    return {
        "predictions": results
    }


